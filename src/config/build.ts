import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import * as net from "node:net";
import type { AddressInfo } from "node:net";
import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";
import { DEFAULT_HOST, DEFAULT_LOG_DIR, DEFAULT_LLAMA_BIN, DEFAULT_LLAMA_BIN_WIN32, DEFAULT_MODEL_LOAD_POLL_INTERVAL_MS, DEFAULT_MODEL_LOAD_POLL_TIMEOUT_MS, DEFAULT_PORT, DEFAULT_ROUTER_POLL_INTERVAL_MS, DEFAULT_ROUTER_POLL_TIMEOUT_MS, DEFAULT_ROUTER_SHUTDOWN_GRACE_MS, DEFAULT_SLEEP_IDLE_SECONDS, DEFAULT_SLOT_SAVE_DIR } from "./defaults.js";
import { RESERVED_VARIANT_NAMES, STRATEGY_IDS, hasMmproj, isMmprojOnCPU, isSpecEnabled, type ConfigSource, type Strategy, type ManagerConfig, type ModelConfig, type ModelState, type ModelVariant, type RawModel, type StrategyId } from "./types.js";
import type { ReloadParams, SpeculativeType } from "../client/types.js";
import { DEFAULT_KV_PRECISION, MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { maybeReject, normalizeFlag } from "./flags.js";
import { parseLadderStep, parseRouterConfig, validateVariantPath, type ParsedArgs } from "./parser.js";

const log: ConsolaInstance = logger.withTag('config');

const CTX_KEY = "ctx-size";

const MANAGER_FIELDS = new Set([
    "ladder",
    "ctx-size",
    "c",
    "cache-type-k",
    "ctk",
    "cache-type-v",
    "ctv",
    "slot-save-path",
    "model-variants",
    "model-variant",
]);

const SUPPORTED_KV_PRECISION = ['f16', 'q8_0', 'q4_0'];
const PRECISION_RANK: Record<string, number> = { 'q4_0': 0, 'q8_0': 1, 'f16': 2 };

type SpecDraft = NonNullable<ReloadParams['spec']>['draft'];

type InterpretedEntry = {
    entry: RawModel;
    aliases: string[];
    variants: Record<string, string>;
    cache_floor: 'f16' | 'q8_0' | 'q4_0';
};

export async function buildConfig(
    parsed: ParsedArgs, 
    project_root: string, 
    preset_out_path: string, 
    default_config_path: string
): Promise<ManagerConfig> {
    const source = buildSource(parsed, default_config_path);
    
    const raw_models = source.raw_models;
    if (Object.keys(raw_models).length === 0) {
        throw new Error(`no models found in config`);
    }

    const interpreted = new Map<string, InterpretedEntry>();
    for (const [id, raw] of Object.entries(raw_models)) {
        interpreted.set(id, interpretEntry(id, raw));
    }

    for (const [id, interp] of interpreted) {
        const has_variants = Object.keys(interp.variants).length > 0;
        const explicit = 'ladder' in interp.entry
            ? interp.entry['ladder'] as Strategy[]
            : source.ladder_override;

        if (has_variants && explicit === undefined) {
            throw new Error(`model '${id}' declares model-variants and requires an explicit ladder with swap-model steps ('ladder' in the model config or --ladder)`);
        }

        const steps = structuredClone(explicit ?? deriveLadder(interp.entry, interp.cache_floor));

        if (has_variants) {
            validateVariantLadder(id, steps, interp.variants, buildInitialState(id, interp.entry));
        } else {
            for (let i = 0; i < steps.length; i++) {
                const step = steps[i];
                if (step.kind === 'swap-model') {
                    throw new Error(`model '${id}': ladder step ${i} switches to variant '${step.variant}' but the model declares no model-variants`);
                }
            }
        }

        interp.entry['ladder'] = steps;
    }

    validateAliases(interpreted);

    const model_variants = resolveVariants(interpreted);

    const models: Record<string, ModelConfig> = {};
    for (const [id, interp] of interpreted) {
        models[id] = buildEntry(id, interp.entry, interp.aliases, model_variants[id]);
    }

    const model_keys = Object.keys(models);
    const default_model = source.default_model ?? model_keys[0];
    if (!models[default_model]) {
        throw new Error(`default-model '${default_model}' not found in models`);
    }

    const raw_router = source.raw_router;
    const llama_log_dir = resolve(project_root, normalizeDir((raw_router['llama-log-dir'] as string) ?? DEFAULT_LOG_DIR));
    const slot_save_path = resolve(project_root, normalizeDir((raw_router['slot-save-path'] as string) ?? DEFAULT_SLOT_SAVE_DIR));

    // resolve llama-server binary. tiered resolution: [--bin flag, LLAMA_BIN env, config.yaml, default location]
    const default_bin = process.platform === 'win32' ? DEFAULT_LLAMA_BIN_WIN32 : DEFAULT_LLAMA_BIN;
    const bin = resolve(project_root, source.bin_override ?? process.env.LLAMA_BIN ?? (raw_router['bin'] as string) ?? default_bin);
    if (!existsSync(bin)) {
        throw new Error(`llama-server binary not found at '${bin}'. build it with scripts/build-llamacpp.sh, or supply one via --bin, the LLAMA_BIN env var, or 'router.bin' in the config`);
    }

    const router_listen = raw_router['listen'] as string | undefined;
    const listen = router_listen ?? `${DEFAULT_HOST}:${await findFreePort()}`;

    const raw_model_load = source.raw_model_load;
    const config: ManagerConfig = {
        mode: source.mode,
        router: {
            bin,
            llama_log_dir,
            slot_save_path,
            listen,
            poll_interval_ms: (raw_router['poll-interval-ms'] as number) ?? DEFAULT_ROUTER_POLL_INTERVAL_MS,
            poll_timeout_ms: (raw_router['poll-timeout-ms'] as number) ?? DEFAULT_ROUTER_POLL_TIMEOUT_MS,
            shutdown_grace_period_ms: (raw_router['shutdown-grace-period-ms'] as number) ?? DEFAULT_ROUTER_SHUTDOWN_GRACE_MS,
        },
        host: source.host ?? DEFAULT_HOST,
        port: source.port ?? DEFAULT_PORT,
        sleep_idle_seconds: source.sleep_idle_seconds ?? DEFAULT_SLEEP_IDLE_SECONDS,
        model_load: {
            poll_interval_ms: (raw_model_load['poll-interval-ms'] as number) ?? DEFAULT_MODEL_LOAD_POLL_INTERVAL_MS,
            poll_timeout_ms: (raw_model_load['poll-timeout-ms'] as number) ?? DEFAULT_MODEL_LOAD_POLL_TIMEOUT_MS,
        },
        models,
        default_model,
    };

    mkdirSync(config.router.llama_log_dir, { recursive: true });
    mkdirSync(config.router.slot_save_path, { recursive: true });

    // write generated router preset to file
    const preset_models: Record<string, PresetModel> = {};
    for (const [id, interp] of interpreted) {
        preset_models[id] = { entry: interp.entry, variants: model_variants[id] };
    }
    writeFileSync(preset_out_path, buildIni(preset_models, config.router.slot_save_path), "utf8");
    return config;
}

// assemble the ConfigSource for both modes: parsed CLI flags override values from the
// config file, which fall back to defaults applied later in buildConfig
function buildSource(parsed: ParsedArgs, default_config_path: string): ConfigSource {
    const config_path = parsed.config_path ?? process.env.MANAGER_CONFIG;

    let source: ConfigSource;
    if (parsed.mode === 'server') {
        let cfg: ConfigSource | undefined;
        if (config_path !== undefined) {
            cfg = parseRouterConfig(config_path);
            const ignored = Object.keys(cfg.raw_models);
            if (ignored.length > 0) {
                log.error(`server mode: ignoring ${ignored.length} model(s) from config '${config_path}': ${ignored.join(', ')}`);
            }
        }

        source = {
            mode: 'server',
            raw_models: parsed.raw_models ?? {},
            raw_router: cfg?.raw_router ?? {},
            raw_model_load: cfg?.raw_model_load ?? {},
            host: parsed.host ?? cfg?.host,
            port: parsed.port ?? cfg?.port,
            sleep_idle_seconds: parsed.sleep_idle_seconds ?? cfg?.sleep_idle_seconds,
            default_model: Object.keys(parsed.raw_models ?? {})[0],
            ladder_override: parsed.ladder,
            bin_override: parsed.bin,
        };
    } else {
        const cfg = parseRouterConfig(config_path ?? default_config_path);
        source = {
            ...cfg,
            host: parsed.host ?? cfg.host,
            port: parsed.port ?? cfg.port,
            sleep_idle_seconds: parsed.sleep_idle_seconds ?? cfg.sleep_idle_seconds,
            ladder_override: parsed.ladder ?? cfg.ladder_override,
            bin_override: parsed.bin ?? cfg.bin_override,
        };
    }

    if (parsed.slot_save_path !== undefined) {
        source.raw_router['slot-save-path'] = parsed.slot_save_path;
    }

    return source;
}

// canonicalizes config keys, rejects unsupported fields, warns about manager-owned fields, etc
// unrecognized keys pass through as llama.cpp args
function interpretEntry(name: string, raw: RawModel): InterpretedEntry {
    const entry: RawModel = {};
    const aliases: string[] = [];
    const variants: Record<string, string> = {};

    let cache_type_k: string | undefined;
    let cache_type_v: string | undefined;

    for (const [raw_key, value] of Object.entries(raw)) {
        const flag = normalizeFlag(raw_key);

        const reason = maybeReject(flag);
        if (reason) throw new Error(reason);

        switch (flag) {
            case 'alias':
                aliases.push(...aliasValues(value));
                entry['alias'] = value;
                break;
            case 'ladder': {
                const tokens = Array.isArray(value)
                    ? value.map(v => String(v))
                    : String(value).split(',');
                entry['ladder'] = tokens
                    .map(s => s.trim())
                    .filter(s => s !== '')
                    .map(parseLadderStep);
                break;
            }
            case 'model-variants':
                parseModelVariants(name, value, variants);
                break;
            case 'ctx-size':
                log.error(`'ctx-size' is ignored; context is sized reactively. run with --calc-breakpoints to see achievable context sizes`);
                break;
            case 'cache-type-k':
            case 'cache-type-v':
                if (typeof value !== 'string' || !SUPPORTED_KV_PRECISION.includes(value)) {
                    throw new Error(`unsupported cache type '${value}' for '${flag}' (supported: ${SUPPORTED_KV_PRECISION.join(', ')})`);
                }
                if (flag === 'cache-type-k') cache_type_k = value;
                else cache_type_v = value;
                break;
            case 'n-gpu-layers':
                if (typeof value === 'number' && value < 99) {
                    log.error(`'n-gpu-layers' < 99 implies CPU-offloaded layers; the memory planner does not model them`);
                }
                entry['n-gpu-layers'] = value;
                break;
            case 'api-key':
            case 'api-key-file':
                log.error(`'${flag}' is ignored; the manager does not support auth`);
                break;
            case 'device':
                log.error(`'device' with CPU devices is not supported by the memory planner`);
                entry['device'] = value;
                break;
            case 'fa':
                if (isTruthy(value)) {
                    entry['fa'] = value;
                } else {
                    log.error(`'fa' is ignored; flash attention cannot be disabled (required for kv quantization)`);
                }
                break;
            case 'no-fa':
                log.error(`'no-fa' is ignored; flash attention cannot be disabled (required for kv quantization)`);
                break;
            case 'mmproj-device':
            case 'no-mmproj-offload':
                log.error(`'${flag}' is ignored; mmproj placement is managed via the ladder`);
                break;
            case 'webui':
            case 'no-webui':
                log.error(`the webui may not be compatible with the manager`);
                entry[flag] = value;
                break;
            case 'no-op-offload':
            case 'no-kv-offload':
            case 'swa-full':
            case 'n-cpu-moe':
            case 'n-cpu-moe-draft':
            case 'override-tensor':
            case 'override-tensor-draft':
            case 'lora':
                log.error(`'${flag}' implies CPU-resident components the memory planner does not model`);
                entry[flag] = value;
                break;
            default:
                entry[flag] = value;
        }
    }

    if (Object.keys(variants).length > 0) {
        const baseline = entry['model'] ?? entry['model-url'];
        if (typeof baseline !== 'string' || baseline === '') {
            throw new Error(`model '${name}': model-variants requires a baseline model source ('model' or 'model-url')`);
        }
    }

    return { entry, aliases, variants, cache_floor: precisionFloor(cache_type_k, cache_type_v) };
}

function parseModelVariants(name: string, value: unknown, out: Record<string, string>): void {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error(`model '${name}': 'model-variants' must be a map of variant name to local GGUF path`);
    }

    for (const [variant, path] of Object.entries(value as Record<string, unknown>)) {
        if (variant.trim() === '' || /\s/.test(variant)) {
            throw new Error(`model '${name}': invalid model-variant name '${variant}'`);
        }
        if (RESERVED_VARIANT_NAMES.has(variant)) {
            throw new Error(`model '${name}': model-variant name '${variant}' is reserved`);
        }
        validateVariantPath(path, `model '${name}': model-variant '${variant}'`);
        out[variant] = path;
    }

    if (Object.keys(out).length === 0) {
        throw new Error(`model '${name}': 'model-variants' must not be empty`);
    }
}

function precisionFloor(ctk?: string, ctv?: string): 'f16' | 'q8_0' | 'q4_0' {
    if (ctk === undefined && ctv === undefined) return 'q4_0';

    const declared = [ctk, ctv].filter((v): v is string => v !== undefined);
    const floor = declared.reduce((a, b) => PRECISION_RANK[b] > PRECISION_RANK[a] ? b : a);

    if (ctk !== undefined && ctv !== undefined && ctk !== ctv) {
        log.error(`mixed k/v cache precision (cache-type-k: ${ctk}, cache-type-v: ${ctv}); treating both as ${floor}`);
    }

    return floor as 'f16' | 'q8_0' | 'q4_0';
}

function deriveLadder(entry: RawModel, floor: 'f16' | 'q8_0' | 'q4_0'): Strategy[] {
    const ladder: StrategyId[] = [];
    if (getHasSpec(entry)) ladder.push('disable-spec');
    if (getHasMmproj(entry)) ladder.push('mmproj-to-cpu');

    if (floor === 'q8_0') {
        ladder.push('quantize-kv-q8');
    } else if (floor === 'q4_0') {
        ladder.push('quantize-kv-q8');
        ladder.push('quantize-kv-q4');
    }

    return ladder.map(id => ({ kind: 'reload-model', id }));
}

function validateVariantLadder(
    name: string,
    steps: Strategy[],
    variants: Record<string, string>,
    initial: ModelState,
): void {
    let state = initial;
    const switched = new Set<string>();

    steps.forEach((step, i) => {
        if (step.kind === 'swap-model') {
            if (!(step.variant in variants)) {
                throw new Error(`model '${name}': ladder step ${i} switches to unknown variant '${step.variant}' (declared: ${Object.keys(variants).join(', ')})`);
            }
            if (switched.has(step.variant)) {
                throw new Error(`model '${name}': ladder step ${i} switches to variant '${step.variant}' more than once`);
            }
            switched.add(step.variant);
            step.router_id = variantRouterId(name, step.variant);
            state = { ...state, model_variant: step.router_id, cache_type_k: 'f16', cache_type_v: 'f16' };
            return;
        }

        if (!strategyApplicable(step.id, state)) {
            throw new Error(`model '${name}': ladder step ${i} ('${step.id}') is not applicable at that point in the ladder`);
        }
        state = applyStrategy(step.id, state);
    });

    for (const variant of Object.keys(variants)) {
        if (!switched.has(variant)) {
            throw new Error(`model '${name}': declared variant '${variant}' is not used by the ladder`);
        }
    }
}

function strategyApplicable(id: StrategyId, state: ModelState): boolean {
    switch (id) {
        case 'disable-spec':
            return isSpecEnabled(state);
        case 'mmproj-to-cpu':
            return hasMmproj(state) && !isMmprojOnCPU(state);
        case 'quantize-kv-q8':
            return state.cache_type_k === 'f16' && state.cache_type_v === 'f16';
        case 'quantize-kv-q4':
            return (state.cache_type_k === 'f16' || state.cache_type_k === 'q8_0')
                && (state.cache_type_v === 'f16' || state.cache_type_v === 'q8_0');
    }
}

function applyStrategy(id: StrategyId, state: ModelState): ModelState {
    const next = structuredClone(state);
    switch (id) {
        case 'disable-spec':
            next.spec = { types: ['none'] };
            break;
        case 'mmproj-to-cpu':
            next.mmproj = { ...next.mmproj, mmproj_offload: false };
            break;
        case 'quantize-kv-q8':
            next.cache_type_k = 'q8_0';
            next.cache_type_v = 'q8_0';
            break;
        case 'quantize-kv-q4':
            next.cache_type_k = 'q4_0';
            next.cache_type_v = 'q4_0';
            break;
    }
    return next;
}

function validateAliases(interpreted: Map<string, InterpretedEntry>): void {
    const ids = new Set(interpreted.keys());
    const owner = new Map<string, string>();

    for (const [id, interp] of interpreted) {
        for (const alias of interp.aliases) {
            if (alias === id) {
                throw new Error(`alias '${alias}' conflicts with the model id`);
            }
            if (ids.has(alias)) {
                throw new Error(`alias '${alias}' conflicts with model id '${alias}'`);
            }
            const prev = owner.get(alias);
            if (prev !== undefined) {
                if (prev === id) {
                    throw new Error(`alias '${alias}' is declared multiple times by model '${id}'`);
                }
                throw new Error(`alias '${alias}' is declared by both '${prev}' and '${id}'`);
            }
            owner.set(alias, id);
        }
    }
}

function aliasValues(value: unknown): string[] {
    if (Array.isArray(value)) return value.map(String);
    return [String(value)];
}

function buildEntry(name: string, raw: RawModel, aliases: string[], variants: Record<string, ModelVariant>): ModelConfig {
    return {
        name,
        aliases,
        variants,
        ladder: (raw['ladder'] as Strategy[]) ?? [],
        initial_state: buildInitialState(name, raw),
    };
}

function buildInitialState(name: string, raw: RawModel): ModelState {
    const state: ModelState = {
        kv_unified: getKvUnified(raw),
        cache_type_k: DEFAULT_KV_PRECISION,
        cache_type_v: DEFAULT_KV_PRECISION,
        model_variant: name,
    };

    if (getHasSpec(raw)) {
        state.spec = {
            types: getSpecTypes(raw),
            draft: getSpecDraft(raw),
        };
    }

    if (getHasMmproj(raw)) {
        state.mmproj = { 
            path: getMmprojPath(raw),
            mmproj_offload: true 
        };
    }

    return state;
}

function resolveVariants(interpreted: Map<string, InterpretedEntry>): Record<string, Record<string, ModelVariant>> {
    const public_names = new Set<string>();
    for (const [id, interp] of interpreted) {
        public_names.add(id);
        for (const alias of interp.aliases) {
            public_names.add(alias);
        }
    }

    const out: Record<string, Record<string, ModelVariant>> = {};
    const private_ids = new Set<string>();

    for (const [id, interp] of interpreted) {
        const map: Record<string, ModelVariant> = {};
        for (const [variant, path] of Object.entries(interp.variants)) {
            const router_id = variantRouterId(id, variant);
            if (public_names.has(router_id)) {
                throw new Error(`generated router id '${router_id}' (model '${id}', variant '${variant}') collides with a public model id or alias`);
            }
            if (private_ids.has(router_id)) {
                throw new Error(`generated router id '${router_id}' (model '${id}', variant '${variant}') collides with another generated router id`);
            }
            private_ids.add(router_id);
            map[variant] = { router_id, path };
        }
        out[id] = map;
    }

    return out;
}

function variantRouterId(model: string, variant: string): string {
    return `${model}__${variant}`;
}

type PresetModel = {
    entry: RawModel;
    variants: Record<string, ModelVariant>;
};

function buildIni(models: Record<string, PresetModel>, slot_save_path: string): string {
    const sections: string[] = [];

    for (const [name, model] of Object.entries(models)) {
        sections.push(presetSection(name, model.entry, slot_save_path));
        for (const variant of Object.values(model.variants)) {
            sections.push(presetSection(variant.router_id, variantSection(model.entry, variant.path), slot_save_path));
        }
    }

    return sections.join("\n\n") + "\n";
}

function presetSection(id: string, entry: RawModel, slot_save_path: string): string {
    const lines = [
        `[${id}]`,
        `${CTX_KEY} = ${MIN_ALLOWED_CTX}`,
        `cache-type-k = ${DEFAULT_KV_PRECISION}`,
        `cache-type-v = ${DEFAULT_KV_PRECISION}`,
        `slot-save-path = ${slot_save_path}`,
    ];

    for (const [key, value] of Object.entries(entry)) {
        if (MANAGER_FIELDS.has(key)) continue;
        lines.push(`${key} = ${iniValue(value)}`);
    }

    return lines.join("\n");
}

function variantSection(entry: RawModel, path: string): RawModel {
    const out: RawModel = { model: path };
    for (const [key, value] of Object.entries(entry)) {
        if (key === 'model' || key === 'model-url' || key === 'alias') continue;
        out[key] = value;
    }
    return out;
}

function iniValue(v: unknown): string {
    if (typeof v === "string") return v;
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    if (Array.isArray(v)) return v.map(String).join(", ");
    return JSON.stringify(v);
}

function normalizeDir(p: string): string {
    return p.endsWith('/') ? p : p + '/';
}

export function getSpecTypes(entry: RawModel): SpeculativeType[] {
    const spec = entry['spec-type'];
    if (typeof spec === 'string') {
        return spec.split(',').map(t => t
            .trim()
            .toLowerCase())
            .filter(t => t !== '');
    }
    if (Array.isArray(spec)) {
        return spec.map(t => String(t)
            .trim()
            .toLowerCase())
            .filter(t => t !== '');
    }
    return [];
}

export function getHasSpec(entry: RawModel): boolean {
    const types = getSpecTypes(entry);
    return types.length > 0 && !types.includes('none');
}

function getSpecDraft(entry: RawModel): SpecDraft | undefined {
    const draft: SpecDraft = {};

    const path = entry['model-draft'] ?? entry['md'] ?? entry['spec-draft-model'];
    if (typeof path === 'string' && path.length > 0) draft.path = path;

    const n_max = entry['spec-draft-n-max'];
    if (typeof n_max === 'number') draft.n_max = n_max;

    const n_min = entry['spec-draft-n-min'];
    if (typeof n_min === 'number') draft.n_min = n_min;

    const p_split = entry['spec-draft-p-split'] ?? entry['draft-p-split'];
    if (typeof p_split === 'number') draft.p_split = p_split;

    const p_min = entry['spec-draft-p-min'] ?? entry['draft-p-min'];
    if (typeof p_min === 'number') draft.p_min = p_min;

    const ngl = entry['spec-draft-ngl'] ?? entry['ngld'] ?? entry['n-gpu-layers-draft'];
    if (typeof ngl === 'number') draft.n_gpu_layers = ngl;

    return Object.keys(draft).length > 0 ? draft : undefined;
}

function getMmprojPath(entry: RawModel): string | undefined {
    const path = entry['mmproj'] ?? entry['mm'];
    if (typeof path === 'string' && path.length > 0) return path;

    const url = entry['mmproj-url'] ?? entry['mmu'];
    if (typeof url === 'string' && url.length > 0) return url;

    return undefined;
}

export function getHasMmproj(entry: RawModel): boolean {
    // TODO: assumes mmproj is on GPU
    if (isTruthy(entry['no-mmproj']) || isTruthy(entry['no-mmproj-auto'])) {
        return false;
    }

    const path = entry['mmproj'] ?? entry['mm'];
    const url = entry['mmproj-url'] ?? entry['mmu'];
    const auto = entry['mmproj-auto'];

    return (typeof path === 'string' && path.length > 0)
        || (typeof url === 'string' && url.length > 0)
        || isTruthy(auto);
}

export function getKvUnified(entry: RawModel): boolean {
    const pos = entry['kv-unified'] ?? entry['kvu'];
    const neg = entry['no-kv-unified'] ?? entry['no-kvu'];

    if (pos !== undefined) return isTruthy(pos);
    if (neg !== undefined) return !isTruthy(neg);

    const par = entry['parallel'] ?? entry['np'];
    if (typeof par === 'number') return par === -1;

    return true;
}

function isTruthy(v: unknown): boolean {
    if (typeof v === 'boolean') return v;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') return ['on', 'enabled', 'true', '1'].includes(v.toLowerCase());
    return false;
}

function findFreePort(): Promise<number> {
    return new Promise((resolve, reject) => {
        const server = net.createServer();
        server.once('error', reject);
        server.listen(0, DEFAULT_HOST, () => {
            const addr = server.address() as AddressInfo;
            server.close(() => resolve(addr.port));
        });
    });
}