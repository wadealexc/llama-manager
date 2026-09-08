import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import * as net from "node:net";
import type { AddressInfo } from "node:net";
import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";
import { canonicalKey, rejectedKeyReason } from "./parser.js";
import { type ConfigSource, type ManagerConfig, type Mode, type ModelConfig, type ModelState, type RawModel, type StrategyId } from "./types.js";
import { DEFAULT_KV_PRECISION, MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";

const log: ConsolaInstance = logger.withTag('config');

const DEFAULT_LOG_DIR = './logs/';
const DEFAULT_SLOT_SAVE_DIR = './slots/';

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
]);

const SUPPORTED_KV_PRECISION = ['f16', 'q8_0', 'q4_0'];
const PRECISION_RANK: Record<string, number> = { 'q4_0': 0, 'q8_0': 1, 'f16': 2 };

type InterpretedEntry = {
    entry: RawModel;
    aliases: string[];
    cache_floor: 'f16' | 'q8_0' | 'q4_0';
};

export async function buildConfig(source: ConfigSource, project_root: string, preset_out_path: string): Promise<ManagerConfig> {
    const raw_models = source.raw_models;
    if (Object.keys(raw_models).length === 0) {
        throw new Error(`no models found in config`);
    }

    const interpreted = new Map<string, InterpretedEntry>();
    for (const [id, raw] of Object.entries(raw_models)) {
        interpreted.set(id, interpretEntry(raw, source.mode));
    }

    if (source.mode === 'server') {
        const interp = interpreted.values().next().value!;
        interp.entry['ladder'] = source.ladder_override ?? deriveLadder(interp.entry, interp.cache_floor);
    }

    validateAliases(interpreted);

    const models: Record<string, ModelConfig> = {};
    for (const [id, interp] of interpreted) {
        models[id] = buildEntry(id, interp.entry, interp.aliases);
    }

    const model_keys = Object.keys(models);
    const default_models = (source.default_models && source.default_models.length > 0)
        ? source.default_models
        : [model_keys[0]];

    for (const id of default_models) {
        if (!models[id]) {
            throw new Error(`default-model '${id}' not found in models`);
        }
    }

    const raw_router = source.raw_router;
    const llama_log_dir = resolve(project_root, normalizeDir((raw_router['llama-log-dir'] as string) ?? DEFAULT_LOG_DIR));
    const slot_save_path = resolve(project_root, normalizeDir((raw_router['slot-save-path'] as string) ?? DEFAULT_SLOT_SAVE_DIR));

    const router_listen = raw_router['listen'] as string | undefined;
    const listen = router_listen ?? `127.0.0.1:${await findFreePort()}`;

    const raw_model_load = source.raw_model_load;
    const config: ManagerConfig = {
        mode: source.mode,
        router: {
            bin: resolve(project_root, (raw_router['bin'] as string) ?? (process.platform === 'win32'
                ? './llama.cpp/build/bin/Release/llama-server.exe'
                : './llama.cpp/build/bin/llama-server')),
            llama_log_dir,
            slot_save_path,
            listen,
            poll_interval_ms: (raw_router['poll-interval-ms'] as number) ?? 50,
            poll_timeout_ms: (raw_router['poll-timeout-ms'] as number) ?? 10000,
            shutdown_grace_period_ms: (raw_router['shutdown-grace-period-ms'] as number) ?? 10000,
        },
        host: source.host,
        port: source.port,
        sleep_idle_seconds: source.sleep_idle_seconds,
        model_load: {
            poll_interval_ms: (raw_model_load['poll-interval-ms'] as number) ?? 100,
            poll_timeout_ms: (raw_model_load['poll-timeout-ms'] as number) ?? 50000,
        },
        models,
        default_models,
    };

    mkdirSync(config.router.llama_log_dir, { recursive: true });
    mkdirSync(config.router.slot_save_path, { recursive: true });

    // write generated router preset to file
    writeFileSync(preset_out_path, buildIni(rawModelsOf(interpreted), config.router.slot_save_path), "utf8");
    return config;
}

// canonicalizes config keys, rejects unsupported fields, warns about manager-owned fields, etc
// unrecognized keys pass through as llama.cpp args
function interpretEntry(raw: RawModel, mode: Mode): InterpretedEntry {
    const entry: RawModel = {};
    const aliases: string[] = [];

    let cache_type_k: string | undefined;
    let cache_type_v: string | undefined;

    for (const [raw_key, value] of Object.entries(raw)) {
        const key = canonicalKey(raw_key) ?? raw_key;

        const rejected = rejectedKeyReason(key);
        if (rejected) throw new Error(rejected);

        switch (key) {
            case 'alias':
                aliases.push(...aliasValues(value));
                entry['alias'] = value;
                break;
            case 'ctx-size':
                if (mode === 'server') {
                    log.error(`'ctx-size' is ignored; context is sized reactively. run with --show-breakpoints to see achievable context sizes`);
                } else {
                    log.error(`'ctx-size' is ignored; context is sized reactively`);
                }
                break;
            case 'cache-type-k':
            case 'cache-type-v':
                if (typeof value !== 'string' || !SUPPORTED_KV_PRECISION.includes(value)) {
                    throw new Error(`unsupported cache type '${value}' for '${key}' (supported: ${SUPPORTED_KV_PRECISION.join(', ')})`);
                }
                if (mode === 'router') {
                    log.error(`'${key}' is manager-controlled; the manager serves an f16 kvcache at baseline`);
                }
                if (key === 'cache-type-k') cache_type_k = value;
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
                log.error(`'${key}' is ignored; the manager does not support auth`);
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
                log.error(`'${key}' is ignored; mmproj placement is managed via the ladder`);
                break;
            case 'webui':
            case 'no-webui':
                log.error(`the webui may not be compatible with the manager`);
                entry[key] = value;
                break;
            case 'no-op-offload':
            case 'no-kv-offload':
            case 'swa-full':
            case 'n-cpu-moe':
            case 'n-cpu-moe-draft':
            case 'override-tensor':
            case 'override-tensor-draft':
            case 'lora':
                log.error(`'${key}' implies CPU-resident components the memory planner does not model`);
                entry[key] = value;
                break;
            default:
                entry[key] = value;
        }
    }

    return { entry, aliases, cache_floor: precisionFloor(cache_type_k, cache_type_v) };
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

function deriveLadder(entry: RawModel, floor: 'f16' | 'q8_0' | 'q4_0'): StrategyId[] {
    const ladder: StrategyId[] = [];
    if (getHasSpec(entry)) ladder.push('disable-spec');
    if (getHasMmproj(entry)) ladder.push('mmproj-to-cpu');

    if (floor === 'f16') return ladder;
    if (floor === 'q8_0') return [...ladder, 'quantize-kv-q8'];
    return [...ladder, 'quantize-kv-q8', 'quantize-kv-q4'];
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

function buildEntry(name: string, raw: RawModel, aliases: string[]): ModelConfig {
    const initial_state: ModelState = {
        mmproj_loaded: getHasMmproj(raw),
        spec_loaded: getHasSpec(raw),
        kv_unified: getKvUnified(raw),
        cache_type_k: DEFAULT_KV_PRECISION,
        cache_type_v: DEFAULT_KV_PRECISION,
    };

    return {
        name,
        aliases,
        ladder: (raw['ladder'] as StrategyId[]) ?? [],
        initial_state,
    };
}

function rawModelsOf(interpreted: Map<string, InterpretedEntry>): Record<string, RawModel> {
    const out: Record<string, RawModel> = {};
    for (const [id, interp] of interpreted) {
        out[id] = interp.entry;
    }
    return out;
}

function buildIni(raw: Record<string, RawModel>, slot_save_path: string): string {
    const sections: string[] = [];

    for (const [name, entry] of Object.entries(raw)) {
        const lines = [
            `[${name}]`,
            `${CTX_KEY} = ${MIN_ALLOWED_CTX}`,
            `cache-type-k = ${DEFAULT_KV_PRECISION}`,
            `cache-type-v = ${DEFAULT_KV_PRECISION}`,
            `slot-save-path = ${slot_save_path}`,
        ];

        for (const [key, value] of Object.entries(entry)) {
            if (MANAGER_FIELDS.has(key)) continue;
            lines.push(`${key} = ${iniValue(value)}`);
        }
        sections.push(lines.join("\n"));
    }

    return sections.join("\n\n") + "\n";
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

export function getHasSpec(entry: RawModel): boolean {
    const spec = entry['spec-type'];
    let types: string[];
    if (typeof spec === 'string') {
        types = spec.split(',').map(t => t.trim().toLowerCase()).filter(t => t !== '');
    } else if (Array.isArray(spec)) {
        types = spec.map(t => String(t).trim().toLowerCase()).filter(t => t !== '');
    } else {
        return false;
    }
    return types.length > 0 && !types.includes('none');
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
        server.listen(0, '127.0.0.1', () => {
            const addr = server.address() as AddressInfo;
            server.close(() => resolve(addr.port));
        });
    });
}
