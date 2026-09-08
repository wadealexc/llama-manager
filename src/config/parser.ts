import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { load as yamlLoad } from "js-yaml";
import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";
import type { ConfigSource, ModelId, RawModel, StrategyId } from "./types.js";

const log: ConsolaInstance = logger.withTag('parser');

export type ParsedArgs = {
    mode: 'router';
    config_path?: string;
    show_breakpoints: boolean;
    calc_max_ctx: boolean;
    help: boolean;
    version: boolean;
} | {
    mode: 'server';
    source: ConfigSource;
    show_breakpoints: boolean;
    calc_max_ctx: boolean;
    help: boolean;
    version: boolean;
};

interface FlagDef {
    aliases: string[];
    value: boolean;
}

// tokenization grammar for llama-server args. canonical flag name -> ini key.
// `value` marks flags that consume the next token; the rest are bool flags
const FLAGS: Record<string, FlagDef> = {
    'model': { aliases: ['m'], value: true },
    'model-url': { aliases: ['mu'], value: true },
    'alias': { aliases: ['a'], value: true },
    'mmproj': { aliases: ['mm'], value: true },
    'mmproj-url': { aliases: ['mmu'], value: true },
    'mmproj-auto': { aliases: [], value: false },
    'no-mmproj': { aliases: [], value: false },
    'no-mmproj-auto': { aliases: [], value: false },
    'mmproj-offload': { aliases: [], value: false },
    'no-mmproj-offload': { aliases: [], value: false },
    'mmproj-device': { aliases: ['mmdev'], value: true },
    'spec-type': { aliases: [], value: true },
    'cache-type-k': { aliases: ['ctk'], value: true },
    'cache-type-v': { aliases: ['ctv'], value: true },
    'ctx-size': { aliases: ['c'], value: true },
    'n-gpu-layers': { aliases: ['ngl', 'gpu-layers'], value: true },
    'api-key': { aliases: [], value: true },
    'api-key-file': { aliases: [], value: true },
    'device': { aliases: ['dev'], value: true },
    'fa': { aliases: ['flash-attn'], value: true },
    'no-fa': { aliases: [], value: false },
    'fit': { aliases: [], value: true },
    'fit-print': { aliases: ['fitp'], value: true },
    'fit-target': { aliases: ['fitt'], value: true },
    'fit-ctx': { aliases: ['fitc'], value: true },
    'parallel': { aliases: ['np'], value: true },
    'kv-unified': { aliases: ['kvu'], value: false },
    'no-kv-unified': { aliases: ['no-kvu'], value: false },
    'kv-offload': { aliases: ['kvo'], value: false },
    'no-kv-offload': { aliases: ['nkvo'], value: false },
    'op-offload': { aliases: [], value: false },
    'no-op-offload': { aliases: [], value: false },
    'swa-full': { aliases: [], value: false },
    'n-cpu-moe': { aliases: ['ncmoe'], value: true },
    'n-cpu-moe-draft': { aliases: ['ncmoed', 'spec-draft-ncmoe', 'spec-draft-n-cpu-moe'], value: true },
    'override-tensor': { aliases: ['ot'], value: true },
    'override-tensor-draft': { aliases: ['otd', 'spec-draft-override-tensor'], value: true },
    'lora': { aliases: [], value: true },
    'webui': { aliases: ['ui'], value: false },
    'no-webui': { aliases: ['no-ui'], value: false },
};

const CANONICAL: Record<string, string> = {};
for (const [canonical, def] of Object.entries(FLAGS)) {
    CANONICAL[canonical] = canonical;
    for (const alias of def.aliases) {
        CANONICAL[alias] = canonical;
    }
}

export function canonicalKey(raw: string): string | undefined {
    return CANONICAL[normalizeFlagName(raw)];
}

const REJECTED_HF = new Set(['hf', 'hfr', 'hf-repo', 'hff', 'hf-file', 'hft', 'hf-token', 'dr', 'docker-repo', 'mtp']);
const REJECTED_MULTI_DEVICE = new Set(['ts', 'tensor-split', 'sm', 'split-mode']);
const REJECTED_ROUTER_LEVEL = new Set(['models-preset', 'models-dir', 'models-max', 'models-autoload', 'no-models-autoload']);

export function rejectedKeyReason(key: string): string | undefined {
    if (REJECTED_HF.has(key)) {
        return `'${key}' is not supported: hf/docker model sources resolve remote router presets`;
    }
    if (REJECTED_MULTI_DEVICE.has(key) || key.startsWith('rpc')) {
        return `'${key}' implies multi-device placement, which is unsupported`;
    }
    if (REJECTED_ROUTER_LEVEL.has(key)) {
        return `'${key}' is a router-level flag and cannot appear in a model config`;
    }
    return undefined;
}

const MODEL_SOURCE_FLAGS = new Set(['model', 'model-url']);

const MANAGER_VALUE_FLAGS = new Set(['config', 'ladder', 'sleep-idle-seconds', 'host', 'port', 'slot-save-path']);
const MANAGER_BOOL_FLAGS = new Set(['show-breakpoints', 'calc-max-ctx', 'serve', 'help', 'version']);

const STRATEGY_IDS: StrategyId[] = ['disable-spec', 'mmproj-to-cpu', 'quantize-kv-q8', 'quantize-kv-q4'];

interface ServerState {
    entry: RawModel;
    raw_router: Record<string, unknown>;
    aliases: string[];
    model_sources: Set<string>;
    host: string;
    port: number;
    sleep_idle_seconds: number;
    show_breakpoints: boolean;
    calc_max_ctx: boolean;
    help: boolean;
    version: boolean;
    ladder?: StrategyId[];
    seen: Set<string>;
}

export function parseArgs(argv: string[]): ParsedArgs {
    for (const tok of argv) {
        if (!tok.startsWith('-')) continue;
        const reason = rejectedKeyReason(normalizeFlagName(tok));
        if (reason) throw new Error(reason);
    }

    const has_model_source = argv.some(tok => tok.startsWith('-') && MODEL_SOURCE_FLAGS.has(normalizeFlagName(tok)));
    if (has_model_source) {
        return parseServerArgs(argv);
    }
    return parseRouterArgs(argv);
}

function parseRouterArgs(argv: string[]): ParsedArgs {
    const parsed: Extract<ParsedArgs, { mode: 'router' }> = {
        mode: 'router',
        show_breakpoints: false,
        calc_max_ctx: false,
        help: false,
        version: false,
    };

    for (let i = 0; i < argv.length; i++) {
        const name = normalizeFlagName(argv[i]);
        switch (name) {
            case 'config':
                parsed.config_path = argv[++i];
                break;
            case 'show-breakpoints':
                parsed.show_breakpoints = true;
                break;
            case 'calc-max-ctx':
                parsed.calc_max_ctx = true;
                break;
            case 'help':
                parsed.help = true;
                break;
            case 'version':
                parsed.version = true;
                break;
        }
    }

    return parsed;
}

function parseServerArgs(argv: string[]): ParsedArgs {
    const state: ServerState = {
        entry: {},
        raw_router: {},
        aliases: [],
        model_sources: new Set(),
        host: '127.0.0.1',
        port: 8080,
        sleep_idle_seconds: 0,
        show_breakpoints: false,
        calc_max_ctx: false,
        help: false,
        version: false,
        seen: new Set(),
    };

    let i = 0;
    while (i < argv.length) {
        const tok = argv[i];
        if (!tok.startsWith('-')) {
            log.error(`server mode: ignoring positional argument '${tok}'`);
            i++;
            continue;
        }

        const name = normalizeFlagName(tok);
        i++;

        if (MANAGER_VALUE_FLAGS.has(name)) {
            const value = consumeValue(argv, i, tok);
            i = value.next;
            handleManagerValue(state, name, value.value);
            continue;
        }

        if (MANAGER_BOOL_FLAGS.has(name)) {
            handleManagerBool(state, name);
            continue;
        }

        const canonical = CANONICAL[name];
        if (!canonical) {
            const consumed = consumeUnknownValue(argv, i);
            i = consumed.next;
            setEntryValue(state, name, coerceValue(consumed.value));
            continue;
        }

        if (FLAGS[canonical].value) {
            const value = consumeValue(argv, i, tok);
            i = value.next;
            setEntryValue(state, canonical, coerceValue(value.value));
        } else {
            setEntryValue(state, canonical, true);
        }
    }

    return finalizeServer(state);
}

function handleManagerValue(state: ServerState, name: string, value: string): void {
    switch (name) {
        case 'config':
            throw new Error(`--config cannot be combined with a model source`);
        case 'ladder':
            state.ladder = parseLadder(value);
            return;
        case 'sleep-idle-seconds': {
            const n = toInt(value, '--sleep-idle-seconds');
            state.sleep_idle_seconds = n > 0 ? n : 0;
            return;
        }
        case 'host':
            state.host = value;
            return;
        case 'port':
            state.port = toInt(value, name);
            return;
        case 'slot-save-path':
            state.raw_router['slot-save-path'] = value;
            return;
    }
}

function handleManagerBool(state: ServerState, name: string): void {
    switch (name) {
        case 'show-breakpoints':
            state.show_breakpoints = true;
            return;
        case 'calc-max-ctx':
            state.calc_max_ctx = true;
            return;
        case 'serve':
            return;
        case 'help':
            state.help = true;
            return;
        case 'version':
            state.version = true;
            return;
    }
}

function setEntryValue(state: ServerState, key: string, value: string | number | boolean): void {
    const accumulates = key === 'spec-type' || key === 'alias';
    if (!accumulates && state.seen.has(key)) {
        log.error(`server mode: repeated flag '--${key}'; using last value`);
    }
    state.seen.add(key);

    if (key === 'model' || key === 'model-url') {
        state.model_sources.add(key);
        state.entry[key] = value;
        return;
    }

    if (key === 'alias') {
        if (value !== '') {
            state.aliases.push(String(value));
        }
        return;
    }

    if (key === 'spec-type' && state.entry['spec-type'] !== undefined) {
        const prev = state.entry['spec-type'];
        state.entry['spec-type'] = [...(Array.isArray(prev) ? prev : [prev]), value];
        return;
    }

    state.entry[key] = value;
}

function finalizeServer(state: ServerState): ParsedArgs {
    if (state.model_sources.size === 0) {
        throw new Error(`server mode requires a model source (-m/--model or -mu/--model-url)`);
    }
    if (state.model_sources.size > 1) {
        throw new Error(`server mode accepts exactly one model source`);
    }

    const source_path = (state.entry['model'] ?? state.entry['model-url']) as string;
    const id = state.aliases.length > 0 ? state.aliases[0] : deriveModelId(source_path);

    const rest = state.aliases.slice(1);
    if (rest.length > 0) {
        state.entry['alias'] = rest;
    }

    const source: ConfigSource = {
        mode: 'server',
        raw_models: { [id]: state.entry },
        raw_router: state.raw_router,
        raw_model_load: {},
        host: state.host,
        port: state.port,
        sleep_idle_seconds: state.sleep_idle_seconds,
        default_models: [id],
        ladder_override: state.ladder,
    };

    return {
        mode: 'server',
        source,
        show_breakpoints: state.show_breakpoints,
        calc_max_ctx: state.calc_max_ctx,
        help: state.help,
        version: state.version,
    };
}

function parseLadder(value: string): StrategyId[] {
    const ids = value.split(',').map(s => s.trim()).filter(s => s !== '');
    if (ids.length === 0) {
        throw new Error(`--ladder requires at least one strategy id (supported: ${STRATEGY_IDS.join(', ')})`);
    }
    for (const id of ids) {
        if (!STRATEGY_IDS.includes(id as StrategyId)) {
            throw new Error(`unknown strategy id '${id}' (--ladder supports: ${STRATEGY_IDS.join(', ')})`);
        }
    }
    return ids as StrategyId[];
}

function deriveModelId(source: string): ModelId {
    const clean = source.split('?')[0].split('#')[0];
    const base = basename(clean);
    return base.toLowerCase().endsWith('.gguf') ? base.slice(0, -5) : base;
}

function consumeValue(argv: string[], i: number, tok: string): { value: string; next: number } {
    if (i >= argv.length) {
        throw new Error(`'${tok}' expects a value`);
    }
    return { value: argv[i], next: i + 1 };
}

function consumeUnknownValue(argv: string[], i: number): { value: string | true; next: number } {
    if (i < argv.length && (!argv[i].startsWith('-') || isNumeric(argv[i]))) {
        return { value: argv[i], next: i + 1 };
    }
    return { value: true, next: i };
}

function coerceValue(v: string | true): string | number | boolean {
    if (v === true) return true;
    return isNumeric(v) ? Number(v) : v;
}

function normalizeFlagName(tok: string): string {
    return tok.replace(/^-+/, '').replaceAll('_', '-');
}

function isNumeric(s: string): boolean {
    return /^-?\d+(\.\d+)?$/.test(s);
}

function toInt(value: string, flag: string): number {
    if (!/^-?\d+$/.test(value)) {
        throw new Error(`'${flag}' expects an integer, got '${value}'`);
    }
    return Number(value);
}

export function parseRouterConfig(path: string): ConfigSource {
    let raw: Record<string, unknown>;
    try {
        raw = yamlLoad(readFileSync(path, "utf8")) as Record<string, unknown>;
    } catch (e) {
        throw new Error(`failed to parse config: ${(e as Error).message}`);
    }

    return {
        mode: 'router',
        raw_models: (raw.models as Record<string, RawModel>) ?? {},
        raw_router: (raw.router ?? {}) as Record<string, unknown>,
        raw_model_load: (raw['model-load'] ?? {}) as Record<string, unknown>,
        host: (raw.host as string) ?? '127.0.0.1',
        port: (raw.port as number) ?? 8080,
        sleep_idle_seconds: (raw['sleep-idle-seconds'] as number) ?? 300,
        default_models: raw['default-models'] as string[] | undefined,
    };
}
