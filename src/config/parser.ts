import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { load as yamlLoad } from "js-yaml";
import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";
import { STRATEGY_IDS, type ConfigSource, type Mode, type ModelId, type RawModel, type StrategyId } from "./types.js";
import { hasValue, isManagerBoolFlag, isManagerValueFlag, isModelSource, isRecognized, maybeReject, normalizeFlag } from "./flags.js";

const log: ConsolaInstance = logger.withTag('parser');

// accumulates the manager flags shared by both modes' parsers
interface ManagerFlags {
    bin?: string;
    config_path?: string;
    host?: string;
    port?: number;
    sleep_idle_seconds?: number;
    ladder?: StrategyId[];
    slot_save_path?: string;
    idle: boolean;
    calc_breakpoints: boolean;
    help: boolean;
    version: boolean;
}

interface ServerState extends ManagerFlags {
    entry: RawModel;
    aliases: string[];
    model_sources: Set<string>;
    seen: Set<string>;
}

export interface ParsedArgs extends ManagerFlags {
    mode: Mode;

    // server mode only
    raw_models?: Record<string, RawModel>;
}

export function parseArgs(argv: string[]): ParsedArgs {
    // first pass - reject unsupported keys
    for (const tok of argv) {
        if (!tok.startsWith('-')) continue;
        const reason = maybeReject(normalizeFlag(tok));
        if (reason) throw new Error(reason);
    }

    // if CLI args pass in an explicit model source, parse and prepare to serve single model
    const has_model_source = argv.some(tok => tok.startsWith('-') && isModelSource(normalizeFlag(tok)));
    if (has_model_source) {
        return parseServerArgs(argv);
    }

    // model source must be in supplied config and may contain multiple models
    return parseRouterArgs(argv);
}

function parseRouterArgs(argv: string[]): ParsedArgs {
    const flags: ManagerFlags = {
        idle: false,
        calc_breakpoints: false,
        help: false,
        version: false,
    };

    parseLlamaArgs(argv, flags, () => { });

    return { mode: 'router', ...flags };
}

function parseServerArgs(argv: string[]): ParsedArgs {
    const state: ServerState = {
        entry: {},
        aliases: [],
        model_sources: new Set(),
        idle: false,
        calc_breakpoints: false,
        help: false,
        version: false,
        seen: new Set(),
    };

    parseLlamaArgs(argv, state, (key, value) => setEntryValue(state, key, value));

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

    return {
        mode: 'server',
        bin: state.bin,
        config_path: state.config_path,
        host: state.host,
        port: state.port,
        sleep_idle_seconds: state.sleep_idle_seconds,
        ladder: state.ladder,
        slot_save_path: state.slot_save_path,
        idle: state.idle,
        calc_breakpoints: state.calc_breakpoints,
        help: state.help,
        version: state.version,
        raw_models: { [id]: state.entry },
    };
}

function parseLlamaArgs(
    argv: string[],
    flags: ManagerFlags,
    sink: (key: string, value: string | number | boolean) => void,
): void {
    let i = 0;
    while (i < argv.length) {
        const tok = argv[i];
        if (!tok.startsWith('-')) {
            log.error(`ignoring positional argument '${tok}'`);
            i++;
            continue;
        }

        const name = normalizeFlag(tok);
        i++;

        if (isManagerValueFlag(name)) {
            const value = consumeValue(argv, i, tok);
            i = value.next;
            handleManagerValue(flags, name, value.value);
            continue;
        }

        if (isManagerBoolFlag(name)) {
            handleManagerBool(flags, name);
            continue;
        }

        // if we don't recognize the flag, try to consume a value
        if (!isRecognized(name)) {
            const consumed = consumeUnknownValue(argv, i);
            i = consumed.next;
            sink(name, coerceValue(consumed.value));
            continue;
        }

        // flag is recognized - consume a value if needed
        if (hasValue(name)) {
            const value = consumeValue(argv, i, tok);
            i = value.next;
            sink(name, coerceValue(value.value));
        } else {
            sink(name, true);
        }
    }
}

function handleManagerValue(state: ManagerFlags, name: string, value: string): void {
    switch (name) {
        case 'bin':
            state.bin = value;
            return;
        case 'config':
            state.config_path = value;
            return;
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
            state.slot_save_path = value;
            return;
    }
}

function handleManagerBool(state: ManagerFlags, name: string): void {
    switch (name) {
        case 'idle':
            state.idle = true;
            return;
        case 'calc-breakpoints':
            state.calc_breakpoints = true;
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
        host: raw.host as string | undefined,
        port: raw.port as number | undefined,
        sleep_idle_seconds: raw['sleep-idle-seconds'] as number | undefined,
        default_model: raw['default-model'] as string | undefined,
    };
}
