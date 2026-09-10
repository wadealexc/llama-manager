interface FlagDef {
    aliases: string[];
    value: boolean;
}

// tokenization grammar for llama-server args. canonical flag name -> ini key.
// `value` marks flags that consume the next token; the rest are bool flags
export const FLAGS: Record<string, FlagDef> = {
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

// resolves all flag names/aliases to a single value
const CANONICAL: Record<string, string> = {};
for (const [flag, def] of Object.entries(FLAGS)) {
    CANONICAL[flag] = flag;
    for (const alias of def.aliases) {
        CANONICAL[alias] = flag;
    }
}

// coerce a raw token to a normalized flag
export function normalizeFlag(tok: string): string {
    const flag = tok.replace(/^-+/, '').replaceAll('_', '-');
    return CANONICAL[flag] ?? flag;
}

// note: expects normalized flag as input
export function isRecognized(flag: string): boolean {
    return !!CANONICAL[flag];
}

// note: expects normalized flag as input
export function hasValue(flag: string): boolean {
    return FLAGS[flag]?.value ?? false;
}

const REJECTED_HF = new Set(['hf', 'hfr', 'hf-repo', 'hff', 'hf-file', 'hft', 'hf-token', 'dr', 'docker-repo', 'mtp']);
const REJECTED_MULTI_DEVICE = new Set(['ts', 'tensor-split', 'sm', 'split-mode']);
const REJECTED_ROUTER_LEVEL = new Set(['models-preset', 'models-dir', 'models-max', 'models-autoload', 'no-models-autoload']);

// note: expects normalized flag as input
export function maybeReject(flag: string): string | undefined {
    if (REJECTED_HF.has(flag)) {
        return `'${flag}' is not supported: hf/docker model sources resolve remote router presets`;
    }
    if (REJECTED_MULTI_DEVICE.has(flag) || flag.startsWith('rpc')) {
        return `'${flag}' implies multi-device placement, which is unsupported`;
    }
    if (REJECTED_ROUTER_LEVEL.has(flag)) {
        return `'${flag}' is a router-level flag and cannot appear in a model config`;
    }
    return undefined;
}

const MODEL_SOURCE_FLAGS = new Set(['model', 'model-url']);
const MANAGER_VALUE_FLAGS = new Set(['bin', 'config', 'ladder', 'sleep-idle-seconds', 'host', 'port', 'slot-save-path']);
const MANAGER_BOOL_FLAGS = new Set(['idle', 'calc-breakpoints', 'help', 'version']);

// note: expects normalized flag as input
export function isModelSource(flag: string): boolean {
    return MODEL_SOURCE_FLAGS.has(flag);
}

// note: expects normalized flag as input
export function isManagerValueFlag(flag: string): boolean {
    return MANAGER_VALUE_FLAGS.has(flag);
}

// note: expects normalized flag as input
export function isManagerBoolFlag(flag: string): boolean {
    return MANAGER_BOOL_FLAGS.has(flag);
}