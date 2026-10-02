import type { ReloadParams } from "../client/types.js";

export type ModelId = string;

export type Tokens = number;

export type StrategyId =
    | "disable-spec"
    | "quantize-kv-q8"
    | "quantize-kv-q4"
    | "mmproj-to-cpu";

const STRATEGY_IDS: StrategyId[] = ['disable-spec', 'mmproj-to-cpu', 'quantize-kv-q8', 'quantize-kv-q4'];
export { STRATEGY_IDS };

export type Strategy =
    | { kind: 'reload-model'; id: StrategyId }
    | { kind: 'swap-model'; variant: string };

export const RESERVED_VARIANT_NAMES = new Set(['model', 'model-url']);

export enum LoadStatus {
    UNLOADED,
    WEIGHTS_ONLY,
    LOADED,
}

export interface LlamaConfig {
    bin: string;
    llama_log_dir: string;
    slot_save_path: string;
    listen: string;
    poll_interval_ms: number;
    poll_timeout_ms: number;
    shutdown_grace_period_ms: number;
}

export interface ModelLoadConfig {
    poll_interval_ms: number;
    poll_timeout_ms: number;
}

export type ModelState = Pick<ReloadParams,
    | 'cache_type_k'
    | 'cache_type_v'
    | 'kv_unified'
    | 'mmproj'
    | 'spec'
> & {
    model_variant: string;
};

export function isSpecEnabled(state: ModelState): boolean {
    return !!state.spec && !!state.spec.types?.length && !state.spec.types.includes('none');
}

export function hasMmproj(state: ModelState): boolean {
    return state.mmproj !== undefined && state.mmproj.path !== '';
}

export function isMmprojOnCPU(state: ModelState): boolean {
    return state.mmproj?.mmproj_offload === false;
}

export interface ModelVariant {
    router_id: ModelId;
    path: string;
}

export interface ModelConfig {
    name: ModelId;
    aliases: string[];
    variants: Record<string, ModelVariant>;
    ladder: Strategy[];
    initial_state: ModelState;
}

export type Mode = 'server' | 'router';

export type RawModel = Record<string, unknown>;

export interface ConfigSource {
    mode: Mode;
    raw_models: Record<string, RawModel>;
    raw_router: Record<string, unknown>;
    raw_model_load: Record<string, unknown>;
    host?: string;
    port?: number;
    sleep_idle_seconds?: number;
    default_model?: ModelId;
    ladder_override?: Strategy[];
    bin_override?: string;
}

export interface ManagerConfig {
    mode: Mode;
    router: LlamaConfig;
    host: string;
    port: number;
    sleep_idle_seconds: number;
    model_load: ModelLoadConfig;
    models: Record<ModelId, ModelConfig>;
    default_model: ModelId;
}

export class ConfigError extends Error {
    constructor(message: string, readonly path?: string) {
        super(message);
        this.name = "ConfigError";
    }
}
