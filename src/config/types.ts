import type { KVPrecision } from "../client/types.js";

export type ModelId = string;

export type Tokens = number;

export type StrategyId =
    | "disable-spec"
    | "quantize-kv-q8"
    | "quantize-kv-q4"
    | "mmproj-to-cpu";

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

export interface ModelState {
    mmproj_loaded: boolean;
    spec_loaded: boolean;
    kv_unified: boolean;
    cache_type_k: KVPrecision;
    cache_type_v: KVPrecision;
}

export interface ModelConfig {
    name: ModelId;
    aliases: string[];
    ladder: StrategyId[];
    initial_state: ModelState;
}

export type Mode = 'server' | 'router';

export type RawModel = Record<string, unknown>;

export interface ConfigSource {
    mode: Mode;
    raw_models: Record<string, RawModel>;
    raw_router: Record<string, unknown>;
    raw_model_load: Record<string, unknown>;
    host: string;
    port: number;
    sleep_idle_seconds?: number;
    default_models?: ModelId[];
    ladder_override?: StrategyId[];
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
    default_models: ModelId[];
}

export class ConfigError extends Error {
    constructor(message: string, readonly path?: string) {
        super(message);
        this.name = "ConfigError";
    }
}
