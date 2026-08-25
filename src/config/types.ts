import type { KVPrecision } from "../client/types.js";

export type ModelId = string;

export type Tokens = number;

export type ModelRole = "main" | "task";

export type StrategyId =
    | "evict-task-model"
    | "disable-spec"
    | "quantize-kv-q8"
    | "quantize-kv-q4"
    | "mmproj-on-demand";

export interface ModelState {
    n_ctx: Tokens;

    mmproj_loaded: boolean;
    spec_loaded: boolean;

    kv_unified: boolean;
    cache_type_k: KVPrecision;
    cache_type_v: KVPrecision;
}

export interface ModelEntry {
    role: ModelRole;
    name: ModelId;
    expected_response_tokens: number;
    fit_target_mib: number;

    is_loaded: boolean;
    ladder: StrategyId[];
    applied: StrategyId[];

    initial_state: ModelState;
    current_state: ModelState;
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

export interface ManagerConfig {
    router: LlamaConfig;
    listen: string;
    idle_timeout: number;
    model_load: ModelLoadConfig;
    models: Partial<Record<ModelRole, ModelEntry>>;
}

export class ConfigError extends Error {
    constructor(message: string, readonly path?: string) {
        super(message);
        this.name = "ConfigError";
    }
}
