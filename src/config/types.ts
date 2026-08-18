import type { StrategyId } from "../planner/types.js";

export type ModelId = string;

export type ModelRole = "main" | "task";

export interface ModelConfig {
    role: ModelRole;
    name: ModelId;
    expected_response_tokens: number;
    ladder: StrategyId[];
    has_spec: boolean;
    has_mmproj: boolean;
    kv_unified: boolean;
    fit_target_mib: number;
}

export interface LlamaConfig {
    bin: string;
    llama_log_dir: string;
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
    models: Partial<Record<ModelRole, ModelConfig>>;
}

export class ConfigError extends Error {
    constructor(message: string, readonly path?: string) {
        super(message);
        this.name = "ConfigError";
    }
}
