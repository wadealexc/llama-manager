export type ModelId = string;

export type StrategyId =
    | "disable-spec"
    // | "quantize-kv-q8"
    // | "quantize-kv-q4"
    // | "mmproj-on-demand"
    // | "evict-kvcache"
    // | "evict-weights"
;

export interface ModelConfig {
    id: ModelId;
    min_ctx: number;
    // Estimated number of tokens to allow for a model's response
    expected_response_tokens: number;    
    // Ordered strategies to apply when under pressure
    ladder: StrategyId[];
}

export interface LlamaConfig {
    bin: string;
    preset: string;
    listen: string;
    poll_interval_ms: number;
    poll_timeout_ms: number;
    shutdown_grace_period_ms: number;
    llama_log_dir: string;
}

export interface ManagerConfig {
    router: LlamaConfig;
    listen: string;
    idle_timeout: number;
    model: ModelConfig;
}

export class ConfigError extends Error {
    constructor(message: string, readonly path?: string) {
        super(message);
        this.name = "ConfigError";
    }
}
