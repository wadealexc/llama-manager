import type { ModelConfig, ModelRole } from "../config/types.js";

export type StrategyId =
    | "evict-task-model"
    | "disable-spec"
    | "quantize-kv-q8"
    | "quantize-kv-q4"
    | "mmproj-on-demand";

export interface StrategyContext {
    target: ModelRole;
    models: Partial<Record<ModelRole, ModelConfig>>;
}

export interface Strategy {
    id: StrategyId;
    canApply(ctx: StrategyContext): boolean;
    apply(ctx: StrategyContext): Promise<void>;
    applyNoSave(ctx: StrategyContext): Promise<void>;
}