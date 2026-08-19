import type { ModelEntry, ModelRole, StrategyId, Tokens } from "../config/types.js";

export interface StrategyContext {
    target: ModelRole;
    models: Partial<Record<ModelRole, ModelEntry>>;
}

export interface Strategy {
    id: StrategyId;
    canApply(ctx: StrategyContext): boolean;
    apply(ctx: StrategyContext, n_ctx: Tokens): Promise<void>;
    applyNoSave(ctx: StrategyContext): Promise<void>;
    beforeRequest?(ctx: StrategyContext, req: unknown): Promise<unknown | void>;
}
