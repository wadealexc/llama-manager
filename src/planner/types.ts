import type { ModelEntry, ModelId, ModelRole, StrategyId, Tokens } from "../config/types.js";

export interface StrategyContext {
    target: ModelRole;
    models: Partial<Record<ModelRole, ModelEntry>>;
}

/**
 * @idea split strategy interface up a little so that we do only 1 reloadModel call rather than several
 * (the same optimization improves kvcache save/restore). Example interface fn:
 * 
 * ```js
 * apply(entry: ModelEntry, cur_params: ReloadParams): ReloadParams
 * ```
 * 
 * This method takes params to reloadModel as input and produces ReloadParams that will apply the strategy.
 * The planner can iterate over strategies, creating parameters to an aggregate reloadModel call:
 * 
 * ```js
 * let params = getCurParams(entry);
 * for (const s of strategies_to_apply) {
 *     params = s.apply(entry, params);
 * }
 * 
 * reload(entry, params);
 * ```
 * 
 * This could also be done with kvcache: provide input state and cur kvcache; strategy computes kvcache
 * modifications and returns the new kvcache. The result should be able to be restored via the llama.cpp API
 * after the strategy is applied.
 * 
 * Note: Currently, strategies also modify current_state. Would need to fold that into this somehow. Maybe
 * change ModelEntry so that `initial_state` is actually ReloadParams, representing "the base reloadParams
 * that will reset the model."
 */
export interface Strategy {
    id: StrategyId;
    canApply(ctx: StrategyContext): boolean;
    apply(ctx: StrategyContext, n_ctx: Tokens): Promise<void>;
    applyNoSave(ctx: StrategyContext): Promise<void>;
    beforeRequest?(ctx: StrategyContext, req: unknown): Promise<unknown | void>;
}