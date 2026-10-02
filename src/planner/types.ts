import type { ModelState, StrategyId } from "../config/types.js";

export interface StrategyImpl {
    id: StrategyId;
    canApply(state: ModelState): boolean;
    getNewState(cur: ModelState): ModelState;
}
