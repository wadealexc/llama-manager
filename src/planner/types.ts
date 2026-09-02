import type { ReloadParams, SlotSave } from "../client/types.js";
import type { ModelState, StrategyId } from "../config/types.js";

export interface ActionContext {
    slots?: SlotSave[];
    signal: AbortSignal;
}

export interface Strategy {
    id: StrategyId;
    canApply(state: ModelState): boolean;
    getNewState(cur: ModelState): ModelState;
    getNewParams(params: ReloadParams): ReloadParams;

    action?(ctx: ActionContext): Promise<void>;
}