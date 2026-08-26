import type { ReloadParams, SlotSave } from "../client/types.js";
import type { ModelEntry, ModelRole, StrategyId } from "../config/types.js";

export interface StrategyContext {
    target: ModelRole;
    models: Partial<Record<ModelRole, ModelEntry>>;
}

export interface Strategy {
    id: StrategyId;
    canApply(ctx: StrategyContext): boolean;
    applyNoSend(ctx: StrategyContext, params: ReloadParams, saves?: SlotSave[]): Promise<ReloadParams>;
    apply(ctx: StrategyContext): Promise<void>;
    beforeRequest?(ctx: StrategyContext, req: unknown): Promise<unknown | void>;
}