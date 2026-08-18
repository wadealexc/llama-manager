import type { LlamaAPI } from "../../client/llama-api.js";
import type { Strategy, StrategyContext, StrategyId } from "../types.js";

export class DisableSpec implements Strategy {

    id: StrategyId = 'disable-spec';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models[ctx.target]?.has_spec;
    }

    async apply(ctx: StrategyContext): Promise<void> {
        await this.applyNoSave(ctx); // TODO - implement save/restore
    }

    async applyNoSave(ctx: StrategyContext): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`DisableSpec.applyNoSave: target not found`);
        }

        // TODO - new max ctx
        await this.client.reloadModel({ 
            n_ctx: 1, 
            spec: { types: ["none"] } 
        }, target.name);
    }
}