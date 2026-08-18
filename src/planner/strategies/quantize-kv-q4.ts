import type { LlamaAPI } from "../../client/llama-api.js";
import type { Strategy, StrategyContext, StrategyId } from "../types.js";

export class QuantizeKvQ4 implements Strategy {

    id: StrategyId = 'quantize-kv-q4';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return false; // TODO - can apply if configured kvcache precision is > q4_0
    }

    async apply(ctx: StrategyContext): Promise<void> {
        await this.applyNoSave(ctx); // TODO - implement save/restore
    }

    async applyNoSave(ctx: StrategyContext): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`QuantizeKvQ4.applyNoSave: target not found`);
        }

        // TODO - new max ctx
        await this.client.reloadModel({ 
            n_ctx: 1, 
            cache_type_k: 'q4_0', 
            cache_type_v: 'q4_0' 
        }, target.name);
    }
}