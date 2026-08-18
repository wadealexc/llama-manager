import type { LlamaAPI } from "../../client/llama-api.js";
import type { Strategy, StrategyContext, StrategyId } from "../types.js";

export class MmprojOnDemand implements Strategy {

    id: StrategyId = 'mmproj-on-demand';

    client: LlamaAPI;
    
    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models[ctx.target]?.has_mmproj;
    }

    /**
     * TODO - right now, we just disable the mmproj. Eventually, the goal is to make mmproj "on-demand,"
     * which means that once the strategy is applied, the mmproj is unloaded and from that point forward:
     * - When a request comes in with an image (and that image isn't already in our kvcache), we:
     *   - Free up space on the GPU (probably via temporary kvcache evict)
     *   - Load the mmproj and run non-cached images through it
     *   - Unload the mmproj, restore the kvcache, and feed the tokenized images in along with the text prompt
     * - Otherwise, we handle the request normally, because we don't need the mmproj
     * 
     * (This probably requires llama.cpp changes, so for now we just disable.)
     */
    async apply(ctx: StrategyContext): Promise<void> {
        await this.applyNoSave(ctx); // TODO - implement save/restore
    }

    async applyNoSave(ctx: StrategyContext): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`MmprojOnDemand.applyNoSave: target not found`);
        }

        // TODO - new max ctx
        await this.client.reloadModel({ 
            n_ctx: 1, 
            mmproj: { path: "" } 
        }, target.name);
    }
}