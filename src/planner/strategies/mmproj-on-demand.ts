import type { LlamaAPI } from "../../client/llama-api.js";
import type { StrategyId, Tokens } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";

export class MmprojOnDemand implements Strategy {

    id: StrategyId = 'mmproj-on-demand';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models[ctx.target]?.current_state.mmproj_loaded;
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
     * 
     * TODO: also implement save/restore
     */
    async apply(ctx: StrategyContext, n_ctx: Tokens): Promise<void> {
        await this.#apply(ctx, n_ctx);
    }

    async applyNoSave(ctx: StrategyContext): Promise<void> {
        await this.#apply(ctx);
    }

    // TODO
    async beforeRequest(ctx: StrategyContext, req: unknown): Promise<unknown> {
        return req;
    }

    async #apply(ctx: StrategyContext, n_ctx?: Tokens): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`MmprojOnDemand.#apply: target not found`);
        }

        await this.client.reloadModel({
            n_ctx: n_ctx,
            mmproj: { path: "" }
        }, target.name);

        if (n_ctx !== undefined) target.current_state.n_ctx = n_ctx;
        target.current_state.mmproj_loaded = false;
    }
}