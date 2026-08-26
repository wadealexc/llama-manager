import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../../client/llama-api.js";
import type { ReloadParams, SlotSave } from "../../client/types.js";
import type { ModelEntry, StrategyId, Tokens } from "../../config/types.js";
import { logger } from "../../logger.js";
import type { Strategy, StrategyContext } from "../types.js";

const log: ConsolaInstance = logger.withTag('disable-mmproj');

export class DisableMmproj implements Strategy {

    id: StrategyId = 'disable-mmproj';

    client: LlamaAPI;

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
    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models[ctx.target]?.current_state.mmproj_loaded;
    }

    async applyNoSend(ctx: StrategyContext, params: ReloadParams, saves?: SlotSave[]): Promise<ReloadParams> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`DisableMmproj.applyNoSend: target not found`);
        }

        this.#updateCurrentState(target);

        return this.#getParams(params);
    }

    async apply(ctx: StrategyContext): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`DisableMmproj.apply: target not found`);
        }

        this.#updateCurrentState(target);

        const params = this.#getParams({});
        await this.client.reloadModel(params, target.name);
    }

    #getParams(params: ReloadParams): ReloadParams {
        return {
            ...params,
            mmproj: { path: "" }
        };
    }

    #updateCurrentState(model: ModelEntry): void {
        model.current_state.mmproj_loaded = false;
    }
}