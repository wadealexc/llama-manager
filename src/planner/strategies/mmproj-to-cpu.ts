import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../../client/llama-api.js";
import { hasMmproj, isMmprojOnCPU, type ModelState, type StrategyId } from "../../config/types.js";
import { logger } from "../../logger.js";
import type { StrategyImpl } from "../types.js";

const log: ConsolaInstance = logger.withTag('mmproj-to-cpu');

export class MmprojToCPU implements StrategyImpl {

    id: StrategyId = 'mmproj-to-cpu';

    client: LlamaAPI;

    /**
     * TODO - Eventually, the goal is to make mmproj "on-demand." Once the strategy is apply,
     * the mmproj is fully unloaded and from that point forward:
     * - When a request comes in with an image (and that image isn't already in our kvcache), we:
     *   - Free up space on the GPU (probably via temporary kvcache evict)
     *   - Load the mmproj and run non-cached images through it
     *   - Unload the mmproj, restore the kvcache, and feed the tokenized images in along with the text prompt
     * - Otherwise, we handle the request normally, because we don't need the mmproj
     * 
     * (This requires llama.cpp changes, so for now we just disable/move to cpu.)
     */
    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(state: ModelState): boolean {
        return hasMmproj(state) && !isMmprojOnCPU(state);
    }

    getNewState(cur: ModelState): ModelState {
        const next = structuredClone(cur);
        next.mmproj = { ...next.mmproj, mmproj_offload: false };
        return next;
    }
}