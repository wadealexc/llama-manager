import type { ConsolaInstance } from "consola";
import { logger } from "../../logger.js";
import type { LlamaAPI } from "../../client/llama-api.js";
import type { ReloadParams } from "../../client/types.js";
import type { ModelState, StrategyId } from "../../config/types.js";
import type { Strategy } from "../types.js";

const log: ConsolaInstance = logger.withTag('disable-spec');

export class DisableSpec implements Strategy {

    id: StrategyId = 'disable-spec';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(state: ModelState): boolean {
        return state.spec_loaded;
    }

    getNewState(cur: ModelState): ModelState {
        const next = structuredClone(cur);
        next.spec_loaded = false;
        return next;
    }

    getNewParams(params: ReloadParams): ReloadParams {
        return {
            ...params,
            spec: { types: ['none'] }
        };
    }
}