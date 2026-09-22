import type { ConsolaInstance } from "consola";
import { logger } from "../../logger.js";
import type { LlamaAPI } from "../../client/llama-api.js";
import { isSpecEnabled, type ModelState, type StrategyId } from "../../config/types.js";
import type { Strategy } from "../types.js";

const log: ConsolaInstance = logger.withTag('disable-spec');

export class DisableSpec implements Strategy {

    id: StrategyId = 'disable-spec';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(state: ModelState): boolean {
        return isSpecEnabled(state);
    }

    getNewState(cur: ModelState): ModelState {
        const next = structuredClone(cur);
        next.spec = { types: ['none'] };
        return next;
    }
}