import type { ConsolaInstance } from "consola";
import { logger } from "../../logger.js";
import type { LlamaAPI } from "../../client/llama-api.js";
import type { ReloadParams } from "../../client/types.js";
import type { ModelState, StrategyId } from "../../config/types.js";
import type { Strategy } from "../types.js";

const log: ConsolaInstance = logger.withTag('quantize-kv-q4');

export class QuantizeKvQ4 implements Strategy {

    id: StrategyId = 'quantize-kv-q4';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(state: ModelState): boolean {
        return isKvMorePreciseThanQ4(state);
    }

    getNewState(cur: ModelState): ModelState {
        const next = structuredClone(cur);
        next.cache_type_k = 'q4_0';
        next.cache_type_v = 'q4_0';
        return next;
    }

    getNewParams(params: ReloadParams): ReloadParams {
        return {
            ...params,
            cache_type_k: 'q4_0',
            cache_type_v: 'q4_0',
        };
    }
}

// TODO: Does not handle ctk/ctv individually being more precise.
function isKvMorePreciseThanQ4(st: ModelState): boolean {
    const ctk = st.cache_type_k;
    const ctv = st.cache_type_v;

    return (ctk === 'f16' || ctk === 'q8_0') && (ctv === 'f16' || ctv === 'q8_0');
}
