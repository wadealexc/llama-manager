import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";
import type { ModelEntry } from "../config/types.js";
import type { LlamaAPI } from "../client/llama-api.js";
import type { MemoryResponse } from "../client/types.js";

const log: ConsolaInstance = logger.withTag('cost-model');

// TODO - currently assuming 1 GPU
function calcFreeBytes(mem: MemoryResponse): number {
    for (const dev of mem.devices) {
        if (dev.type !== "cpu") return dev.free;
    }
    return 0;
}

export class CostModel {

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    /**
     * TODO: 
     * - Math.ceil padding approach assumes kvunified. Needs fix for non-unified
     * - reduce iterations by interpolating fit from two good points
     * - this is generally inefficient; better would be exposing llama.cpp's fit via HTTP
     */
    async findMaxCtx(cur: number, model: ModelEntry): Promise<number> {
        log.info(`finding max ctx for ${model.name}`);

        const t = performance.now();
        const res = await this.#findMax(cur, model);
        const find_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.info(`found max ctx in ${find_sec} sec`);

        return res;
    }

    async #findMax(cur: number, model: ModelEntry): Promise<number> {
        const models = await this.client.getModels();
        const model_info = models.find((m) => m.id === model.name);
        if (!model_info) throw new Error(`model not found: ${model.name}`);
        if (!model_info.meta) throw new Error(`model info did not contain 'meta' key: ${model.name}`);

        // Reload model context, using binary search to find  the highest non-failing value
        let lo = cur;
        let hi = model_info.meta.n_ctx_train;
        let iterations = 0;

        while (lo <= hi) {
            // llama.cpp pads context lengths up to the nearest multiple of 256
            let mid = lo + Math.trunc((hi - lo) / 2);
            mid = Math.ceil(mid / 256) * 256;

            let success = false;
            try {
                const status = await this.client.reloadModel({ n_ctx: mid }, model.name);
                success = status.success;
            } catch { }

            // We reloaded, but we may not be meeting our overhead target
            if (success) {
                // TODO: would be nice to have the bar fill here
                try {
                    const mem = await this.client.getMemory(model.name);
                    if (calcFreeBytes(mem) < model.fit_target_mib * 1024 * 1024) {
                        success = false;
                    }
                } catch { }
            }

            if (success) {
                cur = mid;
                lo = mid + 1;
            } else {
                hi = mid - 256;
            }

            iterations++;
            // TODO: remove paranoid check
            if (iterations > 100) throw new Error(`findMaxCtx reached 100 iterations`);
        }

        // reload to known good max ctx since a failure leaves context unusable
        try {
            await this.client.reloadModel({ n_ctx: cur }, model.name);
        } catch (err) {
            throw new Error(`final reload to known-good max failed with err: ${err}`);
        }

        // use n_ctx_seq as max ctx
        const slots = await this.client.getSlots(model.name);
        if (slots.length === 0) throw new Error(`expected nonzero slots`);
        const max = slots[0].n_ctx;

        log.info(`max ctx for ${model} found after ${iterations} iterations: ${max}`);
        return max;
    }
}