import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../../client/llama-api.js";
import type { ReloadParams, SlotSave } from "../../client/types.js";
import type { ModelEntry, ModelState, StrategyId } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";
import { logger } from "../../logger.js";

const log: ConsolaInstance = logger.withTag('quantize-kv-q4');

export class QuantizeKvQ4 implements Strategy {

    id: StrategyId = 'quantize-kv-q4';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return isKvMorePreciseThanQ4(ctx.models[ctx.target]?.current_state);
    }

    // TODO: restore KVCache
    async applyNoSend(ctx: StrategyContext, params: ReloadParams, saves?: SlotSave[]): Promise<ReloadParams> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`QuantizeKvQ4.applyNoSend: target not found`);
        }

        this.#updateCurrentState(target);

        return this.#getParams(params);
    }

    async apply(ctx: StrategyContext): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`QuantizeKvQ4.apply: target not found`);
        }

        this.#updateCurrentState(target);

        const params = this.#getParams({});
        await this.client.reloadModel(params, target.name);
    }

    #getParams(params: ReloadParams): ReloadParams {
        return {
            ...params,
            cache_type_k: 'q4_0',
            cache_type_v: 'q4_0'
        };
    }

    #updateCurrentState(model: ModelEntry): void {
        model.current_state.cache_type_k = 'q4_0';
        model.current_state.cache_type_v = 'q4_0';
    }
}

// TODO: Does not handle ctk/ctv individually being more precise.
function isKvMorePreciseThanQ4(st?: ModelState): boolean {
    const ctk = st?.cache_type_k;
    const ctv = st?.cache_type_v;

    return (ctk === 'f16' || ctk === 'q8_0') && (ctv === 'f16' || ctv === 'q8_0');
}