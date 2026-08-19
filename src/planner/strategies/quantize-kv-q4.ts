import type { LlamaAPI } from "../../client/llama-api.js";
import type { ModelState, StrategyId, Tokens } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";

export class QuantizeKvQ4 implements Strategy {

    id: StrategyId = 'quantize-kv-q4';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return (isKvMorePreciseThanQ4(ctx.models[ctx.target]?.current_state));
    }

    // TODO - implement save/restore
    async apply(ctx: StrategyContext, n_ctx: Tokens): Promise<void> {
        await this.#apply(ctx, n_ctx);
    }

    async applyNoSave(ctx: StrategyContext): Promise<void> {
        await this.#apply(ctx);
    }

    async #apply(ctx: StrategyContext, n_ctx?: Tokens): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`QuantizeKvQ4.#apply: target not found`);
        }

        await this.client.reloadModel({
            n_ctx: n_ctx,
            cache_type_k: 'q4_0',
            cache_type_v: 'q4_0'
        }, target.name);

        if (n_ctx !== undefined) target.current_state.n_ctx = n_ctx;
        target.current_state.cache_type_k = 'q4_0';
        target.current_state.cache_type_v = 'q4_0';
    }
}

// TODO: Does not handle ctk/ctv individually being more precise.
function isKvMorePreciseThanQ4(st?: ModelState): boolean {
    const ctk = st?.cache_type_k;
    const ctv = st?.cache_type_k;

    return (ctk === 'f16' || ctk === 'q8_0') && (ctv === 'f16' || ctv === 'q8_0');
}