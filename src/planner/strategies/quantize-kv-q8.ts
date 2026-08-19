import type { LlamaAPI } from "../../client/llama-api.js";
import type { ModelState, StrategyId, Tokens } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";

export class QuantizeKvQ8 implements Strategy {

    id: StrategyId = 'quantize-kv-q8';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return (isKvMorePreciseThanQ8(ctx.models[ctx.target]?.current_state));
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
            throw new Error(`QuantizeKvQ8.#apply: target not found`);
        }

        await this.client.reloadModel({
            n_ctx: n_ctx,
            cache_type_k: 'q8_0',
            cache_type_v: 'q8_0'
        }, target.name);

        if (n_ctx !== undefined) target.current_state.n_ctx = n_ctx;
        target.current_state.cache_type_k = 'q8_0';
        target.current_state.cache_type_v = 'q8_0';
    }
}

// TODO: Does not handle ctk/ctv individually being more precise.
function isKvMorePreciseThanQ8(st?: ModelState): boolean {
    const ctk = st?.cache_type_k;
    const ctv = st?.cache_type_k;

    return ctk === 'f16' && ctv === 'f16';
}