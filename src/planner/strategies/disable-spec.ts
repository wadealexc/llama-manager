import type { LlamaAPI } from "../../client/llama-api.js";
import type { StrategyId, Tokens } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";

export class DisableSpec implements Strategy {

    id: StrategyId = 'disable-spec';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models[ctx.target]?.current_state.spec_loaded;
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
            throw new Error(`DisableSpec.#apply: target not found`);
        }

        await this.client.reloadModel({
            n_ctx: n_ctx,
            spec: { types: ["none"] }
        }, target.name);

        if (n_ctx !== undefined) target.current_state.n_ctx = n_ctx;
        target.current_state.spec_loaded = false;
    }
}