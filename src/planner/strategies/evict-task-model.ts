import type { LlamaAPI } from "../../client/llama-api.js";
import type { StrategyId, Tokens } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";

export class EvictTaskModel implements Strategy {

    id: StrategyId = 'evict-task-model';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models['task']?.is_loaded;
    }

    // TODO - implement save/restore
    async apply(ctx: StrategyContext, n_ctx: Tokens): Promise<void> {
        await this.#apply(ctx, n_ctx);
    }

    async applyNoSave(ctx: StrategyContext): Promise<void> {
        await this.#apply(ctx);
    }

    // Evict task model and resize main model kvcache
    async #apply(ctx: StrategyContext, n_ctx?: Tokens): Promise<void> {
        const model_task = ctx.models['task'];
        const model_main = ctx.models['main'];

        if (!model_task) {
            throw new Error(`EvictTaskModel.#apply: expected task model`);
        }

        if (!model_main) {
            throw new Error(`EvictTaskModel.#apply: expected main model`)
        }

        await this.client.unloadModelAndWait(model_task.name);
        await this.client.reloadModel({ n_ctx: n_ctx }, model_main.name);

        if (n_ctx !== undefined) model_main.current_state.n_ctx = n_ctx;
        model_task.is_loaded = false;
    }
}