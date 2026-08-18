import type { LlamaAPI } from "../../client/llama-api.js";
import type { Strategy, StrategyContext, StrategyId } from "../types.js";

export class EvictTaskModel implements Strategy {

    id: StrategyId = 'evict-task-model';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models['task'];
    }

    async apply(ctx: StrategyContext): Promise<void> {
        await this.applyNoSave(ctx); // TODO - implement save/restore for main model
    }

    // Evict task model and resize main model kvcache
    async applyNoSave(ctx: StrategyContext): Promise<void> {
        const model_task = ctx.models['task'];
        const model_main = ctx.models['main'];

        if (!model_task) {
            throw new Error(`EvictTaskModel.applyNoSave: expected task model`);
        }

        if (!model_main) {
            throw new Error(`EvictTaskModel.applyNoSave: expected main model`)
        }

        await this.client.unloadModelAndWait(model_task.name);
        // TODO - new max ctx
        await this.client.reloadModel({ n_ctx: 1 }, model_main.name);
    }
}