import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../../client/llama-api.js";
import { LoadStatus, type ModelEntry, type StrategyId, type Tokens } from "../../config/types.js";
import { logger } from "../../logger.js";
import type { Strategy, StrategyContext } from "../types.js";
import type { ReloadParams, SlotSave } from "../../client/types.js";

const log: ConsolaInstance = logger.withTag('evict-task-model');

export class EvictTaskModel implements Strategy {

    id: StrategyId = 'evict-task-model';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return (
            ctx.models['task'] !== undefined &&
            ctx.models['task'].status !== LoadStatus.UNLOADED
        );
    }

    // unloads the task model
    async applyNoSend(ctx: StrategyContext, params: ReloadParams, saves?: SlotSave[]): Promise<ReloadParams> {
        const model_task = ctx.models['task'];
        if (!model_task) {
            throw new Error(`EvictTaskModel.applyNoSend: task model not found`);
        }

        await this.#unloadTaskModel(model_task);

        return params;
    }

    async apply(ctx: StrategyContext): Promise<void> {
        const model_task = ctx.models['task'];
        if (!model_task) {
            throw new Error(`EvictTaskModel.apply: task model not found`);
        }

        await this.#unloadTaskModel(model_task);
    }

    async #unloadTaskModel(model: ModelEntry): Promise<void> {
        if (model.status === LoadStatus.UNLOADED) {
            log.error(`task model ${model.name} is already unloaded`);
            return;
        }

        model.status = LoadStatus.UNLOADED;

        await this.client.unloadModelAndWait(model.name);
    }
}