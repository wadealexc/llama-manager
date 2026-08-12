import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ManagerConfig, ModelConfig, ModelRole, StrategyId } from "../config/types.js";
import type { ModelId } from "../types.js";
import { logger } from "../logger.js";

const log: ConsolaInstance = logger.withTag('planner');

export type PlannerDecision =
    | { action: "serve" }
    | { action: "serve"; degraded: StrategyId[] }
    | { action: "queued" }
    | { action: "fail"; reason: string };

export class Planner {

    client: LlamaAPI;
    config: ManagerConfig;

    models: Partial<Record<ModelRole, ModelConfig>>;
    max_ctx: Map<ModelId, Map<StrategyId[], number>> = new Map();

    constructor(client: LlamaAPI, config: ManagerConfig) {
        this.client = client;
        this.config = config;
        this.models = config.models;

        const model_main = this.models['main'];
        const model_task = this.models['task'];
        // only main model is required
        if (!model_main) throw new Error(`Planner: expected main model`);

        this.max_ctx.set(model_main.name, new Map());
        if (model_task) this.max_ctx.set(model_task.name, new Map());
    }

    async buildCostModel(): Promise<void> {
        log.info('building cost model');
        const model_main = this.models['main']!;
        const model_task = this.models['task'];

        // Load main and task model sequentially
        try {
            await this.client.loadModelAndWait(model_main.name, this.config.model_load);
            if (model_task) await this.client.loadModelAndWait(model_task.name, this.config.model_load);
        } catch (err) {
            log.error(`buildCostModel error loading models: ${err}`);
            throw err;
        }
        
        const minfo_main = await this.client.getMemory(model_main.name);
        log.info(`main model memory info: ${JSON.stringify(minfo_main, null, 2)}`);

        if (model_task) {
            const minfo_task = await this.client.getMemory(model_task.name);
            log.info(`task model memory info: ${JSON.stringify(minfo_task, null, 2)}`);
        }

        log.info('unloading models');
        await Promise.allSettled([
            this.client.unloadModelAndWait(model_main.name, this.config.model_load),
            model_task ? this.client.unloadModelAndWait(model_task.name, this.config.model_load) : Promise.resolve(),
        ]);

        log.info('done!');
    }

    async decide(role: ModelRole, input_tokens: number): Promise<PlannerDecision> { 
        return { action: "serve" }; 
    }

    async reset(role: ModelRole): Promise<void> {

    }
}
