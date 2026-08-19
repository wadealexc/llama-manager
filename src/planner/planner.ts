import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ManagerConfig, ModelRole } from "../config/types.js";
import type { MemoryResponse } from "../client/types.js";
import type { ModelId } from "../types.js";
import { logger } from "../logger.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { PrintMemory } from "./print-memory.js";
import type { ModelEntry, StrategyId } from "../config/types.js";
import type { Strategy, StrategyContext } from "./types.js";
import { createStrategies } from "./strategies/index.js";

const log: ConsolaInstance = logger.withTag('planner');

export type PlannerDecision =
    | { action: "serve" }
    | { action: "serve"; degraded: StrategyId[] }
    | { action: "queued" }
    | { action: "fail"; reason: string };

// TODO - currently assuming 1 GPU
function calcFreeBytes(mem: MemoryResponse): number {
    for (const dev of mem.devices) {
        if (dev.type !== "cpu") return dev.free;
    }
    return 0;
}

export class Planner {

    client: LlamaAPI;
    config: ManagerConfig;

    models: Partial<Record<ModelRole, ModelEntry>>;
    // Map model -> number of strategies applied from ladder -> measured max ctx
    max_ctx: Map<ModelId, Map<number, number>> = new Map();

    strategies: Map<StrategyId, Strategy>;

    printer: PrintMemory;

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

        this.strategies = createStrategies(client);

        this.printer = new PrintMemory(log, this.client);
    }

    /**
     * TODO think about:
     * - Locking model loads/inference while building cost model
     * - "trying" to load both models, and failing gracefully if they won't fit
     *   - maybe evaluating whether we think a model load will succeed
     * 
     * - Do we need llama-api (or other) to keep track of which models are/are not loaded?
     *   - Does that belong in the planner instead?
     * 
     * - read strategy ladder from config
     */
    async buildCostModel(): Promise<void> {
        log.info('building cost model');
        const model_main = this.models['main']!;
        const model_task = this.models['task'];

        // Load main and task model sequentially
        try {
            await this.client.loadModelAndWait(model_main.name);
            if (model_task) await this.client.loadModelAndWait(model_task.name);
        } catch (err) {
            log.error(`buildCostModel error loading models: ${err}`);
            throw err;
        }

        await this.printer.print("Weights only", this.models);

        // If we have a task model, calculate its max context while the main model is loaded
        if (model_task) {
            const t = performance.now();
            const task_max = await this.#findMaxCtx(MIN_ALLOWED_CTX, model_task);
            const find_sec = ((performance.now() - t) / 1000).toFixed(2);
            log.debug(`time elapsed: (find max: ${find_sec} sec)`);

            this.max_ctx.set(model_task.name, new Map([[model_main.applied.length, task_max]]));

            await this.printer.print(`Task model @ ${task_max.toLocaleString()} ctx`, this.models, model_task.applied);

            // Reset task model ctx
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, model_task.name);
        }

        // Calculate main model max context with no strategies
        const t = performance.now();
        let main_cur_ctx = await this.#findMaxCtx(MIN_ALLOWED_CTX, model_main);
        const find_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.debug(`time elapsed: (find max: ${find_sec} sec)`);

        this.max_ctx.set(model_main.name, new Map([[model_main.applied.length, main_cur_ctx]]));

        await this.printer.print(`Main model @ ${main_cur_ctx.toLocaleString()} ctx`, this.models, model_main.applied);

        const st_context: StrategyContext = {
            target: 'main',
            models: this.models,
        }

        // Apply each configured strategy sequentially and find the main model's new max at each step
        for (const strat_id of model_main.ladder) {
            const strat = this.strategies.get(strat_id)!;
            if (!strat.canApply(st_context)) {
                log.info(`Model ${model_main.name} incompatible with strategy ${strat.id}; skipping`);
                continue;
            }

            const prev_max = main_cur_ctx;
            const timings = { apply_sec: '', find_sec: '' };
            
            {
                // Apply strategy
                const t = performance.now();
                await strat.applyNoSave(st_context);
                timings.apply_sec = ((performance.now() - t) / 1000).toFixed(2);
            }
            
            {
                // Find new max ctx
                const t = performance.now();
                main_cur_ctx = await this.#findMaxCtx(main_cur_ctx, model_main);
                timings.find_sec = ((performance.now() - t) / 1000).toFixed(2);
            }

            const ctx_gain = main_cur_ctx - prev_max;
            if (ctx_gain < 0) {
                throw new Error(`buildCostModel: applying ${strat.id} results in ctx decrease of ${ctx_gain}`);
            }

            // Update max ctx for this point in the strategy ladder
            model_main.applied.push(strat_id);
            this.max_ctx.set(model_main.name, new Map([[model_main.applied.length, main_cur_ctx]]));

            log.info(`applying strategy: ${strat.id} yields ctx gain of ${ctx_gain} tokens`);
            log.debug(`time elapsed: (apply: ${timings.apply_sec} sec | find max: ${timings.find_sec} sec)`);

            await this.printer.print(`Main model @ ${main_cur_ctx.toLocaleString()} ctx`, this.models, model_main.applied);
        }

        // Update in case we skipped some strategies
        model_main.ladder = model_main.applied;
        model_main.applied = [];

        log.info('waiting 20 seconds');
        await new Promise<void>((resolve) => setTimeout(resolve, 20000));

        log.info('unloading models');
        await Promise.allSettled([
            this.client.unloadModelAndWait(model_main.name),
            model_task ? this.client.unloadModelAndWait(model_task.name) : Promise.resolve(),
        ]);

        log.info('done!');
    }

    async decide(role: ModelRole, input_tokens: number): Promise<PlannerDecision> { 
        throw new Error('unimplemented');
    }

    async reset(role: ModelRole): Promise<void> {
        throw new Error('unimplemented');
    }

    /**
     * TODO: 
     * - Math.ceil padding approach assumes kvunified. Needs fix for non-unified
     * - reduce iterations by interpolating fit from two good points
     * - this is generally inefficient; better would be exposing llama.cpp's fit via HTTP
     */
    async #findMaxCtx(cur: number, model: ModelEntry): Promise<number> {
        log.info(`finding max ctx for ${model.name}`);

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
            } catch {}

            // We reloaded, but we may not be meeting our overhead target
            if (success) {
                // TODO: would be nice to have the bar fill here
                try {
                    const mem = await this.client.getMemory(model.name);
                    if (calcFreeBytes(mem) < model.fit_target_mib * 1024 * 1024) {
                        success = false;
                    }
                } catch {}
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