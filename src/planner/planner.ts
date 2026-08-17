import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ManagerConfig, ModelConfig, ModelRole, StrategyId } from "../config/types.js";
import type { MemoryResponse } from "../client/types.js";
import type { ModelId } from "../types.js";
import { logger } from "../logger.js";
import { DEFAULT_FIT_OVERHEAD_MIB, MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { PrintMemory } from "./print-memory.js";

const log: ConsolaInstance = logger.withTag('planner');

export type PlannerDecision =
    | { action: "serve" }
    | { action: "serve"; degraded: StrategyId[] }
    | { action: "queued" }
    | { action: "fail"; reason: string };

// When calculating max ctx, leave at least this amount of space free on the GPU
// TODO - parse from config and set default if unset
const MAX_CTX_OVERHEAD_MIB = DEFAULT_FIT_OVERHEAD_MIB;

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

    models: Partial<Record<ModelRole, ModelConfig>>;
    // Map model -> number of strategies applied from ladder -> measured max ctx
    max_ctx: Map<ModelId, Map<number, number>> = new Map();

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
            await this.client.loadModelAndWait(model_main.name, this.config.model_load);
            if (model_task) await this.client.loadModelAndWait(model_task.name, this.config.model_load);
        } catch (err) {
            log.error(`buildCostModel error loading models: ${err}`);
            throw err;
        }

        const strategies: StrategyId[] = [];

        await this.printer.print("Weights only", this.models);

        // If we have a task model, calculate its max context while the main model is loaded
        if (model_task) {
            const find_start = performance.now();
            const task_max = await this.#findMaxCtx(MIN_ALLOWED_CTX, model_task.name);
            const find_end = performance.now();
            const seconds = (find_end - find_start) / 1000;
            log.info(`found max_ctx in ${seconds.toFixed(2)} sec`);

            // TODO - don't really need this for kvunified. clean up/abstract.
            const slots = await this.client.getSlots(model_task.name);
            if (slots.length === 0) throw new Error(`expected nonzero slots`);

            const n_ctx_seq = slots[0].n_ctx;
            this.max_ctx.set(model_task.name, new Map([[strategies.length, n_ctx_seq]]));

            await this.printer.print(`Task model @ ${n_ctx_seq.toLocaleString()} ctx`, this.models, strategies);

            // Reset task model ctx
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, model_task.name);
        }

        // Calculate main model max context with no strategies
        const find_start = performance.now();
        let main_cur = await this.#findMaxCtx(MIN_ALLOWED_CTX, model_main.name);
        const find_end = performance.now();
        const seconds = (find_end - find_start) / 1000;
        log.info(`found max_ctx in ${seconds.toFixed(2)} sec`);

        // TODO - don't really need this for kvunified. clean up/abstract.
        const slots = await this.client.getSlots(model_main.name);
        if (slots.length === 0) throw new Error(`expected nonzero slots`);

        const n_ctx_seq = slots[0].n_ctx;
        this.max_ctx.set(model_main.name, new Map([[strategies.length, n_ctx_seq]]));

        await this.printer.print(`Main model @ ${n_ctx_seq.toLocaleString()} ctx`, this.models, strategies);

        if (model_task) {
            strategies.push('evict-task-model');

            // Calculate main model max context after unloading task model
            await this.client.unloadModelAndWait(model_task.name, this.config.model_load);

            const find_start = performance.now();
            main_cur = await this.#findMaxCtx(main_cur, model_main.name);
            const find_end = performance.now();
            const seconds = (find_end - find_start) / 1000;
            log.info(`found max_ctx in ${seconds.toFixed(2)} sec`);

            // TODO - don't really need this for kvunified. clean up/abstract.
            const slots = await this.client.getSlots(model_main.name);
            if (slots.length === 0) throw new Error(`expected nonzero slots`);

            const n_ctx_seq = slots[0].n_ctx;
            this.max_ctx.set(model_main.name, new Map([[strategies.length, n_ctx_seq]]));

            log.info(`max-ctx: ${main_cur} | n_ctx_seq: ${n_ctx_seq}`);

            delete this.models['task'];
            await this.printer.print(`Main model @ ${n_ctx_seq.toLocaleString()} ctx`, this.models, strategies);
        }

        // TODO - actually check model config for mmproj
        if (true) {
            strategies.push('mmproj-on-demand');

            let find_start = performance.now();
            await this.client.reloadModel({ mmproj: { path: "" } }, model_main.name);
            let find_end = performance.now();
            let seconds = (find_end - find_start) / 1000;
            log.info(`unloaded mmproj ${seconds.toFixed(2)} sec`);

            find_start = performance.now();
            main_cur = await this.#findMaxCtx(main_cur, model_main.name);
            find_end = performance.now();
            seconds = (find_end - find_start) / 1000;
            log.info(`found max_ctx in ${seconds.toFixed(2)} sec`);

            // TODO - don't really need this for kvunified. clean up/abstract.
            const slots = await this.client.getSlots(model_main.name);
            if (slots.length === 0) throw new Error(`expected nonzero slots`);

            const n_ctx_seq = slots[0].n_ctx;
            this.max_ctx.set(model_main.name, new Map([[strategies.length, n_ctx_seq]]));

            log.info(`max-ctx: ${main_cur} | n_ctx_seq: ${n_ctx_seq}`);
            await this.printer.print(`Main model @ ${n_ctx_seq.toLocaleString()} ctx`, this.models, strategies);
        }

        // TODO - actually check model config for spec
        if (true) {
            strategies.push('disable-spec');

            let find_start = performance.now();
            await this.client.reloadModel({ spec: { types: ["none"] } }, model_main.name);
            let find_end = performance.now();
            let seconds = (find_end - find_start) / 1000;
            log.info(`unloaded spec ${seconds.toFixed(2)} sec`);

            find_start = performance.now();
            main_cur = await this.#findMaxCtx(main_cur, model_main.name);
            find_end = performance.now();
            seconds = (find_end - find_start) / 1000;
            log.info(`found max_ctx in ${seconds.toFixed(2)} sec`);

            // TODO - don't really need this for kvunified. clean up/abstract.
            const slots = await this.client.getSlots(model_main.name);
            if (slots.length === 0) throw new Error(`expected nonzero slots`);

            const n_ctx_seq = slots[0].n_ctx;
            this.max_ctx.set(model_main.name, new Map([[strategies.length, n_ctx_seq]]));

            log.info(`max-ctx: ${main_cur} | n_ctx_seq: ${n_ctx_seq}`);
            await this.printer.print(`Main model @ ${n_ctx_seq.toLocaleString()} ctx`, this.models, strategies);
        }

        log.info('waiting 20 seconds');
        await new Promise<void>((resolve) => setTimeout(resolve, 20000));

        log.info('unloading models');
        await Promise.allSettled([
            this.client.unloadModelAndWait(model_main.name, this.config.model_load),
            model_task ? this.client.unloadModelAndWait(model_task.name, this.config.model_load) : Promise.resolve(),
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
    async #findMaxCtx(cur: number, model: ModelId): Promise<number> {
        log.info(`finding max ctx for ${model}`);

        const models = await this.client.getModels();
        const model_info = models.find((m) => m.id === model);
        if (!model_info) throw new Error(`model not found: ${model}`);
        if (!model_info.meta) throw new Error(`info did not contain 'meta' key: ${model}`);

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
                const status = await this.client.reloadModel({ n_ctx: mid }, model);
                success = status.success;
            } catch {}

            // We reloaded, but we may not be meeting our overhead target
            if (success) {
                // TODO: would be nice to have the bar fill here
                try {
                    const mem = await this.client.getMemory(model);
                    if (calcFreeBytes(mem) < MAX_CTX_OVERHEAD_MIB * 1024 * 1024) {
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
            await this.client.reloadModel({ n_ctx: cur }, model);
        } catch (err) {
            throw new Error(`final reload to known-good max failed with err: ${err}`);
        }

        log.info(`max ctx for ${model} found after ${iterations} iterations: ${cur}`);
        return cur;
    }
}