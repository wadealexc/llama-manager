import type { ConsolaInstance } from "consola";
import { logger } from "../logger.js";
import { LoadStatus, type ModelEntry, type ModelId, type ModelRole, type StrategyId } from "../config/types.js";
import type { LlamaAPI } from "../client/llama-api.js";
import type { MemoryResponse } from "../client/types.js";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { PrintMemory } from "./print-memory.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import type { Strategy, StrategyContext } from "./types.js";

const log: ConsolaInstance = logger.withTag('cost-model');

const COST_MODEL_PATH = "./generated-cost-model.json";

export type CostModelCache = {
    config_hash: string;
    models: Record<string, {
        ladder: StrategyId[];
        max_ctx: Record<string, number>;
    }>;
}

type CostModelParams = {
    config_path: string;
    force_build: boolean;
    models: Models;
    strategies: Map<StrategyId, Strategy>;
    signal: AbortSignal;
}

type Models = Partial<Record<ModelRole, ModelEntry>>;

export class CostModel {

    client: LlamaAPI;
    printer: PrintMemory;

    constructor(client: LlamaAPI) {
        this.client = client;
        this.printer = new PrintMemory(log, this.client);
    }

    async initCostModel(params: CostModelParams): Promise<CostModelCache> {
        const config_hash = createHash('sha256')
            .update(readFileSync(params.config_path))
            .digest('hex');

        if (!params.force_build) {
            const cm = this.#loadCostModel(params.config_path, config_hash);
            if (cm) return cm;

            log.info(`unable to load cost model from cache; rebuilding`);
        }

        const cm = await this.#buildCostModel(params, config_hash);
        writeFileSync(COST_MODEL_PATH, JSON.stringify(cm, null, 2));
        log.info(`cost model cached to ${COST_MODEL_PATH}`);

        return cm;
    }

    #loadCostModel(config_path: string, config_hash: string): CostModelCache | undefined {
        let raw: string;
        try {
            raw = readFileSync(COST_MODEL_PATH, 'utf8');
        } catch (err: any) {
            log.info(`could not read cost model at path ${COST_MODEL_PATH}; err: ${err}`);
            return undefined;
        }

        let cache: CostModelCache;
        try {
            cache = JSON.parse(raw) as CostModelCache;
        } catch {
            log.warn(`cost model cache: failed to parse`);
            return undefined;
        }

        if (cache.config_hash !== config_hash) {
            log.info(`cost model cache: config changed`);
            return undefined;
        }

        return cache;
    }

    /**
     * TODO think about:
     * - Locking model loads/inference while building cost model
     * - "trying" to load both models, and failing gracefully if they won't fit
     *   - maybe evaluating whether we think a model load will succeed
     * 
     * TODO: this is still super messy
     */
    async #buildCostModel(params: CostModelParams, config_hash: string): Promise<CostModelCache> {
        log.info('building cost model');
        const t = performance.now();

        const model_main = params.models['main']!;
        const model_task = params.models['task'];

        // Load main and task model sequentially
        try {
            await this.client.loadModelAndWait(model_main.name, params.signal);
            model_main.status = LoadStatus.WEIGHTS_ONLY;
            if (model_task) {
                await this.client.loadModelAndWait(model_task.name, params.signal);
                model_task.status = LoadStatus.WEIGHTS_ONLY;
            }
        } catch (err) {
            log.error(`buildCostModel: error loading models: ${err}`);
            throw err;
        }

        await this.printer.print("Weights only", params.models);

        const cache: CostModelCache = { config_hash, models: {} };

        // If we have a task model, calculate its max context while the main model is loaded
        if (model_task) {
            const task_max = await this.findMaxCtx(MIN_ALLOWED_CTX, model_task);

            cache.models[model_task.name] = { ladder: [], max_ctx: { '0': task_max } };
            await this.printer.print(`Task model @ ${task_max.toLocaleString()} ctx`, params.models, []);

            // Reset task model ctx
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, model_task.name);
        }

        // Calculate main model max context with no strategies
        let main_cur_ctx = await this.findMaxCtx(MIN_ALLOWED_CTX, model_main);

        const main_ladder: StrategyId[] = [];
        const main_max_ctx: Record<string, number> = { '0': main_cur_ctx };

        await this.printer.print(
            `Main model @ ${main_cur_ctx.toLocaleString()} ctx`,
            params.models,
            main_ladder,
        );

        const st_context: StrategyContext = {
            target: 'main',
            models: params.models,
        }

        // Apply each configured strategy sequentially and find the main model's new max at each step
        for (const strat_id of model_main.ladder) {
            const strat = params.strategies.get(strat_id)!;
            if (!strat.canApply(st_context)) {
                log.info(`Model ${model_main.name} incompatible with strategy ${strat.id}; skipping`);
                continue;
            }

            const strat_t = performance.now();
            await strat.apply(st_context);
            const find_sec = ((performance.now() - strat_t) / 1000).toFixed(2);

            log.debug(`checking free memory after applying ${strat_id}:`);
            const mem = await this.client.getMemory(model_main.name);
            for (const dev of mem.devices) {
                const free = dev.free / (1024 ** 3);
                log.debug(`- ${dev.name}: ${free.toFixed(2)} GiB`);
            }

            const prev_max = main_cur_ctx;
            main_cur_ctx = await this.findMaxCtx(main_cur_ctx, model_main);

            const ctx_gain = main_cur_ctx - prev_max;
            if (ctx_gain < 0) {
                throw new Error(`buildCostModel: applying ${strat.id} results in ctx decrease of ${ctx_gain}`);
            }

            log.info(`strategy ${strat.id} applied in ${find_sec} sec for a gain of ${ctx_gain} tokens`);

            main_ladder.push(strat_id);
            main_max_ctx[String(main_ladder.length)] = main_cur_ctx;

            await this.printer.print(
                `Main model @ ${main_cur_ctx.toLocaleString()} ctx`,
                params.models,
                main_ladder,
            );
        }

        cache.models[model_main.name] = {
            ladder: main_ladder,
            max_ctx: main_max_ctx,
        };

        const build_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.info(`finished cost model in ${build_sec} sec`);

        return cache;
    }

    /**
     * TODO: 
     * - Math.ceil padding approach assumes kvunified. Needs fix for non-unified
     * - reduce iterations by interpolating fit from two good points
     * - this is generally inefficient; better would be exposing llama.cpp's fit via HTTP
     */
    async findMaxCtx(cur: number, model: ModelEntry): Promise<number> {
        log.info(`finding max ctx for ${model.name}`);

        const t = performance.now();
        const res = await this.#findMax(cur, model);
        const find_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.info(`found max ctx in ${find_sec} sec`);

        return res;
    }

    async #findMax(cur: number, model: ModelEntry): Promise<number> {
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
            } catch { }

            // We reloaded, but we may not be meeting our overhead target
            if (success) {
                // TODO: would be nice to have the bar fill here
                try {
                    const mem = await this.client.getMemory(model.name);
                    if (calcFreeBytes(mem) < model.fit_target_mib * 1024 * 1024) {
                        success = false;
                    }
                } catch { }
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

// TODO - currently assuming 1 GPU
function calcFreeBytes(mem: MemoryResponse): number {
    for (const dev of mem.devices) {
        if (dev.type !== "cpu") return dev.free;
    }
    return 0;
}