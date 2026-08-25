import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ManagerConfig, ModelId, ModelRole, Tokens } from "../config/types.js";
import type { MemoryResponse } from "../client/types.js";
import { logger } from "../logger.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { PrintMemory } from "./print-memory.js";
import type { ModelEntry, StrategyId } from "../config/types.js";
import type { Strategy, StrategyContext } from "./types.js";
import { createStrategies } from "./strategies/index.js";

const log: ConsolaInstance = logger.withTag('planner');

export interface Client {
    completions(body: unknown, model: ModelId, signal: AbortSignal): Promise<Response>;
}

type PlannerCallback = (body: unknown, client: Client, signal: AbortSignal) => Promise<void>;

type TokenRequirement = {
    tokens_in: number;
    est_tokens_out?: number;
}

type ActiveState = {
    pending: boolean;
    model: ModelId;
    readers: number;
}

type ReadHandle = {
    release(): void;
}

type Waiter = {
    model: ModelId;
    req: TokenRequirement;
    resolve: (handle: ReadHandle) => void;
    reject: (reason: any) => void;
}

type Plan = {
    strategy: StrategyId;
    n_ctx: number;
}[]

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

    shutdown_ctrl: AbortController = new AbortController();

    models: Partial<Record<ModelRole, ModelEntry>>;
    model_to_role: Map<ModelId, ModelRole> = new Map();

    // TODO: reset on idle
    previous_state: Map<ModelId, ModelEntry> = new Map();

    // Map model -> number of strategies applied from ladder -> measured max ctx
    max_ctx: Map<ModelId, Map<number, number>> = new Map();

    active: ActiveState | null = null;
    waiting: Waiter[] = [];

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
        this.model_to_role.set(model_main.name, 'main');
        this.previous_state.set(model_main.name, structuredClone(model_main));

        this.max_ctx.set(model_main.name, new Map());
        if (model_task) {
            this.max_ctx.set(model_task.name, new Map());
            this.model_to_role.set(model_task.name, 'task');
            this.previous_state.set(model_task.name, structuredClone(model_task));
        }

        this.strategies = createStrategies(client);

        this.printer = new PrintMemory(log, this.client);
    }

    /**
     * TODO think about:
     * - Locking model loads/inference while building cost model
     * - "trying" to load both models, and failing gracefully if they won't fit
     *   - maybe evaluating whether we think a model load will succeed
     */
    async buildCostModel(): Promise<void> {
        log.info('building cost model');
        const model_main = this.models['main']!;
        const model_task = this.models['task'];

        // Load main and task model sequentially
        try {
            await this.#loadModel(model_main);
            if (model_task) await this.#loadModel(model_task);
        } catch (err) {
            log.error(`buildCostModel: error loading models: ${err}`);
            throw err;
        }

        await this.printer.print("Weights only", this.models);

        // If we have a task model, calculate its max context while the main model is loaded
        if (model_task) {
            const t = performance.now();
            const task_max = await this.#findMaxCtx(MIN_ALLOWED_CTX, model_task);
            const find_sec = ((performance.now() - t) / 1000).toFixed(2);
            log.debug(`time elapsed: (find max: ${find_sec} sec)`);

            model_task.initial_state.n_ctx = task_max;
            model_task.current_state.n_ctx = task_max;
            this.max_ctx.set(model_task.name, new Map([[model_main.applied.length, task_max]]));
            this.previous_state.set(model_task.name, model_task);

            await this.printer.print(`Task model @ ${task_max.toLocaleString()} ctx`, this.models, model_task.applied);

            // Reset task model ctx
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, model_task.name);
        }

        // Calculate main model max context with no strategies
        const t = performance.now();
        let main_cur_ctx = await this.#findMaxCtx(MIN_ALLOWED_CTX, model_main);
        const find_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.debug(`time elapsed: (find max: ${find_sec} sec)`);

        model_main.initial_state.n_ctx = main_cur_ctx;
        model_main.current_state.n_ctx = main_cur_ctx;
        this.max_ctx.set(model_main.name, new Map([[model_main.applied.length, main_cur_ctx]]));
        this.previous_state.set(model_main.name, model_main);

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
            this.#unloadModel(model_main),
            model_task ? this.#unloadModel(model_task) : Promise.resolve(),
        ]);

        log.info('done!');
    }

    async decide(body: unknown, model: ModelId, client_signal: AbortSignal, cb: PlannerCallback): Promise<void> {
        const signal = AbortSignal.any([client_signal, this.shutdown_ctrl.signal]);

        // tokenize input and estimate required tokens
        const req = await this.#withModel(model, { tokens_in: 0 }, async (entry: ModelEntry) => {
            const token_ids = await this.client.tokenize(body, model, signal);
            return {
                tokens_in: token_ids.length,
                est_tokens_out: token_ids.length + entry.expected_response_tokens,
            };
        });

        // validate that we are able to serve the request
        const max_tokens_possible = this.#maxTokensForModel(model);
        if (req.tokens_in > max_tokens_possible) {
            throw new Error(`unable to serve request for ${model}; tokens in: ${req.tokens_in} | max tokens: ${max_tokens_possible}`);
        }

        // stream from model when token requirement is met
        await this.#withModel(model, req, async (entry: ModelEntry) => {
            const n_ctx = entry.current_state.n_ctx;
            if (n_ctx < req.est_tokens_out) {
                log.warn(
                    `unable to guarantee entire context for ${model}: `,
                    `in: ${req.tokens_in} | requested: ${req.est_tokens_out} | got: ${n_ctx}`
                );
            }

            await cb(body, this.client, signal);
        });
    }

    async #withModel<T>(model: ModelId, req: TokenRequirement, cb: (entry: ModelEntry) => Promise<T>): Promise<T> {
        let handle: ReadHandle;
        if (this.canServeImmediately(model, req)) {
            this.active!.readers++;
            handle = this.#getHandle();
        } else {
            handle = await new Promise((resolve, reject) => {
                this.waiting.push({ model, req, resolve, reject });
                this.#maybeTransition();
            });
        }

        try {
            const entry = this.entry(model)!;
            return await cb(entry);
        } finally {
            handle.release();
        }
    }

    #getHandle(): ReadHandle {
        return {
            release: () => {
                if (this.active) {
                    this.active.readers--;
                    if (this.active.readers !== 0) return;
                }
                
                this.#maybeTransition();
            }
        }
    }

    async #maybeTransition(): Promise<void> {
        if (this.active && (this.active.readers !== 0 || this.active.pending)) {
            return;
        }

        let model: ModelEntry;
        let plan: Plan | undefined = [];

        while (true) {
            if (this.waiting.length === 0) {
                return;
            }

            const head: Waiter = this.waiting[0];
            model = this.entry(head.model)!;

            const idx = this.#largestRequestForModel(model.name);
            if (idx === undefined) {
                log.warn(`#maybeTransition expected request for ${model.name}`);
                return;
            }

            const largest = this.waiting.at(idx)!;
            plan = this.#getMinAddtlStrategies(model.name, largest.req);
            if (plan) {
                break;
            }

            // unable to apply: reject/remove request and move to the next one
            try {
                largest.reject(`unable to serve request: insufficient tokens when all strategies applied`);
            } catch (err: any) {
                log.error(`error rejecting largest request: ${err}`);
            }

            this.waiting.splice(idx, 1);
            log.info(
                `removed request to ${model.name}; max servable: ${this.#maxTokensForModel(model.name)} ` +
                `req: (tokens_in: ${largest.req.tokens_in} | est_tokens_out: ${largest.req.est_tokens_out})`
            );
        }

        if (plan.length !== 0) {
            log.info(`applying strategies to ${model.name}: ${plan}`);
        }

        /**
         * TODO: there needs to be a clearer distinction between:
         * - can serve a request: "weights are loaded and kvcache is loaded"
         * - can't serve a request (needs kvcache): "weights are loaded"
         * - can't serve a request (needs kvcache AND weights): "nothing is loaded"
         * 
         * currently we only have `this.active` and `ModelEntry.is_loaded`
         */
        if (!this.active) {
            this.active = { pending: true, model: model.name, readers: 0 };
        }

        const prior_active = this.active.model;
        this.active.pending = true;
        this.active.model = model.name;

        if (prior_active !== model.name) {
            await this.#popKV(prior_active);
        }

        if (!model.is_loaded) {
            await this.#loadModel(model);
        }

        for (const step of plan) {
            const impl = this.strategies.get(step.strategy)!;

            try {
                await impl.apply({ target: this.model_to_role.get(model.name)!, models: this.models }, step.n_ctx);
                model.applied.push(step.strategy);
            } catch (err: any) {
                log.error(`#maybeTransition: error applying strategy ${step.strategy} to model ${model.name}: err`);
                break;
            }
        }

        this.active.pending = false;

        // fire any waiters satisfied by new active state
        const ready = this.waiting.filter((w) =>
            w.model === model.name && this.canServeImmediately(model.name, w.req)
        );

        this.waiting = this.waiting.filter(w => !ready.includes(w));
        for (const r of ready) {
            this.active.readers++;
            r.resolve(this.#getHandle());
        }
    }

    async reset(model: ModelId): Promise<void> {
        throw new Error('unimplemented');
    }

    async shutdown(): Promise<void> {
        this.shutdown_ctrl.abort('shutdown request received');

        const waiting = this.waiting.splice(0, this.waiting.length);
        for (const w of waiting) {
            try {
                w.reject('shutting down');
            } catch { }
        }
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

    // Load model and restore to previous state
    // TODO: restore kvcache and use fewer reloadModel calls
    async #loadModel(model: ModelEntry): Promise<void> {
        if (model.is_loaded) return;

        await this.client.loadModelAndWait(model.name, this.shutdown_ctrl.signal);
        model.is_loaded = true;

        const prev_state = this.previous_state.get(model.name);
        if (prev_state) {
            log.info(
                `#loadModel: restoring applied strategies: ${JSON.stringify(prev_state.applied)} |`,
                `ctx: ${prev_state.current_state.n_ctx}`,
            );
            await this.#applyStrategies(model, prev_state.applied);
            await this.client.reloadModel(
                { n_ctx: prev_state.current_state.n_ctx },
                model.name,
                this.shutdown_ctrl.signal,
            );
            model.current_state.n_ctx = prev_state.current_state.n_ctx;
        }
    }

    // Unload model and cache previous state
    // TODO: save kvcache
    async #unloadModel(model: ModelEntry): Promise<void> {
        if (!model.is_loaded) return;

        await this.client.unloadModelAndWait(model.name, this.shutdown_ctrl.signal);

        // checkpoint current state
        const prev = structuredClone(model);
        this.previous_state.set(model.name, prev);

        // reset model
        model.is_loaded = false;
        model.applied = [];
        model.current_state = model.initial_state;
    }

    // TODO: save kvcache
    async #popKV(model: ModelId): Promise<void> {
        const entry = this.entry(model)!;
        if (!entry.is_loaded) return;

        // checkpoint current state
        const prev = structuredClone(entry);
        this.previous_state.set(entry.name, prev);

        try {
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, entry.name, this.shutdown_ctrl.signal);
        } catch (err: any) {
            log.error(`#stashKV: error reloading model: ${err}`);
        }

        entry.current_state.n_ctx = MIN_ALLOWED_CTX;
    }

    async #applyStrategies(model: ModelEntry, strategies: StrategyId[]): Promise<void> {
        for (const strategy of strategies) {
            if (model.applied.includes(strategy)) {
                log.info(`#applyStrategies: ${strategy} already applied; skipping`);
                continue;
            }

            const impl = this.strategies.get(strategy);
            if (!impl) throw new Error(`implementation not found for strategy: ${strategy}`);

            const strategy_context = { target: model.role, models: this.models };
            if (!impl.canApply(strategy_context)) {
                log.error(`#applyStrategies: model ${model.name} incompatible with strategy ${strategy}; skipping`);
                break;
            }

            await impl.applyNoSave(strategy_context);
            model.applied.push(strategy);
        }
    }

    #maxTokensForModel(model: ModelId): number {
        const entry = this.entry(model);
        if (!entry) {
            return 0;
        }

        const n_strats = entry.ladder.length;
        return this.max_ctx.get(model)!.get(n_strats) ?? 0;
    }

    #largestRequestForModel(model: ModelId): number | undefined {
        let max: Waiter | undefined;
        let idx: number = 0;
        for (const [i, waiter] of this.waiting.entries()) {
            if (waiter.model !== model) continue;

            if (waiter.req.tokens_in > (max?.req.tokens_in ?? 0)) {
                max = waiter;
                idx = i;
            }
        }

        return max ? idx : undefined;
    }

    // get the fewest additional strategies on top of currently-applied that will serve the request
    // evaluates using previous_state if model is not active
    // return undefined if unable to serve
    #getMinAddtlStrategies(model: ModelId, req: TokenRequirement): Plan | undefined {
        const plan: Plan = [];
        const entry = this.entry(model)!;
        const ctx_per_strats = this.max_ctx.get(model)!;
        const ctx_required = req.est_tokens_out ?? req.tokens_in;

        // get currently applied, or load from previous state
        let applied: StrategyId[];
        if (entry.is_loaded) {
            applied = entry.applied;
        } else {
            applied = this.previous_state.get(model)!.applied;
        }

        // no additional strategies needed
        const n_applied = applied.length;
        if (ctx_per_strats.get(n_applied)! >= ctx_required) {
            return plan;
        }

        // add strategies until request is satisfied
        let n_ctx = 0;
        for (const [i, strategy] of entry.ladder.entries()) {
            if (applied.includes(strategy)) {
                continue;
            }

            n_ctx = ctx_per_strats.get(i + 1)!;
            plan.push({ strategy, n_ctx });

            if (n_ctx >= ctx_required) {
                break;
            }
        }

        if (n_ctx >= ctx_required) {
            return plan;
        } else if (n_ctx >= req.tokens_in) {
            log.warn(`unable to guarantee requested tokens ${ctx_required} for ${model}; serving ${n_ctx} instead`);
            return plan;
        }

        // unable to serve request
        return undefined;
    }

    entry(model: ModelId): ModelEntry | undefined {
        const role = this.model_to_role.get(model);
        return role ? this.models[role] : undefined;
    }

    canServeImmediately(model: ModelId, req: TokenRequirement): boolean {
        const entry = this.entry(model);
        if (!entry) return false;
        if (!entry.is_loaded) return false;

        const ctx_cap_reached = entry.applied.length === entry.ladder.length;
        const meets_token_req = ctx_cap_reached
            ? entry.current_state.n_ctx >= req.tokens_in
            : entry.current_state.n_ctx >= (req.est_tokens_out ?? req.tokens_in);

        return (
            this.active !== null
            && !this.active.pending
            && this.active.model === model
            && meets_token_req
        );
    }
}