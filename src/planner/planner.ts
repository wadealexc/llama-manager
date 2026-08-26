import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import { LoadStatus, type ManagerConfig, type ModelId, type ModelRole } from "../config/types.js";
import type { ReloadParams, SlotSave } from "../client/types.js";
import { logger } from "../logger.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { PrintMemory } from "./print-memory.js";
import type { ModelEntry, StrategyId } from "../config/types.js";
import type { Strategy, StrategyContext } from "./types.js";
import { createStrategies } from "./strategies/index.js";
import { CostModel } from "./cost-model.js";
import { readdir, unlink } from "node:fs/promises";
import { join } from "node:path";

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
    strategies: StrategyId[];
    n_ctx: number;
}

type RestorePoint = {
    n_ctx: number;
    strategies: StrategyId[];
    slots?: SlotSave[];
}

export class Planner {

    client: LlamaAPI;
    config: ManagerConfig;

    shutdown_ctrl: AbortController = new AbortController();

    models: Partial<Record<ModelRole, ModelEntry>>;
    model_to_role: Map<ModelId, ModelRole> = new Map();

    // TODO: reset on idle
    restore_points: Map<ModelId, RestorePoint> = new Map();

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
        this.max_ctx.set(model_main.name, new Map());

        if (model_task) {
            this.max_ctx.set(model_task.name, new Map());
            this.model_to_role.set(model_task.name, 'task');
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
        const t = performance.now();

        const cm = new CostModel(this.client);

        const model_main = this.models['main']!;
        const model_task = this.models['task'];

        // Load main and task model sequentially
        try {
            await this.#loadWeights(model_main);
            if (model_task) await this.#loadWeights(model_task);
        } catch (err) {
            log.error(`buildCostModel: error loading models: ${err}`);
            throw err;
        }

        await this.printer.print("Weights only", this.models);

        // If we have a task model, calculate its max context while the main model is loaded
        if (model_task) {
            const task_max = await cm.findMaxCtx(MIN_ALLOWED_CTX, model_task);

            model_task.initial_state.n_ctx = task_max;
            model_task.current_state.n_ctx = task_max;
            this.max_ctx.set(model_task.name, new Map([[model_main.applied.length, task_max]]));

            await this.printer.print(`Task model @ ${task_max.toLocaleString()} ctx`, this.models, model_task.applied);

            // Reset task model ctx
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, model_task.name);
        }

        // Calculate main model max context with no strategies
        let main_cur_ctx = await cm.findMaxCtx(MIN_ALLOWED_CTX, model_main);

        model_main.initial_state.n_ctx = main_cur_ctx;
        model_main.current_state.n_ctx = main_cur_ctx;
        this.max_ctx.set(model_main.name, new Map([[model_main.applied.length, main_cur_ctx]]));
        this.restore_points.set(model_main.name, { n_ctx: main_cur_ctx, strategies: [] });

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

            // Apply strategy
            const t = performance.now();
            await strat.apply(st_context);
            const find_sec = ((performance.now() - t) / 1000).toFixed(2);

            // Find new max ctx
            const prev_max = main_cur_ctx;
            main_cur_ctx = await cm.findMaxCtx(main_cur_ctx, model_main);

            const ctx_gain = main_cur_ctx - prev_max;
            if (ctx_gain < 0) {
                throw new Error(`buildCostModel: applying ${strat.id} results in ctx decrease of ${ctx_gain}`);
            }

            log.info(`strategy ${strat.id} applied in ${find_sec} sec for a gain of ${ctx_gain} tokens`);

            // Update max ctx for this point in the strategy ladder
            model_main.applied.push(strat_id);
            this.max_ctx.set(model_main.name, new Map([[model_main.applied.length, main_cur_ctx]]));

            await this.printer.print(`Main model @ ${main_cur_ctx.toLocaleString()} ctx`, this.models, model_main.applied);
        }

        // Update in case we skipped some strategies
        model_main.ladder = model_main.applied;
        model_main.applied = [];

        const build_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.info(`finished cost model in ${build_sec} sec`);

        await this.#unloadAllModels();
    }

    async serveDefault(): Promise<void> {
        const model_main = this.models['main']!;
        const model_task = this.models['task'];

        if (model_main.status !== LoadStatus.UNLOADED) throw new Error(`expected main model to be unloaded`);
        if (model_task?.status !== LoadStatus.UNLOADED) throw new Error(`expected task model to be unloaded`);

        const count = model_task ? 2 : 1;

        const t = performance.now();
        await Promise.all([
            this.#loadAndRestore(model_main),
            model_task ? this.#loadWeights(model_task) : Promise.resolve()
        ]);
        const load_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.info(`loaded ${count} models in ${load_sec} sec`);

        this.active = {
            pending: false,
            model: model_main.name,
            readers: 0,
        };
    }

    async decide(body: unknown, model: ModelId, client_signal: AbortSignal, cb: PlannerCallback): Promise<void> {
        const signal = AbortSignal.any([client_signal, this.shutdown_ctrl.signal]);

        log.info(`decide: ${model}, tokenizing...`);

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

        log.info(`decide: ${model}, completions...`);

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
        let plan: Plan | undefined = {
            n_ctx: 0,
            strategies: [],
        };

        while (true) {
            if (this.waiting.length === 0) {
                return;
            }

            const head: Waiter = this.waiting[0];
            model = this.entry(head.model)!;

            const idx = this.#largestRequestForModel(model.name);
            if (idx === undefined) {
                throw new Error(`#maybeTransition: could not find request for ${model.name}`);
            }

            const largest = this.waiting.at(idx)!;
            plan = this.#getMinAddtlStrategies(model.name, largest.req);
            if (plan !== undefined) {
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

        const prior_active = this.entry(this.active.model)!;
        this.active.pending = true;
        this.active.model = model.name;

        if (prior_active.name !== model.name) {
            await this.#stashKV(prior_active);
        }

        if (model.status === LoadStatus.UNLOADED) {
            await this.#loadWeights(model);
        }

        const restore_point = this.#getOrCreateRestore(model);

        if (plan.strategies.length > 0) {
            restore_point.n_ctx = plan.n_ctx;
            restore_point.strategies = [...restore_point.strategies, ...plan.strategies];

            // if the model has a kvcache, save it
            if (model.status === LoadStatus.LOADED) {
                restore_point.slots = await this.client.saveAllSlots(model.name, this.shutdown_ctrl.signal);
            }
        }

        await this.#applyRestore(model, restore_point);
        this.restore_points.delete(model.name);

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

        log.info(`canceling ${this.waiting.length} jobs`);
        const waiting = this.waiting.splice(0, this.waiting.length);
        for (const w of waiting) {
            try {
                w.reject('shutting down');
            } catch { }
        }

        await this.#unloadAllModels();
        await this.#cleanupSlots();
    }

    async #cleanupSlots(): Promise<void> {
        const dir = this.config.router.slot_save_path;
        let names: string[];
        try {
            names = await readdir(dir);
        } catch (err: any) {
            log.warn(`#cleanupSlots: failed to read ${dir}: ${err}`);
            return;
        }

        const targets = names.filter(n => n.endsWith('.bin'));

        await Promise.allSettled(targets.map(n => unlink(join(dir, n))));
    }

    async #unloadAllModels(): Promise<void> {
        const model_main = this.models['main']!;
        const model_task = this.models['task'];

        let count = 0;
        if (model_main.status !== LoadStatus.UNLOADED) count++;
        if (model_task?.status !== LoadStatus.UNLOADED) count++;

        log.info(`unloading ${count} models`);
        const t = performance.now();
        await Promise.all([
            this.#unloadWeights(model_main),
            model_task ? this.#unloadWeights(model_task) : Promise.resolve(),
        ]);
        const unload_sec = ((performance.now() - t) / 1000).toFixed(2);
        log.info(`unloaded ${count} models in ${unload_sec} sec`);
    }

    // fresh load model weights
    async #loadWeights(model: ModelEntry): Promise<void> {
        if (model.status !== LoadStatus.UNLOADED) return;
        model.status = LoadStatus.WEIGHTS_ONLY;

        await this.client.loadModelAndWait(model.name, this.shutdown_ctrl.signal);
    }

    // unload model weights and kvcache. does not save restore point
    async #unloadWeights(model: ModelEntry): Promise<void> {
        if (model.status === LoadStatus.UNLOADED) return;
        model.status = LoadStatus.UNLOADED;

        await this.client.unloadModelAndWait(model.name, this.shutdown_ctrl.signal);

        model.applied = [];
        model.current_state = model.initial_state;
    }

    // restore a model to its last-seen state. loads weights if needed
    async #loadAndRestore(model: ModelEntry): Promise<void> {
        if (model.status === LoadStatus.LOADED) return;

        if (model.status === LoadStatus.UNLOADED) {
            await this.#loadWeights(model);
        }

        const restore_point = this.#getOrCreateRestore(model);

        await this.#applyRestore(model, restore_point);
        this.restore_points.delete(model.name);
    }

    async #applyRestore(model: ModelEntry, point: RestorePoint): Promise<void> {
        if (model.status === LoadStatus.UNLOADED) {
            throw new Error(`#applyRestore: expected model weights for ${model.name}`);
        }

        model.status = LoadStatus.LOADED;

        let params: ReloadParams = { n_ctx: point.n_ctx };
        log.info(
            `#applyRestore: ${model.name} applying ${point.strategies.length} strategies; `,
            `n_ctx: ${point.n_ctx}`
        );

        for (const strategy of point.strategies) {
            const impl = this.strategies.get(strategy)!;

            const ctx = { 
                target: this.model_to_role.get(model.name)!, 
                models: this.models 
            };

            try {
                params = await impl.applyNoSend(ctx, params, point.slots);
            } catch (err: any) {
                throw new Error(`#applyRestore: error applying ${strategy} to ${model.name}: ${err}`);
            }
        }

        // reload model with new params and restore slot info
        await this.client.reloadModel(params, model.name, this.shutdown_ctrl.signal);
        if (point.slots) {
            log.info(`#applyRestore: ${model.name} restoring slots`);
            await this.client.restoreAllSlots(model.name, point.slots, this.shutdown_ctrl.signal);
        }

        model.applied = [...point.strategies];
        model.current_state.n_ctx = point.n_ctx;
    }

    // unload a model's kvcache and save a restore point for later
    async #stashKV(model: ModelEntry): Promise<void> {
        if (model.status !== LoadStatus.LOADED) return;
        model.status = LoadStatus.WEIGHTS_ONLY;

        const restore: RestorePoint = {
            n_ctx: model.current_state.n_ctx,
            strategies: [...model.applied],
        };

        // we only save kvcaches from main models
        if (model.role === 'main') {
            restore.slots = await this.client.saveAllSlots(model.name, this.shutdown_ctrl.signal);
        }

        try {
            await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, model.name, this.shutdown_ctrl.signal);
        } catch (err: any) {
            throw new Error(`#stashKV: error setting ${model.name} to min ctx: ${err}`);
        }

        this.restore_points.set(model.name, restore);
        model.current_state.n_ctx = MIN_ALLOWED_CTX;
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
        let max = -1;
        let idx: number = -1;
        for (const [i, waiter] of this.waiting.entries()) {
            if (waiter.model !== model) continue;

            if (waiter.req.tokens_in > max) {
                max = waiter.req.tokens_in;
                idx = i;
            }
        }

        return idx === -1 ? undefined : idx;
    }

    // get the fewest additional strategies on top of currently-applied that will serve the request
    // evaluates using previous_state if model is not active
    // return undefined if unable to serve
    #getMinAddtlStrategies(model: ModelId, req: TokenRequirement): Plan | undefined {
        const plan: Plan = { strategies: [], n_ctx: 0 };
        const entry = this.entry(model)!;
        const ctx_per_strats = this.max_ctx.get(model)!;
        const ctx_required = req.est_tokens_out ?? req.tokens_in;

        // get currently applied, or load from previous state
        let applied: StrategyId[] = [];
        if (entry.status !== LoadStatus.UNLOADED) {
            applied = entry.applied;
        } else {
            const restore = this.restore_points.get(model);
            if (restore) {
                applied = restore.strategies;
            }
        }

        // no additional strategies needed
        const n_applied = applied.length;
        if (ctx_per_strats.get(n_applied)! >= ctx_required) {
            return plan;
        }

        // add strategies until request is satisfied
        for (const [i, strategy] of entry.ladder.entries()) {
            if (applied.includes(strategy)) {
                continue;
            }

            plan.strategies.push(strategy);
            plan.n_ctx = ctx_per_strats.get(i + 1)!;

            if (plan.n_ctx >= ctx_required) {
                break;
            }
        }

        if (plan.n_ctx >= ctx_required) {
            return plan;
        } else if (plan.n_ctx >= req.tokens_in) {
            log.warn(`unable to guarantee requested tokens ${ctx_required} for ${model}; serving ${plan.n_ctx} instead`);
            return plan;
        }

        // unable to serve request
        return undefined;
    }

    #getOrCreateRestore(model: ModelEntry): RestorePoint {
        let restore_point = this.restore_points.get(model.name);
        if (!restore_point) {
            restore_point = {
                n_ctx: model.current_state.n_ctx,
                strategies: [...model.applied],
            }
        }

        return restore_point;
    }

    entry(model: ModelId): ModelEntry | undefined {
        const role = this.model_to_role.get(model);
        return role ? this.models[role] : undefined;
    }

    role(model: ModelId): ModelRole | undefined {
        return this.model_to_role.get(model);
    }

    canServeImmediately(model: ModelId, req: TokenRequirement): boolean {
        const entry = this.entry(model);
        if (!entry) return false;
        if (entry.status !== LoadStatus.LOADED) return false;

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