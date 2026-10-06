import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import { LoadStatus, type ManagerConfig, type ModelId } from "../config/types.js";
import { logger } from "../logger.js";
import { Timer } from "./timer.js";
import type { ModelEntry } from "./model-entry.js";
import type { InputTokensResponse, MemoryResponse, PromptMetadata } from "../client/types.js";
import { PromptCache } from "./prompt-cache.js";
import { printModelBreakpoints, walkBreakpoints } from "../show-breakpoints.js";


const log: ConsolaInstance = logger.withTag('planner');

export type PlannerResult =
    | { kind: 'done' }
    | { kind: 'grow'; body: unknown };

type PlannerCallback = (body: unknown, model: ModelEntry, signal: AbortSignal, isFinal: boolean) => Promise<PlannerResult>;

type ActiveState = {
    pending: boolean;
    model?: ModelId;     // may be undefined while pending
    readers: number;
}

type ReadHandle = {
    prompt?: InputTokensResponse;
    exclusive(): boolean;
    write(cb: () => Promise<void>): Promise<void>;
    release(): void;
}

type PausedReader = {
    min_ctx: number;
    prompt: PromptMetadata;
    resolve(): void;
    reject(err?: any): void;
}

type Waiter = {
    model: ModelId;
    body: unknown;
    task: ModelTask;
    prompt?: InputTokensResponse;
    resolve: (handle: ReadHandle) => void;
    reject: (reason: any) => void;
}

type DeviceInfo = {
    bytes_total: number;
    bytes_avail: number;
}

export enum ModelTask {
    WAKE,
    TOKENIZE,
    COMPLETIONS,
}

function taskString(t: ModelTask): string {
    if (t === ModelTask.WAKE) return 'wake';
    else if (t === ModelTask.TOKENIZE) return 'tokenization';
    else return 'completions';
}

export class Planner {

    client: LlamaAPI;
    config: ManagerConfig;

    models: Map<ModelId, ModelEntry> = new Map();
    bytes_needed_min: Map<ModelId, number> = new Map();
    cache: PromptCache;

    active: ActiveState = {
        pending: false,
        readers: 0,
    };

    weights_only: Set<ModelId> = new Set();
    waiting_load: Waiter[] = [];
    waiting_reload: PausedReader[] = [];

    // TODO - may need to incorporate fit_target_overhead for 'available'
    dev_info: DeviceInfo = {
        bytes_total: 0,
        bytes_avail: 0,
    };

    shutdown_ctrl: AbortController = new AbortController();
    idle_timer?: ReturnType<typeof setTimeout>;

    constructor(client: LlamaAPI, config: ManagerConfig, models: Map<ModelId, ModelEntry>) {
        this.client = client;
        this.config = config;
        this.models = models;
        this.cache = new PromptCache(config.router.slot_save_path, config.cache_disk_mib);
        for (const model of models.values()) {
            model.cache = this.cache;
        }
    }

    // Note: not intended to be called twice
    async serveDefault(): Promise<void> {
        const t = new Timer('serveDefault');

        this.active.pending = true;

        const model = this.models.get(this.config.default_model);
        if (!model) {
            throw new Error(`serveDefault: default model '${this.config.default_model}' not found`);
        }

        try {
            log.info(`serveDefault: loading weights for ${model.name}`);
            await model.loadWeights(this.shutdown_ctrl.signal, t);
        } catch (err) {
            throw new Error(`serveDefault: unable to load weights for model ${model.name}: ${err}`);
        }

        await this.#updateMemory(model);

        // first load: calculate breakpoints
        const breakpoints = await walkBreakpoints(this.client, model, this.shutdown_ctrl.signal, t?.child(`walkBreakpoints`));
        console.log(printModelBreakpoints(breakpoints));

        log.info(`serveDefault: loading kvcache for ${model.name}`);
        const n_ctx = await model.applyRung(0, this.shutdown_ctrl.signal, t?.child(`applyRung(0)`));
        log.info(`serving ${model.name} with a context window of ${n_ctx} tokens`);

        await this.#updateMemory(model);

        this.active = {
            pending: false,
            model: model.name,
            readers: 0,
        };

        print(t);
        this.#startIdleTimer();
    }

    async serveModel(body: unknown, model_name: string, task: ModelTask, client_signal: AbortSignal, cb: PlannerCallback): Promise<void> {
        const model = this.resolve(model_name);
        if (!model) throw new Error(`serveModel: unknown model ${model_name}`);
        this.shutdown_ctrl.signal.throwIfAborted();

        this.#cancelIdleTimer();

        const signal = AbortSignal.any([client_signal, this.shutdown_ctrl.signal]);
        const t = new Timer(`serveModel: ${model.name} (task: ${taskString(task)})`);

        // stream from model when token requirement is met
        await this.#withModel(model, body, task, signal, t, async (t?: Timer) => {
            let current_body = body;
            while (true) {
                t?.start(`callback (${taskString(task)})`);
                const result = await cb(current_body, model, signal, false);
                t?.stop();
                if (result.kind === 'done') return;

                try {
                    const prompt = await model.renderPrompt(result.body, signal, t);
                    if (!prompt) throw new Error('unable to render continuation prompt');
                    const min_ctx = Math.max(model.getCurCtx() + 1, prompt.input_tokens);

                    if (model.getMinimumRung(min_ctx) === null) {
                        throw new Error(`unable to increase context window for ${model.name} (cur: ${model.getCurCtx()})`);
                    }

                    await new Promise<void>((resolve, reject) => {
                        this.waiting_reload.push({ min_ctx, prompt, resolve, reject });
                        this.#maybeReload(t);
                    });
                } catch {
                    await cb(current_body, model, signal, true);
                    return;
                }

                current_body = result.body;
            }
        }).finally(() => {
            print(t);
            this.#startIdleTimer();
        });
    }

    async #withModel<T>(model: ModelEntry, body: unknown, task: ModelTask, signal: AbortSignal, t: Timer, cb: (t?: Timer) => Promise<T>): Promise<T> {
        let handle: ReadHandle;

        if (!this.modelIsActive(model)) {
            // load model
            handle = await new Promise((resolve, reject) => {
                this.waiting_load.push({ model: model.name, body, task, resolve, reject });
                this.#maybeSwap(t.child(`maybeSwap (load)`));
            });
        } else {
            handle = this.#getHandle();
        }

        // Model is active. Short-circuit depending on task type:
        if (task === ModelTask.WAKE || task === ModelTask.TOKENIZE) {
            try {
                return await cb(t);
            } finally {
                handle.release();
            }
        }

        // Model is loaded and we have a read handle - count tokens
        let tokens_in;
        let rung;
        let prompt: InputTokensResponse;
        try {
            const rendered = handle.prompt ?? await model.renderPrompt(body, signal, t);
            if (rendered === null) {
                throw new Error(`#withModel: failed to count tokens for request to model ${model.name}`);
            }

            prompt = rendered;
            tokens_in = prompt.input_tokens;
            rung = model.getMinimumRung(tokens_in);
            if (rung === null) {
                throw new Error(`#withModel: model ${model.name} cannot serve request (tokens_in: ${tokens_in} | max capacity: ${model.getMaxCtx()})`);
            }
        } catch (err) {
            handle.release();
            throw err;
        }

        try {
            // model is active. serve immediately if:
            // - model is already at optimal rung for input, OR
            // - model is at a valid rung for input and is currently generating
            //
            // otherwise, reload model to optimal rung
            if (model.ladder_i === rung) {
                // done
            } else if (model.ladder_i > rung && !handle.exclusive()) {
                // done
            } else {
                const cur_ctx = model.getCurCtx();
                const new_ctx = model.getCtxCap(rung)!;
                const ctx_str = `(cur ctx: ${cur_ctx} | new ctx: ${new_ctx})`;

                if (cur_ctx >= new_ctx) {
                    log.info(`${model.name}: tokens_in ${tokens_in} can be served at better rung ${ctx_str}`);
                } else {
                    log.info(`${model.name}: tokens_in ${tokens_in} requires capacity increase ${ctx_str}`);
                }

                // capacity change needed: reload model before serving
                await new Promise<void>((resolve, reject) => {
                    this.waiting_reload.push({ min_ctx: tokens_in, prompt, resolve, reject });
                    this.#maybeReload(t);
                });
            }

            // if model still needs a cache restore, do it here if possible
            if (model.needs_restore) {
                if (handle.exclusive()) {
                    await handle.write(async () => {
                        await model.restorePrompts([prompt], signal, t);
                    });
                } else {
                    model.needs_restore = false;
                }
            }
            return await cb(t);
        } finally {
            handle.release();
        }
    }

    #getHandle(): ReadHandle {
        this.active.readers++;
        return {
            exclusive: () => {
                // returns true if we're the only reader
                return this.active.readers === 1 && !this.active.pending;
            },
            write: async (cb: () => Promise<void>): Promise<void> => {
                if (this.active.pending) throw new Error(`handle.write: simultaneous write`);
                this.active.pending = true;

                try {
                    await cb();
                    return Promise.resolve();
                } catch (err) {
                    return Promise.reject(`handle.write: callback failed: ${err}`);
                } finally {
                    this.active.pending = false;
                }
            },
            release: () => {
                // NOTE: do not call other methods on handle after releasing
                this.active.readers--;

                // reload or swap if needed
                if (this.active.readers === 0) {
                    this.#maybeSwap();
                } else if (this.waiting_reload.length === this.active.readers) {
                    this.#maybeReload();
                }
            }
        }
    }

    // NOTE: failures/crashes in this method are uncaught and result in the server shutting down
    // this is an intentional decision, because failure to unload/load/reload leaves models in 
    // an undefined state. Eventually we should handle crashes gracefully and reset to a known-good
    // state, but for now, loud crashes are more helpful.
    async #maybeReload(t?: Timer): Promise<void> {
        // return early if no reload needed
        if (
            this.active.model === undefined
            || this.active.pending
            || this.active.readers === 0
            || this.active.readers !== this.waiting_reload.length
        ) {
            return;
        }

        if (!t) t = new Timer(`maybeReload`);

        let waiting_reload: PausedReader[];

        const signal = this.shutdown_ctrl.signal;
        const handle = this.#getHandle();
        await handle.write(async () => {
            const model = this.activeModel()!;
            waiting_reload = this.waiting_reload.splice(0);
            const fulfill: PausedReader[] = [];
            let rung_needed = 0;
            let min_ctx = 0;
            for (const waiter of waiting_reload) {
                const min_rung = model.getMinimumRung(waiter.min_ctx);
                if (min_rung === null) {
                    waiter.reject(`unable to expand to ctx ${waiter.min_ctx}`);
                    continue;
                }

                if (min_rung > rung_needed) rung_needed = min_rung;
                min_ctx = Math.max(min_ctx, waiter.min_ctx);
                fulfill.push(waiter);
            }

            if (fulfill.length === 0) return;

            // unload unrelated models and save live model's slots
            await this.#unloadAllModels(true, model.name, t?.child(`unloadAllModels`));
            await model.saveSlots(signal, t);

            // apply desired rung, retrying if ctx did not increase beyond the minimum needed
            let cur_ctx = model.getCurCtx();
            while (true) {
                await model.applyRung(rung_needed, signal, t.child(`applyRung(${rung_needed})`));
                await this.#updateMemory(model);

                cur_ctx = model.getCurCtx();
                if (cur_ctx >= min_ctx || !model.hasRung(rung_needed + 1)) break;

                log.warn(`${model.name}: rung ${rung_needed} reloaded to ctx ${cur_ctx}, below required ${min_ctx}; trying rung ${rung_needed + 1}`);
                rung_needed++;
            }

            // restore model slots based on waiting prompts
            const prompts = fulfill.filter(waiter => cur_ctx >= waiter.min_ctx).map(waiter => waiter.prompt);
            await model.restorePrompts(prompts, signal, t);
            
            // enforce prompt cache budget
            this.cache.enforce();

            // flush waiters; continue generation
            for (const waiter of fulfill) {
                if (cur_ctx >= waiter.min_ctx) {
                    waiter.resolve();
                } else {
                    waiter.reject(`unable to expand to ctx ${waiter.min_ctx} (actual ctx: ${cur_ctx})`);
                }
            }            
        }).catch((err) => {
            // reject requests
            for (const waiter of waiting_reload) {
                waiter.reject(`maybeReload error: ${err}`);
            }

            // re-throw
            throw new Error(`maybeReload: error reloading model: ${err}`);
        }).finally(() => handle.release());
    }

    // NOTE: failures/crashes in this method are uncaught and result in the server shutting down
    // this is an intentional decision, because failure to unload/load/reload leaves models in 
    // an undefined state. Eventually we should handle crashes gracefully and reset to a known-good
    // state, but for now, loud crashes are more helpful.
    async #maybeSwap(t?: Timer): Promise<void> {
        if (
            this.active.pending
            || this.active.readers !== 0
            || this.waiting_load.length === 0
        ) {
            return;
        }

        if (!t) t = new Timer(`maybeSwap`);

        const signal = this.shutdown_ctrl.signal;
        const wait_head = this.waiting_load.at(0)!;
        const target = this.models.get(wait_head.model)!;

        const to_flush: Waiter[] = [];
        const fulfill: Waiter[] = [];
        const flush = (waiters: Waiter[]): void => {
            for (const waiter of waiters) {
                const reader = this.#getHandle();
                reader.prompt = waiter.prompt;
                waiter.resolve(reader);
            }
        };

        const still_waiting: Waiter[] = [];
        for (const w of this.waiting_load) {
            if (w.model === target.name) {
                to_flush.push(w);
            } else {
                still_waiting.push(w);
            }
        }

        this.waiting_load = still_waiting;

        // handle case where a swap request comes in when a reload is pending
        if (this.modelIsActive(target)) {
            flush(to_flush);
            return;
        }

        const handle = this.#getHandle();
        await handle.write(async () => {
            // unload any other active models
            await this.#unloadAllModels(true, target.name, t?.child(`unloadAllModels`));

            // if target model has not been loaded before, we don't know how much space we need.
            // evict all models and load target, then calculate breakpoints
            if (target.getMaxCtx() === null) {
                log.info(`first load of ${target.name}; calculating breakpoints`);

                const breakpoints = await walkBreakpoints(this.client, target, signal, t?.child(`walkBreakpoints`));
                console.log(printModelBreakpoints(breakpoints));

                await target.applyRung(0, signal, t?.child(`applyRung(0)`));
                await this.#updateMemory(target);
            }

            // load model weights if needed
            if (target.status === LoadStatus.UNLOADED) {
                log.info(`maybeSwap: loading weights for ${target.name}`);
                await target.loadWeights(signal, t?.child(`loadWeights`));
                await this.#updateMemory(target);
            }

            // count tokens and get minimum rung to satisfy all requests
            let rung_needed = 0;
            for (const waiter of to_flush) {
                // don't need model at a specific rung for these tasks
                if (waiter.task === ModelTask.WAKE || waiter.task === ModelTask.TOKENIZE) {
                    fulfill.push(waiter);
                    continue;
                }

                const prompt = await target.renderPrompt(waiter.body, signal, t);
                if (prompt === null) {
                    waiter.reject(`failed to count tokens for request`);
                    continue;
                }

                waiter.prompt = prompt;
                const tokens_in = prompt.input_tokens;
                const min_rung = target.getMinimumRung(tokens_in);
                if (min_rung === null) {
                    waiter.reject(`unable to serve at ctx ${tokens_in}`);
                    continue;
                }

                if (min_rung > rung_needed) rung_needed = min_rung;
                fulfill.push(waiter);
            }

            await target.applyRung(rung_needed, signal, t.child(`applyRung`));
            await this.#updateMemory(target);

            // restore model slots based on waiting prompts
            const prompts = fulfill.flatMap(waiter => waiter.prompt ? [waiter.prompt] : []);
            if (prompts.length) {
                await target.restorePrompts(prompts, signal, t);
            } else {
                target.needs_restore = true;
            }

            // enforce prompt cache budget
            this.cache.enforce();

            this.active.model = target.name;
            this.weights_only.delete(target.name);

            flush(fulfill);
        }).catch((err) => {
            // reject requests
            for (const waiter of to_flush) {
                waiter.reject(`maybeSwap error: ${err}`);
            }

            // re-throw
            throw new Error(`maybeSwap: error swapping model: ${err}`);
        }).finally(() => handle.release());
    }

    async shutdown(): Promise<void> {
        this.#cancelIdleTimer();

        const t = new Timer(`Planner.shutdown`);
        this.shutdown_ctrl.abort('shutdown request received');
        t.start('cache shutdown');
        await this.cache.shutdown();
        t.stop();

        t.start('unloadAllModels');
        await Promise.allSettled([...this.models.values()].map(model => model.unloadHard(t)));
        t.stop();

        t.start(`cancel ${this.waiting_load.length} load jobs`);
        const waiting_load = this.waiting_load.splice(0);
        for (const w of waiting_load) {
            try { w.reject('shutting down') } catch { }
        }
        t.stop();

        t.start(`cancel ${this.waiting_reload.length} reload jobs`);
        const waiting_reload = this.waiting_reload.splice(0);
        for (const w of waiting_reload) {
            try { w.reject('shutting down') } catch { }
        }
        t.stop();

        log.info(`shutdown: done`);
    }

    #startIdleTimer(): void {
        this.#cancelIdleTimer();

        if (!this.#isIdle()) {
            return;
        }

        const ms = this.config.sleep_idle_seconds * 1000;
        if (ms <= 0) return;

        this.idle_timer = setTimeout(() => {
            this.#onIdle().catch(err => log.error(`idle unload error: ${err}`));
        }, ms);
    }

    #cancelIdleTimer(): void {
        if (this.idle_timer !== undefined) {
            clearTimeout(this.idle_timer);
            this.idle_timer = undefined;
        }
    }

    async #onIdle(): Promise<void> {
        if (!this.#isIdle()) {
            return;
        }

        log.info(`idle timeout reached; unloading all models`);

        const handle = this.#getHandle();
        await handle.write(async () => {
            await this.#unloadAllModels(true);
            await this.cache.enforce();
        }).finally(() => handle.release());
    }

    #isIdle(): boolean {
        return !(
            this.active.pending
            || this.active.readers !== 0
            || this.waiting_load.length !== 0
            || this.waiting_reload.length !== 0
        );
    }

    async #unloadAllModels(stash_kv: boolean, exclude?: ModelId, t?: Timer): Promise<void> {
        let promises: Promise<void>[] = [];

        const active = this.activeModel();
        if (active && active.name !== exclude) {
            if (!stash_kv) {
                promises.push(this.#unloadNoRestore(active, t));
            } else {
                promises.push(this.#unloadWithRestore(active, t));
            }
        }

        for (const id of [...this.weights_only.keys()]) {
            if (id === exclude) continue;

            const model = this.models.get(id)!;
            if (!stash_kv) {
                promises.push(this.#unloadNoRestore(model, t));
            } else {
                promises.push(this.#unloadWithRestore(model, t));
            }
        }

        if (promises.length > 0) {
            log.info(`unloading ${promises.length} models${stash_kv ? ' with restore' : ''}`);
        }

        await Promise.allSettled(promises);
    }

    async #unloadNoRestore(model: ModelEntry, t?: Timer): Promise<void> {
        try {
            const mem = await model.getMemory(this.shutdown_ctrl.signal);
            const bytes_used = calcTotalBytesForModel(mem);

            await model.unloadHard(t);

            // remove model from tracking
            this.weights_only.delete(model.name);
            if (this.active.model === model.name) {
                this.active.model = undefined;
            }

            this.#setMemFreed(bytes_used);
        } catch (err) {
            throw new Error(`#unloadNoRestore: error unloading weights for model ${model.name}: ${err}`);
        }
    }

    async #unloadWithRestore(model: ModelEntry, t?: Timer): Promise<void> {
        try {
            const mem = await model.getMemory(this.shutdown_ctrl.signal);
            const bytes_used = calcTotalBytesForModel(mem);

            await model.unloadWeights(this.shutdown_ctrl.signal, t);

            // remove model from tracking
            this.weights_only.delete(model.name);
            if (this.active.model === model.name) {
                this.active.model = undefined;
            }

            this.#setMemFreed(bytes_used);
        } catch (err) {
            throw new Error(`#unloadWithRestore: error unloading weights for model ${model.name}: ${err}`);
        }
    }

    #setMemFreed(bytes_freed: number): void {
        this.dev_info.bytes_avail += bytes_freed;

        if (this.dev_info.bytes_avail > this.dev_info.bytes_total) {
            log.warn(`#setMemFreed shows available > total `,
                `(avail: ${fmtBytes(this.dev_info.bytes_avail)} |`,
                ` total: ${fmtBytes(this.dev_info.bytes_total)})`
            );

            this.dev_info.bytes_avail = this.dev_info.bytes_total;
        }
    }

    async #updateMemory(model: ModelEntry): Promise<void> {
        const mem = await model.getMemory(this.shutdown_ctrl.signal);

        if (this.dev_info.bytes_total === 0) {
            log.info(`first model loaded`);
            this.dev_info.bytes_total = calcTotalBytesOnDevice(mem);
        }

        this.dev_info.bytes_avail = calcFreeBytes(mem);

        if (calcTotalBytesOnDevice(mem) != this.dev_info.bytes_total) {
            log.error(`#updateMem shows total device bytes changed; ignoring (`,
                `cur: ${fmtBytes(this.dev_info.bytes_total)} | `,
                `new: ${fmtBytes(calcTotalBytesOnDevice(mem))})`
            );
        }

        if (this.dev_info.bytes_avail > this.dev_info.bytes_total) {
            log.warn(`#updateMem shows available > total (`,
                `avail: ${fmtBytes(this.dev_info.bytes_avail)} | `,
                `total: ${fmtBytes(this.dev_info.bytes_total)})`
            );

            this.dev_info.bytes_avail = this.dev_info.bytes_total;
        }

        log.info(`available memory: (${fmtBytes(this.dev_info.bytes_avail)} / ${fmtBytes(this.dev_info.bytes_total)})`);
    }

    // server mode: return the only model we're serving, regardless of input
    // router mode: resolve model id, fall back to aliases
    resolve(name: ModelId): ModelEntry | undefined {
        if (this.config.mode === 'server') {
            return [...this.models.values()][0];
        }

        const exact = this.models.get(name);
        if (exact) return exact;

        for (const entry of this.models.values()) {
            if (entry.aliases.includes(name)) return entry;
        }

        return undefined;
    }

    isModelQueued(name: ModelId): boolean {
        return this.waiting_load.some(w => w.model === name);
    }

    activeModel(): ModelEntry | undefined {
        const name = this.active.model;
        if (!name) return undefined;
        else return this.models.get(name);
    }

    // Note: assumes model is loaded
    modelIsActive(model: ModelEntry): boolean {
        return !this.active.pending && this.active.model === model.name;
    }

    modelIsGenerating(model: ModelEntry): boolean {
        return this.modelIsActive(model)
            && this.active.readers !== 0;
    }
}

// TODO - assumes single GPU
function calcFreeBytes(mem: MemoryResponse): number {
    for (const dev of mem.devices) {
        if (dev.type !== "cpu") return dev.free;
    }
    return 0;
}

// TODO - assumes single GPU
function calcTotalBytesOnDevice(mem: MemoryResponse): number {
    for (const dev of mem.devices) {
        if (dev.type !== "cpu") return dev.total;
    }
    return 0;
}

function calcTotalBytesForModel(mem: MemoryResponse): number {
    let total = 0;

    const calc = (cmp?: { model: number, context: number, compute: number }): number => {
        if (!cmp) return 0;

        return cmp.model + cmp.context + cmp.compute;
    };

    for (const dev of mem.devices) {
        if (dev.type === "cpu") continue;

        const cmp = dev.components;
        total += calc(cmp.main);
        total += calc(cmp.spec);
        total += calc(cmp.mmproj);
    }

    return total;
}

function fmtBytes(bytes: number): string {
    const gib = bytes / (1024 ** 3);
    return `${gib.toFixed(2)} GiB`;
}

function print(t?: Timer) {
    if (!t) return;
    log.info(t.fmt());
}