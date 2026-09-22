import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import { LoadStatus, type ManagerConfig, type ModelId, type Tokens } from "../config/types.js";
import { logger } from "../logger.js";
import { readdir, unlink } from "node:fs/promises";
import { join } from "node:path";
import { Timer } from "./timer.js";
import type { ModelEntry, RestorePoint } from "./model-entry.js";
import type { MemoryResponse } from "../client/types.js";
import { printModelBreakpoints, walkBreakpoints } from "../show-breakpoints.js";


const log: ConsolaInstance = logger.withTag('planner');

export interface Client {
    completions(body: unknown, model: ModelId, signal: AbortSignal): Promise<Response>;
    countTokens(body: unknown, model: ModelId, signal: AbortSignal): Promise<number>;
}

type PlannerCallback = (body: unknown, client: Client, signal: AbortSignal, isFinal: boolean) => Promise<boolean>;

type ActiveState = {
    pending: boolean;
    model?: ModelId;     // may be undefined while pending
    readers: number;
}

type ReadHandle = {
    exclusive(): boolean;
    write(cb: () => Promise<void>): Promise<void>;
    release(): void;
}

type PausedReader = {
    min_ctx: number;
    resolve(): void;
    reject(err?: any): void;
}

type Waiter = {
    model: ModelId;
    body: unknown;
    resolve: (handle: ReadHandle) => void;
    reject: (reason: any) => void;
}

type RestoreInfo = {
    point: RestorePoint;
    bytes_needed: number;
}

type DeviceInfo = {
    bytes_total: number;
    bytes_avail: number;
}

export class Planner {

    client: LlamaAPI;
    config: ManagerConfig;

    models: Map<ModelId, ModelEntry> = new Map();
    bytes_needed_min: Map<ModelId, number> = new Map();

    active: ActiveState = {
        pending: false,
        readers: 0,
    };

    weights_only: Set<ModelId> = new Set();
    waiting_load: Waiter[] = [];
    waiting_reload: PausedReader[] = [];

    restore_info: Map<ModelId, RestoreInfo> = new Map();

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

        this.#updateMem(await this.client.getMemory(model.name, this.shutdown_ctrl.signal));

        // first load: calculate breakpoints
        const breakpoints = await walkBreakpoints(this.client, model, this.shutdown_ctrl.signal, t?.child(`walkBreakpoints`));
        console.log(printModelBreakpoints(breakpoints));

        log.info(`serveDefault: loading kvcache for ${model.name}`);
        const n_ctx = await model.moveToRung(0, this.shutdown_ctrl.signal, t?.child(`moveToRung(0)`));
        log.info(`serving ${model.name} with a context window of ${n_ctx} tokens`);

        // update memory info
        this.#updateMem(await this.client.getMemory(model.name, this.shutdown_ctrl.signal));

        this.active = {
            pending: false,
            model: model.name,
            readers: 0,
        };

        print(t);
        this.#startIdleTimer();
    }

    async serveModel(body: unknown, model: ModelEntry, client_signal: AbortSignal, cb: PlannerCallback): Promise<void> {
        this.#cancelIdleTimer();

        const signal = AbortSignal.any([client_signal, this.shutdown_ctrl.signal]);
        const t = new Timer(`serveModel: ${model.name}`);

        // stream from model when token requirement is met
        await this.#withModel(model, body, signal, t, async (t?: Timer) => {
            // TODO: this timer name is wrong when caller is from tokenize/input_tokens/etc
            // FIX: allow caller to pass label.
            t?.start('completions callback');
            let success = await cb(body, this.client, signal, false);
            t?.stop();

            // failure indicates the response was truncated and we need to increase the ctx window
            while (!success) {
                try {
                    // cur_cap + 1 will push the model to the next rung
                    const cur_cap = model.getCtxCap(model.ladder_i)!;
                    const min_ctx = cur_cap + 1;

                    if (model.getMinimumRung(min_ctx) === null) {
                        throw new Error(`unable to further context window for ${model.name} (cur: ${cur_cap})`);
                    }

                    // queue reload to increase cap
                    await new Promise<void>((resolve, reject) => {
                        this.waiting_reload.push({ min_ctx, resolve, reject });
                        this.#maybeReload(t);
                    });
                } catch {
                    // if unable to grow, inform via callback
                    success = await cb(body, this.client, signal, true);
                    break;
                }

                t?.start('completions callback');
                success = await cb(body, this.client, signal, false);
                t?.stop();
            }
        }).finally(() => {
            print(t);
            this.#startIdleTimer();
        });
    }

    async #withModel<T>(model: ModelEntry, body: unknown, signal: AbortSignal, t: Timer, cb: (t?: Timer) => Promise<T>): Promise<T> {
        let handle: ReadHandle;

        if (this.modelIsActive(model)) {
            handle = this.#getHandle();
            let tokens_in;
            let rung;
            try {
                tokens_in = await model.countTokens(body, signal, t);
                if (tokens_in === null) {
                    throw new Error(`#withModel: failed to count tokens for request to model ${model.name}`);
                }

                rung = model.getMinimumRung(tokens_in);
                if (rung === null) {
                    throw new Error(`#withModel: model ${model.name} cannot serve request (tokens_in: ${tokens_in} | max capacity: ${model.getMaxCtx()})`);
                }
            } catch (err) {
                handle.release();
                throw err;
            }

            // model is active. serve immediately if:
            // - model is already at optimal rung for input, OR
            // - model is at a valid rung for input and is currently generating
            //
            // otherwise, reload model
            if (model.ladder_i === rung) {
                // done
            } else if (model.ladder_i > rung && !handle.exclusive()) {
                // done
            } else {
                const new_cap = model.getCtxCap(rung);
                log.info(`${model.name}: tokens_in ${tokens_in} requires capacity increase to ${new_cap}`);

                // capacity change needed: reload model before serving
                await new Promise<void>((resolve, reject) => {
                    this.waiting_reload.push({ min_ctx: tokens_in, resolve, reject });
                    this.#maybeReload(t);
                });
            }
        } else {
            // load model
            handle = await new Promise((resolve, reject) => {
                this.waiting_load.push({ model: model.name, body, resolve, reject });
                this.#maybeSwap(t.child(`maybeSwap (load)`));
            });
        }

        try {
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

        const signal = this.shutdown_ctrl.signal;
        const handle = this.#getHandle();
        await handle.write(async () => {
            const model = this.activeModel()!;
            const waiting_reload = this.waiting_reload.splice(0);
            const fulfill = [];

            // get the minimum rung that will satisfy all readers
            let rung_needed = 0;
            for (const waiter of waiting_reload) {
                const min_rung = model.getMinimumRung(waiter.min_ctx);
                if (min_rung === null) {
                    waiter.reject(`unable to expand to ctx ${waiter.min_ctx}`);
                    continue;
                }

                if (min_rung > rung_needed) rung_needed = min_rung;
                fulfill.push(waiter);
            }

            const cur_cap = model.getCtxCap(model.ladder_i);
            const new_cap = model.getCtxCap(rung_needed);

            const strats = model.ladder.map(r => r.strategy as string);
            const diff: string[] = rung_needed > model.ladder_i
                ? strats.slice(model.ladder_i, rung_needed)
                : strats.slice(rung_needed, model.ladder_i).reverse();

            // free space for kv, if needed
            const bytes_needed = model.getBytesNeeded(rung_needed)!;
            if (bytes_needed < 0) {
                log.info(`loading ${model.name} to rung ${rung_needed} frees ${fmtBytes(bytes_needed)}`);
            } else if (bytes_needed > this.dev_info.bytes_avail) {
                log.info(`freeing ${fmtBytes(bytes_needed)} for model ${model.name}`);
                await this.#free(model, bytes_needed, t?.child(`free (kv)`));

                if (bytes_needed > this.dev_info.bytes_avail) {
                    log.warn(`maybeReload: unable to free all requested bytes; attempting reload (still needed: ${fmtBytes(bytes_needed - this.dev_info.bytes_avail)})`);
                }
            }

            // reload model
            if (rung_needed !== model.ladder_i) {
                const action = rung_needed > model.ladder_i ? "applying" : "removing";
                log.info(`${model.name}: ${action} strategies [${diff.join(", ")}]; new ctx cap: ${new_cap}`);

                await model.moveToRung(rung_needed, signal, t.child(`moveToRung`));
                this.#updateMem(await this.client.getMemory(model.name, signal));
            } else {
                log.info(`maybeReload: ${model.name} does not need reload (cur ctx cap: ${cur_cap} | rung: ${rung_needed})`);
            }

            // flush queue
            for (const waiter of fulfill) {
                waiter.resolve();
            }
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

        const fulfill: Waiter[] = [];

        const handle = this.#getHandle();
        await handle.write(async () => {
            // filter out load requests that want the head model
            const to_serve: Waiter[] = [];
            const still_waiting: Waiter[] = [];
            for (const w of this.waiting_load) {
                if (w.model === target.name) {
                    to_serve.push(w);
                } else {
                    still_waiting.push(w);
                }
            }

            this.waiting_load = still_waiting;

            // if target model has not been loaded before, we don't know how much space we need.
            // evict all models and load target, then calculate breakpoints
            if (target.getMaxCtx() === null) {
                log.info(`first load of ${target.name}; clearing GPU and calculating breakpoints`);
                await this.#unloadAllModels(true, target.name, t?.child(`unloadAllModels`));

                const breakpoints = await walkBreakpoints(this.client, target, signal, t?.child(`walkBreakpoints`));
                console.log(printModelBreakpoints(breakpoints));

                await target.moveToRung(0, signal, t?.child(`moveToRung(0)`));
                this.#updateMem(await this.client.getMemory(target.name, signal));
            }

            // load model weights if needed
            if (target.status === LoadStatus.UNLOADED) {
                log.info(`maybeSwap: freeing space for ${target.name} weights`);
                await this.#free(target, target.bytes_needed_no_kv!, t?.child(`free (weights)`));

                log.info(`maybeSwap: loading weights for ${target.name}`);
                await target.loadWeights(signal, t?.child(`loadWeights`));
                this.#updateMem(await this.client.getMemory(target.name, signal));
            }

            // count tokens and get minimum rung to satisfy all requests
            let rung_needed = 0;
            const fulfill = [];
            for (const waiter of to_serve) {
                const tokens_in = await target.countTokens(waiter.body, signal, t?.child(`countTokens`));
                if (tokens_in === null) {
                    waiter.reject(`failed to count tokens for request`);
                    continue;
                }

                const min_rung = target.getMinimumRung(tokens_in);
                if (min_rung === null) {
                    waiter.reject(`unable to serve at ctx ${tokens_in}`);
                    continue;
                }

                if (min_rung > rung_needed) rung_needed = min_rung;
                fulfill.push(waiter);
            }

            // free space for kv, if needed
            const bytes_needed = target.getBytesNeeded(rung_needed)!;
            if (bytes_needed < 0) {
                log.info(`loading ${target.name} to rung ${rung_needed} frees ${fmtBytes(bytes_needed)}`);
            } else if (bytes_needed > this.dev_info.bytes_avail) {
                log.info(`freeing ${fmtBytes(bytes_needed)} for model ${target.name}`);
                await this.#free(target, bytes_needed, t?.child(`free (kv)`));

                if (bytes_needed > this.dev_info.bytes_avail) {
                    throw new Error(`maybeSwap: unable to free all requested bytes`);
                }
            }

            // reload model to target rung
            const strats = target.ladder.map(r => r.strategy as string).slice(0, rung_needed);
            const new_cap = target.getCtxCap(rung_needed);
            log.info(`${target.name}: applying strategies [${strats.join(", ")}]; new ctx cap: ${new_cap}`);
            await target.moveToRung(rung_needed, signal, t.child(`moveToRung`));
            this.#updateMem(await this.client.getMemory(target.name, signal));

            this.active.model = target.name;
            this.weights_only.delete(target.name);
        }).finally(() => {
            // fulfill requests
            for (const waiter of fulfill) {
                waiter.resolve(this.#getHandle());
            }

            handle.release();
        });
    }

    // free space on the gpu by stashing idle models' kvcaches and/or evicting weights
    async #free(target_model: ModelEntry, bytes_needed: number, t?: Timer): Promise<void> {
        if (this.dev_info.bytes_avail > bytes_needed) {
            log.info(`#free not needed (avail: ${fmtBytes(this.dev_info.bytes_avail)} | needed: ${fmtBytes(bytes_needed)})`);
            return;
        }

        // stash kv for currently-loaded model
        const active = this.activeModel();
        if (active && active.name !== target_model.name) {
            let mem = await this.client.getMemory(active.name, this.shutdown_ctrl.signal);
            const free_before = calcFreeBytes(mem);

            log.info(`unloading kv for model: ${active.name}`);
            const restore_point = await active.unloadKV(this.shutdown_ctrl.signal, t);

            // update device memory and tracking
            mem = await this.client.getMemory(active.name, this.shutdown_ctrl.signal);
            const free_after = calcFreeBytes(mem);

            if (restore_point) {
                // bytes needed to restore = bytes freed when unloading KV
                this.restore_info.set(active.name, {
                    point: restore_point,
                    bytes_needed: free_after - free_before,
                });
            }

            this.#updateMem(mem);
            this.active.model = undefined;
            this.weights_only.add(active.name);

            if (this.dev_info.bytes_avail > bytes_needed) return;
        }

        if (this.weights_only.size === 0) {
            const b = bytes_needed - this.dev_info.bytes_avail;
            log.error(`#free requires ${fmtBytes(b)} bytes, but found no more models to unload`);
            return;
        }

        log.info(`unloading weights for idle models`);
        t = t?.child('unloadWeights');

        // evict weights for all weights-only models
        for (const id of [...this.weights_only.keys()]) {
            if (id === target_model.name) continue;
            const model = this.models.get(id)!;

            await this.#unloadWithRestore(model, t);

            if (this.dev_info.bytes_avail > bytes_needed) return;
        }

        const b = bytes_needed - this.dev_info.bytes_avail;
        log.error(`#free requires ${fmtBytes(b)} bytes, but found no more models to unload`);
    }

    async shutdown(): Promise<void> {
        this.#cancelIdleTimer();

        const t = new Timer(`Planner.shutdown`);

        t.start('unloadAllModels');
        await this.#unloadAllModels(false).catch(err => {
            log.warn(`shutdown: unloadAllModels error: ${err}`);
        });
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

        t.start(`cleanupSlots`);
        await this.#cleanupSlots().catch(err => {
            log.warn(`shutdown: cleanupSlots error: ${err}`);
        });
        t.stop();

        this.shutdown_ctrl.abort('shutdown request received');
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
            await this.#unloadAllModels(false);
            this.restore_info.clear();
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

    async #cleanupSlots(): Promise<void> {
        const dir = this.config.router.slot_save_path;
        let names: string[];
        try {
            names = await readdir(dir);
        } catch (err: any) {
            log.warn(`#cleanupSlots: failed to read ${dir}: ${err}`);
            return;
        }

        const targets = names.filter(n => n.endsWith('.bin') || n.endsWith('.ckpt'));

        await Promise.allSettled(targets.map(n => unlink(join(dir, n))));
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
            const mem = await this.client.getMemory(model.name, this.shutdown_ctrl.signal);
            const bytes_used = calcTotalBytesForModel(mem);

            await model.unloadHard(t);

            // remove model from tracking
            this.weights_only.delete(model.name);
            if (this.active.model === model.name) {
                this.active.model = undefined;
            }

            this.restore_info.delete(model.name);

            this.#setMemFreed(bytes_used);
        } catch (err) {
            throw new Error(`#unloadNoRestore: error unloading weights for model ${model.name}: ${err}`);
        }
    }

    async #unloadWithRestore(model: ModelEntry, t?: Timer): Promise<void> {
        try {
            const mem = await this.client.getMemory(model.name, this.shutdown_ctrl.signal);
            const bytes_used = calcTotalBytesForModel(mem);

            // unload model and update restore point, if created
            const restore_point = await model.unloadWeights(this.shutdown_ctrl.signal, t);
            if (restore_point) {
                this.restore_info.set(model.name, {
                    point: restore_point,
                    bytes_needed: bytes_used,
                });
            } else if (this.restore_info.has(model.name)) {
                // if we already had a restore point and didn't get one back from unloadWeights,
                // this model was weights-only. we add bytes_used to the existing restore point
                this.restore_info.get(model.name)!.bytes_needed += bytes_used;
            }

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

    #updateMem(mem: MemoryResponse): void {
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