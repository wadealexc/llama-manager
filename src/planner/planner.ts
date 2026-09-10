import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import { type ManagerConfig, type ModelId } from "../config/types.js";
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
    release(): void;
}

type PausedReader = {
    resolve(): void;
    reject(err?: any): void;
}

type Waiter = {
    model: ModelId;
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

    active: ActiveState = {
        pending: false,
        readers: 0,
    };

    weights_only: Set<ModelId> = new Set();
    waiting: Waiter[] = [];
    waiting_grow: PausedReader[] = [];

    restore_info: Map<ModelId, RestoreInfo> = new Map();

    // TODO - may need to incorporate fit_target_overhead for 'available'
    dev_info: DeviceInfo = {
        bytes_total: 0,
        bytes_avail: 0,
    };

    max_ctx: Map<ModelId, number> = new Map();

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

        await this.#onFirstLoad(model, t?.child('onFirstLoad'));

        log.info(`serveDefault: loading kvcache for ${model.name}`);
        const n_ctx = await model.loadWithKV(null, this.shutdown_ctrl.signal, t?.child('loadWithKV'));
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

        if (!this.canServeNow(model)) {
            log.debug(`req ${model.name}: can't serve request, queuing`);
        }

        // stream from model when token requirement is met
        await this.#withModel(model, t, async (t?: Timer) => {
            t?.start('completions callback');
            let success = await cb(body, this.client, signal, false);
            t?.stop();

            // failure indicates the response was truncated and we need to increase the ctx window
            while (!success) {
                try {
                    await this.#waitForGrow(model, t?.child('waitForGrow'));
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

    // attempt to expand the model's context window. if there are other active readers,
    // add a waiter to `waiting_grow` until all readers are waiting for grow.
    async #waitForGrow(model: ModelEntry, t?: Timer): Promise<void> {
        if (this.waiting_grow.length + 1 < this.active.readers) {
            await new Promise<void>((resolve, reject) => {
                this.waiting_grow.push({ resolve, reject });
            });
            return;
        }

        await this.#growModel(model, t);
    }

    async #growModel(model: ModelEntry, t?: Timer): Promise<void> {
        if (this.active.model !== model.name) {
            throw new Error(`#growModel: expected ${model.name} to be active`);
        }

        this.active.pending = true;

        log.info(`growing model ${model.name} to fill available space`);
        const prev_ctx = model.curCtx();

        try {
            // if there are idle weights_only models, evict to make space, then expand to fill
            // we only apply strategies if we can't evict other models
            if (this.weights_only.size !== 0) {
                await this.#unloadAllModels(true, model.name, t?.child('unloadAllModels'));
                await model.expandToFit(this.shutdown_ctrl.signal, t?.child('growModel'));
            } else if (model.hasNextStrategy()) {
                await model.applyNextStrategy(true, this.shutdown_ctrl.signal, t?.child('applyNextStrategy'));
            } else {
                throw new Error(`strategies exhausted`);
            }
        } catch (err) {
            log.info(`${model.name} already at max ctx of ${model.curCtx()} tokens (strategies exhausted)`);
            // cannot grow further. release waiters and throw
            const msg = err instanceof Error ? err.message : String(err);
            const waiting = this.waiting_grow.splice(0);
            for (const waiter of waiting) waiter.reject(msg);
            throw new Error(`#growModel error: ${msg}`);
        } finally {
            this.active.pending = false;
        }

        log.info(`${model.name} context window expanded from ${prev_ctx} to ${model.curCtx()}`);

        // update available memory
        const mem = await this.client.getMemory(model.name, this.shutdown_ctrl.signal);
        this.#updateMem(mem);

        const waiting = this.waiting_grow.splice(0);
        for (const w of waiting) w.resolve();
    }

    async #withModel<T>(model: ModelEntry, t: Timer, cb: (t?: Timer) => Promise<T>): Promise<T> {
        let handle: ReadHandle;
        if (this.canServeNow(model)) {
            this.active.readers++;
            handle = this.#getHandle();
        } else {
            handle = await new Promise((resolve, reject) => {
                this.waiting.push({ model: model.name, resolve, reject });
                this.#maybeSwap(t.child(`maybeSwap`));
            });
        }

        try {
            return await cb(t);
        } finally {
            handle.release();
        }
    }

    #getHandle(): ReadHandle {
        return {
            release: () => {
                this.active.readers--;

                // we still have active readers and no grow required
                if (this.active.readers !== 0 && this.waiting_grow.length !== this.active.readers) {
                    return;
                }

                // we have active readers and all are waiting for grow
                if (this.waiting_grow.length > 0 && this.waiting_grow.length === this.active.readers) {
                    const model = this.models.get(this.active.model!)!;
                    this.#growModel(model).catch(err => log.error(`#growModel error: ${err}`));
                    return;
                }

                // we don't have active readers; swap model
                this.#maybeSwap();
            }
        }
    }

    async #maybeSwap(t?: Timer): Promise<void> {
        if (this.active.readers !== 0 || this.active.pending) {
            return;
        }

        if (this.waiting.length === 0) return;
        const waiter = this.waiting.shift()!;
        const target = this.models.get(waiter.model)!;

        const flush_queue = () => {
            const serve: Waiter[] = [];
            const wait: Waiter[] = [];

            for (const w of this.waiting) {
                if (w.model === target.name) {
                    serve.push(w);
                } else {
                    wait.push(w);
                }
            }

            this.waiting = wait;
            for (const w of serve) {
                this.active.readers++;
                w.resolve(this.#getHandle());
            }
        }

        if (this.active.model === target.name) {
            this.active.readers++;
            waiter.resolve(this.#getHandle());
            flush_queue();
            return;
        }

        this.active.pending = true;
        this.active.readers = 0;

        const restore = this.restore_info.get(target.name);
        const bytes_needed = restore?.bytes_needed;

        // free space on the device for the target model
        // if we don't know how much space is needed, unload all other models
        if (bytes_needed === undefined) {
            log.info(`unknown bytes needed for model ${target.name}; unloading idle models`);
            await this.#unloadAllModels(true, target.name, t?.child('unloadAllModels'));
        } else if (bytes_needed > this.dev_info.bytes_avail) {
            log.info(`freeing ${fmtBytes(bytes_needed)} for model ${target.name}`);
            await this.#free(target, bytes_needed, t?.child('free'));

            if (bytes_needed > this.dev_info.bytes_avail) {
                throw new Error(`maybeSwap: unable to free all requested bytes`);
            }
        }

        // on first load, calculate breakpoints and max ctx
        if (!this.max_ctx.has(target.name)) {
            await this.#onFirstLoad(target, t?.child('onFirstLoad'));
        }

        // load target model
        log.info(`loading kvcache for model ${target.name}`);
        const n_ctx = await target.loadWithKV(restore?.point ?? null, this.shutdown_ctrl.signal, t?.child(`loadWithKV(${target.name})`));
        log.info(`${target.name} kvcache loaded; ctx window: ${n_ctx} tokens`);
        this.#updateMem(await this.client.getMemory(target.name, this.shutdown_ctrl.signal));

        this.active.pending = false;
        this.active.model = target.name;
        this.active.readers++;
        this.weights_only.delete(target.name);

        // serve request
        waiter.resolve(this.#getHandle());

        // serve any other requests waiting for this model
        flush_queue();
    }

    // free space on the gpu by stashing idle models' kvcaches and/or evicting weights
    async #free(target_model: ModelEntry, bytes_needed: number, t?: Timer): Promise<void> {
        if (this.dev_info.bytes_avail > bytes_needed) {
            log.info(`#free not needed (avail: ${fmtBytes(this.dev_info.bytes_avail)} | needed: ${fmtBytes(bytes_needed)})`);
            return;
        }

        // stash kv for currently-loaded model
        const active_id = this.active.model;
        if (active_id !== undefined && active_id !== target_model.name) {
            const active_model = this.models.get(active_id)!;

            let mem = await this.client.getMemory(active_id, this.shutdown_ctrl.signal);
            const free_before = calcFreeBytes(mem);

            log.info(`unloading kv for model: ${active_id}`);
            const restore_point = await active_model.unloadKV(this.shutdown_ctrl.signal, t);

            // update device memory and tracking
            mem = await this.client.getMemory(active_id, this.shutdown_ctrl.signal);
            const free_after = calcFreeBytes(mem);

            if (restore_point) {
                // bytes needed to restore = bytes freed when unloading KV
                this.restore_info.set(active_id, {
                    point: restore_point,
                    bytes_needed: free_after - free_before,
                });
            }

            this.#updateMem(mem);
            this.active.model = undefined;
            this.weights_only.add(active_id);

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

    // calculate and print a model's breakpoints the first time it is loaded
    // the model is fully unloaded afterwards.
    // TODO: a little janky. we should be able to do this without a hard unload (ctx reset?)
    async #onFirstLoad(model: ModelEntry, t?: Timer): Promise<void> {
        if (this.max_ctx.has(model.name)) return;

        log.info(`first load of ${model.name}; calculating max ctx`);
        const breakpoints = await walkBreakpoints(this.client, model, this.shutdown_ctrl.signal, t?.child(`walkBreakpoints(${model.name})`));
        const ctx = breakpoints.rungs.at(-1)!.n_ctx;

        this.max_ctx.set(model.name, ctx);
        console.log(printModelBreakpoints(breakpoints));
    }

    async shutdown(): Promise<void> {
        this.#cancelIdleTimer();

        const t = new Timer(`Planner.shutdown`);

        t.start('unloadAllModels');
        await this.#unloadAllModels(false).catch(err => {
            log.warn(`shutdown: unloadAllModels error: ${err}`);
        });
        t.stop();

        t.start(`cancel ${this.waiting.length} jobs`);
        const waiting = this.waiting.splice(0, this.waiting.length);
        for (const w of waiting) {
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

        // not idle; return
        if (
            this.active.readers !== 0 || 
            this.waiting.length !== 0 || 
            this.waiting_grow.length !== 0 || 
            this.active.pending
        ) {
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
        if (this.active.readers !== 0 || this.waiting.length !== 0 || this.waiting_grow.length !== 0 || this.active.pending) {
            return;
        }

        this.active.pending = true;

        log.info(`idle timeout reached; unloading all models`);
        await this.#unloadAllModels(false);
        this.restore_info.clear();

        this.active.pending = false;
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

        const active_id = this.active?.model;
        if (active_id && active_id !== exclude) {
            const model = this.models.get(active_id)!;

            if (!stash_kv) {
                promises.push(this.#unloadNoRestore(model, t));
            } else {
                promises.push(this.#unloadWithRestore(model, t));
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
        return this.waiting.some(w => w.model === name);
    }

    canServeNow(model: ModelEntry): boolean {
        return (
            this.active !== null
            && !this.active.pending
            && this.active.model === model.name
            && model.canPrompt()
        );
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