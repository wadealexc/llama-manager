import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ReloadParams, SlotSave } from "../client/types.js";
import { LoadStatus, type ModelId, type ModelState, type StrategyId, type Tokens } from "../config/types.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { logger } from "../logger.js";
import { Timer } from "./timer.js";
import type { Strategy } from "./types.js";

export interface RestorePoint {
    ladder_i: number;
    n_ctx: number;
    slots?: SlotSave[];
}

export interface Rung {
    strategy: StrategyId | 'none';
    impl: Strategy;
    state: ModelState;
    n_ctx_cap: Tokens;
    bytes_needed: number;
}

export class ModelEntry {

    log: ConsolaInstance;

    client: LlamaAPI;

    name: ModelId;
    aliases: string[];

    ladder: Rung[];
    ladder_i: number = -1;
    n_ctx: number = 0;
    status: LoadStatus = LoadStatus.UNLOADED;

    // TODO: clean up first-load semantics
    bytes_needed_no_kv?: number;

    constructor(client: LlamaAPI, name: ModelId, aliases: string[], ladder: Rung[]) {
        this.log = logger.withTag(name);

        this.client = client;
        this.name = name;
        this.aliases = aliases;
        this.ladder = ladder;
    }

    async loadWeights(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.UNLOADED) return;

        t?.start(`loadModelAndWait`);
        await this.client.loadModelAndWait(this.name, signal);
        t?.stop();

        // record bytes for weights-only, if needed
        // TODO: combine with breakpoint calculation
        if (!this.bytes_needed_no_kv) {
            const mem = await this.client.getMemory(this.name, signal);
            let weight_bytes = 0;
            let context_bytes = 0;
            for (const dev of mem.devices) {
                if (dev.type === "cpu") continue;
                for (const name of ["main", "spec", "mmproj"] as const) {
                    const c = dev.components[name];
                    if (!c) continue;
                    weight_bytes += c.model;
                    context_bytes += c.context + c.compute;
                }
            }

            this.bytes_needed_no_kv = weight_bytes + context_bytes;
        }

        this.#setWeightsOnly();
    }

    async loadWithKV(restore: RestorePoint | null, signal: AbortSignal, t?: Timer, target_rung?: number): Promise<number> {
        const rung_i = target_rung ?? restore?.ladder_i ?? 0;

        const params = this.paramsForRung(rung_i);
        if (!params) throw new Error(`loadWithKV: rung index out of bounds`);

        if (this.status === LoadStatus.LOADED) return this.curCtx();

        if (this.status === LoadStatus.UNLOADED) {
            await this.loadWeights(signal, t);
        }

        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, this.name, signal);
        t?.stop();

        if (restore?.slots) {
            t?.start('restoreAllSlots');
            await this.client.restoreAllSlots(this.name, restore.slots, signal);
            t?.stop();
        }

        this.ladder_i = rung_i;
        this.status = LoadStatus.LOADED;
        return this.curCtx();
    }

    async moveToRung(rung_i: number, signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status !== LoadStatus.LOADED) throw new Error(`moveToRung: model must be loaded`);

        const params = this.paramsForRung(rung_i);
        if (!params) throw new Error(`moveToRung: rung index out of bounds`);

        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, this.name, signal);
        t?.stop();

        this.ladder_i = rung_i;
        return this.curCtx();
    }

    async expandToFit(signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status !== LoadStatus.LOADED) throw new Error(`expandToFit: model must be loaded`);

        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel({ n_ctx: 0 }, this.name, signal);
        t?.stop();

        return this.curCtx();
    }

    async applyNextStrategy(signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status !== LoadStatus.LOADED) throw new Error(`applyNextStrategy: model must be loaded`);

        const next_rung = this.ladder_i + 1;
        const params = this.paramsForRung(next_rung);
        if (!params) throw new Error(`applyNextStrategy: model has no more strategies`);

        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, this.name, signal);
        t?.stop();

        this.ladder_i = next_rung;
        return this.curCtx();
    }

    async unloadKV(signal: AbortSignal, t?: Timer): Promise<RestorePoint | null> {
        if (this.status !== LoadStatus.LOADED) return null;

        // create a restore point
        const restore = await this.#createRestorePoint(signal, t);

        // unload kv
        t?.start('reloadModel');
        await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, this.name, signal);
        t?.stop();

        this.#setWeightsOnly();
        return restore;
    }

    // unload model, saving slots and returning a restore point if model was loaded
    async unloadWeights(signal: AbortSignal, t?: Timer): Promise<RestorePoint | null> {
        if (this.status === LoadStatus.UNLOADED) return null;

        // if model is currently loaded, create restore point
        let restore: RestorePoint | null = null;
        if (this.status === LoadStatus.LOADED) {
            restore = await this.#createRestorePoint(signal, t)
        }

        t?.start('unloadModelAndWait');
        await this.client.unloadModelAndWait(this.name);
        t?.stop();

        this.#setUnloaded();
        return restore;
    }

    // unload a model without saving slots
    async unloadHard(t?: Timer): Promise<void> {
        if (this.status === LoadStatus.UNLOADED) return;

        t?.start('unloadModelAndWait');
        await this.client.unloadModelAndWait(this.name);
        t?.stop();

        this.#setUnloaded();
    }

    async #createRestorePoint(signal: AbortSignal, t?: Timer): Promise<RestorePoint> {
        t?.start('saveAllSlots');
        const restore: RestorePoint = {
            ladder_i: this.ladder_i,
            n_ctx: this.n_ctx,
            slots: await this.client.saveAllSlots(this.name, signal),
        };
        t?.stop();

        return restore;
    }

    curCtx(): number {
        return this.n_ctx;
    }

    async countTokens(body: unknown, signal: AbortSignal, t?: Timer): Promise<number | null> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`countTokens: model ${this.name} is not loaded`);

        t?.start('countTokens');
        try {
            return await this.client.countTokens(body, this.name, signal);
        } catch (err: any) {
            this.log.error(`countTokens error: ${err}`);
            return null;
        } finally {
            t?.stop();
        }
    }

    paramsForRung(rung_i: number): ReloadParams | undefined {
        const state = this.ladder[rung_i]?.state;
        if (!state) return undefined;

        return {
            ...structuredClone(state),
            n_ctx: 0,
        };
    }

    getMinimumRung(tokens: Tokens): number | null {
        const i = this.ladder.findIndex(rung => rung.n_ctx_cap >= tokens);
        return i === -1 ? null : i;
    }

    // NOTE: returns null if model has not been loaded for the first time
    getMaxCtx(): number | null {
        const cap = this.ladder.at(-1)!.n_ctx_cap;
        if (cap === -1) return null;

        return cap;
    }

    getCtxCap(rung: number): number | null {
        const cap = this.ladder.at(rung)?.n_ctx_cap;
        if (cap === undefined) return null;
        if (cap === -1) return null;

        return cap;
    }

    // returns the free space needed to load the model to `rung`, in bytes,
    // considering the model's current load status.
    // NOTE: returns null if model has not been first-loaded yet.
    // NOTE: this may return a negative value if loading to `rung` would free space
    getBytesNeeded(rung: number): number | null {
        const total = this.ladder.at(rung)?.bytes_needed;
        if (total === undefined) return null;

        if (this.status === LoadStatus.UNLOADED) return total;

        // model is partially or fully loaded; subtract bytes for weights
        if (this.status === LoadStatus.WEIGHTS_ONLY) {
            return total - this.bytes_needed_no_kv!;
        }

        // model is fully loaded; subtract bytes for current rung
        return total - this.ladder[this.ladder_i]!.bytes_needed;
    }

    hasNextStrategy(): boolean {
        return this.ladder.length - 1 > this.ladder_i;
    }

    #setWeightsOnly() {
        this.status = LoadStatus.WEIGHTS_ONLY;
        this.n_ctx = MIN_ALLOWED_CTX;
    }

    #setUnloaded() {
        this.status = LoadStatus.UNLOADED;
        this.n_ctx = 0;
        this.ladder_i = -1;
    }
}