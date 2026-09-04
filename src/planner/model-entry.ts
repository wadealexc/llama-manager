import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ReloadParams, SlotSave } from "../client/types.js";
import { LoadStatus, type ModelId, type ModelState, type StrategyId } from "../config/types.js";
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
}

export class ModelEntry {

    log: ConsolaInstance;

    client: LlamaAPI;

    name: ModelId;

    readonly ladder: Rung[];
    ladder_i: number = -1;
    n_ctx: number = 0;
    status: LoadStatus = LoadStatus.UNLOADED;

    constructor(client: LlamaAPI, name: ModelId, ladder: Rung[]) {
        this.log = logger.withTag(name);

        this.client = client;
        this.name = name;
        this.ladder = ladder;
    }

    async loadWeights(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.UNLOADED) return;

        t?.start(`loadModelAndWait`);
        await this.client.loadModelAndWait(this.name, signal);
        t?.stop();

        this.#setWeightsOnly();
    }

    async loadWithKV(restore: RestorePoint | null, signal: AbortSignal, t?: Timer): Promise<number> {
        if (restore && restore.ladder_i >= this.ladder.length) {
            throw new Error(`loadWithKV: strategy index out of bounds`);
        }

        if (this.status === LoadStatus.LOADED) return this.curCtx();

        if (this.status === LoadStatus.UNLOADED) {
            await this.loadWeights(signal, t);
        }

        let params: ReloadParams = { n_ctx: 0 };

        // handle restore point from prior load
        if (restore) {
            for (const [i, rung] of this.ladder.entries()) {
                // stop when target is reached
                if (i > restore.ladder_i) break;

                // initial 'empty' strategy
                if (rung.strategy === 'none') continue;

                this.log.debug(`loadWithKV: applying strategy ${rung.strategy}`);
                params = rung.impl.getNewParams(params);

                if (!rung.impl.action) continue;

                try {
                    t?.start(`${rung.strategy}: action`);
                    await rung.impl.action({
                        slots: restore?.slots,
                        signal,
                    });
                    t?.stop();
                } catch (err) {
                    this.log.error(`${rung.strategy}.action error: ${err}`);
                }
            }

            this.ladder_i = restore.ladder_i;
        } else {
            this.ladder_i = 0;
        }

        // load kvcache, applying restore point params if supplied
        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, this.name, signal);
        t?.stop();

        if (restore && restore.slots) {
            t?.start('restoreAllSlots');
            await this.client.restoreAllSlots(this.name, restore.slots, signal);
            t?.stop();
        }

        this.status = LoadStatus.LOADED;
        return this.curCtx();
    }

    async expandToFit(signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status !== LoadStatus.LOADED) throw new Error(`expandToFit: model must be loaded`);

        // stash existing kvcache
        const restore = await this.#createRestorePoint(signal, t);

        // reload to fill available space
        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel({ n_ctx: 0 }, this.name, signal);
        t?.stop();

        // restore kvcache
        if (restore.slots) {
            t?.start('restoreAllSlots');
            await this.client.restoreAllSlots(this.name, restore.slots, signal);
            t?.stop();
        }

        return this.curCtx();
    }

    async applyNextStrategy(create_restore: boolean, signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status !== LoadStatus.LOADED) throw new Error(`applyNextStrategy: model must be loaded`);

        const next_rung = this.ladder.at(this.ladder_i + 1);
        if (!next_rung) throw new Error(`applyNextStrategy: model has no more strategies`);

        // stash existing kvcache
        let restore: RestorePoint | undefined;
        if (create_restore) {
            restore = await this.#createRestorePoint(signal, t);
        }

        let params: ReloadParams = { n_ctx: 0 };

        if (next_rung.strategy !== 'none') {
            // get reload params for strategy
            this.log.debug(`model ${this.name} applying strategy ${next_rung.strategy}`);
            params = next_rung.impl.getNewParams(params);

            if (next_rung.impl.action) {
                try {
                    t?.start(`${next_rung.strategy}: action`);
                    await next_rung.impl.action({
                        slots: restore?.slots,
                        signal,
                    });
                    t?.stop();
                } catch (err) {
                    this.log.error(`${next_rung.strategy}.action error: ${err}`);
                }
            }
        }

        // reload to fill available space
        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, this.name, signal);
        t?.stop();

        // restore slots
        if (restore?.slots) {
            t?.start('restoreAllSlots');
            await this.client.restoreAllSlots(this.name, restore.slots, signal);
            t?.stop();
        }

        this.ladder_i++;
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

    currentState(): ModelState | undefined {
        return this.ladder.at(this.ladder_i)?.state ?? undefined;
    }

    canPrompt(): boolean {
        return this.status === LoadStatus.LOADED;
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