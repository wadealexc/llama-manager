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

        // if we have a restore point from a prior load, restore first
        if (restore) {
            let params: ReloadParams = { n_ctx: restore.n_ctx };

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

            t?.start('reloadModel');
            await this.client.reloadModel(params, this.name, signal);
            t?.stop();

            this.ladder_i = restore.ladder_i;
            this.n_ctx = restore.n_ctx;
        }

        // get max ctx given current config and available space
        t?.start('fitModel');
        const fit_ctx = await this.client.fitModel(this.name, signal);
        t?.stop();

        // if fit returns a larger ctx window, fill available space
        if (fit_ctx > this.curCtx()) {
            t?.start('reloadModel');
            await this.client.reloadModel({ n_ctx: fit_ctx }, this.name, signal);
            t?.stop();

            this.n_ctx = fit_ctx;
        }

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

        // get max ctx given current config and available space
        t?.start('fitModel');
        const fit_ctx = await this.client.fitModel(this.name, signal);
        t?.stop();

        if (fit_ctx <= this.curCtx()) {
            throw new Error(`expandToFit: unable to grow`);
        }

        // create restore point
        const restore = await this.#createRestorePoint(signal, t);

        // reload to fill available space
        t?.start('reloadModel');
        await this.client.reloadModel({ n_ctx: fit_ctx }, this.name, signal);
        t?.stop();

        // restore slots
        if (restore.slots) {
            t?.start('restoreAllSlots');
            await this.client.restoreAllSlots(this.name, restore.slots, signal);
            t?.stop();
        }

        this.n_ctx = fit_ctx;
        return this.curCtx();
    }

    async applyNextStrategy(signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status !== LoadStatus.LOADED) throw new Error(`applyNextStrategy: model must be loaded`);

        const next_rung = this.ladder.at(this.ladder_i + 1);
        if (!next_rung) throw new Error(`applyNextStrategy: model has no more strategies`);

        const restore = await this.#createRestorePoint(signal, t);

        if (next_rung.strategy !== 'none') {
            // get reload params for strategy
            this.log.debug(`model ${this.name} applying strategy ${next_rung.strategy}`);
            const params = next_rung.impl.getNewParams({});

            // apply strategy's additional action if needed
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

            // reload to new config
            t?.start('reloadModel');
            await this.client.reloadModel(params, this.name, signal);
            t?.stop();
        }

        // check fit
        t?.start('fitModel');
        const fit_ctx = await this.client.fitModel(this.name, signal);
        t?.stop();

        // reload to fill available space
        t?.start('reloadModel');
        await this.client.reloadModel({ n_ctx: fit_ctx }, this.name, signal);
        t?.stop();

        // restore slots
        if (restore.slots) {
            t?.start('restoreAllSlots');
            await this.client.restoreAllSlots(this.name, restore.slots, signal);
            t?.stop();
        }

        this.ladder_i++;
        this.n_ctx = fit_ctx;
        return fit_ctx;
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