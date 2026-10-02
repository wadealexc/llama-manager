import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { MemoryResponse, ReloadParams, Slot, SlotSave } from "../client/types.js";
import { LoadStatus, type ModelId, type ModelState, type Strategy, type StrategyId, type Tokens } from "../config/types.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { logger } from "../logger.js";
import { Timer } from "./timer.js";
import type { StrategyImpl } from "./types.js";

export interface Rung {
    strategy: StrategyId | 'swap-model' | 'baseline';
    variant_name: string;
    impl?: StrategyImpl;
    state: ModelState;
    n_ctx_cap: Tokens;
    bytes_needed: number;
    last_slots?: SlotSave[];
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

    constructor(
        client: LlamaAPI,
        name: ModelId,
        aliases: string[],
        initial_state: ModelState,
        ladder: Strategy[],
        impls: Map<StrategyId, StrategyImpl>
    ) {
        this.log = logger.withTag(name);

        this.client = client;
        this.name = name;
        this.aliases = aliases;
        this.ladder = this.#buildLadder(initial_state, ladder, impls);
    }

    async loadWeights(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.UNLOADED) return;

        t?.start(`loadModelAndWait`);
        await this.client.loadModelAndWait(this.curVariant(), signal);
        t?.stop();

        // record bytes for weights-only, if needed
        // TODO: combine with breakpoint calculation
        if (!this.bytes_needed_no_kv) {
            const mem = await this.client.getMemory(this.curVariant(), signal);
            let weight_bytes = 0;
            let context_bytes = 0;
            for (const dev of mem.devices) {
                if (dev.type === "cpu") continue;
                for (const comp of ["main", "spec", "mmproj"] as const) {
                    const c = dev.components[comp];
                    if (!c) continue;
                    weight_bytes += c.model;
                    context_bytes += c.context + c.compute;
                }
            }

            this.bytes_needed_no_kv = weight_bytes + context_bytes;
        }

        this.#setWeightsOnly();
    }

    async moveToRung(rung_i: number, signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`moveToRung: model must be loaded`);
        if (!this.hasRung(rung_i)) throw new Error(`moveToRung: rung index out of bounds`);

        const cur_i = this.ladder_i;
        this.#logTransition(rung_i);

        // create a restore point at the current rung
        await this.#createRestorePoint(signal, t);

        await this.#applyRung(rung_i, signal, t);

        // if we're increasing the rung or keeping the current rung, retain live kvcache
        // otherwise, restore kvcache at new rung
        const kv_restore_i = (rung_i >= cur_i && cur_i >= 0) ? cur_i : rung_i;
        await this.#restoreSlots(signal, kv_restore_i, t);

        return this.getCurCtx();
    }

    // identical to moveToRung, except without creating or using a restore point
    async applyRung(rung_i: number, signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`applyRung: model must be loaded`);
        if (!this.hasRung(rung_i)) throw new Error(`applyRung: rung index out of bounds`);

        await this.#applyRung(rung_i, signal, t);
        return this.getCurCtx();
    }

    // identical to applyRung, except that it applies the next rung in the ladder
    async applyNextRung(signal: AbortSignal, t?: Timer): Promise<number> {
        return await this.applyRung(this.ladder_i + 1, signal, t);
    }

    async #applyRung(i: number, signal: AbortSignal, t?: Timer): Promise<void> {
        const dest_state = this.stateAtRung(i)!;
        const cur_variant = this.curVariant();
        const new_variant = dest_state.model_variant;

        // swap model variant
        if (cur_variant !== new_variant) {
            t?.start(`unloadModelAndWait ${cur_variant}`);
            await this.client.unloadModelAndWait(cur_variant);
            t?.stop();

            t?.start(`loadModelAndWait ${new_variant}`);
            await this.client.loadModelAndWait(new_variant, signal);
            t?.stop();
        }

        const { model_variant, ...state } = structuredClone(dest_state);
        const params: ReloadParams = { ...state, n_ctx: 0 };

        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, new_variant, signal);
        t?.stop();

        this.ladder_i = i;
        this.status = LoadStatus.LOADED;
    }

    async unloadKV(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.LOADED) return;

        // create a restore point at the current rung
        await this.#createRestorePoint(signal, t);

        // unload kv
        t?.start('reloadModel');
        await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, this.curVariant(), signal);
        t?.stop();

        this.#setWeightsOnly();
    }

    // unload model, saving slots and returning a restore point if model was loaded
    async unloadWeights(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status === LoadStatus.UNLOADED) return;

        // if model is currently loaded, create restore point
        if (this.status === LoadStatus.LOADED) {
            await this.#createRestorePoint(signal, t)
        }

        t?.start('unloadModelAndWait');
        await this.client.unloadModelAndWait(this.curVariant());
        t?.stop();

        this.#setUnloaded();
    }

    // unload a model without saving slots
    async unloadHard(t?: Timer): Promise<void> {
        if (this.status === LoadStatus.UNLOADED) return;

        t?.start('unloadModelAndWait');
        await this.client.unloadModelAndWait(this.curVariant());
        t?.stop();

        this.#setUnloaded();
    }

    async #createRestorePoint(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.LOADED) return;

        const path_base = this.#getRestoreName();

        try {
            t?.start('saveAllSlots');
            const slots = await this.client.saveAllSlots(this.curVariant(), path_base, signal);
            this.ladder[this.ladder_i].last_slots = slots;
            this.log.info(`saved slots at rung ${this.ladder_i} | ${slots.map(slot => `${slot.id_slot}=${slot.n_saved} tokens`).join(', ')}`);
        } catch (err) {
            this.log.error(`error creating restore point: ${err}`);
        } finally {
            t?.stop();
        }
    }

    async #restoreSlots(signal: AbortSignal, src_rung: number, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.LOADED) return;
        if (src_rung >= this.ladder.length) throw new Error(`restoreSlots: bad src rung ${src_rung}`);

        const slots = this.ladder[src_rung].last_slots;
        if (!slots) return;

        try {
            t?.start('restoreAllSlots');
            const restored = await this.client.restoreAllSlots(this.curVariant(), slots, signal);
            this.log.info(`restored rung ${src_rung} slots into rung ${this.ladder_i} | ${restored.map(slot => `${slot.id_slot}=${slot.n_restored} tokens`).join(', ')}`);
        } catch (err) {
            this.log.error(`error restoring last slots: ${err}`);
        } finally {
            t?.stop();
        }
    }

    #getRestoreName(): string {
        return `${this.name}-rung-${this.ladder_i}`;
    }

    async completions(body: unknown, signal: AbortSignal, t?: Timer): Promise<Response> {
        return await this.client.completions(body, this.curVariant(), signal);
    }

    async countTokens(body: unknown, signal: AbortSignal, t?: Timer): Promise<number | null> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`countTokens: model ${this.name} is not loaded`);

        t?.start('countTokens');
        try {
            return await this.client.countTokens(body, this.curVariant(), signal);
        } catch (err: any) {
            this.log.error(`countTokens error: ${err}`);
            return null;
        } finally {
            t?.stop();
        }
    }

    async getMemory(signal: AbortSignal): Promise<MemoryResponse> {
        return await this.client.getMemory(this.curVariant(), signal);
    }

    async getSlots(signal: AbortSignal): Promise<Slot[]> {
        return await this.client.getSlots(this.curVariant(), signal);
    }

    curVariant(): string {
        if (this.ladder_i === -1) return this.name;
        else return this.ladder[this.ladder_i].state.model_variant;
    }

    rungLabel(i: number): string {
        const rung = this.ladder[i];
        return rung.strategy === 'swap-model'
            ? `swap-model:${rung.variant_name}`
            : rung.strategy;
    }

    #logTransition(dest_i: number): void {
        const src_i = this.ladder_i;
        const labels = this.ladder.map((_, i) => this.rungLabel(i));
        const crossed = dest_i > src_i
            ? labels.slice(Math.max(0, src_i) + 1, dest_i + 1)
            : labels.slice(dest_i + 1, src_i + 1).reverse();
        const action = dest_i === src_i ? 'reloading' : dest_i > src_i ? 'applying' : 'removing';
        const strategies = crossed.join(', ');
        if (src_i === -1) {
            this.log.info(`initializing model to rung ${dest_i}${strategies ? `: applying [${strategies}]` : ''}`);
        } else if (src_i === dest_i) {
            this.log.info(`reloading at current rung (${src_i})`);
        } else {
            this.log.info(`rung ${src_i} → ${dest_i}: ${action}${strategies ? ` [${strategies}]` : ''}`);
        }
    }

    stateAtRung(i: number): ModelState | undefined {
        return this.ladder[i]?.state;
    }

    paramsForRung(rung_i: number): ReloadParams | undefined {
        const state = this.ladder[rung_i]?.state;
        if (!state) return undefined;

        const { model_variant, ...rest } = structuredClone(state);
        return {
            ...rest,
            n_ctx: 0,
        }
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

    getCurCtx(): number {
        return this.n_ctx;
    }

    hasRung(i: number): boolean {
        return i >= 0 && i < this.ladder.length;
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

    hasNextRung(): boolean {
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

    #buildLadder(initial_state: ModelState, ladder: Strategy[], strats: Map<StrategyId, StrategyImpl>): Rung[] {
        if (initial_state.model_variant === undefined) {
            throw new Error(`#buildLadder: expected initial model variant for ${this.name}`);
        }

        let state = initial_state;
        let variant_name = 'baseline';

        const rungs: Rung[] = [{
            strategy: 'baseline',
            variant_name,
            state,
            n_ctx_cap: -1,
            bytes_needed: 0,
        }];

        for (const [i, item] of ladder.entries()) {
            if (item.kind === 'reload-model') {
                const s = strats.get(item.id);
                if (!s) throw new Error(`#buildLadder: unrecognized strategy: ${item.id}`);

                if (!s.canApply(state)) {
                    this.log.warn(`cannot apply strategy ${item.id} at pos ${i}; skipping`);
                    continue;
                }

                state = s.getNewState(state);
                rungs.push({
                    strategy: item.id,
                    variant_name,
                    impl: s,
                    state,
                    n_ctx_cap: -1,
                    bytes_needed: 0,
                });
            } else {
                if (!item.router_id) throw new Error(`#buildLadder: expected router id for variant ${item.variant}`);
                state = structuredClone(state);
                state.model_variant = item.router_id;
                variant_name = item.variant;
                state.cache_type_k = 'f16';
                state.cache_type_v = 'f16';

                rungs.push({
                    strategy: 'swap-model',
                    variant_name,
                    state,
                    n_ctx_cap: -1,
                    bytes_needed: 0,
                })
            }
        }

        return rungs;
    }
}
