import type { ConsolaInstance } from "consola";
import type { LlamaAPI } from "../client/llama-api.js";
import type { InputTokensResponse, MemoryResponse, PromptMetadata, ReloadParams, Slot, SlotRestore, SlotSave } from "../client/types.js";
import { LoadStatus, type ModelId, type ModelState, type Strategy, type StrategyId, type Tokens } from "../config/types.js";
import { MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";
import { logger } from "../logger.js";
import { Timer } from "./timer.js";
import type { StrategyImpl } from "./types.js";
import { isValid, type PromptCache } from "./prompt-cache.js";

export interface Rung {
    strategy: StrategyId | 'swap-model' | 'baseline';
    variant_name: string;
    impl?: StrategyImpl;
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
    cache?: PromptCache;
    needs_restore: boolean = false;

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

    async applyRung(rung_i: number, signal: AbortSignal, t?: Timer): Promise<number> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`applyRung: model must be loaded`);
        if (!this.hasRung(rung_i)) throw new Error(`applyRung: rung index out of bounds`);

        this.#logTransition(rung_i);
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

        const { model_variant, ...state } = dest_state;
        const params: ReloadParams = { ...state, n_ctx: 0 };

        t?.start('reloadModel');
        this.n_ctx = await this.client.reloadModel(params, new_variant, signal);
        t?.stop();

        this.ladder_i = i;
        this.status = LoadStatus.LOADED;
    }

    async unloadKV(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.LOADED) return;

        await this.saveSlots(signal, t);

        // unload kv
        t?.start('reloadModel');
        await this.client.reloadModel({ n_ctx: MIN_ALLOWED_CTX }, this.curVariant(), signal);
        t?.stop();

        this.#setWeightsOnly();
    }

    async unloadWeights(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status === LoadStatus.UNLOADED) return;

        await this.saveSlots(signal, t);

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

    // save active slots to cache
    async saveSlots(signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.LOADED || !this.cache) return;
        await this.cache.save(this, signal, t);
    }

    // restore prompts from cache into live slot state
    async restorePrompts(prompts: PromptMetadata[], signal: AbortSignal, t?: Timer): Promise<void> {
        if (this.status !== LoadStatus.LOADED) return;
        this.needs_restore = false;
        await this.cache?.restore(this, prompts, signal, t);
    }

    async completions(body: unknown, signal: AbortSignal, t?: Timer): Promise<Response> {
        return await this.client.completions(body, this.curVariant(), signal);
    }

    async countTokens(body: unknown, signal: AbortSignal, t?: Timer): Promise<number | null> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`countTokens: model ${this.name} is not loaded`);

        return (await this.renderPrompt(body, signal, t))?.input_tokens ?? null;
    }

    async renderPrompt(body: unknown, signal: AbortSignal, t?: Timer): Promise<InputTokensResponse | null> {
        if (this.status === LoadStatus.UNLOADED) throw new Error(`renderPrompt: model ${this.name} is not loaded`);
        t?.start('renderPrompt');
        try {
            const prompt = await this.client.renderPrompt(body, this.curVariant(), signal);
            if (!isValid(prompt) || prompt.tokens.length !== prompt.input_tokens) {
                this.log.warn('invalid rendered prompt metadata; cache matching disabled for this request');
                return { ...prompt, tokens: [], media: [] };
            }
            return prompt;
        } catch (err) {
            this.log.error(`renderPrompt error: ${err}`);
            return null;
        } finally {
            t?.stop();
        }
    }

    async saveSlot(slot_id: number, filename: string, signal?: AbortSignal): Promise<SlotSave> {
        return await this.client.saveSlot(this.curVariant(), slot_id, filename, signal);
    }

    async restoreSlot(slot_id: number, filename: string, signal?: AbortSignal): Promise<SlotRestore> {
        return await this.client.restoreSlot(this.curVariant(), slot_id, filename, signal);
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

    curState(): ModelState | undefined {
        if (this.ladder_i < 0) return undefined;
        else return structuredClone(this.ladder[this.ladder_i].state);
    }

    stateAtRung(i: number): ModelState | undefined {
        return structuredClone(this.ladder[i]?.state);
    }

    paramsForRung(rung_i: number): ReloadParams | undefined {
        const state = this.ladder[rung_i]?.state;
        if (!state) return undefined;

        const { model_variant, ...rest } = state;
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
        this.needs_restore = true;
    }

    #setUnloaded() {
        this.status = LoadStatus.UNLOADED;
        this.n_ctx = 0;
        this.ladder_i = -1;
        this.needs_restore = false;
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
