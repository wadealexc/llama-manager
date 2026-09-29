import { LlamaAPI } from '../client/llama-api.js';
import type { MemoryResponse, ReloadParams, Slot, SlotRestore, SlotSave } from '../client/types.js';
import type { ModelId } from '../config/types.js';
import { MIN_ALLOWED_CTX } from '../llama-cpp-constants.js';

export type Operation =
    | { kind: 'load' | 'unload' | 'memory'; model: ModelId }
    | { kind: 'count'; model: ModelId; tokens: number }
    | { kind: 'save'; model: ModelId; basename: string }
    | { kind: 'reload'; model: ModelId; params: ReloadParams }
    | { kind: 'restore'; model: ModelId; filenames: string[] };

type Failure = 'save' | 'restore' | 'reload';

type ModelRuntime = {
    loaded: boolean;
    kv_loaded: boolean;
    n_ctx: number;
    live_slots: string[];
};

export class LlamaAPIMock extends LlamaAPI {
    operations: Operation[] = [];
    snapshots: Map<string, string[]> = new Map();
    models: Map<ModelId, ModelRuntime> = new Map();
    total_bytes: number = 32768;
    next_failure?: Failure;

    constructor() {
        super('http://127.0.0.1:0', { poll_interval_ms: 0, poll_timeout_ms: 0 });
    }

    async loadModelAndWait(model: ModelId, _signal?: AbortSignal): Promise<void> {
        const runtime = this.#runtime(model);
        if (runtime.loaded) throw new Error(`model ${model} is already loaded`);
        this.#record({ kind: 'load', model });
        runtime.loaded = true;
        runtime.kv_loaded = false;
        runtime.n_ctx = MIN_ALLOWED_CTX;
    }

    async unloadModelAndWait(model: ModelId, _signal?: AbortSignal): Promise<void> {
        const runtime = this.#requireLoaded(model);
        this.#record({ kind: 'unload', model });
        runtime.loaded = false;
        runtime.kv_loaded = false;
        runtime.n_ctx = 0;
        runtime.live_slots = [];
    }

    async getMemory(model: ModelId, _signal?: AbortSignal): Promise<MemoryResponse> {
        const runtime = this.#requireLoaded(model);
        this.#record({ kind: 'memory', model });
        let used = 0;
        for (const state of this.models.values()) {
            if (state.loaded) used += 4096 + state.n_ctx;
        }
        return {
            devices: [{
                name: 'fake-gpu',
                type: 'gpu',
                total: this.total_bytes,
                free: this.total_bytes - used,
                components: {
                    main: { model: 4096, context: runtime.n_ctx, compute: 0 },
                },
            }],
        };
    }

    async getSlots(model: ModelId, _signal?: AbortSignal): Promise<Slot[]> {
        const runtime = this.#requireLoaded(model);
        return [{ id: 0, n_ctx: runtime.n_ctx, speculative: false, is_processing: false }];
    }

    async countTokens(body: unknown, model: ModelId, _signal?: AbortSignal): Promise<number> {
        this.#requireLoaded(model);
        const tokens = (body as { tokens?: unknown })?.tokens;
        if (typeof tokens !== 'number' || !Number.isFinite(tokens)) {
            throw new Error('expected a numeric tokens field');
        }
        this.#record({ kind: 'count', model, tokens });
        return tokens;
    }

    async saveAllSlots(model: ModelId, basename: string, _signal?: AbortSignal): Promise<SlotSave[]> {
        const runtime = this.#requireKV(model);
        this.#record({ kind: 'save', model, basename });
        this.#maybeFail('save');

        const filename = `${basename}-0.bin`;
        this.snapshots.set(filename, [...runtime.live_slots]);
        return [this.#makeSlotSave(filename)];
    }

    async reloadModel(params: ReloadParams, model: ModelId, _signal?: AbortSignal): Promise<number> {
        const runtime = this.#requireLoaded(model);
        this.#record({ kind: 'reload', model, params: structuredClone(params) });
        this.#maybeFail('reload');

        if (params.n_ctx === 0) {
            for (const [other_id, other] of this.models) {
                if (other_id !== model && other.kv_loaded) {
                    throw new Error(`model ${other_id} already has a KV cache`);
                }
            }
        }

        const capacity = params.cache_type_k === 'q8_0'
            ? 8192
            : params.cache_type_k === 'q4_0'
                ? 16384
                : 4096;
        runtime.n_ctx = params.n_ctx === 0 ? capacity : params.n_ctx ?? runtime.n_ctx;
        runtime.kv_loaded = runtime.n_ctx !== MIN_ALLOWED_CTX;
        runtime.live_slots = [];
        return runtime.n_ctx;
    }

    async restoreAllSlots(model: ModelId, saves: SlotSave[], _signal?: AbortSignal): Promise<SlotRestore[]> {
        const runtime = this.#requireKV(model);
        this.#record({ kind: 'restore', model, filenames: saves.map(save => save.filename) });
        this.#maybeFail('restore');

        for (const save of saves) {
            if (save.id_slot !== 0) throw new Error(`unknown slot ${save.id_slot}`);
            const snapshot = this.snapshots.get(save.filename);
            if (!snapshot) throw new Error(`snapshot not found: ${save.filename}`);
            runtime.live_slots = [...snapshot];
        }

        return saves.map(save => ({
            id_slot: save.id_slot,
            filename: save.filename,
            n_restored: runtime.live_slots.length,
            n_read: runtime.live_slots.length,
            timings: { restore_ms: 0 },
        }));
    }

    setLiveSlots(contents: string[], model: ModelId = 'model'): void {
        const runtime = this.#requireKV(model);
        runtime.live_slots = [...contents];
    }

    getLiveSlots(model: ModelId = 'model'): string[] {
        return [...this.#runtime(model).live_slots];
    }

    seedSnapshot(model: ModelId, rung_i: number, contents: string[]): SlotSave[] {
        const filename = `${model}-rung-${rung_i}-0.bin`;
        this.snapshots.set(filename, [...contents]);
        return [this.#makeSlotSave(filename)];
    }

    getSnapshot(model: ModelId, rung_i: number): string[] | undefined {
        const snapshot = this.snapshots.get(`${model}-rung-${rung_i}-0.bin`);
        return snapshot ? [...snapshot] : undefined;
    }

    hasKV(model: ModelId): boolean {
        return this.#runtime(model).kv_loaded;
    }

    isLoaded(model: ModelId): boolean {
        return this.#runtime(model).loaded;
    }

    failNext(kind: Failure): void {
        this.next_failure = kind;
    }

    clearOperations(): void {
        this.operations.length = 0;
    }

    #record(operation: Operation): void {
        this.operations.push(operation);
    }

    #makeSlotSave(filename: string): SlotSave {
        return {
            id_slot: 0,
            filename,
            n_saved: 0,
            n_written: 0,
            timings: { save_ms: 0 },
        };
    }

    #maybeFail(kind: Failure): void {
        if (this.next_failure !== kind) return;
        this.next_failure = undefined;
        throw new Error(`${kind} failed`);
    }

    #runtime(model: ModelId): ModelRuntime {
        let runtime = this.models.get(model);
        if (!runtime) {
            runtime = { loaded: false, kv_loaded: false, n_ctx: 0, live_slots: [] };
            this.models.set(model, runtime);
        }
        return runtime;
    }

    #requireLoaded(model: ModelId): ModelRuntime {
        const runtime = this.#runtime(model);
        if (!runtime.loaded) throw new Error(`model ${model} is not loaded`);
        return runtime;
    }

    #requireKV(model: ModelId): ModelRuntime {
        const runtime = this.#requireLoaded(model);
        if (!runtime.kv_loaded) throw new Error(`model ${model} has no KV cache`);
        return runtime;
    }
}
