import { LlamaAPI } from '../client/llama-api.js';
import type { InputTokensResponse, MemoryResponse, PromptMetadata, ReloadParams, Slot, SlotRestore, SlotSave } from '../client/types.js';
import type { ModelId } from '../config/types.js';
import { MIN_ALLOWED_CTX } from '../llama-cpp-constants.js';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export type Operation =
    | { kind: 'load' | 'unload' | 'memory'; model: ModelId }
    | { kind: 'count'; model: ModelId; tokens: number }
    | { kind: 'save'; model: ModelId; basename: string }
    | { kind: 'reload'; model: ModelId; params: ReloadParams }
    | { kind: 'restore'; model: ModelId; filenames: string[]; slot_id?: number };

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
    slot_directory?: string;
    save_bytes: number = 32;
    sidecar_bytes: number = 0;
    live_metadata: Map<ModelId, PromptMetadata> = new Map();
    saved_metadata: Map<string, PromptMetadata> = new Map();
    capacities: Map<ModelId, number> = new Map();
    slot_ids: number[] = [0];

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
        return this.slot_ids.map(id => ({ id, n_ctx: runtime.n_ctx, speculative: false, is_processing: false, n_prompt_tokens: id === 0 ? runtime.live_slots.length : 0 }));
    }

    async renderPrompt(body: unknown, model: ModelId, signal?: AbortSignal): Promise<InputTokensResponse> {
        const input_tokens = await this.countTokens(body, model, signal);
        const request = body as { prompt?: number[]; media?: PromptMetadata['media'] };
        return {
            input_tokens,
            object: 'response.input_tokens',
            tokens: request.prompt ?? Array(input_tokens).fill(1),
            media: request.media ?? [],
        };
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

    async saveSlot(model: ModelId, slot_id: number, filename: string, _signal?: AbortSignal): Promise<SlotSave> {
        const runtime = this.#requireKV(model);
        this.#record({ kind: 'save', model, basename: filename.replace(/-\d+\.bin$/, '') });
        this.#maybeFail('save');
        const metadata = this.live_metadata.get(model) ?? { tokens: runtime.live_slots.map(() => 1), media: [] };
        this.snapshots.set(filename, [...runtime.live_slots]);
        this.saved_metadata.set(filename, structuredClone(metadata));
        if (this.slot_directory) {
            await writeFile(join(this.slot_directory, filename), Buffer.alloc(this.save_bytes));
            if (this.sidecar_bytes) await writeFile(join(this.slot_directory, `${filename}.ckpt`), Buffer.alloc(this.sidecar_bytes));
        }
        return { ...this.#makeSlotSave(filename), id_slot: slot_id };
    }

    async saveAllSlots(model: ModelId, basename: string, signal?: AbortSignal): Promise<SlotSave[]> {
        return [await this.saveSlot(model, 0, `${basename}-0.bin`, signal)];
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
        runtime.n_ctx = params.n_ctx === 0 ? this.capacities.get(model) ?? capacity : params.n_ctx ?? runtime.n_ctx;
        runtime.kv_loaded = runtime.n_ctx !== MIN_ALLOWED_CTX;
        runtime.live_slots = [];
        this.live_metadata.delete(model);
        return runtime.n_ctx;
    }

    async restoreSlot(model: ModelId, slot_id: number, filename: string, _signal?: AbortSignal): Promise<SlotRestore> {
        const runtime = this.#requireKV(model);
        this.#record({ kind: 'restore', model, filenames: [filename], slot_id });
        this.#maybeFail('restore');
        if (!this.slot_ids.includes(slot_id)) throw new Error(`unknown slot ${slot_id}`);
        const snapshot = this.snapshots.get(filename);
        if (!snapshot) throw new Error(`snapshot not found: ${filename}`);
        runtime.live_slots = [...snapshot];
        const metadata = this.saved_metadata.get(filename);
        if (metadata) this.live_metadata.set(model, structuredClone(metadata));
        return {
            id_slot: slot_id,
            filename,
            n_restored: metadata?.tokens.length ?? snapshot.length,
            n_read: this.save_bytes,
            timings: { restore_ms: 0 },
        };
    }

    async restoreAllSlots(model: ModelId, saves: SlotSave[], signal?: AbortSignal): Promise<SlotRestore[]> {
        return await Promise.all(saves.map(save => this.restoreSlot(model, save.id_slot, save.filename, signal)));
    }

    setLiveSlots(contents: string[], model: ModelId = 'model'): void {
        const runtime = this.#requireKV(model);
        runtime.live_slots = [...contents];
        this.live_metadata.delete(model);
    }

    getLiveSlots(model: ModelId = 'model'): string[] {
        return [...this.#runtime(model).live_slots];
    }

    getSnapshot(model: ModelId, rung_i: number): string[] | undefined {
        const saves = this.operations.filter(operation => operation.kind === 'save' && operation.model === model);
        const save = saves[rung_i];
        const snapshot = save?.kind === 'save' ? this.snapshots.get(`${save.basename}-0.bin`) : undefined;
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
        const metadata = this.saved_metadata.get(filename) ?? { tokens: [], media: [] };
        return {
            ...metadata,
            id_slot: 0,
            filename,
            n_saved: metadata.tokens.length,
            n_written: this.save_bytes,
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
