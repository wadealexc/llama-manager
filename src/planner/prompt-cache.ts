import { readdir, stat, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import type { PromptMetadata, Slot, SlotSave } from '../client/types.js';
import { hasMmproj, type ModelState } from '../config/types.js';
import { logger } from '../logger.js';
import type { ModelEntry } from './model-entry.js';

const log = logger.withTag('prompt-cache');

const CACHE_FILENAME = /^lm-cache-\d+-\d+\.bin(?:\.ckpt)?$/;
const MEDIA_MARKER = -1;

export interface CacheEntry extends PromptMetadata {
    seq: number;

    // model info
    model: string;
    variant: string;
    rung: number;
    state: ModelState;

    slot_id: number;
    filename: string;
    bytes: number;

    saved_at: number;
    last_used: number;
}

type Assignment = {
    entry: CacheEntry;
    slot: Slot;
    matched: number
}

export class PromptCache {

    // map file names to size in bytes
    files: Map<string, number> = new Map();
    directory: string;

    entries: CacheEntry[] = [];
    sequence = Date.now();

    // allowed total size of prompt cache across all models
    budget_bytes: number;

    // queue of pending actions
    pending: Promise<unknown> = Promise.resolve();

    stopped = false;

    constructor(directory: string, budget_mib: number) {
        this.directory = directory;
        this.budget_bytes = budget_mib * 1024 ** 2;
    }

    async start(): Promise<void> {
        await this.#queue(async () => {
            await this.#cleanup();
            log.info(`startup | ${this.#sizeLabel()}`);
        });
    }

    async shutdown(): Promise<void> {
        this.stopped = true;
        await this.#queue(async () => {
            await this.#cleanup();
            log.info(`shutdown | ${this.#sizeLabel()}`);
        });
    }

    // Save a model's live slots to the prompt cache
    //
    // NOTE: Method assumes it is being called inside planner's write lock
    // 
    // NOTE: saving is provisional. we assume saves might push us over budget,
    // and rely on restore and subsequent calls to enforce to ensure we come back
    // under budget.
    async save(model: ModelEntry, signal: AbortSignal): Promise<void> {
        if (this.stopped || this.budget_bytes === 0) return;

        await this.#queue(async () => {
            const slots = await model.getSlots(signal);

            for (const slot of slots) {
                // operations are expected to be called within the context of planner's
                // write lock, so we should never get a busy slot here.
                if (slot.is_processing) {
                    log.warn(`save: got busy slot: ${slot.id} | n_tokens: ${slot.n_prompt_tokens}`);
                    continue;
                }

                if (slot.n_prompt_tokens === 0) continue;

                const seq = ++this.sequence;
                const filename = `lm-cache-${seq}-${slot.id}.bin`;

                const saved = await model.saveSlot(slot.id, filename, signal)
                    .catch((err) => log.error(`${model.name}: failed to save slot ${slot.id}| filename: ${filename} | err: ${err}`));

                if (!saved) continue;

                try {
                    // validate file saved
                    await this.#syncSize(filename);
                    if (!this.files.has(filename)) throw new Error(`PromptCache.save: error saving file`);
                    if (!isValid(saved) || saved.n_saved !== saved.tokens.length) {
                        throw new Error(`PromptCache.save: invalid saved prompt metadata`);
                    }

                    // check that the file is not over budget on its own
                    const bytes = this.#getFileBytes(filename);
                    if (saved.n_saved === 0 || bytes > this.budget_bytes) {
                        await this.#deleteFiles(filename);
                        log.info(`${model.name}: slot save ${filename} exceeds total budget; skipping | size: ${fmtBytes(bytes)} | budget: ${fmtBytes(this.budget_bytes)}`);
                        continue;
                    }

                    // remove duplicate entries and underlying files
                    const state = model.curState()!;
                    for (const entry of [...this.entries].filter(entry => entry.model === model.name)) {
                        if (JSON.stringify(entry.state) === JSON.stringify(state) && isIdentical(entry, saved)) {
                            await this.#remove(entry);
                        }
                    }

                    const now = Date.now();
                    this.entries.push({
                        seq,
                        model: model.name,
                        variant: model.curVariant(),
                        rung: model.ladder_i,
                        state,
                        slot_id: slot.id,
                        filename,
                        bytes,
                        saved_at: now,
                        last_used: now,
                        tokens: saved.tokens,
                        media: saved.media,
                    });

                    log.info(`${model.name}: saved ${saved.n_saved} tokens to ${filename} (size: ${fmtBytes(bytes)}) | ${this.#sizeLabel()}`);
                } catch (err) {
                    await this.#syncSize(filename)
                        .catch(err => log.warn(`syncSize ${filename}: ${err}`));
                    await this.#deleteFiles(filename);
                    log.warn(`${model.name}: slot save ${filename} failed: ${err} | ${this.#sizeLabel()}`);
                }
            }
        });
    }

    // Restore slots from the cache that best match the given prompts
    //
    // NOTE: Method assumes it is being called inside planner's write lock
    async restore(model: ModelEntry, prompts: PromptMetadata[], signal: AbortSignal): Promise<void> {
        if (this.stopped || this.budget_bytes === 0) return;

        await this.#queue(async () => {
            const slots = await model.getSlots(signal);
            const state = model.curState()!;

            const seen: PromptMetadata[] = [];
            const assigned: Assignment[] = [];

            const filter_slots = (entry: CacheEntry) => {
                return slots.filter(slot => 
                    !slot.is_processing
                        && entry.tokens.length <= slot.n_ctx
                        && !assigned.some(item => item.slot.id === slot.id));
            }

            for (const prompt of prompts) {
                if (!isValid(prompt) || seen.some(prev => isIdentical(prev, prompt))) continue;
                seen.push(prompt);

                // find valid cache entries we can restore from, and sort in order of best match
                const candidates = this.entries
                    .filter(entry => entry.model === model.name)
                    .filter(entry => !assigned.some(item => item.entry === entry))
                    .map(entry => ({ entry, matched: commonPrefix(entry, prompt) }))
                    .filter(({ entry, matched }) => {
                        if (matched <= 0) return false;

                        if (entry.media.length && !hasMmproj(state)) {
                            log.info(`${model.name}: skipped ${entry.filename} | entry has media and model does not have mmproj`);
                            return false;
                        }

                        // filter for available slots with sufficient capacity to hold the cache entry
                        return filter_slots(entry).length !== 0;
                    })
                    .sort((a, b) => {
                        // sort candidates first by # of tokens matched
                        // next, favor the least-degraded entry
                        // finally, favor the most recently used entry
                        return b.matched - a.matched
                            || a.entry.rung - b.entry.rung
                            || b.entry.last_used - a.entry.last_used;
                    });

                const best = candidates[0];
                if (!best) {
                    log.info(`${model.name}: no cache match for ${prompt.tokens.length} input tokens | ${this.#sizeLabel()}`);
                    continue;
                }

                // try to place the slot where it was originally, else fall back to first valid slot
                const available = filter_slots(best.entry);
                const slot = available.find(slot => slot.id === best.entry.slot_id) ?? available[0];

                assigned.push({ ...best, slot });
            }

            // restore slots
            for(const { entry, slot, matched } of assigned) {
                try {
                    const restored = await model.restoreSlot(slot.id, entry.filename, signal);
                    entry.last_used = Date.now();
                    const pct = restored.n_restored > 0
                        ? Math.min(100, (matched / restored.n_restored) * 100).toFixed(2)
                        : '0.00';
                    log.info(`${model.name}: restored ${restored.n_restored} tokens from ${entry.filename} | ${pct}% matched (size: ${fmtBytes(this.#getFileBytes(entry.filename))})`);
                    log.info(this.#sizeLabel());
                } catch (err) {
                    await this.#remove(entry);
                    log.warn(`${model.name}: restore ${entry.filename} failed: ${err} | ${this.#sizeLabel()}`);
                }
            }
        });
    }

    async enforce(): Promise<void> {
        await this.#queue(() => this.#enforceBudget());
    }

    bytesUsed(): number {
        return [...this.files.values()]
            .reduce((total, bytes) => total + bytes, 0);
    }

    async #queue(cb: () => Promise<void>): Promise<void> {
        const pending = this.pending.then(cb)
            .catch(err => log.warn(`cache operation failed: ${err} | ${this.#sizeLabel()}`));

        this.pending = pending;
        await pending;
    }

    async #remove(entry: CacheEntry): Promise<void> {
        this.entries = this.entries.filter(candidate => candidate !== entry);
        await this.#deleteFiles(entry.filename);
    }

    async #deleteFiles(filename: string): Promise<void> {
        for (const name of [filename, `${filename}.ckpt`]) {
            try {
                await unlink(join(this.directory, name));
                this.files.delete(name);
            } catch (err: any) {
                if (err.code === 'ENOENT') this.files.delete(name);
                else log.warn(`delete ${name} failed: ${err} | ${this.#sizeLabel()}`);
            }
        }
    }

    #getFileBytes(filename: string): number {
        return (this.files.get(filename) ?? 0) + (this.files.get(`${filename}.ckpt`) ?? 0);
    }

    // get the size of a cache file (incl checkpoint)
    async #syncSize(filename: string): Promise<void> {
        for (const name of [filename, `${filename}.ckpt`]) {
            try {
                this.files.set(name, (await stat(join(this.directory, name))).size);
            } catch (err: any) {
                if (err.code !== 'ENOENT') throw err;
                this.files.delete(name);
            }
        }
    }

    // ensure we are not exceeding our size budget
    // evict entries until we are under-budget, starting with the LRU
    async #enforceBudget(): Promise<void> {
        const index = new Set(this.entries.flatMap(entry => [entry.filename, `${entry.filename}.ckpt`]));

        // TODO: ?
        for (const filename of [...this.files.keys()]) {
            if (!index.has(filename)) {
                await this.#deleteFiles(filename.replace(/\.ckpt$/, ''));
            }
        }

        // get eligible entries for deletion, sorted by timestamp
        // TODO: better eviction criteria
        const candidates = this.entries.sort((a, b) => a.last_used - b.last_used || a.seq - b.seq);

        let evicted = false;
        for (const entry of candidates) {
            if (this.bytesUsed() <= this.budget_bytes) break;

            evicted = true;
            await this.#remove(entry);
            log.info(`${entry.model}: evicted ${entry.filename} | ${entry.bytes} bytes | ${this.#sizeLabel()}`);
        }

        if (evicted) {
            log.info(`enforceBudget: after eviction | ${this.#sizeLabel()}`);
        } else {
            log.info(`enforceBudget: no eviction needed | ${this.#sizeLabel()}`);
        }
    }

    // delete cache files
    async #cleanup(): Promise<void> {
        this.entries = [];
        let names: string[];
        try {
            names = await readdir(this.directory);
        } catch (err: any) {
            if (err.code !== 'ENOENT') throw err;
            return;
        }

        for (const name of names.filter(name => CACHE_FILENAME.test(name))) {
            await this.#deleteFiles(name.replace(/\.ckpt$/, ''));
        }
    }

    #sizeLabel(): string {
        const str_cur = fmtBytes(this.bytesUsed());
        const str_cap = fmtBytes(this.budget_bytes);
        return `cache ${str_cur} / ${str_cap} | ${this.entries.length} entries`;
    }
}

// validate the object we grabbed from llama.cpp to prevent malformed metadata from entering the cache
export function isValid(prompt: PromptMetadata): boolean {
    if (!prompt || !Array.isArray(prompt.tokens) || !Array.isArray(prompt.media)) {
        return false;
    }

    let media_i = 0;
    for (let i = 0; i < prompt.tokens.length; i++) {
        const token = prompt.tokens[i];
        if (!Number.isInteger(token) || token < -1 || token > 2147483647) return false;

        const chunk = prompt.media[media_i];
        if (token !== -1) {
            if (chunk && (!Number.isSafeInteger(chunk.start) || chunk.start <= i)) return false;
            continue;
        }

        if (!chunk || chunk.start !== i || typeof chunk.id !== 'string'
            || !Number.isSafeInteger(chunk.n_tokens) || chunk.n_tokens <= 0
            || i + chunk.n_tokens > prompt.tokens.length) {
            return false;
        }

        for (let j = i; j < i + chunk.n_tokens; j++) {
            if (prompt.tokens[j] !== -1) return false;
        }

        i += chunk.n_tokens - 1;
        media_i++;
    }

    return media_i === prompt.media.length;
}

export function commonPrefix(a: PromptMetadata, b: PromptMetadata): number {
    const limit = Math.min(a.tokens.length, b.tokens.length);

    let media_a = 0;
    let media_b = 0;
    let i = 0;
    while (i < limit) {
        if (a.tokens[i] !== b.tokens[i]) break;
        if (a.tokens[i] !== MEDIA_MARKER) {
            i++;
            continue;
        }

        const chunk_a = a.media[media_a++];
        const chunk_b = b.media[media_b++];

        if (
            !chunk_a || !chunk_b
            || chunk_a.start !== i || chunk_b.start !== i
            || !chunk_a.id || chunk_a.id !== chunk_b.id
            || chunk_a.n_tokens !== chunk_b.n_tokens
            || i + chunk_a.n_tokens > limit
        ) break;

        i += chunk_a.n_tokens;
    }

    return i;
}

export function isIdentical(a: PromptMetadata, b: PromptMetadata): boolean {
    return a.tokens.length === b.tokens.length
        && a.tokens.every((token, i) => token === b.tokens[i])
        && a.media.length === b.media.length
        && a.media.every((chunk, i) => chunk.start === b.media[i].start
            && chunk.id === b.media[i].id && chunk.n_tokens === b.media[i].n_tokens);
}

// format bytes as a string
function fmtBytes(bytes: number): string {
    return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}