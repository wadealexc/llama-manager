import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { LoadStatus, type ModelState } from '../config/types.js';
import { MIN_ALLOWED_CTX } from '../llama-cpp-constants.js';
import { ModelEntry } from '../planner/model-entry.js';
import { PromptCache } from '../planner/prompt-cache.js';
import { LlamaAPIMock } from './llama-api-mock.js';

async function withFixture(cb: (entry: ModelEntry, client: LlamaAPIMock, cache: PromptCache) => Promise<void>): Promise<void> {
    const directory = await mkdtemp(join(tmpdir(), 'llama-cache-entry-'));
    const client = new LlamaAPIMock();
    client.slot_directory = directory;
    const state: ModelState = { model_variant: 'model', kv_unified: true, cache_type_k: 'f16', cache_type_v: 'f16' };
    const entry = new ModelEntry(client, 'model', [], state, [], new Map());
    entry.ladder[0].n_ctx_cap = 4096;
    entry.ladder.push({
        strategy: 'quantize-kv-q8',
        variant_name: 'baseline',
        state: { ...state, cache_type_k: 'q8_0', cache_type_v: 'q8_0' },
        n_ctx_cap: 8192,
        bytes_needed: 12288,
    });
    const cache = new PromptCache(directory, 1);
    entry.cache = cache;
    try {
        await cb(entry, client, cache);
    } finally {
        await cache.shutdown();
        await rm(directory, { recursive: true, force: true });
    }
}

const signal = new AbortController().signal;
const prompt = { tokens: [1, 1, 1], media: [] };

describe('ModelEntry cache lifecycle', () => {
    test('saves before growth and restores only when given an incoming prompt', async () => {
        await withFixture(async (entry, client, cache) => {
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            client.setLiveSlots(['live', 'generated']);
            client.clearOperations();
            await entry.saveSlots(signal);
            await entry.applyRung(1, signal);
            assert.equal(entry.needs_restore, true);
            assert.deepEqual(client.operations.map(operation => operation.kind), ['save', 'reload']);
            assert.deepEqual(client.getLiveSlots(), []);
            await entry.restorePrompts([prompt], signal);
            await cache.enforce();
            assert.deepEqual(client.getLiveSlots(), ['live', 'generated']);
            assert.equal(cache.entries.length, 1);
            assert.equal(entry.needs_restore, false);
        });
    });

    test('retains independent snapshots and selects by prefix rather than direction', async () => {
        await withFixture(async (entry, client, cache) => {
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            client.setLiveSlots(['baseline']);
            client.live_metadata.set('model', { tokens: [1, 2], media: [] });
            await entry.saveSlots(signal);
            await entry.applyRung(1, signal);
            await entry.restorePrompts([{ tokens: [1, 2, 3], media: [] }], signal);
            await cache.enforce();
            client.setLiveSlots(['long']);
            client.live_metadata.set('model', { tokens: [1, 2, 3, 4], media: [] });
            await entry.saveSlots(signal);
            await entry.applyRung(0, signal);
            await entry.restorePrompts([{ tokens: [1, 2, 3, 4, 5], media: [] }], signal);
            await cache.enforce();
            assert.deepEqual(client.getLiveSlots(), ['long']);
            assert.equal(cache.entries.length, 2);
        });
    });

    test('continues when saving or restoring fails', async () => {
        await withFixture(async (entry, client, cache) => {
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            client.setLiveSlots(['live']);
            client.failNext('save');
            await entry.saveSlots(signal);
            await entry.applyRung(1, signal);
            await cache.enforce();
            assert.equal(entry.status, LoadStatus.LOADED);
            assert.equal(cache.entries.length, 0);
            client.setLiveSlots(['new']);
            await entry.saveSlots(signal);
            await entry.applyRung(0, signal);
            client.failNext('restore');
            await entry.restorePrompts([prompt], signal);
            await cache.enforce();
            assert.equal(entry.status, LoadStatus.LOADED);
            assert.equal(cache.entries.length, 0);
        });
    });

    test('propagates reload failure without committing a new rung', async () => {
        await withFixture(async (entry, client, cache) => {
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            client.setLiveSlots(['live']);
            await entry.saveSlots(signal);
            client.failNext('reload');
            await assert.rejects(entry.applyRung(1, signal));
            await cache.enforce();
            assert.equal(entry.ladder_i, 0);
            assert.equal(cache.entries.length, 1);
        });
    });

    test('stashes before KV or weight unload and restores on reactivation', async () => {
        await withFixture(async (entry, client, cache) => {
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            client.setLiveSlots(['conversation']);
            await entry.unloadKV(signal);
            assert.equal(entry.status, LoadStatus.WEIGHTS_ONLY);
            assert.equal(entry.getCurCtx(), MIN_ALLOWED_CTX);
            await entry.applyRung(0, signal);
            await entry.restorePrompts([prompt], signal);
            await cache.enforce();
            assert.deepEqual(client.getLiveSlots(), ['conversation']);
            await entry.unloadWeights(signal);
            await cache.enforce();
            assert.equal(entry.status, LoadStatus.UNLOADED);
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            await entry.restorePrompts([prompt], signal);
            await cache.enforce();
            assert.deepEqual(client.getLiveSlots(), ['conversation']);
            assert.equal(cache.entries.length, 1);
        });
    });

    test('hard unload and measurement do not create snapshots', async () => {
        await withFixture(async (entry, client, cache) => {
            await entry.loadWeights(signal);
            await entry.applyRung(0, signal);
            client.setLiveSlots(['conversation']);
            await entry.applyRung(1, signal);
            await entry.unloadHard();
            assert.equal(entry.status, LoadStatus.UNLOADED);
            assert.equal(cache.entries.length, 0);
            assert.equal(client.operations.some(operation => operation.kind === 'save'), false);
        });
    });
});
