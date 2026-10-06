import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { PromptMetadata } from '../client/types.js';
import type { ModelState } from '../config/types.js';
import { ModelEntry } from '../planner/model-entry.js';
import { commonPrefix, isValid, PromptCache } from '../planner/prompt-cache.js';
import { LlamaAPIMock } from './llama-api-mock.js';

const signal = new AbortController().signal;

async function withFixture(cb: (cache: PromptCache, client: LlamaAPIMock, model: ModelEntry, directory: string) => Promise<void>): Promise<void> {
    const directory = await mkdtemp(join(tmpdir(), 'llama-prompt-cache-'));
    const client = new LlamaAPIMock();
    client.slot_directory = directory;
    const cache = new PromptCache(directory, 1);
    const state: ModelState = { model_variant: 'a', kv_unified: true, cache_type_k: 'f16', cache_type_v: 'f16' };
    const model = new ModelEntry(client, 'a', [], state, [], new Map());
    model.cache = cache;
    try {
        await model.loadWeights(signal);
        await model.applyRung(0, signal);
        await cb(cache, client, model, directory);
    } finally {
        await cache.shutdown();
        await rm(directory, { recursive: true, force: true });
    }
}

async function save(cache: PromptCache, client: LlamaAPIMock, model: ModelEntry, tokens: number[], media: PromptMetadata['media'] = []): Promise<void> {
    client.setLiveSlots(['state'], model.curVariant());
    client.live_metadata.set(model.curVariant(), { tokens, media });
    await cache.save(model, signal);
}

test('multimodal prefixes match entire identified chunks and stop at changed or unknown media', () => {
    const a: PromptMetadata = {
        tokens: [10, -1, -1, 20, -1, 30],
        media: [{ start: 1, id: 'image', n_tokens: 2 }, { start: 4, id: 'audio', n_tokens: 1 }],
    };
    assert.equal(isValid(a), true);
    assert.equal(commonPrefix(a, a), 6);
    const changed = structuredClone(a);
    changed.media[0].id = 'other';
    assert.equal(commonPrefix(a, changed), 1);
    changed.media[0].id = 'image';
    changed.media[1].id = '';
    assert.equal(commonPrefix(a, changed), 4);
    assert.equal(commonPrefix(changed, changed), 4);
    const malformed = structuredClone(a);
    malformed.media[0].n_tokens = 3;
    assert.equal(isValid(malformed), false);
    assert.equal(isValid({ tokens: [-1], media: [] }), false);
});

test('selects longest prefix first and source quality on ties', async () => {
    await withFixture(async (cache, client, model) => {
        await save(cache, client, model, [1, 2]);
        model.ladder.push({
            strategy: 'quantize-kv-q8',
            variant_name: 'baseline',
            state: { ...model.ladder[0].state, cache_type_k: 'q8_0', cache_type_v: 'q8_0' },
            n_ctx_cap: 8192,
            bytes_needed: 0,
        });
        await model.applyRung(1, signal);
        await save(cache, client, model, [1, 2]);
        await save(cache, client, model, [1, 2, 3]);
        assert.equal(cache.entries.length, 3);
        const longest = cache.entries.find(entry => entry.tokens.length === 3)!.filename;
        const baseline = cache.entries.find(entry => entry.rung === 0)!.filename;
        await cache.restore(model, [{ tokens: [1, 2, 3, 4], media: [] }], signal);
        await cache.enforce();
        const restored = client.operations.at(-1);
        assert.ok(restored?.kind === 'restore');
        assert.equal(restored.filenames[0], longest);
        await cache.restore(model, [{ tokens: [1, 2, 8], media: [] }], signal);
        await cache.enforce();
        const quality_restore = client.operations.at(-1);
        assert.ok(quality_restore?.kind === 'restore');
        assert.equal(quality_restore.filenames[0], baseline);
    });
});

test('filters full saved state by actual slot capacity and remaps destination slots', async () => {
    await withFixture(async (cache, client, model) => {
        await save(cache, client, model, [1, 2, 3, 4]);
        const filename = cache.entries[0].filename;
        client.slot_ids = [7];
        client.capacities.set('a', 3);
        await model.applyRung(0, signal);
        await cache.restore(model, [{ tokens: [1, 2], media: [] }], signal);
        await cache.enforce();
        assert.equal(client.operations.some(operation => operation.kind === 'restore'), false);
        assert.equal(cache.entries.length, 1);
        client.capacities.set('a', 8);
        await model.applyRung(0, signal);
        await cache.restore(model, [{ tokens: [1, 2, 3], media: [] }], signal);
        await cache.enforce();
        assert.deepEqual(client.operations.at(-1), { kind: 'restore', model: 'a', filenames: [filename], slot_id: 7 });
    });
});

test('enforces one LRU budget across models and accounts for checkpoint sidecars', async () => {
    await withFixture(async (cache, client, model) => {
        client.save_bytes = 40;
        client.sidecar_bytes = 20;
        cache.budget_bytes = 100;
        await save(cache, client, model, [1, 2]);
        await cache.enforce();
        const first = cache.entries[0].filename;
        cache.entries[0].last_used = 0;
        assert.equal(cache.bytesUsed(), 60);
        await model.unloadHard();
        const state = { ...model.ladder[0].state, model_variant: 'b' };
        const other = new ModelEntry(client, 'b', [], state, [], new Map());
        other.cache = cache;
        await other.loadWeights(signal);
        await other.applyRung(0, signal);
        await save(cache, client, other, [9, 8]);
        await cache.enforce();
        assert.equal(cache.bytesUsed(), 60);
        assert.equal(cache.entries[0].model, 'b');
        assert.equal((await readdir(client.slot_directory!)).includes(first), false);
        assert.equal((await readdir(client.slot_directory!)).includes(`${first}.ckpt`), false);
    });
});

test('restores a provisional entry before explicit LRU enforcement', async () => {
    await withFixture(async (cache, client, model) => {
        cache.budget_bytes = 32;
        await save(cache, client, model, [1, 2, 3]);
        await cache.enforce();
        const returning = cache.entries[0].filename;
        await save(cache, client, model, [9]);
        cache.entries[0].last_used = 0;
        cache.entries[1].last_used = 0;
        assert.equal(cache.entries.length, 2);
        assert.equal(cache.bytesUsed(), 64);
        await model.applyRung(0, signal);
        await cache.restore(model, [{ tokens: [1, 2, 3, 4], media: [] }], signal);
        assert.equal(cache.entries.length, 2);
        assert.equal(cache.bytesUsed(), 64);
        const restored = client.operations.at(-1);
        assert.ok(restored?.kind === 'restore');
        assert.equal(restored.filenames[0], returning);
        await cache.enforce();
        assert.equal(cache.entries.length, 1);
        assert.equal(cache.entries[0].filename, returning);
        assert.equal(cache.bytesUsed(), 32);
        assert.deepEqual(client.getLiveSlots(model.name), ['state']);
    });
});

test('restores multiple entries exceeding the disk budget before eviction', async () => {
    await withFixture(async (cache, client, model) => {
        cache.budget_bytes = 32;
        await save(cache, client, model, [1, 2]);
        await save(cache, client, model, [3, 4]);
        const filenames = cache.entries.map(entry => entry.filename);
        client.slot_ids = [0, 1];
        await model.applyRung(0, signal);
        await cache.restore(model, [
            { tokens: [1, 2, 5], media: [] },
            { tokens: [3, 4, 6], media: [] },
        ], signal);
        const restores = client.operations.filter(operation => operation.kind === 'restore');
        assert.deepEqual(restores.flatMap(operation => operation.kind === 'restore' ? operation.filenames : []), filenames);
        assert.equal(cache.bytesUsed(), 64);
        assert.equal(cache.entries.length, 2);
        await cache.enforce();
        assert.equal(cache.bytesUsed(), 32);
        assert.equal(cache.entries.length, 1);
        assert.deepEqual(client.getLiveSlots(model.name), ['state']);
    });
});

test('eviction after restore does not discard the installed live state', async () => {
    await withFixture(async (cache, client, model) => {
        await save(cache, client, model, [1, 2, 3]);
        const filename = cache.entries[0].filename;
        await model.applyRung(0, signal);
        await cache.restore(model, [{ tokens: [1, 2, 3, 4], media: [] }], signal);
        cache.budget_bytes = 0;
        await cache.enforce();
        assert.equal(cache.entries.length, 0);
        assert.equal(cache.bytesUsed(), 0);
        assert.equal((await readdir(client.slot_directory!)).includes(filename), false);
        assert.deepEqual(client.getLiveSlots(model.name), ['state']);
    });
});

test('skips oversized or invalid saves and falls back after restore failure', async () => {
    await withFixture(async (cache, client, model) => {
        cache.budget_bytes = 16;
        await save(cache, client, model, [1]);
        assert.equal(cache.entries.length, 0);
        assert.equal(cache.bytesUsed(), 0);
        cache.budget_bytes = 64;
        await save(cache, client, model, [-1]);
        assert.equal(cache.entries.length, 0);
        await save(cache, client, model, [1]);
        client.failNext('restore');
        await cache.restore(model, [{ tokens: [1, 2], media: [] }], signal);
        await cache.enforce();
        assert.equal(cache.entries.length, 0);
        assert.equal(cache.bytesUsed(), 0);
    });
});

test('deduplicates exact-config snapshots but retains shorter prefixes', async () => {
    await withFixture(async (cache, client, model) => {
        await save(cache, client, model, [1, 2]);
        const old = cache.entries[0].filename;
        await save(cache, client, model, [1, 2]);
        assert.equal(cache.entries.length, 1);
        assert.notEqual(cache.entries[0].filename, old);
        await cache.enforce();
        await save(cache, client, model, [1, 2, 3]);
        await cache.enforce();
        assert.equal(cache.entries.length, 2);
    });
});

test('restores independent media prompts into available slots', async () => {
    await withFixture(async (cache, client, model) => {
        model.ladder[0].state.mmproj = { path: 'projector.gguf', mmproj_offload: true };
        const image: PromptMetadata = { tokens: [1, -1, -1, 2], media: [{ start: 1, id: 'image', n_tokens: 2 }] };
        const audio: PromptMetadata = { tokens: [3, -1, 4], media: [{ start: 1, id: 'audio', n_tokens: 1 }] };
        await save(cache, client, model, image.tokens, image.media);
        await save(cache, client, model, audio.tokens, audio.media);
        client.slot_ids = [0, 1];
        await model.applyRung(0, signal);
        await cache.restore(model, [image, audio, image], signal);
        await cache.enforce();
        const restores = client.operations.filter(operation => operation.kind === 'restore');
        assert.equal(restores.length, 2);
        assert.equal(new Set(restores.map(operation => operation.kind === 'restore' ? operation.slot_id : undefined)).size, 2);
    });
});

test('restores through a compatible weight-variant switch', async () => {
    await withFixture(async (cache, client, model) => {
        await save(cache, client, model, [1, 2, 3]);
        model.ladder.push({
            strategy: 'swap-model',
            variant_name: 'q4',
            state: { ...model.ladder[0].state, model_variant: 'a__q4' },
            n_ctx_cap: 8192,
            bytes_needed: 0,
        });
        await model.applyRung(1, signal);
        await cache.restore(model, [{ tokens: [1, 2, 3, 4], media: [] }], signal);
        await cache.enforce();
        assert.deepEqual(client.getLiveSlots('a__q4'), ['state']);
        const operation = client.operations.at(-1);
        assert.ok(operation?.kind === 'restore');
        assert.equal(operation.model, 'a__q4');
        assert.equal(model.name, 'a');
    });
});

test('zero budget disables saves and restores', async () => {
    await withFixture(async (cache, client, model) => {
        cache.budget_bytes = 0;
        await save(cache, client, model, [1]);
        await cache.restore(model, [{ tokens: [1], media: [] }], signal);
        await cache.enforce();
        assert.equal(client.operations.some(operation => operation.kind === 'save' || operation.kind === 'restore'), false);
    });
});

test('startup and shutdown cleanup leave unrelated slot files untouched', async () => {
    await withFixture(async (cache, client, model, directory) => {
        await writeFile(join(directory, 'user.bin'), 'user');
        await writeFile(join(directory, 'user.bin.ckpt'), 'checkpoint');
        await writeFile(join(directory, 'lm-cache-55-0.bin'), 'stale');
        await cache.start();
        assert.deepEqual((await readdir(directory)).sort(), ['user.bin', 'user.bin.ckpt']);
        await save(cache, client, model, [1]);
        await cache.enforce();
        await cache.shutdown();
        assert.deepEqual((await readdir(directory)).sort(), ['user.bin', 'user.bin.ckpt']);
    });
});
