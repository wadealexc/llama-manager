import assert from 'node:assert/strict';
import { after, describe, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ManagerConfig, ModelState } from '../config/types.js';
import { ModelEntry } from '../planner/model-entry.js';
import { ModelTask, Planner } from '../planner/planner.js';
import { LlamaAPIMock } from './llama-api-mock.js';

const fixtures: { directory: string; planner: Planner }[] = [];
after(async () => {
    await Promise.all(fixtures.map(async ({ directory, planner }) => {
        await planner.shutdown();
        await rm(directory, { recursive: true, force: true });
    }));
});

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>(done => {
        resolve = done;
    });
    return { promise, resolve };
}

async function waitFor(condition: () => boolean): Promise<void> {
    for (let i = 0; i < 100; i++) {
        if (condition()) return;
        await new Promise<void>(resolve => setImmediate(resolve));
    }
    assert.fail('expected planner state was not reached');
}

function createModel(client: LlamaAPIMock, name: string): ModelEntry {
    const initial_state: ModelState = {
        kv_unified: true,
        cache_type_k: 'f16',
        cache_type_v: 'f16',
        model_variant: name,
    };

    const entry = new ModelEntry(client, name, [], initial_state, [], new Map());
    entry.ladder[0].n_ctx_cap = 4096;
    entry.ladder[0].bytes_needed = 8192;

    entry.ladder.push({
        strategy: 'quantize-kv-q8',
        variant_name: 'baseline',
        state: { model_variant: name, kv_unified: true, cache_type_k: 'q8_0', cache_type_v: 'q8_0' },
        n_ctx_cap: 8192,
        bytes_needed: 12288,
    });
    
    return entry;
}

async function createFixture(rung_i: number = 0) {
    const client = new LlamaAPIMock();
    const model = createModel(client, 'model');
    const directory = await mkdtemp(join(tmpdir(), 'llama-cache-planner-'));
    client.slot_directory = directory;
    const config: ManagerConfig = {
        mode: 'router',
        router: {
            bin: '',
            llama_log_dir: '',
            slot_save_path: directory,
            listen: '',
            poll_interval_ms: 0,
            poll_timeout_ms: 0,
            shutdown_grace_period_ms: 0,
        },
        host: '127.0.0.1',
        port: 0,
        sleep_idle_seconds: 0,
        cache_disk_mib: 1,
        model_load: { poll_interval_ms: 0, poll_timeout_ms: 0 },
        models: {},
        default_model: model.name,
    };
    const planner = new Planner(client, config, new Map([[model.name, model]]));
    fixtures.push({ directory, planner });
    const signal = new AbortController().signal;

    await model.loadWeights(signal);
    await model.applyRung(rung_i, signal);
    model.needs_restore = false;
    const memory = await model.getMemory(signal);
    planner.active.model = model.name;
    planner.dev_info.bytes_total = memory.devices[0].total;
    planner.dev_info.bytes_avail = memory.devices[0].free;
    client.clearOperations();

    return { client, model, planner };
}

describe('Planner non-generation tasks', () => {
    test('tokenizes a short input without downshifting or disturbing the live conversation', async () => {
        const { client, model, planner } = await createFixture(1);
        client.setLiveSlots(['long conversation', 'latest generated tokens'], model.name);
        const snapshots_before = structuredClone(client.snapshots);
        const saved_slots_before = structuredClone(planner.cache.entries);
        let callback_count = 0;
        let counted_tokens: number | null = null;

        await planner.serveModel({ tokens: 1000 }, model.name, ModelTask.TOKENIZE, new AbortController().signal, async (body, active_model, signal, is_final) => {
            callback_count++;
            assert.equal(is_final, false);
            assert.equal(active_model.ladder_i, 1);
            counted_tokens = await active_model.countTokens(body, signal);
            return { kind: 'done' };
        });

        assert.equal(callback_count, 1);
        assert.equal(counted_tokens, 1000);
        assert.equal(model.ladder_i, 1);
        assert.equal(model.getCurCtx(), 8192);
        assert.deepEqual(client.getLiveSlots(model.name), ['long conversation', 'latest generated tokens']);
        assert.deepEqual(client.snapshots, snapshots_before);
        assert.deepEqual(planner.cache.entries, saved_slots_before);
        assert.deepEqual(client.operations, [{ kind: 'count', model: model.name, tokens: 1000 }]);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.waiting_reload.length, 0);

        let completion_count = 0;
        await planner.serveModel({ tokens: 6000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (_body, active_model, _signal, is_final) => {
            completion_count++;
            assert.equal(is_final, false);
            assert.equal(active_model.ladder_i, 1);
            assert.deepEqual(client.getLiveSlots(active_model.name), ['long conversation', 'latest generated tokens']);
            return { kind: 'done' };
        });

        assert.equal(completion_count, 1);
        assert.equal(model.ladder_i, 1);
        assert.equal(model.getCurCtx(), 8192);
        assert.deepEqual(client.getLiveSlots(model.name), ['long conversation', 'latest generated tokens']);
        assert.deepEqual(client.snapshots, snapshots_before);
        assert.deepEqual(planner.cache.entries, saved_slots_before);
        assert.deepEqual(client.operations, [
            { kind: 'count', model: model.name, tokens: 1000 },
            { kind: 'count', model: model.name, tokens: 6000 },
        ]);
        assert.equal(planner.active.pending, false);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('tokenizes an input above the current context without upshifting', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['baseline conversation'], model.name);
        let callback_count = 0;
        let counted_tokens: number | null = null;

        await planner.serveModel({ tokens: 5000 }, model.name, ModelTask.TOKENIZE, new AbortController().signal, async (body, active_model, signal, is_final) => {
            callback_count++;
            assert.equal(is_final, false);
            assert.equal(active_model.ladder_i, 0);
            counted_tokens = await active_model.countTokens(body, signal);
            return { kind: 'done' };
        });

        assert.equal(callback_count, 1);
        assert.equal(counted_tokens, 5000);
        assert.equal(model.ladder_i, 0);
        assert.equal(model.getCurCtx(), 4096);
        assert.deepEqual(client.getLiveSlots(model.name), ['baseline conversation']);
        assert.equal(client.snapshots.size, 0);
        assert.deepEqual(client.operations, [{ kind: 'count', model: model.name, tokens: 5000 }]);
        assert.equal(planner.active.pending, false);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('tokenizes an input above maximum measured capacity without rejecting it', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['baseline conversation'], model.name);
        assert.equal(model.getMaxCtx(), 8192);
        let callback_count = 0;
        let counted_tokens: number | null = null;

        await planner.serveModel({ tokens: 9000 }, model.name, ModelTask.TOKENIZE, new AbortController().signal, async (body, active_model, signal, is_final) => {
            callback_count++;
            assert.equal(is_final, false);
            assert.equal(active_model.ladder_i, 0);
            counted_tokens = await active_model.countTokens(body, signal);
            return { kind: 'done' };
        });

        assert.equal(callback_count, 1);
        assert.equal(counted_tokens, 9000);
        assert.equal(model.ladder_i, 0);
        assert.equal(model.getCurCtx(), 4096);
        assert.deepEqual(client.getLiveSlots(model.name), ['baseline conversation']);
        assert.equal(client.snapshots.size, 0);
        assert.deepEqual(client.operations, [{ kind: 'count', model: model.name, tokens: 9000 }]);
        assert.equal(planner.active.pending, false);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('wakes an active model with empty messages without counting tokens or reloading', async () => {
        const { client, model, planner } = await createFixture(1);
        client.setLiveSlots(['long conversation'], model.name);
        const snapshots_before = structuredClone(client.snapshots);
        const saved_slots_before = structuredClone(planner.cache.entries);
        let callback_count = 0;

        await planner.serveModel({ messages: [] }, model.name, ModelTask.WAKE, new AbortController().signal, async (_body, active_model, _signal, is_final) => {
            callback_count++;
            assert.equal(is_final, false);
            assert.equal(active_model.ladder_i, 1);
            return { kind: 'done' };
        });

        assert.equal(callback_count, 1);
        assert.equal(model.ladder_i, 1);
        assert.equal(model.getCurCtx(), 8192);
        assert.deepEqual(client.getLiveSlots(model.name), ['long conversation']);
        assert.deepEqual(client.snapshots, snapshots_before);
        assert.deepEqual(planner.cache.entries, saved_slots_before);
        assert.deepEqual(client.operations, []);
        assert.equal(planner.active.pending, false);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('serves tokenization queued during a reload without reinitializing the active model', { timeout: 5000 }, async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['live conversation', 'generated tokens'], model.name);
        const release_reload = deferred<void>();
        const reload_model = client.reloadModel.bind(client);
        let reload_entered = false;
        let completion_count = 0;
        let tokenization_count = 0;
        let completion_rung = -1;
        let tokenization_rung = -1;
        let counted_tokens: number | null = null;
        let tokenizing: Promise<void> | undefined;

        client.reloadModel = async (params, model_name, signal) => {
            reload_entered = true;
            await release_reload.promise;
            return await reload_model(params, model_name, signal);
        };

        const generating = planner.serveModel({ tokens: 5000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (_body, active_model, _signal, is_final) => {
            completion_count++;
            completion_rung = active_model.ladder_i;
            assert.equal(is_final, false);
            assert.deepEqual(client.getLiveSlots(active_model.name), ['live conversation', 'generated tokens']);
            return { kind: 'done' };
        });

        try {
            await waitFor(() => reload_entered);
            assert.equal(planner.active.pending, true);
            assert.equal(completion_count, 0);

            tokenizing = planner.serveModel({ tokens: 1000 }, model.name, ModelTask.TOKENIZE, new AbortController().signal, async (body, active_model, signal, is_final) => {
                tokenization_count++;
                tokenization_rung = active_model.ladder_i;
                assert.equal(is_final, false);
                assert.deepEqual(client.getLiveSlots(active_model.name), ['live conversation', 'generated tokens']);
                counted_tokens = await active_model.countTokens(body, signal);
                return { kind: 'done' };
            });

            await waitFor(() => planner.waiting_load.length === 1);
            assert.equal(planner.active.pending, true);
            assert.equal(completion_count, 0);
            assert.equal(tokenization_count, 0);

            release_reload.resolve();
            await Promise.all([generating, tokenizing]);
        } finally {
            release_reload.resolve();
            await Promise.allSettled(tokenizing ? [generating, tokenizing] : [generating]);
            client.reloadModel = reload_model;
        }

        assert.equal(completion_count, 1);
        assert.equal(tokenization_count, 1);
        assert.equal(completion_rung, 1);
        assert.equal(tokenization_rung, 1);
        assert.equal(counted_tokens, 1000);
        assert.equal(model.ladder_i, 1);
        assert.equal(model.getCurCtx(), 8192);
        assert.deepEqual(client.getLiveSlots(model.name), ['live conversation', 'generated tokens']);
        assert.deepEqual(client.getSnapshot(model.name, 0), ['live conversation', 'generated tokens']);
        assert.equal(client.getSnapshot(model.name, 1), undefined);
        assert.deepEqual(client.operations.map(operation => operation.kind), ['count', 'save', 'reload', 'memory', 'restore', 'count']);
        assert.deepEqual(client.operations.filter(operation => operation.kind === 'count'), [
            { kind: 'count', model: model.name, tokens: 5000 },
            { kind: 'count', model: model.name, tokens: 1000 },
        ]);
        assert.equal(planner.active.model, model.name);
        assert.equal(planner.active.pending, false);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });
});

describe('Planner warm-model scheduling', () => {
    test('restores a long conversation after an unrelated completion', async () => {
        const { client, model, planner } = await createFixture(1);
        const long_prompt = Array(5000).fill(1);
        client.setLiveSlots(['long conversation'], model.name);
        client.live_metadata.set(model.name, { tokens: long_prompt, media: [] });
        await planner.serveModel({ tokens: 3, prompt: [8, 9, 10] }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.equal(model.ladder_i, 0);
            client.setLiveSlots(['unrelated'], model.name);
            client.live_metadata.set(model.name, { tokens: [8, 9, 10, 11], media: [] });
            return { kind: 'done' };
        });
        await planner.serveModel({ tokens: 5001, prompt: [...long_prompt, 2] }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.equal(model.ladder_i, 1);
            assert.deepEqual(client.getLiveSlots(model.name), ['long conversation']);
            return { kind: 'done' };
        });
        await planner.cache.enforce();
        assert.equal(planner.cache.entries.length, 2);
    });

    test('restores after wake activation without another rung change', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['returning conversation'], model.name);
        await model.unloadWeights(new AbortController().signal);
        planner.active.model = undefined;
        await planner.serveModel({}, model.name, ModelTask.WAKE, new AbortController().signal, async () => ({ kind: 'done' }));
        assert.equal(model.needs_restore, true);
        const reloads = client.operations.filter(operation => operation.kind === 'reload').length;
        await planner.serveModel({ tokens: 100 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.deepEqual(client.getLiveSlots(model.name), ['returning conversation']);
            return { kind: 'done' };
        });
        assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, reloads);
    });

    test('restores only after the final capacity-satisfying reload', async () => {
        const { client, model, planner } = await createFixture();
        model.ladder.push({
            strategy: 'quantize-kv-q4',
            variant_name: 'baseline',
            state: { ...model.ladder[0].state, cache_type_k: 'q4_0', cache_type_v: 'q4_0' },
            n_ctx_cap: 16384,
            bytes_needed: 20480,
        });
        client.setLiveSlots(['live conversation'], model.name);
        const reload_model = client.reloadModel.bind(client);
        client.reloadModel = async (params, model_id, signal) => {
            if (params.cache_type_k === 'q8_0') client.capacities.set(model_id, 4500);
            else client.capacities.delete(model_id);
            return await reload_model(params, model_id, signal);
        };
        await planner.serveModel({ tokens: 5000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.equal(model.ladder_i, 2);
            assert.deepEqual(client.getLiveSlots(model.name), ['live conversation']);
            return { kind: 'done' };
        });
        assert.equal(client.operations.filter(operation => operation.kind === 'save').length, 1);
        assert.equal(client.operations.filter(operation => operation.kind === 'restore').length, 1);
        assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, 2);
    });

    test('finishes without rendering a retry when the callback returns done', async () => {
        const { client, model, planner } = await createFixture();
        const body = { tokens: 1000 };
        let attempts = 0;
        await planner.serveModel(body, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (request_body, _model, _signal, is_final) => {
            attempts++;
            assert.equal(request_body, body);
            assert.equal(is_final, false);
            return { kind: 'done' };
        });
        assert.equal(attempts, 1);
        assert.deepEqual(client.operations, [{ kind: 'count', model: model.name, tokens: 1000 }]);
    });

    test('returns the current request with isFinal when growth is unavailable', async () => {
        const { client, model, planner } = await createFixture(1);
        const body = { tokens: 5000 };
        const retry_body = { tokens: 9000 };
        const calls: { body: unknown; is_final: boolean }[] = [];
        await planner.serveModel(body, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (request_body, _model, _signal, is_final) => {
            calls.push({ body: request_body, is_final });
            return is_final ? { kind: 'done' } : { kind: 'grow', body: retry_body };
        });
        assert.deepEqual(calls, [{ body, is_final: false }, { body, is_final: true }]);
        assert.equal(client.operations.some(operation => operation.kind === 'reload'), false);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('keeps the last successful retry body when later growth is unavailable', async () => {
        const { client, model, planner } = await createFixture();
        const body = { tokens: 1000 };
        const first_retry = { tokens: 5000 };
        const second_retry = { tokens: 9000 };
        const calls: { body: unknown; is_final: boolean }[] = [];
        await planner.serveModel(body, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (request_body, _model, _signal, is_final) => {
            calls.push({ body: request_body, is_final });
            if (is_final) return { kind: 'done' };
            return { kind: 'grow', body: request_body === body ? first_retry : second_retry };
        });
        assert.deepEqual(calls, [
            { body, is_final: false },
            { body: first_retry, is_final: false },
            { body: first_retry, is_final: true },
        ]);
        assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, 1);
        assert.equal(planner.active.readers, 0);
    });

    test('selects continuation cache state using the returned grow body', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['older original'], model.name);
        client.live_metadata.set(model.name, { tokens: [1, 2, 3], media: [] });
        await model.saveSlots(new AbortController().signal);
        await planner.cache.enforce();
        const retry_body = { tokens: 5, prompt: [1, 2, 4, 5, 6] };
        let attempts = 0;
        await planner.serveModel({ tokens: 2, prompt: [1, 2] }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (request_body, _model, _signal, is_final) => {
            attempts++;
            assert.equal(is_final, false);
            if (attempts === 1) {
                client.setLiveSlots(['continuation'], model.name);
                client.live_metadata.set(model.name, { tokens: [1, 2, 4, 5], media: [] });
                return { kind: 'grow', body: retry_body };
            }
            assert.equal(request_body, retry_body);
            assert.deepEqual(client.getLiveSlots(model.name), ['continuation']);
            return { kind: 'done' };
        });
        assert.equal(attempts, 2);
    });

    test('downshifts an unoccupied model to the minimum sufficient rung', async () => {
        const { client, model, planner } = await createFixture(1);
        await model.applyRung(0, new AbortController().signal);
        client.setLiveSlots(['baseline conversation'], model.name);
        await model.saveSlots(new AbortController().signal);
        await planner.cache.enforce();
        await model.applyRung(1, new AbortController().signal);
        client.setLiveSlots(['long conversation'], model.name);
        client.live_metadata.set(model.name, { tokens: [2], media: [] });
        client.clearOperations();
        let seen_rung = -1;

        await planner.serveModel({ tokens: 1000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            seen_rung = model.ladder_i;
            return { kind: 'done' };
        });

        assert.equal(seen_rung, 0);
        assert.equal(model.ladder_i, 0);
        assert.deepEqual(client.getLiveSlots(model.name), ['baseline conversation']);
        assert.ok(planner.cache.entries.some(entry => entry.rung === 1 && entry.tokens[0] === 2));
        assert.deepEqual(client.operations.filter(operation => operation.kind === 'reload').map(operation => operation.model), ['model']);
        assert.deepEqual(client.operations.flatMap(operation => operation.kind === 'restore' ? [operation.filenames] : []), [[planner.cache.entries.find(entry => entry.rung === 0)!.filename]]);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('upshifts before serving an input that exceeds the current rung', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['live conversation'], model.name);
        let seen_rung = -1;

        await planner.serveModel({ tokens: 5000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            seen_rung = model.ladder_i;
            return { kind: 'done' };
        });

        assert.equal(seen_rung, 1);
        assert.equal(model.ladder_i, 1);
        assert.deepEqual(client.getSnapshot(model.name, 0), ['live conversation']);
        assert.deepEqual(client.getLiveSlots(model.name), ['live conversation']);
        assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, 1);
        assert.equal(planner.active.readers, 0);
    });

    test('serves a contending short request at the current sufficient rung', async () => {
        const { client, model, planner } = await createFixture(1);
        const started = deferred<void>();
        const finish = deferred<void>();
        const busy = planner.serveModel({ tokens: 6000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            started.resolve();
            await finish.promise;
            return { kind: 'done' };
        });

        try {
            await started.promise;
            let seen_rung = -1;
            await planner.serveModel({ tokens: 1000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
                seen_rung = model.ladder_i;
                return { kind: 'done' };
            });

            assert.equal(seen_rung, 1);
            assert.equal(model.ladder_i, 1);
            assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, 0);
            assert.equal(planner.active.readers, 1);
        } finally {
            finish.resolve();
            await busy;
        }

        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('unloads outgoing model on a swap even when both models fit in memory', async () => {
        const { client, model, planner } = await createFixture();
        const second = createModel(client, 'second');
        second.cache = planner.cache;
        planner.models.set(second.name, second);
        await second.loadWeights(new AbortController().signal);
        planner.weights_only.add(second.name);
        client.setLiveSlots(['first conversation'], model.name);
        const memory = await second.getMemory(new AbortController().signal);
        planner.dev_info.bytes_avail = memory.devices[0].free;
        client.clearOperations();

        await planner.serveModel({ tokens: 1000 }, second.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.equal(planner.active.model, second.name);
            assert.equal(client.hasKV(model.name), false);
            assert.equal(client.hasKV(second.name), true);
            return { kind: 'done' };
        });

        assert.equal(planner.weights_only.has(model.name), false);
        assert.equal(client.isLoaded(model.name), false);
        assert.ok(planner.cache.entries.some(entry => entry.model === model.name));
        assert.equal(client.operations.filter(operation => operation.kind === 'unload').length, 1);
        client.setLiveSlots(['second conversation'], second.name);
        client.clearOperations();

        await planner.serveModel({ tokens: 1000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.equal(planner.active.model, model.name);
            assert.equal(client.hasKV(second.name), false);
            assert.equal(client.hasKV(model.name), true);
            assert.deepEqual(client.getLiveSlots(model.name), ['first conversation']);
            return { kind: 'done' };
        });

        assert.equal(planner.weights_only.has(second.name), false);
        assert.ok(planner.cache.entries.some(entry => entry.model === second.name));
        assert.equal(client.operations.filter(operation => operation.kind === 'unload').length, 1);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
    });

    test('evicts idle weights when a new model needs room for its KV', async () => {
        const { client, model, planner } = await createFixture();
        const second = createModel(client, 'second');
        second.cache = planner.cache;
        second.bytes_needed_no_kv = 5120;
        planner.models.set(second.name, second);
        client.total_bytes = 12000;
        planner.dev_info.bytes_total = client.total_bytes;
        planner.dev_info.bytes_avail = (await model.getMemory(new AbortController().signal)).devices[0].free;
        client.clearOperations();

        await planner.serveModel({ tokens: 1000 }, second.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            assert.equal(planner.active.model, second.name);
            assert.equal(client.hasKV(second.name), true);
            assert.equal(client.isLoaded(model.name), false);
            return { kind: 'done' };
        });

        assert.equal(planner.weights_only.has(model.name), false);
        assert.equal(client.operations.filter(operation => operation.kind === 'unload').length, 1);
        assert.deepEqual(client.operations.flatMap(operation => operation.kind === 'unload' ? [operation.model] : []), [model.name]);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.active.readers, 0);
    });

    test('waits for generation to finish before swapping models', async () => {
        const { client, model, planner } = await createFixture();
        const second = createModel(client, 'second');
        second.cache = planner.cache;
        planner.models.set(second.name, second);
        await second.loadWeights(new AbortController().signal);
        planner.weights_only.add(second.name);
        const memory = await second.getMemory(new AbortController().signal);
        planner.dev_info.bytes_avail = memory.devices[0].free;
        client.clearOperations();

        const started = deferred<void>();
        const finish = deferred<void>();
        const generating = planner.serveModel({ tokens: 1000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => {
            started.resolve();
            await finish.promise;
            return { kind: 'done' };
        });

        try {
            await started.promise;
            const queued = planner.serveModel({ tokens: 1000 }, second.name, ModelTask.COMPLETIONS, new AbortController().signal, async () => ({ kind: 'done' }));
            await waitFor(() => planner.waiting_load.length === 1);

            assert.equal(planner.active.model, model.name);
            assert.equal(client.hasKV(model.name), true);
            assert.equal(client.hasKV(second.name), false);
            assert.equal(client.operations.filter(operation => operation.kind === 'save').length, 0);

            finish.resolve();
            await Promise.all([generating, queued]);
        } finally {
            finish.resolve();
        }

        assert.equal(planner.active.model, second.name);
        assert.equal(client.hasKV(model.name), false);
        assert.equal(client.hasKV(second.name), true);
        assert.equal(planner.waiting_load.length, 0);
        assert.equal(planner.active.readers, 0);
    });

    test('waits for all generating readers before one upshift and restores the live slots', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['live prompt', 'generated tokens'], model.name);
        const started = deferred<void>();
        const finish = deferred<void>();
        const slow_rungs: number[] = [];
        const fast_rungs: number[] = [];
        const slow = planner.serveModel({ tokens: 1000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (request_body, _client, _signal, isFinal) => {
            assert.equal(isFinal, false);
            slow_rungs.push(model.ladder_i);
            if (slow_rungs.length === 1) {
                started.resolve();
                await finish.promise;
                return { kind: 'grow', body: request_body };
            }
            return { kind: 'done' };
        });

        try {
            await started.promise;
            const fast = planner.serveModel({ tokens: 1000 }, model.name, ModelTask.COMPLETIONS, new AbortController().signal, async (request_body, _client, _signal, isFinal) => {
                assert.equal(isFinal, false);
                fast_rungs.push(model.ladder_i);
                return fast_rungs.length > 1
                    ? { kind: 'done' }
                    : { kind: 'grow', body: request_body };
            });

            await waitFor(() => planner.waiting_reload.length === 1);
            assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, 0);
            assert.equal(planner.active.readers, 2);

            finish.resolve();
            await Promise.all([slow, fast]);
        } finally {
            finish.resolve();
        }

        assert.deepEqual(slow_rungs, [0, 1]);
        assert.deepEqual(fast_rungs, [0, 1]);
        assert.equal(client.operations.filter(operation => operation.kind === 'reload').length, 1);
        assert.deepEqual(client.getSnapshot(model.name, 0), ['live prompt', 'generated tokens']);
        assert.deepEqual(client.getLiveSlots(model.name), ['live prompt', 'generated tokens']);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });
});
