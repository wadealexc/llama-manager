import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import type { ManagerConfig, ModelState } from '../config/types.js';
import { ModelEntry } from '../planner/model-entry.js';
import { Planner } from '../planner/planner.js';
import { LlamaAPIMock } from './llama-api-mock.js';

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
    const config: ManagerConfig = {
        mode: 'router',
        router: {
            bin: '',
            llama_log_dir: '',
            slot_save_path: '',
            listen: '',
            poll_interval_ms: 0,
            poll_timeout_ms: 0,
            shutdown_grace_period_ms: 0,
        },
        host: '127.0.0.1',
        port: 0,
        sleep_idle_seconds: 0,
        model_load: { poll_interval_ms: 0, poll_timeout_ms: 0 },
        models: {},
        default_model: model.name,
    };
    const planner = new Planner(client, config, new Map([[model.name, model]]));
    const signal = new AbortController().signal;

    await model.loadWeights(signal);
    await model.applyRung(rung_i, signal);
    const memory = await model.getMemory(signal);
    planner.active.model = model.name;
    planner.dev_info.bytes_total = memory.devices[0].total;
    planner.dev_info.bytes_avail = memory.devices[0].free;
    client.clearOperations();

    return { client, model, planner };
}

describe('Planner warm-model scheduling', () => {
    test('downshifts an unoccupied model to the minimum sufficient rung', async () => {
        const { client, model, planner } = await createFixture(1);
        model.ladder[0].last_slots = client.seedSnapshot(model.name, 0, ['baseline conversation']);
        client.setLiveSlots(['long conversation'], model.name);
        let seen_rung = -1;

        await planner.serveModel({ tokens: 1000 }, model.name, new AbortController().signal, async () => {
            seen_rung = model.ladder_i;
            return true;
        });

        assert.equal(seen_rung, 0);
        assert.equal(model.ladder_i, 0);
        assert.deepEqual(client.getLiveSlots(model.name), ['baseline conversation']);
        assert.deepEqual(client.getSnapshot(model.name, 1), ['long conversation']);
        assert.deepEqual(client.operations.filter(operation => operation.kind === 'reload').map(operation => operation.model), ['model']);
        assert.deepEqual(client.operations.flatMap(operation => operation.kind === 'restore' ? [operation.filenames] : []), [['model-rung-0-0.bin']]);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_reload.length, 0);
    });

    test('upshifts before serving an input that exceeds the current rung', async () => {
        const { client, model, planner } = await createFixture();
        client.setLiveSlots(['live conversation'], model.name);
        let seen_rung = -1;

        await planner.serveModel({ tokens: 5000 }, model.name, new AbortController().signal, async () => {
            seen_rung = model.ladder_i;
            return true;
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
        const busy = planner.serveModel({ tokens: 6000 }, model.name, new AbortController().signal, async () => {
            started.resolve();
            await finish.promise;
            return true;
        });

        try {
            await started.promise;
            let seen_rung = -1;
            await planner.serveModel({ tokens: 1000 }, model.name, new AbortController().signal, async () => {
                seen_rung = model.ladder_i;
                return true;
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
        planner.models.set(second.name, second);
        await second.loadWeights(new AbortController().signal);
        planner.weights_only.add(second.name);
        client.setLiveSlots(['first conversation'], model.name);
        const memory = await second.getMemory(new AbortController().signal);
        planner.dev_info.bytes_avail = memory.devices[0].free;
        client.clearOperations();

        await planner.serveModel({ tokens: 1000 }, second.name, new AbortController().signal, async () => {
            assert.equal(planner.active.model, second.name);
            assert.equal(client.hasKV(model.name), false);
            assert.equal(client.hasKV(second.name), true);
            return true;
        });

        assert.equal(planner.weights_only.has(model.name), false);
        assert.equal(client.isLoaded(model.name), false);
        assert.deepEqual(client.getSnapshot(model.name, 0), ['first conversation']);
        assert.equal(client.operations.filter(operation => operation.kind === 'unload').length, 1);
        client.setLiveSlots(['second conversation'], second.name);
        client.clearOperations();

        await planner.serveModel({ tokens: 1000 }, model.name, new AbortController().signal, async () => {
            assert.equal(planner.active.model, model.name);
            assert.equal(client.hasKV(second.name), false);
            assert.equal(client.hasKV(model.name), true);
            assert.deepEqual(client.getLiveSlots(model.name), ['first conversation']);
            return true;
        });

        assert.equal(planner.weights_only.has(second.name), false);
        assert.deepEqual(client.getSnapshot(second.name, 0), ['second conversation']);
        assert.equal(client.operations.filter(operation => operation.kind === 'unload').length, 1);
        assert.equal(planner.active.readers, 0);
        assert.equal(planner.waiting_load.length, 0);
    });

    test('evicts idle weights when a new model needs room for its KV', async () => {
        const { client, model, planner } = await createFixture();
        const second = createModel(client, 'second');
        second.bytes_needed_no_kv = 5120;
        planner.models.set(second.name, second);
        client.total_bytes = 12000;
        planner.dev_info.bytes_total = client.total_bytes;
        planner.dev_info.bytes_avail = (await model.getMemory(new AbortController().signal)).devices[0].free;
        client.clearOperations();

        await planner.serveModel({ tokens: 1000 }, second.name, new AbortController().signal, async () => {
            assert.equal(planner.active.model, second.name);
            assert.equal(client.hasKV(second.name), true);
            assert.equal(client.isLoaded(model.name), false);
            return true;
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
        planner.models.set(second.name, second);
        await second.loadWeights(new AbortController().signal);
        planner.weights_only.add(second.name);
        const memory = await second.getMemory(new AbortController().signal);
        planner.dev_info.bytes_avail = memory.devices[0].free;
        client.clearOperations();

        const started = deferred<void>();
        const finish = deferred<void>();
        const generating = planner.serveModel({ tokens: 1000 }, model.name, new AbortController().signal, async () => {
            started.resolve();
            await finish.promise;
            return true;
        });

        try {
            await started.promise;
            const queued = planner.serveModel({ tokens: 1000 }, second.name, new AbortController().signal, async () => true);
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
        const slow = planner.serveModel({ tokens: 1000 }, model.name, new AbortController().signal, async (_body, _client, _signal, isFinal) => {
            assert.equal(isFinal, false);
            slow_rungs.push(model.ladder_i);
            if (slow_rungs.length === 1) {
                started.resolve();
                await finish.promise;
                return false;
            }
            return true;
        });

        try {
            await started.promise;
            const fast = planner.serveModel({ tokens: 1000 }, model.name, new AbortController().signal, async (_body, _client, _signal, isFinal) => {
                assert.equal(isFinal, false);
                fast_rungs.push(model.ladder_i);
                return fast_rungs.length > 1;
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
