import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import { LoadStatus, type ModelState } from '../config/types.js';
import { MIN_ALLOWED_CTX } from '../llama-cpp-constants.js';
import { ModelEntry } from '../planner/model-entry.js';
import { LlamaAPIMock, type Operation } from './llama-api-mock.js';

type Fixture = {
    entry: ModelEntry;
    operations: Operation[];
    setLiveSlots(contents: string[]): void;
    getLiveSlots(): string[];
    seedSnapshot(rung_i: number, contents: string[]): void;
    getSnapshot(rung_i: number): string[] | undefined;
    failNext(kind: 'save' | 'restore' | 'reload'): void;
    clearOperations(): void;
};

function createFixture(): Fixture {
    const client = new LlamaAPIMock();
    const name = 'model';

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

    return {
        entry,
        operations: client.operations,
        setLiveSlots: contents => client.setLiveSlots(contents),
        getLiveSlots: () => client.getLiveSlots(),
        seedSnapshot: (rung_i, contents) => {
            entry.ladder[rung_i].last_slots = client.seedSnapshot(name, rung_i, contents);
        },
        getSnapshot: rung_i => client.getSnapshot(name, rung_i),
        failNext: kind => client.failNext(kind),
        clearOperations: () => client.clearOperations(),
    };
}

const signal = new AbortController().signal;

describe('ModelEntry', () => {
    test('activates a weights-only model and restores the destination rung snapshot', async () => {
        const fixture = createFixture();
        fixture.seedSnapshot(1, ['saved prompt']);

        await fixture.entry.loadWeights(signal);
        fixture.clearOperations();

        const n_ctx = await fixture.entry.moveToRung(1, signal);

        assert.equal(n_ctx, 8192);
        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.equal(fixture.entry.ladder_i, 1);
        assert.equal(fixture.entry.getCurCtx(), 8192);
        assert.deepEqual(fixture.getLiveSlots(), ['saved prompt']);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['reload', 'restore']);

        const reload = fixture.operations.find(operation => operation.kind === 'reload');
        assert.ok(reload && reload.kind === 'reload');
        assert.equal(reload.params.n_ctx, 0);
        assert.equal(reload.params.cache_type_k, 'q8_0');
        assert.equal(reload.params.cache_type_v, 'q8_0');

        const restore = fixture.operations.find(operation => operation.kind === 'restore');
        assert.ok(restore && restore.kind === 'restore');
        assert.deepEqual(restore.filenames, ['model-rung-1-0.bin']);
    });

    test('upshifts with the just-saved live slots rather than an older destination snapshot', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.setLiveSlots(['live prompt', 'generated tokens']);
        fixture.seedSnapshot(1, ['older prompt']);
        fixture.clearOperations();

        await fixture.entry.moveToRung(1, signal);

        assert.equal(fixture.entry.ladder_i, 1);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['save', 'reload', 'restore']);
        assert.deepEqual(fixture.getSnapshot(0), ['live prompt', 'generated tokens']);
        assert.deepEqual(fixture.getSnapshot(1), ['older prompt']);
        assert.deepEqual(fixture.getLiveSlots(), ['live prompt', 'generated tokens']);

        const save = fixture.operations.find(operation => operation.kind === 'save');
        assert.ok(save && save.kind === 'save');
        assert.equal(save.basename, 'model-rung-0');

        const restore = fixture.operations.find(operation => operation.kind === 'restore');
        assert.ok(restore && restore.kind === 'restore');
        assert.deepEqual(restore.filenames, ['model-rung-0-0.bin']);
    });

    test('downshifts to the destination rung snapshot while retaining the source snapshot', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.setLiveSlots(['baseline conversation']);
        await fixture.entry.moveToRung(1, signal);
        fixture.setLiveSlots(['long conversation']);
        fixture.clearOperations();

        await fixture.entry.moveToRung(0, signal);

        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.equal(fixture.entry.ladder_i, 0);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['save', 'reload', 'restore']);
        assert.deepEqual(fixture.getSnapshot(1), ['long conversation']);
        assert.deepEqual(fixture.getSnapshot(0), ['baseline conversation']);
        assert.deepEqual(fixture.getLiveSlots(), ['baseline conversation']);

        const restore = fixture.operations.find(operation => operation.kind === 'restore');
        assert.ok(restore && restore.kind === 'restore');
        assert.deepEqual(restore.filenames, ['model-rung-0-0.bin']);
    });

    test('repeated saves replace only the snapshot belonging to that rung', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.seedSnapshot(1, ['rung one snapshot']);
        fixture.setLiveSlots(['first rung zero snapshot']);
        await fixture.entry.unloadKV(signal);

        assert.deepEqual(fixture.getSnapshot(0), ['first rung zero snapshot']);
        assert.deepEqual(fixture.getSnapshot(1), ['rung one snapshot']);

        await fixture.entry.moveToRung(0, signal);
        fixture.setLiveSlots(['second rung zero snapshot']);
        await fixture.entry.unloadKV(signal);

        assert.deepEqual(fixture.getSnapshot(0), ['second rung zero snapshot']);
        assert.deepEqual(fixture.getSnapshot(1), ['rung one snapshot']);
    });

    test('continues a rung transition when saving fails', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.setLiveSlots(['live prompt']);
        fixture.failNext('save');
        fixture.clearOperations();

        await fixture.entry.moveToRung(1, signal);

        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.equal(fixture.entry.ladder_i, 1);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['save', 'reload']);
        assert.equal(fixture.getSnapshot(0), undefined);
        assert.deepEqual(fixture.getLiveSlots(), []);
    });

    test('continues a rung transition when restoring fails', async () => {
        const fixture = createFixture();
        fixture.seedSnapshot(0, ['saved prompt']);
        await fixture.entry.loadWeights(signal);
        fixture.failNext('restore');
        fixture.clearOperations();

        await fixture.entry.moveToRung(0, signal);

        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.equal(fixture.entry.ladder_i, 0);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['reload', 'restore']);
        assert.deepEqual(fixture.getLiveSlots(), []);
        assert.deepEqual(fixture.getSnapshot(0), ['saved prompt']);
    });

    test('propagates a reload failure without committing a new rung', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.setLiveSlots(['live prompt']);
        fixture.failNext('reload');
        fixture.clearOperations();

        await assert.rejects(fixture.entry.moveToRung(1, signal));

        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.equal(fixture.entry.ladder_i, 0);
        assert.deepEqual(fixture.getSnapshot(0), ['live prompt']);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['save', 'reload']);
    });

    test('stashes slots on KV unload and restores them on reactivation', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.setLiveSlots(['conversation']);
        fixture.clearOperations();

        await fixture.entry.unloadKV(signal);

        assert.equal(fixture.entry.status, LoadStatus.WEIGHTS_ONLY);
        assert.equal(fixture.entry.getCurCtx(), MIN_ALLOWED_CTX);
        assert.deepEqual(fixture.getSnapshot(0), ['conversation']);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['save', 'reload']);

        await fixture.entry.moveToRung(0, signal);

        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.deepEqual(fixture.getLiveSlots(), ['conversation']);
    });

    test('unloads weights with a saved snapshot for the next activation', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.setLiveSlots(['conversation before unload']);
        fixture.clearOperations();

        await fixture.entry.unloadWeights(signal);

        assert.equal(fixture.entry.status, LoadStatus.UNLOADED);
        assert.equal(fixture.entry.ladder_i, -1);
        assert.equal(fixture.entry.getCurCtx(), 0);
        assert.deepEqual(fixture.getSnapshot(0), ['conversation before unload']);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['save', 'unload']);

        await fixture.entry.loadWeights(signal);
        await fixture.entry.moveToRung(0, signal);

        assert.equal(fixture.entry.status, LoadStatus.LOADED);
        assert.deepEqual(fixture.getLiveSlots(), ['conversation before unload']);
    });

    test('hard unload resets runtime state without saving or erasing existing snapshots', async () => {
        const fixture = createFixture();
        await fixture.entry.loadWeights(signal);
        await fixture.entry.applyRung(0, signal);
        fixture.seedSnapshot(0, ['previous snapshot']);
        fixture.setLiveSlots(['new conversation']);
        fixture.clearOperations();

        await fixture.entry.unloadHard();

        assert.equal(fixture.entry.status, LoadStatus.UNLOADED);
        assert.equal(fixture.entry.ladder_i, -1);
        assert.equal(fixture.entry.getCurCtx(), 0);
        assert.deepEqual(fixture.getSnapshot(0), ['previous snapshot']);
        assert.deepEqual(fixture.operations.map(operation => operation.kind), ['unload']);
    });
});
