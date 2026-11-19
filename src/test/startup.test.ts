import assert from 'node:assert/strict';
import { after, describe, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { LoadStatus, type ManagerConfig, type ModelState } from '../config/types.js';
import { ModelEntry } from '../planner/model-entry.js';
import { ModelTask, Planner } from '../planner/planner.js';
import { printModelBreakpoints, showBreakpoints } from '../show-breakpoints.js';
import { captureOutput } from './capture-output.js';
import { LlamaAPIMock } from './llama-api-mock.js';

const fixtures: { directory: string; planner: Planner }[] = [];
after(async () => {
    await Promise.all(fixtures.map(async ({ directory, planner }) => {
        await planner.shutdown();
        await rm(directory, { recursive: true, force: true });
    }));
});

function createModel(client: LlamaAPIMock, name: string): ModelEntry {
    const initial_state: ModelState = {
        kv_unified: true,
        cache_type_k: 'f16',
        cache_type_v: 'f16',
        model_variant: name,
    };

    const entry = new ModelEntry(client, name, [], initial_state, [], new Map());
    entry.ladder.push({
        strategy: 'quantize-kv-q8',
        variant_name: 'baseline',
        state: { model_variant: name, kv_unified: true, cache_type_k: 'q8_0', cache_type_v: 'q8_0' },
        n_ctx_cap: -1,
        bytes_needed: 0,
    });
    return entry;
}

async function createFixture(names: string[], default_model: string) {
    const client = new LlamaAPIMock();
    const models = new Map(names.map(name => [name, createModel(client, name)]));
    const directory = await mkdtemp(join(tmpdir(), 'llama-manager-startup-'));
    client.slot_directory = directory;
    const config: ManagerConfig = {
        mode: 'router',
        router: { bin: '', llama_log_dir: '', slot_save_path: directory, listen: '', poll_interval_ms: 0, poll_timeout_ms: 0, shutdown_grace_period_ms: 0 },
        host: '127.0.0.1',
        port: 0,
        idle_load_default_seconds: 0,
        downshift_grace_seconds: 0,
        cache_disk_mib: 1,
        model_load: { poll_interval_ms: 0, poll_timeout_ms: 0 },
        models: {},
        default_model,
    };
    const planner = new Planner(client, config, models);
    fixtures.push({ directory, planner });
    return { client, config, models, planner };
}

async function wake(planner: Planner, model: string): Promise<void> {
    await planner.serveModel({ messages: [] }, model, ModelTask.WAKE, new AbortController().signal, async () => ({ kind: 'done' }));
}

describe('startup', () => {
    test('calc-breakpoints walks the default model last and leaves only it loaded', async () => {
        const { client, config, models } = await createFixture(['default', 'other'], 'default');

        await captureOutput(() => showBreakpoints(client, config, models));

        assert.deepEqual(client.operations.filter(op => op.kind === 'load' || op.kind === 'unload').map(op => `${op.kind}:${op.model}`),
            ['load:other', 'unload:other', 'load:default']);
        assert.equal(models.get('default')!.status, LoadStatus.LOADED);
        assert.equal(models.get('other')!.status, LoadStatus.UNLOADED);
        assert.equal(models.get('default')!.getMaxCtx(), 8192);
        assert.equal(models.get('other')!.getMaxCtx(), 8192);
    });

    test('unload-all after calc-breakpoints unloads the default model left loaded by the walk', async () => {
        const { client, config, models, planner } = await createFixture(['default', 'other'], 'default');
        await captureOutput(() => showBreakpoints(client, config, models));
        client.clearOperations();

        const output = await captureOutput(async () => { assert.equal(await planner.unloadAll(), 'success'); });

        assert.deepEqual(client.operations.filter(op => op.kind === 'unload').map(op => op.model), ['default']);
        assert.equal(models.get('default')!.status, LoadStatus.UNLOADED);
        assert.equal(models.get('default')!.getMaxCtx(), 8192);
        assert.match(output, /unloading 1 models with restore/);
    });

    test('waking the default model after calc-breakpoints reuses its weights and walks once', async () => {
        const { client, config, models, planner } = await createFixture(['default'], 'default');
        await captureOutput(() => showBreakpoints(client, config, models));
        client.clearOperations();

        await captureOutput(() => wake(planner, 'default'));

        assert.equal(client.operations.filter(op => op.kind === 'load').length, 0);
        assert.deepEqual(client.operations.flatMap(op => op.kind === 'reload' ? [op.params.cache_type_k] : []), ['f16']);
        assert.equal(planner.active.model, 'default');
        assert.equal(models.get('default')!.ladder_i, 0);
    });

    test('first-load wake walks the ladder and applies rung 0 once', async () => {
        const { client, models, planner } = await createFixture(['default'], 'default');

        const output = await captureOutput(() => wake(planner, 'default'));

        assert.equal(client.operations.filter(op => op.kind === 'load').length, 1);
        assert.deepEqual(client.operations.flatMap(op => op.kind === 'reload' ? [op.params.cache_type_k] : []), ['f16', 'q8_0', 'f16']);
        assert.equal(models.get('default')!.ladder_i, 0);
        assert.match(output, /serving default with a context window of 4096 tokens/);
        assert.match(output, /available memory: gpu \(.*\) \| cpu \(.*\)/);
    });
});

describe('printModelBreakpoints', () => {
    const mem = (weight_gib: number, context_gib: number) => ({ weight_gib, context_gib });
    const table = printModelBreakpoints({
        name: 'model-with-an-extremely-long-name-that-would-otherwise-run-into-the-gpu-column',
        gpu_total_gib: 31.36,
        cpu_total_gib: 114.95,
        rungs: [
            { i: 0, strategy_name: 'baseline', n_ctx: 90624, gain: 0, gpu: mem(24.12, 5.62), cpu: mem(86.43, 0.31) },
            { i: 1, strategy_name: 'disable-spec', n_ctx: 246784, gain: 156160, gpu: mem(120.91, 9.22), cpu: mem(86.43, 0.3) },
        ],
    });
    const lines = table.trimEnd().split('\n');

    test('shows gpu and cpu capacity and per-device memory', () => {
        assert.match(lines[1], /│ +gpu: 31 GiB │ +cpu: 115 GiB$/);
        assert.match(lines[3], /│ +weights \/ ctx │ +weights \/ ctx$/);
        assert.match(lines[5], /│ +24\.12 \/ 5\.62 │ +86\.43 \/ 0\.31$/);
        assert.match(lines[6], /│ 120\.91 \/ 9\.22 │ +86\.43 \/ 0\.30$/);
        for (const line of lines.slice(3)) assert.doesNotMatch(line, /GiB/);
    });

    test('aligns device dividers on every line and truncates long names', () => {
        const positions = (line: string) => [...line].flatMap((ch, i) => '│╤╪┼┴'.includes(ch) ? [i] : []);
        const expected = positions(lines[3]);
        assert.equal(expected.length, 2);
        for (const line of lines.slice(0, -1)) {
            assert.deepEqual(positions(line), expected, line);
        }
        assert.match(lines[1], /^ model-with-an-extremely-long-name.*… │/);
        const rules = [lines[0], lines[2], lines[4], lines[7]];
        assert.equal(new Set(rules.map(line => line.length)).size, 1);
    });
});
