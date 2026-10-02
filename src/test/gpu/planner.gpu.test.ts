import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { buildConfig } from '../../config/build.js';
import { parseArgs } from '../../config/parser.js';
import { LoadStatus, type ManagerConfig, type ModelConfig, type RawModel, type StrategyId } from '../../config/types.js';
import type { LlamaAPI } from '../../client/llama-api.js';
import { RouterProcess } from '../../client/router-process.js';
import type { SlotRestore } from '../../client/types.js';
import { ModelEntry } from '../../planner/model-entry.js';
import { Planner } from '../../planner/planner.js';
import { createStrategies } from '../../planner/strategies/index.js';
import type { StrategyImpl } from '../../planner/types.js';
import { walkBreakpoints } from '../../show-breakpoints.js';

function requiredPath(key: string): string {
    const value = process.env[key]?.trim();
    assert.ok(value, `set ${key} in .env.gpu-test`);
    const path = resolve(value);
    assert.ok(existsSync(path), `${key} does not exist: ${path}`);
    return path;
}

function optionalPath(key: string): string | undefined {
    return process.env[key]?.trim() ? requiredPath(key) : undefined;
}

function modelFields(path: string, mmproj?: string, spec_type?: string, draft?: string): RawModel {
    const fields: RawModel = {
        model: path,
        ladder: [],
        'fit-target': Number(process.env.FIT_TARGET?.trim() || '512'),
        'n-gpu-layers': Number(process.env.N_GPU_LAYERS?.trim() || '99'),
        parallel: 1,
    };
    if (mmproj) fields.mmproj = mmproj;
    if (spec_type) fields['spec-type'] = spec_type;
    if (draft) fields['model-draft'] = draft;
    if (spec_type && process.env.SPEC_DRAFT_N_MAX?.trim()) {
        fields['spec-draft-n-max'] = Number(process.env.SPEC_DRAFT_N_MAX.trim());
    }
    return fields;
}

function makeEntry(client: LlamaAPI, cfg: ModelConfig, strategies: Map<StrategyId, StrategyImpl>): ModelEntry {
    return new ModelEntry(
        client,
        cfg.name,
        cfg.aliases,
        cfg.initial_state,
        cfg.ladder,
        strategies,
    );
}

type Fixture = {
    directory: string;
    client: LlamaAPI;
    router: RouterProcess;
    config: ManagerConfig;
    a: ModelEntry;
    b: ModelEntry;
    restores: { model: string; slots: SlotRestore[] }[];
};

async function createFixture(): Promise<Fixture> {
    const root = resolve(import.meta.dirname, '../../..');
    const model = requiredPath('MODEL');
    const second_model = requiredPath('SECOND_MODEL');
    const mmproj = optionalPath('MMPROJ');
    const draft = optionalPath('DRAFT_MODEL');
    const second_mmproj = optionalPath('SECOND_MMPROJ');
    const spec_type = process.env.SPEC_TYPE?.trim();
    assert.ok(!draft || (spec_type && spec_type !== 'none'), 'DRAFT_MODEL requires SPEC_TYPE');
    const default_bin = process.platform === 'win32'
        ? './llama.cpp/build/bin/Release/llama-server.exe'
        : './llama.cpp/build/bin/llama-server';
    const bin = resolve(root, process.env.LLAMA_BIN?.trim() || default_bin);
    assert.ok(existsSync(bin), `llama-server binary does not exist: ${bin}`);

    const artifacts_dir = join(root, 'logs', 'gpu-tests');
    await mkdir(artifacts_dir, { recursive: true });
    const directory = await mkdtemp(join(artifacts_dir, 'planner-'));
    const config_path = join(directory, 'config.yaml');
    const preset_path = join(directory, 'preset.ini');
    let router: RouterProcess | undefined;
    try {
        await writeFile(config_path, JSON.stringify({
            router: {
                bin,
                'llama-log-dir': join(directory, 'logs'),
                'slot-save-path': join(directory, 'slots'),
            },
            models: {
                'gpu-test-a': modelFields(model, mmproj, spec_type, draft),
                'gpu-test-b': modelFields(second_model, second_mmproj),
            },
            'default-model': 'gpu-test-a',
            'sleep-idle-seconds': 0,
        }));
        const config = await buildConfig(parseArgs(['--config', config_path]), root, preset_path, config_path);
        config.router.poll_timeout_ms = 30000;
        config.model_load.poll_timeout_ms = 180000;
        router = new RouterProcess(config.router, config.model_load);
        const client = await router.start(preset_path);
        const strategies = createStrategies(client);
        const a = makeEntry(client, config.models['gpu-test-a'], strategies);
        const b = makeEntry(client, config.models['gpu-test-b'], strategies);
        const restores: Fixture['restores'] = [];
        const original_restore = client.restoreAllSlots.bind(client);
        client.restoreAllSlots = async (model_id, saves, signal) => {
            const slots = await original_restore(model_id, saves, signal);
            restores.push({ model: model_id, slots });
            return slots;
        };
        return { directory, client, router, config, a, b, restores };
    } catch (err) {
        try {
            await router?.shutdown();
        } finally {
            console.error(`GPU test artifacts preserved: ${directory}`);
        }
        throw err;
    }
}

async function withFixture(run: (fixture: Fixture) => Promise<void>): Promise<void> {
    const fixture = await createFixture();
    let succeeded = false;
    try {
        await run(fixture);
        succeeded = true;
    } finally {
        try {
            await fixture.router.shutdown();
        } catch (err) {
            succeeded = false;
            throw err;
        } finally {
            if (succeeded) {
                await rm(fixture.directory, { recursive: true, force: true });
            } else {
                console.error(`GPU test artifacts preserved: ${fixture.directory}`);
            }
        }
    }
}

let iter = 0;

async function serve(planner: Planner, model: ModelEntry, messages: { role: string; content: string }[], min_ctx?: number): Promise<void> {
    const body = { model: model.name, messages, stream: false, max_tokens: 32, temperature: 0 };
    await planner.serveModel(body, model.name, new AbortController().signal, async (_body, model, signal, isFinal) => {
        iter++;
        if (iter > 10) throw new Error(`max iter reached`);
        console.log(`${model.name} is at ctx: ${model.getCurCtx()}`);

        const response = await model.completions(body, signal);
        const text = await response.text();
        if (!response.ok) {
            let context_exceeded = false;
            try {
                const payload = JSON.parse(text) as { error?: { type?: string } };
                context_exceeded = payload.error?.type === 'exceed_context_size_error';
            } catch { }
            if (context_exceeded && !isFinal) return false;
            assert.fail(`${model.name} completion failed (${response.status}): ${text}`);
        }
        assert.equal(isFinal, false, `${model.name} could not serve the request`);
        const result = JSON.parse(text) as { choices?: { message?: unknown }[] };
        assert.ok(result.choices?.[0]?.message, `${model.name} completion has no assistant message: ${text}`);
        if (min_ctx !== undefined) {
            const live = (await model.getSlots(signal))[0]?.n_ctx ?? 0;
            assert.ok(live >= min_ctx, `${model.name} live context ${live} is smaller than required ${min_ctx}`);
        }
        return true;
    });
}

async function measure(client: LlamaAPI, entry: ModelEntry, signal: AbortSignal): Promise<void> {
    await walkBreakpoints(client, entry, signal);
    await entry.unloadHard();
    assert.ok(entry.getCtxCap(0)! > 0, `${entry.name} has no measured baseline capacity`);
}

async function activateA(fixture: Fixture, signal: AbortSignal): Promise<Planner> {
    const { a, b, client, config } = fixture;
    const planner = new Planner(client, config, new Map([[a.name, a], [b.name, b]]));
    await a.loadWeights(signal);
    await a.moveToRung(0, signal);
    const memory = await a.getMemory(signal);
    const gpu = memory.devices.find(device => device.type !== 'cpu');
    assert.ok(gpu, 'no GPU memory reported');
    planner.dev_info.bytes_total = gpu.total;
    planner.dev_info.bytes_avail = gpu.free;
    planner.active.model = a.name;
    return planner;
}

test('GPU: planner serves A → B → A with one KV and restores A slots', { timeout: 1200000 }, async () => {
    await withFixture(async fixture => {
        const { a, b, client, restores } = fixture;
        const signal = new AbortController().signal;
        await measure(client, a, signal);
        await measure(client, b, signal);
        const planner = await activateA(fixture, signal);
        const prompt = [{ role: 'user', content: 'What is two plus two? Answer briefly.' }];

        await serve(planner, a, prompt);
        await serve(planner, b, prompt);
        assert.notEqual(a.status, LoadStatus.LOADED, 'A still holds a KV cache');
        assert.equal(b.status, LoadStatus.LOADED);
        const saved = a.ladder[0].last_slots;
        assert.ok(saved && saved.some(slot => slot.n_saved > 0), 'A did not save a populated slot on swap');
        await serve(planner, a, prompt);
        assert.equal(a.status, LoadStatus.LOADED);
        assert.notEqual(b.status, LoadStatus.LOADED, 'B still holds a KV cache');
        assert.equal(planner.active.model, a.name);
        assert.ok(restores.some(restore => restore.model === a.name && restore.slots.some(slot =>
            saved.some(source => source.filename === slot.filename && slot.n_restored > 0))),
        'A was not restored from its saved slot');
        console.log('GPU planner swap covered: A → B → A');
    });
});

async function findPrompt(model: ModelEntry, lower: number, upper: number): Promise<{
    messages: { role: string; content: string }[];
    tokens: number;
}> {
    let low = 1;
    let high = upper * 2;
    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const messages = [{ role: 'user', content: 'hello '.repeat(mid) }];
        const tokens = await model.countTokens({ model: model.name, messages, max_tokens: 32 }, new AbortController().signal);
        assert.ok(tokens !== null, `countTokens failed for model ${model.name}`);
        if (tokens <= lower) {
            low = mid + 1;
        } else if (tokens + 32 >= upper) {
            high = mid - 1;
        } else {
            return { messages, tokens };
        }
    }
    assert.fail(`could not construct a prompt between live ctx ${lower} and measured ctx ${upper}`);
}

test('GPU: planner serves an input between live and measured capacity after evicting idle weights', { timeout: 1200000 }, async () => {
    await withFixture(async fixture => {
        const { a, b, client } = fixture;
        const signal = new AbortController().signal;
        await measure(client, a, signal);
        const measured = a.getCtxCap(0)!;
        await a.loadWeights(signal);
        await b.loadWeights(signal);
        await a.moveToRung(0, signal);
        const live = (await client.getSlots(a.name, signal))[0]?.n_ctx;
        assert.ok(live && measured > live + 64,
            `setup requires measured capacity > live capacity + 64 (measured: ${measured}, live: ${live})`);
        const { messages, tokens } = await findPrompt(a, live, measured);
        console.log(`GPU capacity case: measured=${measured}, live=${live}, prompt=${tokens}`);

        const planner = new Planner(client, fixture.config, new Map([[a.name, a], [b.name, b]]));
        planner.active.model = a.name;
        planner.weights_only.add(b.name);
        const memory = await client.getMemory(a.name, signal);
        const gpu = memory.devices.find(device => device.type !== 'cpu');
        assert.ok(gpu, 'no GPU memory reported');
        planner.dev_info.bytes_total = gpu.total;
        planner.dev_info.bytes_avail = gpu.free;

        await serve(planner, a, messages, tokens + 32);
        assert.equal(b.status, LoadStatus.UNLOADED, 'idle B weights were not evicted');
        const grown = (await client.getSlots(a.name, signal))[0]?.n_ctx ?? 0;
        assert.ok(grown >= tokens + 32, `active context ${grown} did not grow to fit ${tokens + 32} tokens`);
    });
});