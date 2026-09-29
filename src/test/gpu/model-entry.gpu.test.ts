import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { parseArgs } from '../../config/parser.js';
import { buildConfig } from '../../config/build.js';
import { LoadStatus, type ModelState, type StrategyId } from '../../config/types.js';
import type { LlamaAPI } from '../../client/llama-api.js';
import type { SlotRestore } from '../../client/types.js';
import { RouterProcess } from '../../client/router-process.js';
import { ModelEntry, type Rung } from '../../planner/model-entry.js';
import { createStrategies } from '../../planner/strategies/index.js';
import type { Strategy } from '../../planner/types.js';

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

function buildLadder(initial_state: ModelState, ids: StrategyId[], strategies: Map<StrategyId, Strategy>): Rung[] {
    const rungs: Rung[] = [{ strategy: 'none', impl: null!, state: initial_state, n_ctx_cap: -1, bytes_needed: 0 }];
    let state = initial_state;
    for (const id of ids) {
        const strategy = strategies.get(id);
        assert.ok(strategy, `missing strategy ${id}`);
        if (!strategy.canApply(state)) continue;
        state = strategy.getNewState(state);
        rungs.push({ strategy: id, impl: strategy, state, n_ctx_cap: -1, bytes_needed: 0 });
    }
    return rungs;
}

async function chat(client: LlamaAPI, model: string): Promise<void> {
    const response = await client.completions({
        messages: [
            { role: 'system', content: 'Answer briefly.' },
            { role: 'user', content: 'What is two plus two?' },
        ],
        stream: false,
        max_tokens: 32,
        temperature: 0,
    }, model);
    const text = await response.text();
    assert.ok(response.ok, `completion failed (${response.status}): ${text}`);
    const result = JSON.parse(text) as { choices?: { message?: unknown }[] };
    assert.ok(result.choices?.[0]?.message, `completion has no assistant message: ${text}`);
}

function assertRestored(entry: ModelEntry, rung_i: number, restores: SlotRestore[][]): void {
    const saved = entry.ladder[rung_i].last_slots;
    assert.ok(saved?.length, `no slot snapshot saved at rung ${rung_i}`);
    assert.ok(saved.some(slot => slot.n_saved > 0), `rung ${rung_i} saved no prompt tokens`);
    assert.equal(restores.length, 1, `expected one successful restore after leaving rung ${rung_i}`);
    for (const slot of saved) {
        const restored = restores[0].find(result => result.id_slot === slot.id_slot && result.filename === slot.filename);
        assert.ok(restored, `slot ${slot.id_slot} was not restored from ${slot.filename}`);
        if (slot.n_saved > 0) {
            assert.ok(restored.n_restored > 0, `slot ${slot.id_slot} restored no prompt tokens from ${slot.filename}`);
        }
    }
}

test('GPU: real router preserves slots through configured rung transitions', { timeout: 600000 }, async () => {
    const project_root = resolve(import.meta.dirname, '../../..');
    const model_path = requiredPath('MODEL');
    const mmproj_path = optionalPath('MMPROJ');
    const draft_path = optionalPath('DRAFT_MODEL');
    const spec_type = process.env.SPEC_TYPE?.trim();
    const default_bin = process.platform === 'win32'
        ? './llama.cpp/build/bin/Release/llama-server.exe'
        : './llama.cpp/build/bin/llama-server';
    const bin = resolve(project_root, process.env.LLAMA_BIN?.trim() || default_bin);
    assert.ok(existsSync(bin), `llama-server binary does not exist: ${bin}`);
    const floor = process.env.KV_FLOOR?.trim() || 'q8_0';
    assert.ok(['f16', 'q8_0', 'q4_0'].includes(floor), `unsupported KV_FLOOR: ${floor}`);
    if (draft_path) assert.ok(spec_type && spec_type !== 'none', 'DRAFT_MODEL requires SPEC_TYPE');
    // NOTE: disable hadamard rotation
    if (floor !== 'f16') process.env.LLAMA_ATTN_ROT_DISABLE = '1';

    const artifacts_dir = join(project_root, 'logs', 'gpu-tests');
    await mkdir(artifacts_dir, { recursive: true });
    const directory = await mkdtemp(join(artifacts_dir, 'model-entry-'));
    const preset_path = join(directory, 'preset.ini');
    const config_path = join(directory, 'config.yaml');
    const slot_path = join(directory, 'slots');
    let router: RouterProcess | undefined;
    let succeeded = false;

    try {
        await writeFile(config_path, JSON.stringify({ router: { 'llama-log-dir': join(directory, 'logs') } }));
        const argv = [
            '--config', config_path,
            '--model', model_path,
            '--alias', 'gpu-test-model',
            '--bin', bin,
            '--slot-save-path', slot_path,
            '--fit-target', process.env.FIT_TARGET?.trim() || '512',
            '--n-gpu-layers', process.env.N_GPU_LAYERS?.trim() || '99',
            '--parallel', '1',
            '--cache-type-k', floor,
            '--cache-type-v', floor,
        ];
        if (mmproj_path) argv.push('--mmproj', mmproj_path);
        if (spec_type) {
            argv.push('--spec-type', spec_type);
            const draft_n_max = process.env.SPEC_DRAFT_N_MAX?.trim();
            if (draft_n_max) argv.push('--spec-draft-n-max', draft_n_max);
        }
        if (draft_path) argv.push('--model-draft', draft_path);

        const parsed = parseArgs(argv);
        const config = await buildConfig(parsed, project_root, preset_path, config_path);
        config.router.poll_timeout_ms = 30000;
        config.model_load.poll_timeout_ms = 180000;
        router = new RouterProcess(config.router, config.model_load);
        const client = await router.start(preset_path);
        const restores: SlotRestore[][] = [];
        const restore_slots = client.restoreAllSlots.bind(client);
        client.restoreAllSlots = async (model, saves, signal) => {
            const results = await restore_slots(model, saves, signal);
            restores.push(results);
            return results;
        };
        const model_cfg = config.models[config.default_model];
        assert.ok(model_cfg);
        const entry = new ModelEntry(client, model_cfg.name, model_cfg.aliases,
            buildLadder(model_cfg.initial_state, model_cfg.ladder, createStrategies(client)));
        const rung_ids = entry.ladder.map(rung => rung.strategy);
        if (mmproj_path) assert.ok(rung_ids.includes('mmproj-to-cpu'), 'mmproj strategy was not configured');
        if (spec_type && spec_type !== 'none') assert.ok(rung_ids.includes('disable-spec'), 'spec strategy was not configured');
        if (floor !== 'f16') assert.ok(rung_ids.includes('quantize-kv-q8'), 'KV quantization strategy was not configured');
        assert.ok(entry.ladder.length > 1, 'configure at least one strategy to exercise a rung transition');
        const signal = new AbortController().signal;
        await entry.loadWeights(signal);
        await entry.moveToRung(0, signal);
        assert.equal(entry.status, LoadStatus.LOADED);
        await chat(client, entry.name);

        for (let rung_i = 1; rung_i < entry.ladder.length; rung_i++) {
            console.log(`${entry.name} moving to rung ${rung_i} (${entry.ladder[rung_i].strategy})`);
            const source = rung_i - 1;
            restores.length = 0;
            const n_ctx = await entry.moveToRung(rung_i, signal);
            assert.ok(n_ctx > 0, `rung ${rung_i} has no context`);
            assert.equal(entry.ladder_i, rung_i);
            assertRestored(entry, source, restores);
            await chat(client, entry.name);
            console.log(`GPU transition covered: ${entry.ladder[rung_i].strategy} (${n_ctx} tokens)`);
        }

        const final_rung = entry.ladder_i;
        restores.length = 0;
        await entry.moveToRung(0, signal);
        assert.equal(entry.ladder_i, 0);
        assertRestored(entry, 0, restores);
        await chat(client, entry.name);
        console.log(`GPU transition covered: ${entry.ladder[final_rung].strategy} → baseline`);
        succeeded = true;
    } finally {
        try {
            await router?.shutdown();
        } catch (err) {
            succeeded = false;
            throw err;
        } finally {
            if (succeeded) {
                await rm(directory, { recursive: true, force: true });
            } else {
                console.error(`GPU test artifacts preserved: ${directory}`);
            }
        }
    }
});
