import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { parseArgs } from '../../config/parser.js';
import { buildConfig } from '../../config/build.js';
import { LoadStatus } from '../../config/types.js';
import type { SlotRestore } from '../../client/types.js';
import { RouterProcess } from '../../client/router-process.js';
import { ModelEntry } from '../../planner/model-entry.js';
import { createStrategies } from '../../planner/strategies/index.js';
import { PromptCache } from '../../planner/prompt-cache.js';
import { GpuTestResources } from './resources.js';
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

async function chat(model: ModelEntry, signal: AbortSignal): Promise<void> {
    const response = await model.completions({
        messages: [
            { role: 'system', content: 'Answer briefly.' },
            { role: 'user', content: 'What is two plus two?' },
        ],
        stream: false,
        max_tokens: 32,
        temperature: 0,
    }, signal);
    const text = await response.text();
    assert.ok(response.ok, `completion failed (${response.status}): ${text}`);
    const result = JSON.parse(text) as { choices?: { message?: unknown }[] };
    assert.ok(result.choices?.[0]?.message, `completion has no assistant message: ${text}`);
}

function assertRestored(entry: ModelEntry, restores: SlotRestore[][]): void {
    const restored = restores.flat();
    assert.ok(restored.some(slot => slot.n_restored > 0), 'expected a populated cache restore');
    assert.ok(restored.every(slot => entry.cache?.entries.some(snapshot => snapshot.filename === slot.filename)), 'restore must use an indexed snapshot');
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
    const resources = new GpuTestResources(directory);
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
        resources.trackRouter(router);
        const client = await router.start(preset_path);
        const restores: SlotRestore[][] = [];
        const restore_slot = client.restoreSlot.bind(client);
        client.restoreSlot = async (model, slot_id, filename, signal) => {
            const result = await restore_slot(model, slot_id, filename, signal);
            restores.push([result]);
            return result;
        };
        const model_cfg = config.models[config.default_model];
        assert.ok(model_cfg);
        const entry = new ModelEntry(
            client, 
            model_cfg.name, 
            model_cfg.aliases, 
            model_cfg.initial_state, 
            model_cfg.ladder, 
            createStrategies(client)
        );
        entry.cache = new PromptCache(slot_path, config.cache_disk_mib);
        resources.trackCache(entry.cache);
        const rung_ids = entry.ladder.map(rung => rung.strategy);
        if (mmproj_path) assert.ok(rung_ids.includes('mmproj-to-cpu'), 'mmproj strategy was not configured');
        if (spec_type && spec_type !== 'none') assert.ok(rung_ids.includes('disable-spec'), 'spec strategy was not configured');
        if (floor !== 'f16') assert.ok(rung_ids.includes('quantize-kv-q8'), 'KV quantization strategy was not configured');
        assert.ok(entry.ladder.length > 1, 'configure at least one strategy to exercise a rung transition');
        const signal = resources.signal;
        await entry.loadWeights(signal);
        await entry.applyRung(0, signal);
        assert.equal(entry.status, LoadStatus.LOADED);
        await chat(entry, signal);
        const incoming = await entry.renderPrompt({ messages: [
            { role: 'system', content: 'Answer briefly.' },
            { role: 'user', content: 'What is two plus two?' },
        ] }, signal);
        assert.ok(incoming);

        for (let rung_i = 1; rung_i < entry.ladder.length; rung_i++) {
            console.log(`${entry.name} moving to rung ${rung_i} (${entry.ladder[rung_i].strategy})`);
            restores.length = 0;
            await entry.saveSlots(signal);
            const n_ctx = await entry.applyRung(rung_i, signal);
            assert.ok(n_ctx > 0, `rung ${rung_i} has no context`);
            assert.equal(entry.ladder_i, rung_i);
            await entry.restorePrompts([incoming], signal);
            await entry.cache.enforce();
            assertRestored(entry, restores);
            await chat(entry, signal);
            console.log(`GPU transition covered: ${entry.ladder[rung_i].strategy} (${n_ctx} tokens)`);
        }

        const final_rung = entry.ladder_i;
        restores.length = 0;
        await entry.saveSlots(signal);
        await entry.applyRung(0, signal);
        assert.equal(entry.ladder_i, 0);
        await entry.restorePrompts([incoming], signal);
        await entry.cache.enforce();
        assertRestored(entry, restores);
        await chat(entry, signal);
        console.log(`GPU transition covered: ${entry.ladder[final_rung].strategy} → baseline`);
        succeeded = true;
    } finally {
        try {
            await resources.shutdown();
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
