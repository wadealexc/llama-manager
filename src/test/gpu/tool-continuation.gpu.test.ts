import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { buildConfig } from '../../config/build.js';
import { parseArgs } from '../../config/parser.js';
import { RouterProcess } from '../../client/router-process.js';
import { ModelEntry, type Rung } from '../../planner/model-entry.js';
import { Planner } from '../../planner/planner.js';
import { createStrategies } from '../../planner/strategies/index.js';

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

test('GPU: llama-server accepts a tool-call continuation after planner growth', { timeout: 1200000 }, async () => {
    const root = resolve(import.meta.dirname, '../../..');
    const model_path = requiredPath('MODEL');
    const mmproj_path = optionalPath('MMPROJ');
    const draft_path = optionalPath('DRAFT_MODEL');
    const spec_type = process.env.SPEC_TYPE?.trim();
    assert.ok(!draft_path || (spec_type && spec_type !== 'none'), 'DRAFT_MODEL requires SPEC_TYPE');
    const default_bin = process.platform === 'win32'
        ? './llama.cpp/build/bin/Release/llama-server.exe'
        : './llama.cpp/build/bin/llama-server';
    const bin = resolve(root, process.env.LLAMA_BIN?.trim() || default_bin);
    assert.ok(existsSync(bin), `llama-server binary does not exist: ${bin}`);
    process.env.LLAMA_ATTN_ROT_DISABLE = '1';

    const artifacts_dir = join(root, 'logs', 'gpu-tests');
    await mkdir(artifacts_dir, { recursive: true });
    const directory = await mkdtemp(join(artifacts_dir, 'tool-continuation-'));
    const config_path = join(directory, 'config.yaml');
    const preset_path = join(directory, 'preset.ini');
    const trace_path = join(directory, 'trace.json');
    let router: RouterProcess | undefined;
    let succeeded = false;

    try {
        await writeFile(config_path, JSON.stringify({ router: { 'llama-log-dir': join(directory, 'logs') } }));
        const argv = [
            '--config', config_path,
            '--model', model_path,
            '--alias', 'gpu-test-tool-continuation',
            '--bin', bin,
            '--slot-save-path', join(directory, 'slots'),
            '--fit-target', process.env.FIT_TARGET?.trim() || '512',
            '--n-gpu-layers', process.env.N_GPU_LAYERS?.trim() || '99',
            '--parallel', '1',
            '--jinja',
            '--ladder', 'quantize-kv-q8',
        ];
        if (mmproj_path) argv.push('--mmproj', mmproj_path);
        if (spec_type) {
            argv.push('--spec-type', spec_type);
            const draft_n_max = process.env.SPEC_DRAFT_N_MAX?.trim();
            if (draft_n_max) argv.push('--spec-draft-n-max', draft_n_max);
        }
        if (draft_path) argv.push('--model-draft', draft_path);

        const config = await buildConfig(parseArgs(argv), root, preset_path, config_path);
        config.router.poll_timeout_ms = 30000;
        config.model_load.poll_timeout_ms = 180000;
        config.sleep_idle_seconds = 0;
        router = new RouterProcess(config.router, config.model_load);
        const client = await router.start(preset_path);
        const model_cfg = config.models[config.default_model];
        assert.ok(model_cfg);
        const strategy = createStrategies(client).get('quantize-kv-q8');
        assert.ok(strategy);
        const rungs: Rung[] = [
            { strategy: 'none', impl: null!, state: model_cfg.initial_state, n_ctx_cap: -1, bytes_needed: 0 },
            { strategy: 'quantize-kv-q8', impl: strategy, state: strategy.getNewState(model_cfg.initial_state), n_ctx_cap: -1, bytes_needed: 0 },
        ];
        const entry = new ModelEntry(client, model_cfg.name, model_cfg.aliases, rungs);
        const signal = new AbortController().signal;
        await entry.loadWithKV(signal);
        await entry.applyNextStrategy(signal);
        rungs[1].n_ctx_cap = (await client.getSlots(entry.name, signal))[0].n_ctx;
        assert.ok(rungs[1].n_ctx_cap > 1024, `q8 rung must exceed 1024 tokens, got ${rungs[1].n_ctx_cap}`);
        await entry.unloadHard();
        await entry.loadWeights(signal);
        await entry.moveToRung(0, signal);
        entry.n_ctx = await client.reloadModel({ ...entry.paramsForRung(0), n_ctx: 1024 }, entry.name, signal);
        rungs[0].n_ctx_cap = (await client.getSlots(entry.name, signal))[0].n_ctx;
        assert.ok(rungs[0].n_ctx_cap >= 1024 && rungs[0].n_ctx_cap < rungs[1].n_ctx_cap);

        const initial_body = {
            model: entry.name,
            messages: [{ role: 'user', content: 'Say hello in one short sentence.' }],
            stream: false,
            max_tokens: 32,
            temperature: 0,
            chat_template_kwargs: { enable_thinking: false },
        };
        const tokens = await client.countTokens(initial_body, entry.name, signal);
        assert.ok(tokens < rungs[0].n_ctx_cap - 32,
            `initial request requires ${tokens} tokens; baseline capacity is ${rungs[0].n_ctx_cap}`);

        const planner = new Planner(client, config, new Map([[entry.name, entry]]));
        const memory = await client.getMemory(entry.name, signal);
        const gpu = memory.devices.find(device => device.type !== 'cpu');
        assert.ok(gpu, 'no GPU memory reported');
        planner.dev_info.bytes_total = gpu.total;
        planner.dev_info.bytes_avail = gpu.free;
        planner.active.model = entry.name;

        const attempts: { rung: number; status: number; body: unknown; response: string }[] = [];
        await planner.serveModel(initial_body, entry, signal, async (_body, completion_client, request_signal, isFinal) => {
            assert.equal(isFinal, false, 'planner could not grow the context for the continuation');
            if (attempts.length === 0) {
                assert.equal(entry.ladder_i, 0);
                const response = await completion_client.completions(initial_body, entry.name, request_signal);
                const text = await response.text();
                attempts.push({ rung: entry.ladder_i, status: response.status, body: initial_body, response: text });
                await writeFile(trace_path, JSON.stringify(attempts, null, 2));
                assert.ok(response.ok, `initial completion failed (${response.status}): ${text}`);
                const result = JSON.parse(text) as { choices?: { message?: unknown }[] };
                assert.ok(result.choices?.[0]?.message, `initial completion has no assistant message: ${text}`);
                return false;
            }

            assert.equal(entry.ladder_i, 1, 'planner did not advance to the next rung');
            const continuation_body = {
                ...initial_body,
                messages: [
                    ...initial_body.messages,
                    {
                        role: 'assistant',
                        content: '',
                        tool_calls: [{
                            id: 'call_partial',
                            type: 'function',
                            function: { name: 'say_hello', arguments: '{"name":"hel' },
                        }],
                    },
                ],
                tools: [{
                    type: 'function',
                    function: {
                        name: 'say_hello',
                        description: 'Say hello to someone.',
                        parameters: {
                            type: 'object',
                            properties: { name: { type: 'string' } },
                            required: ['name'],
                        },
                    },
                }],
                tool_choice: 'required',
                continue_final_message: true,
                add_generation_prompt: false,
            };
            const response = await completion_client.completions(continuation_body, entry.name, request_signal);
            const text = await response.text();
            attempts.push({ rung: entry.ladder_i, status: response.status, body: continuation_body, response: text });
            await writeFile(trace_path, JSON.stringify(attempts, null, 2));
            assert.ok(response.ok, `tool-call continuation rejected (${response.status}): ${text}`);
            const result = JSON.parse(text) as { choices?: { message?: unknown }[] };
            assert.ok(result.choices?.[0]?.message, `tool-call continuation has no assistant message: ${text}`);
            return true;
        });
        assert.equal(attempts.length, 2, 'expected an initial completion and one continuation');
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
