import assert from 'node:assert/strict';
import { once } from 'node:events';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import type { AddressInfo } from 'node:net';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { ApiServer } from '../../api/server.js';
import type { CompletionChunk, CompletionRequest } from '../../api/types.js';
import { RouterProcess } from '../../client/router-process.js';
import type { InputTokensResponse, SlotRestore } from '../../client/types.js';
import { buildConfig } from '../../config/build.js';
import { parseArgs } from '../../config/parser.js';
import { ModelEntry } from '../../planner/model-entry.js';
import { Planner } from '../../planner/planner.js';
import { commonPrefix } from '../../planner/prompt-cache.js';
import { createStrategies } from '../../planner/strategies/index.js';
import { GpuTestResources } from './resources.js';

const BASELINE_CTX = 1024;
const Q8_CTX = 8192;
const MAX_TOKENS = 64;

type CompletionResponse = CompletionChunk & {
    timings: { prompt_n: number; cache_n: number };
};

function requiredPath(key: string): string {
    const value = process.env[key]?.trim();
    assert.ok(value, `set ${key} in .env.gpu-test`);
    const path = resolve(value);
    assert.ok(existsSync(path), `${key} does not exist: ${path}`);
    return path;
}

async function findLongPrompt(model: ModelEntry, base: CompletionRequest, signal: AbortSignal): Promise<{
    body: CompletionRequest;
    prompt: InputTokensResponse;
}> {
    let low = 0;
    let high = Q8_CTX;
    let best: { body: CompletionRequest; prompt: InputTokensResponse } | undefined;
    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const body = {
            ...base,
            messages: [
                { role: 'system', content: 'Answer briefly. Do not repeat the supplied reference text.' },
                { role: 'user', content: `Reference text: ${'The quick brown fox jumps over the lazy dog. '.repeat(mid)}\nWhat is two plus two?` },
            ],
        };
        const prompt = await model.renderPrompt(body, signal);
        assert.ok(prompt, 'unable to render long prompt');
        if (prompt.input_tokens <= 3072) {
            best = { body, prompt };
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    assert.ok(best && best.prompt.input_tokens > BASELINE_CTX * 2, 'unable to construct a prompt that requires q8');
    return best;
}

test('GPU: long conversation survives a baseline title detour with main-model cache reuse', { timeout: 1200000 }, async () => {
    const root = resolve(import.meta.dirname, '../../..');
    const model_path = requiredPath('MODEL');
    const default_bin = process.platform === 'win32'
        ? './llama.cpp/build/bin/Release/llama-server.exe'
        : './llama.cpp/build/bin/llama-server';
    const bin = resolve(root, process.env.LLAMA_BIN?.trim() || default_bin);
    assert.ok(existsSync(bin), `llama-server binary does not exist: ${bin}`);
    process.env.LLAMA_ATTN_ROT_DISABLE = '1';

    const artifacts_dir = join(root, 'logs', 'gpu-tests');
    await mkdir(artifacts_dir, { recursive: true });
    const directory = await mkdtemp(join(artifacts_dir, 'prompt-cache-'));
    const resources = new GpuTestResources(directory);
    const signal = resources.signal;
    const trace: Record<string, unknown> = {};
    let succeeded = false;

    try {
        const config_path = join(directory, 'config.yaml');
        const preset_path = join(directory, 'preset.ini');
        await writeFile(config_path, JSON.stringify({
            router: { 'llama-log-dir': join(directory, 'logs') },
        }));
        const config = await buildConfig(parseArgs([
            '--config', config_path,
            '--model', model_path,
            '--alias', 'gpu-test-prompt-cache',
            '--bin', bin,
            '--slot-save-path', join(directory, 'slots'),
            '--fit-target', process.env.FIT_TARGET?.trim() || '512',
            '--n-gpu-layers', process.env.N_GPU_LAYERS?.trim() || '99',
            '--parallel', '1',
            '--batch-size', '256',
            '--ubatch-size', '256',
            '--checkpoint-min-step', '0',
            '--cache-ram', '0',
            '--no-cache-idle-slots',
            '--no-mmproj',
            '--spec-type', 'none',
            '--jinja',
            '--ladder', 'quantize-kv-q8',
        ]), root, preset_path, config_path);
        config.router.poll_timeout_ms = 30000;
        config.model_load.poll_timeout_ms = 180000;
        config.sleep_idle_seconds = 0;
        config.port = 0;

        const router = new RouterProcess(config.router, config.model_load);
        resources.trackRouter(router);
        const client = await router.start(preset_path);
        const model_cfg = config.models[config.default_model];
        const entry = new ModelEntry(client, model_cfg.name, model_cfg.aliases, model_cfg.initial_state,
            model_cfg.ladder, createStrategies(client));
        assert.equal(entry.ladder.length, 2);

        const reload_model = client.reloadModel.bind(client);
        client.reloadModel = async (params, model_id, request_signal) => reload_model({
            ...params,
            n_ctx: params.n_ctx === 0
                ? params.cache_type_k === 'q8_0' ? Q8_CTX : BASELINE_CTX
                : params.n_ctx,
        }, model_id, request_signal);
        await entry.loadWeights(signal);
        await entry.applyRung(0, signal);
        entry.ladder[0].n_ctx_cap = (await entry.getSlots(signal))[0].n_ctx;
        await entry.applyRung(1, signal);
        entry.ladder[1].n_ctx_cap = (await entry.getSlots(signal))[0].n_ctx;
        assert.equal(entry.getCtxCap(0), BASELINE_CTX);
        assert.equal(entry.getCtxCap(1), Q8_CTX);
        await entry.applyRung(0, signal);

        const planner = new Planner(client, config, new Map([[entry.name, entry]]));
        resources.trackPlanner(planner);
        await planner.cache.start();
        planner.active.model = entry.name;
        const memory = await entry.getMemory(signal);
        const gpu = memory.devices.find(device => device.type !== 'cpu');
        assert.ok(gpu, 'no GPU memory reported');
        planner.dev_info.bytes_total = gpu.total;
        planner.dev_info.bytes_avail = gpu.free;

        const restores: SlotRestore[] = [];
        const restore_slot = client.restoreSlot.bind(client);
        client.restoreSlot = async (model_id, slot_id, filename, request_signal) => {
            const restored = await restore_slot(model_id, slot_id, filename, request_signal);
            restores.push(restored);
            return restored;
        };
        const api = new ApiServer(planner, config);
        resources.trackApi(api);
        await api.start();
        if (!api.server!.listening) await once(api.server!, 'listening');
        const port = (api.server!.address() as AddressInfo).port;
        const url = `http://127.0.0.1:${port}/v1/chat/completions`;
        const complete = async (body: CompletionRequest): Promise<CompletionResponse> => {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify(body),
                signal,
            });
            const text = await response.text();
            assert.equal(response.status, 200, text);
            const result = JSON.parse(text) as CompletionResponse;
            assert.ok(result.choices?.[0]?.message, text);
            assert.ok(!('error' in result), text);
            await planner.cache.enforce();
            return result;
        };
        const base: CompletionRequest = {
            model: entry.name,
            messages: [],
            stream: false,
            max_tokens: MAX_TOKENS,
            temperature: 0,
            reasoning_effort: 'none',
            chat_template_kwargs: { enable_thinking: false },
            cache_prompt: true,
        };

        const long = await findLongPrompt(entry, base, signal);
        const first = await complete(long.body);
        assert.equal(entry.ladder_i, 1);
        assert.equal(first.choices[0].finish_reason, 'stop', 'initial answer must finish within its token budget');
        const assistant = first.choices[0].message!;
        assert.ok(assistant.content, 'initial answer must include text');
        assert.ok(!assistant.tool_calls?.length, 'this regression uses plain chat messages');
        trace.first = first;
        trace.long_input_tokens = long.prompt.input_tokens;

        const title_body: CompletionRequest = {
            ...base,
            messages: [
                { role: 'system', content: 'Return only a short title.' },
                { role: 'user', content: 'Give a three-word title for a conversation about basic arithmetic.' },
            ],
        };
        const title_prompt = await entry.renderPrompt(title_body, signal);
        assert.ok(title_prompt && title_prompt.input_tokens + MAX_TOKENS < BASELINE_CTX);
        const title = await complete(title_body);
        assert.equal(entry.ladder_i, 0);
        trace.title = title;
        const long_entry = planner.cache.entries.find(snapshot => snapshot.rung === 1
            && commonPrefix(snapshot, long.prompt) > BASELINE_CTX);
        assert.ok(long_entry, 'downshift must save the populated long conversation');
        assert.ok(long_entry.tokens.length > BASELINE_CTX, 'long snapshot must not fit baseline');

        const followup_body: CompletionRequest = {
            ...long.body,
            messages: [...long.body.messages, {
                role: 'assistant',
                content: assistant.content,
                reasoning_content: assistant.reasoning_content,
            }, { role: 'user', content: 'And what is three plus three? Answer briefly.' }],
        };
        const followup_prompt = await entry.renderPrompt(followup_body, signal);
        assert.ok(followup_prompt && followup_prompt.input_tokens > BASELINE_CTX
            && followup_prompt.input_tokens + MAX_TOKENS < Q8_CTX);
        const expected_prefix = commonPrefix(long_entry, followup_prompt);
        assert.ok(expected_prefix >= long.prompt.input_tokens - 64, 'followup must retain the original long prefix');
        restores.length = 0;
        const followup = await complete(followup_body);
        trace.followup = followup;
        trace.expected_prefix = expected_prefix;
        trace.restores = restores;
        trace.entries = planner.cache.entries.map(snapshot => ({
            filename: snapshot.filename, rung: snapshot.rung, n_tokens: snapshot.tokens.length,
        }));
        await writeFile(join(directory, 'trace.json'), JSON.stringify(trace, null, 2));

        assert.equal(entry.ladder_i, 1);
        assert.equal(followup.choices[0].finish_reason, 'stop');
        assert.ok(followup.choices[0].message?.content, 'followup must complete successfully');
        const title_entry = planner.cache.entries.find(snapshot => snapshot.rung === 0
            && commonPrefix(snapshot, title_prompt) > 0);
        assert.ok(title_entry, 'upshift must save the independent title state');
        assert.notEqual(title_entry.filename, long_entry.filename);
        assert.ok(restores.some(restored => restored.filename === long_entry.filename && restored.n_restored > BASELINE_CTX),
            'followup must restore the long conversation');
        assert.ok(restores.every(restored => restored.filename !== title_entry.filename), 'followup must not restore the title');

        const cached = followup.usage?.prompt_tokens_details?.cached_tokens;
        assert.ok(cached !== undefined && followup.timings, 'upstream must expose main-model reuse counters');
        assert.equal(cached, followup.timings.cache_n);
        assert.ok(cached >= expected_prefix - 512, `too little main-model reuse: ${cached} cached vs ${expected_prefix} matching`);
        assert.ok(followup.timings.prompt_n <= followup_prompt.input_tokens - expected_prefix + 512,
            `unexpected full prefill: ${followup.timings.prompt_n} processed of ${followup_prompt.input_tokens}`);
        console.log(`GPU title detour covered: ${cached} main-model tokens cached, ${followup.timings.prompt_n} processed`);
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
                await writeFile(join(directory, 'trace.json'), JSON.stringify(trace, null, 2));
                console.error(`GPU test artifacts preserved: ${directory}`);
            }
        }
    }
});
