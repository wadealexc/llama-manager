import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { AddressInfo } from 'node:net';
import { test } from 'node:test';
import { ApiServer } from '../api/server.js';
import type { CompletionChunk, CompletionRequest } from '../api/types.js';
import type { ManagerConfig, ModelState } from '../config/types.js';
import { ModelEntry } from '../planner/model-entry.js';
import { Planner } from '../planner/planner.js';
import { LlamaAPIMock } from './llama-api-mock.js';

async function checkContinuation(stream: boolean, can_grow: boolean): Promise<void> {
    const directory = await mkdtemp(join(tmpdir(), 'llama-cache-api-'));
    const client = new LlamaAPIMock();
    client.slot_directory = directory;
    const state: ModelState = { model_variant: 'model', kv_unified: true, cache_type_k: 'f16', cache_type_v: 'f16' };
    const model = new ModelEntry(client, 'model', [], state, [], new Map());
    model.ladder[0].n_ctx_cap = 4096;
    if (can_grow) {
        model.ladder.push({
            strategy: 'quantize-kv-q8',
            variant_name: 'baseline',
            state: { ...state, cache_type_k: 'q8_0', cache_type_v: 'q8_0' },
            n_ctx_cap: 8192,
            bytes_needed: 0,
        });
    }
    const config: ManagerConfig = {
        mode: 'router',
        router: { bin: '', listen: '', llama_log_dir: '', slot_save_path: directory, poll_interval_ms: 0, poll_timeout_ms: 0, shutdown_grace_period_ms: 0 },
        host: '127.0.0.1',
        port: 0,
        sleep_idle_seconds: 0,
        cache_disk_mib: 1,
        model_load: { poll_interval_ms: 0, poll_timeout_ms: 0 },
        models: {},
        default_model: model.name,
    };
    const planner = new Planner(client, config, new Map([[model.name, model]]));
    const api = new ApiServer(planner, config);
    const signal = new AbortController().signal;
    const attempts: CompletionRequest[] = [];
    client.renderPrompt = async body => {
        const request = body as CompletionRequest;
        const continued = request.messages.at(-1)?.content === 'partial';
        const tokens = continued ? [1, 2, 4] : [1, 2];
        return { input_tokens: tokens.length, object: 'response.input_tokens', tokens, media: [] };
    };
    const completionResponse = (content: string, finish_reason: string, prompt_tokens: number, completion_tokens: number): Response => {
        const usage = { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens };
        if (!stream) {
            return Response.json({ choices: [{ index: 0, finish_reason, message: { role: 'assistant', content } }], usage });
        }
        const frames = [
            { choices: [{ index: 0, delta: { role: 'assistant', content } }] },
            { choices: [{ index: 0, delta: {}, finish_reason }] },
            { choices: [], usage },
        ];
        return new Response(frames.map(frame => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n', {
            headers: { 'content-type': 'text/event-stream' },
        });
    };
    client.completions = async body => {
        const request = body as CompletionRequest;
        attempts.push(structuredClone(request));
        if (attempts.length === 1) {
            client.setLiveSlots(['current partial']);
            client.live_metadata.set('model', { tokens: [1, 2, 4], media: [] });
            return completionResponse('partial', 'length', 2, 1);
        }
        assert.deepEqual(client.getLiveSlots(), ['current partial']);
        assert.equal(model.ladder_i, 1);
        return completionResponse(' final', 'stop', 3, 2);
    };
    try {
        await model.loadWeights(signal);
        await model.applyRung(0, signal);
        planner.active.model = model.name;
        client.setLiveSlots(['older original']);
        client.live_metadata.set('model', { tokens: [1, 2, 3], media: [] });
        await model.saveSlots(signal);
        await planner.cache.enforce();
        await api.start();
        if (!api.server!.listening) await once(api.server!, 'listening');
        const port = (api.server!.address() as AddressInfo).port;
        const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: 'model', messages: [{ role: 'user', content: 'question' }], max_tokens: 100, stream }),
        });
        assert.equal(response.status, 200);
        const text = await response.text();
        const expected_content = can_grow ? 'partial final' : 'partial';
        const expected_finish = can_grow ? 'stop' : 'length';
        const expected_usage = can_grow
            ? { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 }
            : { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 };
        if (stream) {
            const events = text.split('\n\n').filter(event => event.startsWith('data: ')).map(event => event.slice(6));
            assert.equal(events.filter(event => event === '[DONE]').length, 1);
            const frames = events.filter(event => event !== '[DONE]').map(event => JSON.parse(event) as CompletionChunk);
            assert.equal(frames.flatMap(frame => frame.choices.map(choice => choice.delta?.content ?? '')).join(''), expected_content);
            assert.deepEqual(frames.flatMap(frame => frame.choices.flatMap(choice => choice.finish_reason ? [choice.finish_reason] : [])), [expected_finish]);
            assert.deepEqual(frames.flatMap(frame => frame.usage ? [frame.usage] : []), [expected_usage]);
        } else {
            const result = JSON.parse(text) as CompletionChunk;
            assert.equal(result.choices[0].message?.content, expected_content);
            assert.equal(result.choices[0].finish_reason, expected_finish);
            assert.deepEqual(result.usage, expected_usage);
        }
        assert.equal(attempts.length, can_grow ? 2 : 1);
        if (can_grow) {
            assert.equal(attempts[1].continue_final_message, true);
            assert.equal(attempts[1].add_generation_prompt, false);
            assert.equal(attempts[1].max_tokens, 99);
        }
    } finally {
        await api.shutdown();
        await planner.shutdown();
        await rm(directory, { recursive: true, force: true });
    }
}

for (const stream of [false, true]) {
    for (const can_grow of [false, true]) {
        test(`API ${stream ? 'streaming' : 'non-streaming'} continuation ${can_grow ? 'prepares the retry and preserves final usage' : 'returns the pending response when growth is unavailable'}`, async () => {
            await checkContinuation(stream, can_grow);
        });
    }
}
