import assert from 'node:assert/strict';
import { once } from 'node:events';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import type { AddressInfo } from 'node:net';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { ApiServer } from '../../api/server.js';
import type { CompletionChunk, CompletionRequest } from '../../api/types.js';
import { buildConfig } from '../../config/build.js';
import { parseArgs } from '../../config/parser.js';
import { RouterProcess } from '../../client/router-process.js';
import { ModelEntry } from '../../planner/model-entry.js';
import { Planner, type PlannerResult } from '../../planner/planner.js';
import { createStrategies } from '../../planner/strategies/index.js';
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

type StreamFrame = CompletionChunk & { done?: boolean };

type UpstreamAttempt = {
    rung: number;
    body: CompletionRequest;
    status: number;
    text: string;
};

type CallbackAttempt = {
    rung: number;
    is_final: boolean;
    result: PlannerResult['kind'];
};

function parseFrames(text: string): StreamFrame[] {
    return text.split(/\r?\n\r?\n/).flatMap(event => {
        const data = event.split(/\r?\n/).find(line => line.startsWith('data: '))?.slice(6);
        if (!data) return [];
        if (data === '[DONE]') return [{ choices: [], done: true }];
        return [JSON.parse(data) as StreamFrame];
    });
}

function toolDeltas(frames: StreamFrame[]): unknown[] {
    return frames.flatMap(frame => frame.choices?.flatMap(choice => choice.delta?.tool_calls ?? []) ?? []);
}

function finishReasons(frames: StreamFrame[]): string[] {
    return frames.flatMap(frame => frame.choices?.flatMap(choice => choice.finish_reason ? [choice.finish_reason] : []) ?? []);
}

async function findBody(model: ModelEntry, base: Record<string, unknown>, payload: string, margin: number, signal: AbortSignal): Promise<{ body: CompletionRequest; tokens: number }> {
    const target = 1024 - margin;
    let low = 0;
    let high = target * 2;
    let best: { body: CompletionRequest; tokens: number } | undefined;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const body = {
            ...base,
            messages: [{ role: 'user', content: `Ignore this context filler: ${'hello '.repeat(mid)}\nUse the record_text tool to record this exact text without summarizing or shortening it: ${payload}` }],
        };
        const tokens = await model.countTokens(body, signal);
        assert.ok(tokens !== null, `countTokens failed for model: ${model.name}`);
        if (tokens <= target) {
            best = { body, tokens };
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    assert.ok(best, `request exceeds baseline capacity with a margin of ${margin} tokens`);
    assert.ok(best.tokens > target - 8, `unable to place prompt near the baseline boundary: ${best.tokens} vs ${target}`);
    best.body.model = model.curVariant();
    return best;
}

test('GPU: manager discards an interrupted tool call and streams the regenerated call after growth', { timeout: 1200000 }, async () => {
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
    const resources = new GpuTestResources(directory);
    const config_path = join(directory, 'config.yaml');
    const preset_path = join(directory, 'preset.ini');
    const trace_path = join(directory, 'trace.json');
    let router: RouterProcess | undefined;
    let planner: Planner | undefined;
    let api: ApiServer | undefined;
    let succeeded = false;
    const attempts: { margin: number; tokens: number; upstream: UpstreamAttempt[]; callbacks: CallbackAttempt[]; outward_status: number; outward_text: string }[] = [];

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
        config.port = 0;
        router = new RouterProcess(config.router, config.model_load);
        resources.trackRouter(router);
        const client = await router.start(preset_path);
        const model_cfg = config.models[config.default_model];
        assert.ok(model_cfg);
        const strategies = createStrategies(client);
        assert.ok(strategies.get('quantize-kv-q8'));
        const entry = new ModelEntry(
            client, 
            model_cfg.name, 
            model_cfg.aliases, 
            model_cfg.initial_state,
            model_cfg.ladder,
            strategies,
        );
        const rungs = entry.ladder;
        const signal = resources.signal;
        await entry.loadWeights(signal);
        await entry.applyRung(0, signal);
        await entry.applyNextRung(signal);
        rungs[1].n_ctx_cap = (await entry.getSlots(signal))[0].n_ctx;
        assert.ok(rungs[1].n_ctx_cap > 2048, `q8 rung needs enough room to complete a tool call: ${rungs[1].n_ctx_cap}`);
        await entry.unloadHard();
        await entry.loadWeights(signal);
        await entry.applyRung(0, signal);
        entry.n_ctx = await client.reloadModel({ ...entry.paramsForRung(0), n_ctx: 1024 }, entry.name, signal);
        rungs[0].n_ctx_cap = (await entry.getSlots(signal))[0].n_ctx;
        assert.equal(rungs[0].n_ctx_cap, 1024);

        planner = new Planner(client, config, new Map([[entry.name, entry]]));
        resources.trackPlanner(planner);
        const memory = await entry.getMemory(signal);
        const gpu = memory.devices.find(device => device.type !== 'cpu');
        assert.ok(gpu, 'no GPU memory reported');
        planner.dev_info.bytes_total = gpu.total;
        planner.dev_info.bytes_avail = gpu.free;
        planner.active.model = entry.name;

        let current_attempt: typeof attempts[number] | undefined;
        const captures: Promise<void>[] = [];
        const complete = client.completions.bind(client);
        client.completions = async (body, model, request_signal) => {
            const rung = entry.ladder_i;
            const response = await complete(body, model, request_signal);
            const upstream: UpstreamAttempt = { rung, body: structuredClone(body) as CompletionRequest, status: response.status, text: '' };
            current_attempt?.upstream.push(upstream);
            captures.push(response.clone().text().then(text => { upstream.text = text; }));
            return response;
        };

        const serve_model = planner.serveModel.bind(planner);
        planner.serveModel = async (body, model, task, client_signal, cb) => serve_model(body, model, task, client_signal, async (request_body, request_client, request_signal, is_final) => {
            const rung = entry.ladder_i;
            const result = await cb(request_body, request_client, request_signal, is_final);
            current_attempt?.callbacks.push({ rung, is_final, result: result.kind });
            return result;
        });

        api = new ApiServer(planner, config);
        resources.trackApi(api);
        await api.start();
        const server = api.server!;
        if (!server.listening) await once(server, 'listening');
        const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1/chat/completions`;
        const payload = 'The quick brown fox jumps over the lazy dog. '.repeat(24);
        const base = {
            model: entry.name,
            stream: true,
            max_tokens: 512,
            temperature: 0,
            seed: 42,
            reasoning_effort: 'none',
            chat_template_kwargs: { enable_thinking: false },
            tools: [{
                type: 'function',
                function: {
                    name: 'record_text',
                    description: 'Record the supplied text without modification.',
                    parameters: {
                        type: 'object',
                        properties: { text: { type: 'string', description: 'The complete text to record, copied exactly from the user message.' } },
                        required: ['text'],
                    },
                },
            }],
            tool_choice: 'required',
        };

        let covered = false;
        for (const margin of [96, 128, 160, 224, 64]) {
            if (entry.ladder_i !== 0) {
                await entry.saveSlots(signal);
                await entry.applyRung(0, signal);
                await planner.cache.enforce();
            }
            entry.n_ctx = await client.reloadModel({ ...entry.paramsForRung(0), n_ctx: 1024 }, entry.name, signal);
            const { body, tokens } = await findBody(entry, base, payload, margin, signal);
            assert.ok(tokens < rungs[0].n_ctx_cap);
            const attempt = { margin, tokens, upstream: [] as UpstreamAttempt[], callbacks: [] as CallbackAttempt[], outward_status: 0, outward_text: '' };
            attempts.push(attempt);
            current_attempt = attempt;

            const response = await fetch(url, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify(body),
                signal,
            });
            attempt.outward_status = response.status;
            attempt.outward_text = await response.text();
            await Promise.all(captures.splice(0));
            current_attempt = undefined;
            await writeFile(trace_path, JSON.stringify(attempts, null, 2));

            const first = attempt.upstream[0];
            const second = attempt.upstream[1];
            if (!first || !second || first.status !== 200 || second.status !== 200) continue;
            const first_frames = parseFrames(first.text);
            const second_frames = parseFrames(second.text);
            if (first.rung !== 0 || second.rung !== 1 ||
                !finishReasons(first_frames).includes('length') || toolDeltas(first_frames).length === 0 ||
                !finishReasons(second_frames).includes('tool_calls')) continue;

            assert.equal(response.status, 200, attempt.outward_text);
            assert.equal(attempt.callbacks[0]?.rung, 0);
            assert.equal(attempt.callbacks[0]?.result, 'grow');
            assert.equal(attempt.callbacks[1]?.rung, 1);
            assert.equal(attempt.callbacks[1]?.result, 'done');
            assert.ok(attempt.callbacks.every(call => !call.is_final));
            assert.ok(!second.body.messages.at(-1)?.tool_calls, 'partial tool calls were included in the continuation request');
            assert.equal(second.body.tool_choice, 'required');
            assert.deepEqual(second.body.tools, base.tools);

            const outward = parseFrames(attempt.outward_text);
            assert.deepEqual(toolDeltas(outward), toolDeltas(second_frames), 'the outward stream included abandoned tool-call deltas');
            assert.deepEqual(finishReasons(outward), ['tool_calls']);
            const usage = outward.filter(frame => frame.usage);
            assert.equal(usage.length, 1);
            assert.equal(usage[0].usage?.prompt_tokens, tokens);
            assert.equal(usage[0].usage?.total_tokens, tokens + usage[0].usage!.completion_tokens);
            assert.equal(outward.filter(frame => frame.done).length, 1);
            const first_tool = first_frames.findIndex(frame => frame.choices?.[0]?.delta?.tool_calls?.length);
            const prefix_frames = first_frames.slice(0, first_tool);
            const first_content = prefix_frames.map(frame => frame.choices?.[0]?.delta?.content ?? '').join('');
            const first_reasoning = prefix_frames.map(frame => frame.choices?.[0]?.delta?.reasoning_content ?? '').join('');
            if (first_content || first_reasoning) {
                const outward_content = outward.map(frame => frame.choices?.[0]?.delta?.content ?? '').join('');
                const outward_reasoning = outward.map(frame => frame.choices?.[0]?.delta?.reasoning_content ?? '').join('');
                assert.ok(outward_content.startsWith(first_content));
                assert.ok(outward_reasoning.startsWith(first_reasoning));
                const continuation = second.body.messages.at(-1);
                assert.equal(continuation?.role, 'assistant');
                assert.ok(continuation.content?.includes(first_content));
                assert.ok(continuation.reasoning_content?.includes(first_reasoning));
            }
            covered = true;
            break;
        }

        assert.ok(covered, `no prompt margin produced a truncated tool call followed by a completed retry; see ${trace_path}`);
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
                await writeFile(trace_path, JSON.stringify(attempts, null, 2));
                console.error(`GPU test artifacts preserved: ${directory}`);
            }
        }
    }
});
