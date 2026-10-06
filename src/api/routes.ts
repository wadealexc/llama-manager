import type { Express, Request, Response as ExpressResponse } from "express";
import type { ApiServer } from "./server.js";
import { SSERelay } from "./sse-relay.js";
import type { CompletionChunk, CompletionRequest } from "./types.js";
import type { ModelEntry } from "../planner/model-entry.js";
import { logger } from "../logger.js";
import { ModelTask, type PlannerResult } from "../planner/planner.js";

const log = logger.withTag('api');

export function registerRoutes(app: Express, server: ApiServer): void {
    app.get('/v1/models', (req, res) => listModels(server, req, res));
    app.post('/v1/models/load', (req, res) => loadModel(server, req, res));
    app.post('/v1/chat/completions', (req, res) => completions(server, req, res));
    app.post('/v1/chat/completions/input_tokens', (req, res) => countTokens(server, req, res));
}

// TODO - clean up status return
async function listModels(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const data = [...server.planner.models.values()]
        .filter((e): e is NonNullable<typeof e> => !!e)
        .map(e => ({
            id: e.name,
            object: 'model',
            created: 0,
            owned_by: 'llama-manager',
            aliases: e.aliases,
            status: e.status,
            queued: server.planner.isModelQueued(e.name),
            active: server.planner.active.model === e.name && !server.planner.active.pending,
            n_ctx: e.getMaxCtx() ?? e.getCurCtx(),
        }));

    res.json({ object: 'list', data });
}

async function loadModel(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const model_name = req.body?.model;
    if (typeof model_name !== 'string') {
        sendError(res, 400, 'invalid_request', 'missing or invalid "model" field');
        return;
    }

    if (!server.planner.resolve(model_name)) {
        sendError(res, 404, 'invalid_request', `unable to resolve model ${model_name}`);
        return;
    }

    const ac = new AbortController();
    res.on('close', () => ac.abort());
    req.on('aborted', () => ac.abort());

    await server.planner.serveModel({ messages: [] }, model_name, ModelTask.WAKE, ac.signal, async (_b: unknown, model: ModelEntry, signal: AbortSignal, _isFinal: boolean) => {
        res.json({ success: true });
        return { kind: 'done' };
    });
}

async function countTokens(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const body = req.body as CompletionRequest | undefined;
    if (body === undefined) {
        sendError(res, 400, 'invalid_request', 'expected request body');
        return;
    }

    const model_name = body.model;
    if (typeof model_name !== 'string') {
        sendError(res, 400, 'invalid_request', 'missing or invalid "model" field');
        return;
    }

    if (!server.planner.resolve(model_name)) {
        sendError(res, 404, 'invalid_request', `unable to resolve model ${model_name}`);
        return;
    }

    const ac = new AbortController();
    res.on('close', () => ac.abort());
    req.on('aborted', () => ac.abort());

    await server.planner.serveModel(body, model_name, ModelTask.TOKENIZE, ac.signal, async (_body: unknown, model: ModelEntry, signal: AbortSignal, _isFinal: boolean) => {
        let tokens;
        try {
            tokens = await model.renderPrompt(body, signal);
            if (tokens === null) throw new Error(`error counting tokens for: ${model_name}`);
        } catch (err) {
            sendUpstreamError(res, false, err);
            return { kind: 'done' };
        }

        res.json(tokens);
        return { kind: 'done' };
    });
}

async function completions(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const body = req.body as CompletionRequest | undefined;
    if (body === undefined) {
        sendError(res, 400, 'invalid_request', 'expected request body');
        return;
    }

    const model_name = body.model;
    if (typeof model_name !== 'string') {
        sendError(res, 400, 'invalid_request', 'missing or invalid "model" field');
        return;
    }

    const logical_model = server.planner.resolve(model_name);
    if (!logical_model) {
        sendError(res, 404, 'invalid_request', `unable to resolve model ${model_name}`);
        return;
    }

    const is_stream = body.stream === true;

    const ac = new AbortController();
    res.on('close', () => {
        if (!res.writableEnded) ac.abort();
    });
    req.on('aborted', () => ac.abort());

    let headers_set = false;
    let prepared_retry: ReqInfo | undefined;
    const req_info: ReqInfo = {
        body,
        logical_model: logical_model.name,
        outcome: 'error',
        completion_tokens_total: 0,
        isFinal: false,
    };

    await server.planner.serveModel(body, model_name, ModelTask.COMPLETIONS, ac.signal, async (request_body: unknown, model: ModelEntry, signal: AbortSignal, isFinal: boolean): Promise<PlannerResult> => {
        if (res.writableEnded) return { kind: 'done' };

        if (prepared_retry && !isFinal) Object.assign(req_info, prepared_retry);
        prepared_retry = undefined;
        req_info.body = request_body as CompletionRequest;
        req_info.isFinal = isFinal;

        if (isFinal && req_info.pending) {
            log.info(`${model.curVariant()}: context growth unavailable; returning pending response`);
            finishPending(req_info, res);
            return { kind: 'done' };
        }

        if (!headers_set) {
            if (is_stream) {
                res.setHeader('content-type', 'text/event-stream');
                res.setHeader('cache-control', 'no-cache');
                res.setHeader('connection', 'keep-alive');
            } else {
                res.setHeader('content-type', 'application/json');
            }
            res.status(200);
            res.flushHeaders();
            headers_set = true;
        }

        req_info.upstream_usage = undefined;
        req_info.finish_reason = undefined;
        req_info.physical_model = model.curVariant();

        let upstream: Response;
        try {
            const req_body = is_stream
                ? { ...(req_info.body as {}), stream_options: { include_usage: true } }
                : req_info.body;
            upstream = await model.completions(req_body, signal);
        } catch (err) {
            sendUpstreamError(res, is_stream, err);
            return { kind: 'done' };
        }

        if (!upstream.ok || (is_stream && !upstream.body)) {
            const text = await upstream.text().catch(() => '');
            if (isContextExceeded(text) && !req_info.isFinal) {
                log.info(`${req_info.physical_model}: context exceeded; requesting growth`);
                return { kind: 'grow', body: req_info.body };
            }

            sendUpstreamError(res, is_stream, text || `upstream returned ${upstream.status}`);
            return { kind: 'done' };
        }

        const attempt = is_stream
            ? await streamCompletion(req_info, upstream, res)
            : await nonStreamCompletion(req_info, upstream, res);
        if (attempt === 'done') return { kind: 'done' };

        prepared_retry = structuredClone(req_info);
        try {
            const exhausted = await continuePending(prepared_retry, model, signal);
            if (exhausted) {
                finishAtTokenLimit(prepared_retry, res, exhausted);
                Object.assign(req_info, prepared_retry);
                prepared_retry = undefined;
                return { kind: 'done' };
            }

            return { kind: 'grow', body: prepared_retry.body };
        } catch (err) {
            prepared_retry = undefined;
            sendUpstreamError(res, is_stream, err);
            return { kind: 'done' };
        }
    }).finally(() => {
        const outcome = !res.writableEnded && ac.signal.aborted ? 'aborted' : req_info.outcome;
        logCompletion(req_info, req_info.finish_reason ?? outcome, true);
    });
}

type ReqInfo = {
    body: CompletionRequest;
    logical_model: string;
    physical_model?: string;
    input_tokens?: number;
    upstream_usage?: CompletionChunk['usage'];
    final_usage?: CompletionChunk['usage'];
    finish_reason?: string;
    outcome: 'completed' | 'error';
    completion_tokens_total: number;
    continued?: boolean;
    original_prompt_tokens?: number;
    pending?: PendingAttempt;
    isFinal: boolean;
};

type PendingAttempt = {
    content: string;
    reasoning: string;
} & ({
    kind: 'stream';
    frames: string[];
    finish: CompletionChunk;
    usage?: CompletionChunk;
} | {
    kind: 'non-stream';
    response: CompletionChunk;
});

async function streamCompletion(
    req: ReqInfo,
    upstream: Response,
    res: ExpressResponse,
): Promise<'done' | 'truncated'> {
    let partial_content = "";
    let partial_reasoning = "";
    const buffered_frames: string[] = [];
    let buffering = false;
    let pending_finish: CompletionChunk | undefined;

    try {
        const relay = new SSERelay(upstream.body!);
        for await (const frame of relay) {
            if (frame.done) {
                if (pending_finish) {
                    finishStream(req, res, buffered_frames, pending_finish);
                } else {
                    flushFrames(res, buffered_frames);
                    req.outcome = 'completed';
                    finishSSEResponse(res);
                }
                return 'done';
            }

            const json = frame.data as CompletionChunk;
            json.model = req.logical_model;
            captureUsage(req, json, pending_finish?.choices[0]?.finish_reason);
            const choice = json.choices[0];

            if (pending_finish) {
                const completion_tokens = json.usage?.completion_tokens ?? 0;
                if (!req.isFinal && isTruncated(completion_tokens, pending_finish.choices[0]?.finish_reason, req.body.max_tokens)) {
                    req.pending = {
                        kind: 'stream',
                        content: partial_content,
                        reasoning: partial_reasoning,
                        frames: buffered_frames,
                        finish: pending_finish,
                        usage: json,
                    };
                    return 'truncated';
                }

                finishStream(req, res, buffered_frames, pending_finish, json);
                return 'done';
            }

            if (!choice) continue;

            if (choice.finish_reason) {
                pending_finish = json;
                continue;
            }

            const delta = choice.delta;
            if (delta?.tool_calls?.length && !buffering && (delta.content || delta.reasoning_content)) {
                const prefix = { ...json, choices: [{ ...choice, delta: { ...delta, tool_calls: undefined } }] };
                res.write(`data: ${JSON.stringify(prefix)}\n\n`);
                partial_content += delta.content ?? "";
                partial_reasoning += delta.reasoning_content ?? "";
                const tool = { ...json, choices: [{ ...choice, delta: { ...delta, content: undefined, reasoning_content: undefined } }] };
                buffered_frames.push(JSON.stringify(tool));
                buffering = true;
            } else if (buffering || delta?.tool_calls?.length) {
                buffering = true;
                buffered_frames.push(JSON.stringify(json));
            } else {
                res.write(`data: ${JSON.stringify(json)}\n\n`);
                partial_content += delta?.content ?? "";
                partial_reasoning += delta?.reasoning_content ?? "";
            }
        }

        if (pending_finish) {
            finishStream(req, res, buffered_frames, pending_finish);
        } else {
            flushFrames(res, buffered_frames);
            req.outcome = 'completed';
            finishSSEResponse(res);
        }
        return 'done';
    } catch (err) {
        if (res.writableEnded) return 'done';
        sendUpstreamError(res, true, err);
        return 'done';
    }
}

async function nonStreamCompletion(
    req: ReqInfo,
    upstream: Response,
    res: ExpressResponse,
): Promise<'done' | 'truncated'> {
    let json: CompletionChunk;
    try {
        json = await upstream.json() as CompletionChunk;
    } catch (err) {
        sendUpstreamError(res, false, err);
        return 'done';
    }

    json.model = req.logical_model;
    captureUsage(req, json);
    const choice = json.choices[0];
    const finish_reason = choice?.finish_reason;
    const completion_tokens = json.usage?.completion_tokens ?? 0;

    if (!req.isFinal && isTruncated(completion_tokens, finish_reason, req.body.max_tokens)) {
        req.pending = {
            kind: 'non-stream',
            content: choice?.message?.content ?? "",
            reasoning: choice?.message?.reasoning_content ?? "",
            response: json,
        };
        return 'truncated';
    }

    correctUsage(req, json);

    if (req.continued) {
        const last = req.body.messages[req.body.messages.length - 1];
        if (last) {
            if (choice?.message?.content) {
                last.content = (last.content ?? "") + choice.message.content;
            }
            if (choice?.message?.reasoning_content) {
                last.reasoning_content = (last.reasoning_content ?? "") + choice.message.reasoning_content;
            }
            if (json.choices[0].message) {
                json.choices[0].message.content = last.content as string;
                json.choices[0].message.reasoning_content = last.reasoning_content as string | undefined;
            }
        }
    }

    req.outcome = 'completed';
    res.write(JSON.stringify(json));
    res.end();
    return 'done';
}

function captureUsage(req: ReqInfo, chunk: CompletionChunk, finish_reason?: string): void {
    req.finish_reason = chunk.choices[0]?.finish_reason ?? finish_reason ?? req.finish_reason;
    if (!chunk.usage || req.upstream_usage) return;

    req.upstream_usage = structuredClone(chunk.usage);
    req.input_tokens ??= chunk.usage.prompt_tokens;
}

function logCompletion(req: ReqInfo, finish_reason: string, is_final: boolean = false): void {
    const usage = req.upstream_usage;
    const input_tokens = is_final ? req.input_tokens : usage?.prompt_tokens;
    const output_tokens = is_final
        ? req.final_usage?.completion_tokens ?? (usage ? req.completion_tokens_total + usage.completion_tokens : undefined)
        : usage?.completion_tokens;
    const cached_tokens = usage?.prompt_tokens_details?.cached_tokens;
    log.info(`${req.physical_model ?? req.logical_model}: finish (${finish_reason}) | input ${input_tokens ?? 'unknown'} toks (${cached_tokens ?? 'unknown'} cached) | output ${output_tokens ?? 'unknown'} toks`);
}

function logDiscardedToolCalls(req: ReqInfo, pending: PendingAttempt): void {
    const discarded = pending.kind === 'stream'
        ? pending.frames.length
        : pending.response.choices[0]?.message?.tool_calls?.length ?? 0;
    if (discarded > 0) {
        const unit = pending.kind === 'stream' ? 'buffered frames' : 'tool calls';
        log.info(`${req.physical_model}: discarded interrupted tool-call output (${discarded} ${unit})`);
    }
}

function flushFrames(res: ExpressResponse, frames: string[]): void {
    for (const frame of frames) {
        res.write(`data: ${frame}\n\n`);
    }
}

function correctUsage(req: ReqInfo, chunk: CompletionChunk): void {
    if (!chunk.usage) return;
    if (req.original_prompt_tokens !== undefined) {
        chunk.usage.prompt_tokens = req.original_prompt_tokens;
        chunk.usage.completion_tokens += req.completion_tokens_total;
        chunk.usage.total_tokens = chunk.usage.prompt_tokens + chunk.usage.completion_tokens;
    }
    req.final_usage = structuredClone(chunk.usage);
}

function finishStream(req: ReqInfo, res: ExpressResponse, frames: string[], finish: CompletionChunk, usage?: CompletionChunk): void {
    req.outcome = 'completed';
    flushFrames(res, frames);
    res.write(`data: ${JSON.stringify(finish)}\n\n`);
    if (usage) {
        correctUsage(req, usage);
        res.write(`data: ${JSON.stringify(usage)}\n\n`);
    }
    finishSSEResponse(res);
}

function finishPending(req: ReqInfo, res: ExpressResponse): void {
    const pending = req.pending!;
    req.pending = undefined;

    if (pending.kind === 'stream') {
        finishStream(req, res, pending.frames, pending.finish, pending.usage);
    } else {
        const last = req.body.messages.at(-1);
        const message = pending.response.choices[0]?.message;
        if (req.continued && last?.role === 'assistant' && message) {
            message.content = (last.content ?? "") + (message.content ?? "");
            message.reasoning_content = (last.reasoning_content ?? "") + (message.reasoning_content ?? "");
        }
        correctUsage(req, pending.response);
        req.outcome = 'completed';
        res.write(JSON.stringify(pending.response));
        res.end();
    }
}

async function continuePending(req: ReqInfo, model: ModelEntry, signal: AbortSignal): Promise<PendingAttempt | undefined> {
    const pending = req.pending!;
    const original = await model.countTokens(req.body, signal);
    if (original === null) throw new Error(`continuePending: error counting tokens`);
    req.original_prompt_tokens ??= original;

    if (!pending.content && !pending.reasoning) {
        logCompletion(req, req.finish_reason ?? 'unknown');
        logDiscardedToolCalls(req, pending);
        req.pending = undefined;
        return;
    }

    const base = structuredClone(req.body);
    if (!base.continue_final_message) {
        applyTruncation({ ...req, body: base }, "", "");
    }
    const before = await model.countTokens(base, signal);
    if (before === null) throw new Error(`continuePending: error counting tokens`);

    const next = structuredClone(req.body);
    applyTruncation({ ...req, body: next }, pending.content, pending.reasoning);
    const after = await model.countTokens(next, signal);
    if (after === null) throw new Error(`continuePending: error counting tokens`);
    const retained_tokens = Math.max(0, after - before);

    req.body = next;
    req.completion_tokens_total += retained_tokens;
    if (req.body.max_tokens !== undefined) {
        req.body.max_tokens = Math.max(0, req.body.max_tokens - retained_tokens);
    }
    req.continued = true;
    req.pending = undefined;
    if (req.body.max_tokens !== 0) logCompletion(req, req.finish_reason ?? 'unknown');
    logDiscardedToolCalls(req, pending);
    if (req.body.max_tokens === 0) return pending;
}

function finishAtTokenLimit(req: ReqInfo, res: ExpressResponse, pending: PendingAttempt): void {
    req.outcome = 'completed';
    const completion_tokens = req.completion_tokens_total;
    const prompt_tokens = req.original_prompt_tokens!;
    const usage = {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };
    req.final_usage = usage;

    if (pending.kind === 'stream') {
        res.write(`data: ${JSON.stringify(pending.finish)}\n\n`);
        res.write(`data: ${JSON.stringify({ ...(pending.usage ?? { choices: [] }), usage })}\n\n`);
        finishSSEResponse(res);
    } else {
        const response = pending.response;
        const message = req.body.messages.at(-1);
        if (response.choices[0]) {
            response.choices[0].message = {
                role: 'assistant',
                content: message?.content ?? "",
                reasoning_content: message?.reasoning_content,
            };
        }
        response.usage = usage;
        res.write(JSON.stringify(response));
        res.end();
    }
}

function isTruncated(completion_tokens: number, finish_reason?: string, max_tokens?: number): boolean {
    if (finish_reason !== "length") return false;
    return max_tokens === undefined || completion_tokens < max_tokens;
}

function isContextExceeded(text: string): boolean {
    try {
        const parsed = JSON.parse(text);
        return parsed?.error?.type === "exceed_context_size_error";
    } catch {
        return false;
    }
}

function applyTruncation(req: ReqInfo, content: string, reasoning: string): void {
    const messages = req.body.messages;
    const last = messages.at(-1);

    if (last?.role === 'assistant' && req.body.continue_final_message) {
        last.content = (last.content ?? "") + content;
        last.reasoning_content = (last.reasoning_content ?? "") + reasoning;
        delete last.tool_calls;
    } else {
        messages.push({ role: 'assistant', content, reasoning_content: reasoning });
    }

    req.body.continue_final_message = true;
    req.body.add_generation_prompt = false;
}

function sendError(res: ExpressResponse, status: number, type: string, message: string): void {
    res.status(status).json({ error: { message, type } });
}

function finishSSEResponse(res: ExpressResponse): void {
    res.write("data: [DONE]\n\n");
    res.end();
}

function sendUpstreamError(res: ExpressResponse, is_stream: boolean, err: unknown): void {
    const message = err instanceof Error ? (err.message || '(no message)') : String(err);
    const payload = JSON.stringify({ error: { message, type: 'upstream' } });
    if (is_stream) {
        res.write(`data: ${payload}\n\ndata: [DONE]\n\n`);
    } else {
        res.write(payload);
    }
    res.end();
}