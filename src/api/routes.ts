import type { Express, Request, Response as ExpressResponse } from "express";
import type { ApiServer } from "./server.js";
import type { Client } from "../planner/planner.js";
import { SSERelay } from "./sse-relay.js";
import type { CompletionChunk, CompletionRequest } from "./types.js";

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

    const model = server.planner.resolve(model_name);
    if (!model) {
        sendError(res, 404, 'invalid_request', `unable to resolve model ${model_name}`);
        return;
    }

    const ac = new AbortController();
    res.on('close', () => ac.abort());
    req.on('aborted', () => ac.abort());

    await server.planner.serveModel({ messages: [] }, model, ac.signal, async (_b: unknown, client: Client, signal: AbortSignal, _isFinal: boolean) => {
        res.json({ success: true });
        return true;
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

    const model = server.planner.resolve(model_name);
    if (!model) {
        sendError(res, 400, 'invalid_request', `unable to resolve model ${model_name}`);
        return;
    }

    const ac = new AbortController();
    res.on('close', () => ac.abort());
    req.on('aborted', () => ac.abort());

    await server.planner.serveModel(body, model, ac.signal, async (_body: unknown, client: Client, signal: AbortSignal, _isFinal: boolean) => {
        let tokens: number;
        try {
            tokens = await client.countTokens(body, model.name, signal);
        } catch (err) {
            sendUpstreamError(res, false, err);
            return true;
        }

        res.json({ input_tokens: tokens, object: 'response.input_tokens' });
        return true;
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

    const model = server.planner.resolve(model_name);
    if (!model) {
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
    const req_info: ReqInfo = {
        body,
        completion_tokens_total: 0,
        isFinal: false,
    };

    await server.planner.serveModel(body, model, ac.signal, async (_body: unknown, client: Client, signal: AbortSignal, isFinal: boolean) => {
        if (res.writableEnded) return true;

        req_info.isFinal = isFinal;

        if (isFinal && req_info.pending) {
            finishPending(req_info, res);
            return true;
        }

        if (req_info.pending) {
            try {
                const exhausted = await continuePending(req_info, client, model.name, signal);
                if (exhausted) {
                    finishAtTokenLimit(req_info, res, exhausted);
                    return true;
                }
            } catch (err) {
                sendUpstreamError(res, is_stream, err);
                return true;
            }
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

        let upstream: Response;
        try {
            const req_body = is_stream
                ? { ...(req_info.body as {}), stream_options: { include_usage: true } }
                : req_info.body;
            upstream = await client.completions(req_body, model.name, signal);
        } catch (err) {
            sendUpstreamError(res, is_stream, err);
            return true;
        }

        if (!upstream.ok || (is_stream && !upstream.body)) {
            const text = await upstream.text().catch(() => '');
            if (isContextExceeded(text) && !req_info.isFinal) {
                return false;
            }
            sendUpstreamError(res, is_stream, text || `upstream returned ${upstream.status}`);
            return true;
        }

        if (is_stream) {
            return await streamCompletion(req_info, upstream, res);
        } else {
            return await nonStreamCompletion(req_info, upstream, res);
        }
    });
}

type ReqInfo = {
    body: CompletionRequest;
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
): Promise<boolean> {
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
                    finishSSEResponse(res);
                }
                return true;
            }

            const json = frame.data as CompletionChunk;
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
                    return false;
                }

                finishStream(req, res, buffered_frames, pending_finish, json);
                return true;
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
            finishSSEResponse(res);
        }
        return true;
    } catch (err) {
        if (res.writableEnded) return true;
        sendUpstreamError(res, true, err);
        return true;
    }
}

async function nonStreamCompletion(
    req: ReqInfo,
    upstream: Response,
    res: ExpressResponse,
): Promise<boolean> {
    let json: CompletionChunk;
    try {
        json = await upstream.json() as CompletionChunk;
    } catch (err) {
        sendUpstreamError(res, false, err);
        return true;
    }

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
        return false;
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

    res.write(JSON.stringify(json));
    res.end();
    return true;
}

function flushFrames(res: ExpressResponse, frames: string[]): void {
    for (const frame of frames) {
        res.write(`data: ${frame}\n\n`);
    }
}

function correctUsage(req: ReqInfo, chunk: CompletionChunk): void {
    if (!chunk.usage || req.original_prompt_tokens === undefined) return;

    chunk.usage.prompt_tokens = req.original_prompt_tokens;
    chunk.usage.completion_tokens += req.completion_tokens_total;
    chunk.usage.total_tokens = chunk.usage.prompt_tokens + chunk.usage.completion_tokens;
}

function finishStream(req: ReqInfo, res: ExpressResponse, frames: string[], finish: CompletionChunk, usage?: CompletionChunk): void {
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
        res.write(JSON.stringify(pending.response));
        res.end();
    }
}

async function continuePending(req: ReqInfo, client: Client, model: string, signal: AbortSignal): Promise<PendingAttempt | undefined> {
    const pending = req.pending!;
    const original = await client.countTokens(req.body, model, signal);
    req.original_prompt_tokens ??= original;

    if (!pending.content && !pending.reasoning) {
        req.pending = undefined;
        return;
    }

    const base = structuredClone(req.body);
    if (!base.continue_final_message) {
        applyTruncation({ ...req, body: base }, "", "");
    }
    const before = await client.countTokens(base, model, signal);

    const next = structuredClone(req.body);
    applyTruncation({ ...req, body: next }, pending.content, pending.reasoning);
    const after = await client.countTokens(next, model, signal);
    const retained_tokens = Math.max(0, after - before);

    req.body = next;
    req.completion_tokens_total += retained_tokens;
    if (req.body.max_tokens !== undefined) {
        req.body.max_tokens = Math.max(0, req.body.max_tokens - retained_tokens);
    }
    req.continued = true;
    req.pending = undefined;
    if (req.body.max_tokens === 0) return pending;
}

function finishAtTokenLimit(req: ReqInfo, res: ExpressResponse, pending: PendingAttempt): void {
    const completion_tokens = req.completion_tokens_total;
    const prompt_tokens = req.original_prompt_tokens!;
    const usage = {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

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