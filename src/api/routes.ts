import type { Express, Request, Response as ExpressResponse } from "express";
import type { ApiServer } from "./server.js";
import type { Client } from "../planner/planner.js";
import { SSERelay } from "./sse-relay.js";
import type { CompletionChunk, CompletionRequest, ToolCall } from "./types.js";

export function registerRoutes(app: Express, server: ApiServer): void {
    app.get('/v1/models', (req, res) => listModels(server, req, res));
    app.post('/v1/chat/completions', (req, res) => completions(server, req, res));
}

async function listModels(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const data = [...server.planner.models.values()]
        .filter((e): e is NonNullable<typeof e> => !!e)
        .map(e => ({
            id: e.name,
            object: 'model',
            created: 0,
            owned_by: 'llama-manager',
        }));

    res.json({ object: 'list', data });
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
        sendError(res, 400, 'invalid_request', `unable to resolve model ${model_name}`);
        return;
    }

    const is_stream = body.stream === true;

    const ac = new AbortController();
    res.on('close', () => ac.abort());
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
    isFinal: boolean;
};

type PartialToolCall = {
    id: string;
    type: string;
    function_name: string;
    function_args: string;
}

async function streamCompletion(
    req: ReqInfo,
    upstream: Response,
    res: ExpressResponse,
): Promise<boolean> {
    let partial_content = "";
    let partial_reasoning = "";
    const partial_tool_calls = new Map<number, PartialToolCall>();

    try {
        let pending_frame: { raw: string; json: CompletionChunk } | null = null;

        const relay = new SSERelay(upstream.body!);
        for await (const frame of relay) {
            if (frame.done) {
                if (pending_frame) {
                    res.write(`data: ${pending_frame.raw}\n\n`);
                }
                finishSSEResponse(res);
                return true;
            }

            const json = frame.data as CompletionChunk;
            const raw = JSON.stringify(json);

            // handle truncated response due to context window length
            if (pending_frame) {
                const completion_tokens = json.usage?.completion_tokens ?? 0;
                const finish_reason = pending_frame.json.choices[0]?.finish_reason;

                // accumulate usage from prior turns
                if (req.completion_tokens_total > 0 && json.usage) {
                    json.usage.completion_tokens += req.completion_tokens_total;
                    json.usage.total_tokens += req.completion_tokens_total;
                }

                // unable to expand ctx window, send what we have
                if (req.isFinal) {
                    res.write(`data: ${pending_frame.raw}\n\n`);
                    res.write(`data: ${JSON.stringify(json)}\n\n`);
                    finishSSEResponse(res);
                    return true;
                }

                if (isTruncated(completion_tokens, finish_reason, req.body.max_tokens)) {
                    applyTruncation(req, completion_tokens, partial_content, partial_reasoning || undefined, collapseToolCalls(partial_tool_calls));
                    return false;
                }

                res.write(`data: ${pending_frame.raw}\n\n`);
                res.write(`data: ${JSON.stringify(json)}\n\n`);
                pending_frame = null;
                continue;
            }

            const choice = json.choices[0];
            if (!choice) continue;

            const delta = choice.delta;
            const finish_reason = choice.finish_reason;

            // if we get a finish reason, add it as a pending frame so we can detect truncation
            if (finish_reason) {
                pending_frame = { raw, json };
                continue;
            }

            res.write(`data: ${raw}\n\n`);

            if (delta?.content) {
                partial_content += delta.content;
            }

            if (delta?.reasoning_content) {
                partial_reasoning += delta.reasoning_content;
            }

            if (delta?.tool_calls) {
                for (const tc of delta.tool_calls) {
                    mergeToolCall(partial_tool_calls, tc.index ?? 0, tc);
                }
            }
        }

        finishSSEResponse(res);
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
        const message = choice?.message;
        const content = message?.content ?? "";
        const reasoning = message?.reasoning_content;
        const tool_calls = message?.tool_calls;
        applyTruncation(req, completion_tokens, content, reasoning, tool_calls);
        return false;
    }

    if (req.completion_tokens_total > 0 && json.usage) {
        json.usage.completion_tokens += req.completion_tokens_total;
        json.usage.total_tokens += req.completion_tokens_total;
    }

    if (req.completion_tokens_total > 0) {
        const last = req.body.messages[req.body.messages.length - 1];
        if (last) {
            if (choice?.message?.content) {
                last.content = (last.content ?? "") + choice.message.content;
            }
            if (choice?.message?.reasoning_content) {
                last.reasoning_content = (last.reasoning_content ?? "") + choice.message.reasoning_content;
            }
            if (choice?.message?.tool_calls) {
                last.tool_calls = mergeToolCallLists(last.tool_calls as ToolCall[] | undefined, choice.message.tool_calls);
            }
            if (json.choices[0].message) {
                json.choices[0].message.content = last.content as string;
                json.choices[0].message.reasoning_content = last.reasoning_content as string | undefined;
                json.choices[0].message.tool_calls = last.tool_calls as ToolCall[] | undefined;
            }
        }
    }

    res.write(JSON.stringify(json));
    res.end();
    return true;
}

function mergeToolCall(tool_calls: Map<number, PartialToolCall>, idx: number, tc: ToolCall): void {
    let cur = tool_calls.get(idx);
    if (!cur) {
        cur = { id: "", type: "function", function_name: "", function_args: "" };
        tool_calls.set(idx, cur);
    }

    if (tc.id) cur.id = tc.id;
    if (tc.type) cur.type = tc.type;

    if (tc.function) {
        if (tc.function.name) cur.function_name = tc.function.name;
        if (tc.function.arguments) cur.function_args += tc.function.arguments;
    }
}

function collapseToolCalls(tool_calls: Map<number, PartialToolCall>): ToolCall[] {
    if (tool_calls.size === 0) return [];

    const result: ToolCall[] = [];
    for (const [, tc] of [...tool_calls.entries()].sort(([a], [b]) => a - b)) {
        result.push({
            id: tc.id,
            type: tc.type,
            function: {
                name: tc.function_name,
                arguments: tc.function_args,
            },
        });
    }
    return result;
}

function mergeToolCallLists(prev: ToolCall[] | undefined, next: ToolCall[] | undefined): ToolCall[] | undefined {
    if (!prev || prev.length === 0) return next;
    if (!next || next.length === 0) return prev;

    const merged = new Map<number, ToolCall>();
    for (const tc of prev) merged.set(tc.index ?? 0, tc);
    for (const tc of next) {
        const idx = tc.index ?? 0;
        const existing = merged.get(idx);
        if (existing && existing.function && tc.function) {
            merged.set(idx, {
                ...tc,
                function: {
                    name: tc.function.name ?? existing.function.name,
                    arguments: (existing.function.arguments ?? "") + (tc.function.arguments ?? ""),
                },
            });
        } else {
            merged.set(idx, tc);
        }
    }
    return [...merged.values()];
}

// A response is truncated due to context window size if the pending frame
// contains finish_reason === 'length'.
//
// If the original request specified a max_tokens, we also evaluate whether
// completion_tokens < max_tokens
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

function applyTruncation(
    req: ReqInfo,
    completion_tokens: number,
    content: string,
    reasoning?: string,
    tool_calls?: ToolCall[],
): void {
    const messages = req.body.messages;

    const msg: Record<string, unknown> = { role: "assistant", content };
    if (reasoning) {
        msg.reasoning_content = reasoning;
    }
    if (tool_calls && tool_calls.length > 0) {
        msg.tool_calls = tool_calls;
    }

    const last = messages[messages.length - 1];
    if (last && last.role === "assistant") {
        msg.content = (last.content ?? "") + content;
        msg.reasoning_content = (last.reasoning_content ?? "") + (reasoning ?? "");
        msg.tool_calls = mergeToolCallLists(last.tool_calls as ToolCall[] | undefined, tool_calls);
        messages[messages.length - 1] = msg as any;
    } else {
        messages.push(msg as any);
    }

    if (req.body.max_tokens !== undefined) {
        req.body.max_tokens -= completion_tokens;
    }

    req.body.continue_final_message = true;
    req.body.add_generation_prompt = false;
    req.completion_tokens_total += completion_tokens;
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