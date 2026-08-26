import { Readable } from "node:stream";
import type { Express, Request, Response as ExpressResponse } from "express";
import type { ApiServer } from "./server.js";
import type { Client } from "../planner/planner.js";
import { pipeline } from "node:stream/promises";

export function registerRoutes(app: Express, server: ApiServer): void {
    app.get('/v1/models', (req, res) => listModels(server, req, res));
    app.post('/v1/chat/completions', (req, res) => completions(server, req, res));
}

async function listModels(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const data = Object.values(server.planner.models)
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
    const body = req.body as Record<string, unknown> | undefined;
    const model = body?.model;
    if (typeof model !== 'string') {
        sendError(res, 400, 'invalid_request', 'missing or invalid "model" field');
        return;
    }

    const is_stream = body?.stream === true;

    const ac = new AbortController();
    res.on('close', () => ac.abort());
    req.on('aborted', () => ac.abort());

    await server.planner.decide(body, model, ac.signal, async (body: unknown, client: Client, signal: AbortSignal) => {
        if (res.writableEnded) return;

        if (is_stream) {
            res.setHeader('content-type', 'text/event-stream');
            res.setHeader('cache-control', 'no-cache');
            res.setHeader('connection', 'keep-alive');
        } else {
            res.setHeader('content-type', 'application/json');    
        }

        res.status(200);
        res.flushHeaders();

        let upstream: Response;
        try {
            upstream = await client.completions(body, model, signal);
        } catch (err) {
            sendUpstreamError(res, is_stream, err);
            return;
        }

        if (!upstream.ok || !upstream.body) {
            const text = await upstream.text().catch(() => '');
            sendUpstreamError(res, is_stream, text || `upstream returned ${upstream.status}`);
            return;
        }

        try {
            await pipeline(Readable.fromWeb(upstream.body), res);
        } catch (err) {
            if (res.writableEnded) return;
            sendUpstreamError(res, is_stream, err);
        }
    });
}

function sendError(res: ExpressResponse, status: number, type: string, message: string): void {
    res.status(status).json({ error: { message, type } });
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
