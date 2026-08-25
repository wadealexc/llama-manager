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
    throw new Error('unimplemented');
}

async function completions(server: ApiServer, req: Request, res: ExpressResponse): Promise<void> {
    const body = req.body as Record<string, unknown> | undefined;
    const model = body?.model;
    if (typeof model !== 'string') {
        sendError(res, 400, 'invalid_request', 'missing or invalid "model" field');
        return;
    }

    const ac = new AbortController();
    req.on('close', () => { if (!res.writableEnded) ac.abort(); });

    await server.planner.decide(body, model, ac.signal, async (body: unknown, client: Client, signal: AbortSignal) => {
        const upstream = await client.completions(body, model, signal);
        res.status(upstream.status);

        const ct = upstream.headers.get('content-type');
        if (ct) res.setHeader('content-type', ct);

        if (!upstream.ok || !upstream.body) {
            res.send(await upstream.text());
            return;
        }

        await pipeline(Readable.fromWeb(upstream.body), res);
    });
}

function sendError(res: ExpressResponse, status: number, type: string, message: string): void {
    res.status(status).json({ error: { message, type } });
}
