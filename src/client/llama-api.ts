import type { ModelId, TokenIds } from "../types.js";
import { HttpError, type CompletionRequest, type ModelInfo, type ReloadRequest, type StatusResponse, type Slot, type TokenizeRequest, type TokenizeResponse, type LoadModelParams, type LoadModelRequest, type UnloadModelRequest, type HealthResponse, type MemoryResponse } from "./types.js";

/**
 * HTTP API wrapper around llama-server
 * 
 * @note expects llama-server to:
 * - run in router mode
 * - have slots enabled
 */
export class LlamaAPI {

    base_url: string;

    constructor(base_url: string) {
        this.base_url = base_url;
    }

    async completions(params: CompletionRequest, model: ModelId): Promise<void> {
        const url = this.#buildURL('/chat/completions', model);

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.reqBody),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /chat/completions', msg, res.status);
        }

        return Promise.reject();
    }

    async tokenize(params: TokenizeRequest, model: ModelId): Promise<TokenIds> {
        const url = this.#buildURL('/tokenize', model);

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.reqBody),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /tokenize', msg, res.status);
        }

        return (await res.json() as TokenizeResponse).tokens;
    }

    async getSlots(model: ModelId): Promise<Slot[]> {
        const url = this.#buildURL('/slots', model);

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('GET /slots', msg, res.status);
        }

        return await res.json() as Slot[];
    }

    // async setSlots(): Promise<void> {
    //     return Promise.reject(); TODO
    // }

    async reloadModel(params: ReloadRequest, model: ModelId): Promise<StatusResponse> {
        const url = this.#buildURL('/reload', model);

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.reqBody),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /reload', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async loadModel(params: LoadModelRequest): Promise<StatusResponse> {
        const url = this.#buildURL('/models/load');

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.reqBody),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /models/load', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async unloadModel(params: UnloadModelRequest): Promise<StatusResponse> {
        const url = this.#buildURL('/models/unload');

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.reqBody),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /models/unload', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async getModels(): Promise<ModelInfo[]> {
        const url = this.#buildURL('/models');

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('GET /models', msg, res.status);
        }

        return await res.json() as ModelInfo[];
    }

    async getMemory(model: ModelId): Promise<MemoryResponse> {
        const url = this.#buildURL('/memory', model);

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('GET /memory', msg, res.status);
        }

        return await res.json() as MemoryResponse;
    }

    async getHealth(signal: AbortSignal): Promise<boolean> {
        const url = this.#buildURL('/health');

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
            signal: signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('GET /health', msg, res.status);
        }

        const health = await res.json() as HealthResponse;
        return health.status === "ok";
    }

    #buildURL(route: string, model?: ModelId): string {
        let url = this.base_url + route;
        if (model) url += `?model=${model}`;

        return url;
    }
}