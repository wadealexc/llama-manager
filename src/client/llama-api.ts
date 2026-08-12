import type { ConsolaInstance } from "consola";
import type { ModelLoadConfig } from "../config/types.js";
import type { ModelId, TokenIds } from "../types.js";
import { HttpError, type CompletionRequest, type ModelInfo, type ReloadRequest, type StatusResponse, type Slot, type TokenizeRequest, type TokenizeResponse, type LoadModelParams, type LoadModelRequest, type UnloadModelRequest, type HealthResponse, type MemoryResponse, type RouterModelStatus } from "./types.js";
import { logger } from "../logger.js";

const log: ConsolaInstance = logger.withTag('llama-api');

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
            body: JSON.stringify(params.req_body),
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
            body: JSON.stringify(params.req_body),
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
            body: JSON.stringify(params.req_body),
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

        log.info(`loading model: ${params.req_body.model}`);
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.req_body),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            if (msg.includes("model is already running")) {
                log.debug(`loadModel: ${params.req_body.model} already running`);
                return { success: true }
            }
            
            throw new HttpError('POST /models/load', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async unloadModel(params: UnloadModelRequest): Promise<StatusResponse> {
        const url = this.#buildURL('/models/unload');

        log.info(`unloading model: ${params.req_body.model}`);
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(params.req_body),
            signal: params.signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /models/unload', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async loadModelAndWait(params: LoadModelRequest, opts: ModelLoadConfig): Promise<void> {
        const model = params.req_body.model;
        const status = await this.loadModel(params);
        if (!status.success) {
            throw new Error(`loadModelAndWait: loadModel failed with error: ${status.message}`);
        }
        
        log.info(`polling load status: ${model}`);

        const poll_start = performance.now();
        await this.#pollModelStatus(model, opts, 'loaded', (status: string) => {
            log.debug(`${model} status: ${status}`);
        });
        const poll_end = performance.now();
        const seconds = (poll_end - poll_start) / 1000;

        log.info(`loaded ${model} [elapsed: ${seconds.toFixed(2)}s]`);
    }

    async unloadModelAndWait(params: UnloadModelRequest, opts: ModelLoadConfig): Promise<void> {
        const model = params.req_body.model;
        const status = await this.unloadModel(params);
        if (!status.success) {
            throw new Error(`unloadModelAndWait: unloadModel failed with error: ${status.message}`);
        }

        log.info(`polling unload status: ${model}`);

        const poll_start = performance.now();
        await this.#pollModelStatus(model, opts, 'unloaded', (status: string) => {
            log.debug(`${model} status: ${status}`);
        });
        const poll_end = performance.now();
        const seconds = (poll_end - poll_start) / 1000;

        log.info(`unloaded ${model} [elapsed: ${seconds.toFixed(2)}s]`);
    }

    async #pollModelStatus(model: ModelId, opts: ModelLoadConfig, success: RouterModelStatus, on_change: (status: string) => void): Promise<void> {
        const deadline = Date.now() + opts.poll_timeout_ms;
        let last_status: RouterModelStatus = 'unknown';

        while (Date.now() < deadline) {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 100);

            try {
                const infos = await this.getModels(controller.signal);

                // Find our model and exit when its status === success
                let found = false;
                for (const info of infos) {
                    if (info.id === model) {
                        found = true;

                        if (info.status.failed) {
                            throw new Error(`load failed for ${model}`);
                        }

                        if (info.status.value !== last_status) {
                            last_status = info.status.value;
                            on_change(info.status.value);
                        }

                        // Model has reached desired status
                        if (info.status.value === success) {
                            return;
                        }

                        break;
                    }
                }

                if (!found) throw new Error(`info for model ${model} not returned by server`);
            } catch (err) {
                if (err instanceof Error && err.name === 'AbortError') {
                    // Swallow per-request abort
                } else {
                    throw new Error(`pollModelStatus error: ${err}`);
                }
            } finally {
                clearTimeout(timeout);
            }

            await new Promise((r) => setTimeout(r, opts.poll_interval_ms));
        }

        throw new Error(`model ${model} did not reach status ${success} (timeout elapsed: ${opts.poll_timeout_ms} ms)`);
    }

    async getModels(signal?: AbortSignal): Promise<ModelInfo[]> {
        const url = this.#buildURL('/models');

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
            signal: signal,
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

    async getHealth(signal?: AbortSignal): Promise<boolean> {
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