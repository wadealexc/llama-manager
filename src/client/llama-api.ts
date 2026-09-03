import type { ConsolaInstance } from "consola";
import type { ModelId, ModelLoadConfig } from "../config/types.js";
import { HttpError, type ModelInfo, type StatusResponse, type Slot, type HealthResponse, type MemoryResponse, type RouterModelStatus, type ReloadParams, type SlotSave, type SlotRestore, type InputTokensResponse, type ReloadResponse } from "./types.js";
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
    config: ModelLoadConfig;

    constructor(base_url: string, config: ModelLoadConfig) {
        this.base_url = base_url;
        this.config = config;
    }

    async completions(body: unknown, model: ModelId, signal?: AbortSignal): Promise<Response> {
        const url = this.#buildURL('/chat/completions');

        return await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ ...body as {}, model: model }),
            signal: signal,
        });
    }

    async countTokens(body: unknown, model: ModelId, signal?: AbortSignal): Promise<number> {
        const url = this.#buildURL('/v1/chat/completions/input_tokens');

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ ...body as {}, model: model }),
            signal: signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /v1/chat/completions/input_tokens', msg, res.status);
        }

        return (await res.json() as InputTokensResponse).input_tokens;
    }

    async getSlots(model: ModelId, signal?: AbortSignal): Promise<Slot[]> {
        const url = this.#buildURL('/slots', model);

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
            signal: signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('GET /slots', msg, res.status);
        }

        return await res.json() as Slot[];
    }

    async saveAllSlots(model: ModelId, signal?: AbortSignal): Promise<SlotSave[]> {
        const slots = await this.getSlots(model, signal);

        return await Promise.all(slots.map(async (slot) => {
            const filename = `slot-${model}-${slot.id}.bin`;
            const slot_url = this.#buildURL(`/slots/${slot.id}`) + '?action=save';

            const res = await fetch(slot_url, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify({ filename, model: model }),
                signal,
            });

            if (!res.ok) {
                const msg = await res.text();
                throw new HttpError('POST /slots?action=save', msg, res.status);
            }

            return await res.json() as SlotSave; 
        }));
    }

    async restoreAllSlots(model: ModelId, saves: SlotSave[], signal?: AbortSignal): Promise<SlotRestore[]> {
        return await Promise.all(saves.map(async (save) => {
            const slot_url = this.#buildURL(`/slots/${save.id_slot}`) + '?action=restore';

            const res = await fetch(slot_url, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify({ filename: save.filename, model: model }),
                signal,
            });

            if (!res.ok) {
                const msg = await res.text();
                throw new HttpError('POST /slots?action=restore', msg, res.status);
            }

            return await res.json() as SlotRestore; 
        }));
    }

    async reloadModel(params: ReloadParams, model: ModelId, signal?: AbortSignal): Promise<number> {
        const url = this.#buildURL('/reload');

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ ...params, model: model }),
            signal: signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /reload', msg, res.status);
        }

        const status = await res.json() as ReloadResponse;
        if (!status.success) {
            throw new Error(`POST /reload: unsuccessful, returned error: ${status.message}`);
        }

        return status.n_ctx;
    }

    async loadModel(model: ModelId, signal?: AbortSignal): Promise<StatusResponse> {
        const url = this.#buildURL('/models/load');

        log.info(`loading model: ${model}`);
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: model }),
            signal: signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            if (msg.includes("model is already running")) {
                log.debug(`loadModel: ${model} already running`);
                return { success: true }
            }

            throw new HttpError('POST /models/load', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async unloadModel(model: ModelId, signal?: AbortSignal): Promise<StatusResponse> {
        const url = this.#buildURL('/models/unload');

        log.info(`unloading model: ${model}`);
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: model }),
            signal: signal,
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new HttpError('POST /models/unload', msg, res.status);
        }

        return await res.json() as StatusResponse;
    }

    async loadModelAndWait(model: ModelId, signal?: AbortSignal): Promise<void> {
        const status = await this.loadModel(model, signal);
        if (!status.success) {
            throw new Error(`loadModelAndWait: loadModel failed with error: ${status.message}`);
        }

        log.info(`polling load status: ${model}`);

        const poll_start = performance.now();
        await this.#pollModelStatus(model, 'loaded', (status: string) => {
            log.debug(`${model} status: ${status}`);
        });
        const poll_end = performance.now();
        const seconds = (poll_end - poll_start) / 1000;

        log.info(`loaded ${model} [elapsed: ${seconds.toFixed(2)}s]`);
    }

    async unloadModelAndWait(model: ModelId, signal?: AbortSignal): Promise<void> {
        const status = await this.unloadModel(model, signal);
        if (!status.success) {
            throw new Error(`unloadModelAndWait: unloadModel failed with error: ${status.message}`);
        }

        log.info(`polling unload status: ${model}`);

        const poll_start = performance.now();
        await this.#pollModelStatus(model, 'unloaded', (status: string) => {
            log.debug(`${model} status: ${status}`);
        });
        const poll_end = performance.now();
        const seconds = (poll_end - poll_start) / 1000;

        log.info(`unloaded ${model} [elapsed: ${seconds.toFixed(2)}s]`);
    }

    async #pollModelStatus(model: ModelId, success: RouterModelStatus, on_change: (status: string) => void): Promise<void> {
        const deadline = Date.now() + this.config.poll_timeout_ms;
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
                log.debug(`pollModelStatus err: ${err}`);
                if (err instanceof Error && err.name === 'AbortError') {
                    // Swallow per-request abort
                } else {
                    throw new Error(`pollModelStatus error: ${err}`);
                }
            } finally {
                clearTimeout(timeout);
            }

            await new Promise((r) => setTimeout(r, this.config.poll_interval_ms));
        }

        throw new Error(`model ${model} did not reach status ${success} (timeout elapsed: ${this.config.poll_timeout_ms} ms)`);
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

        const body = await res.json() as { data: ModelInfo[] };
        return body.data;
    }

    async getMemory(model: ModelId, signal?: AbortSignal): Promise<MemoryResponse> {
        const url = this.#buildURL('/memory', model);

        const res = await fetch(url, {
            method: 'GET',
            headers: { 'content-type': 'application/json' },
            signal,
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