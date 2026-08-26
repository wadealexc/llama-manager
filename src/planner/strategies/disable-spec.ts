import type { ConsolaInstance } from "consola";
import { logger } from "../../logger.js";
import type { LlamaAPI } from "../../client/llama-api.js";
import type { ReloadParams, SlotSave } from "../../client/types.js";
import type { ModelEntry, StrategyId } from "../../config/types.js";
import type { Strategy, StrategyContext } from "../types.js";

const log: ConsolaInstance = logger.withTag('disable-spec');

export class DisableSpec implements Strategy {

    id: StrategyId = 'disable-spec';

    client: LlamaAPI;

    constructor(client: LlamaAPI) {
        this.client = client;
    }

    canApply(ctx: StrategyContext): boolean {
        return !!ctx.models[ctx.target]?.current_state.spec_loaded;
    }

    // update the target model's state and saved slot info, if needed
    // returns modified ReloadParams
    async applyNoSend(ctx: StrategyContext, params: ReloadParams, saves?: SlotSave[]): Promise<ReloadParams> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`DisableSpec.applyNoSend: target not found`);
        }

        this.#updateCurrentState(target);

        return this.#getParams(params);
    }

    async apply(ctx: StrategyContext): Promise<void> {
        const target = ctx.models[ctx.target];
        if (!target) {
            throw new Error(`DisableSpec.apply: target not found`);
        }
        
        this.#updateCurrentState(target);

        const params = this.#getParams({});
        await this.client.reloadModel(params, target.name);
    }

    #getParams(params: ReloadParams): ReloadParams {
        return {
            ...params,
            spec: { types: ['none'] }
        };
    }

    #updateCurrentState(model: ModelEntry): void {
        model.current_state.spec_loaded = false;
    }
}