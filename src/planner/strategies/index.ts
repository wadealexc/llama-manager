import type { LlamaAPI } from "../../client/llama-api.js";
import type { StrategyId } from "../../config/types.js";
import type { Strategy } from "../types.js";
import { DisableSpec } from "./disable-spec.js";
import { EvictTaskModel } from "./evict-task-model.js";
import { MmprojOnDemand } from "./mmproj-on-demand.js";
import { QuantizeKvQ4 } from "./quantize-kv-q4.js";
import { QuantizeKvQ8 } from "./quantize-kv-q8.js";

export function createStrategies(client: LlamaAPI): Map<StrategyId, Strategy> {
    return new Map<StrategyId, Strategy>([
        ['evict-task-model', new EvictTaskModel(client)],
        ['disable-spec', new DisableSpec(client)],
        ['mmproj-on-demand', new MmprojOnDemand(client)],
        ['quantize-kv-q8', new QuantizeKvQ8(client)],
        ['quantize-kv-q4', new QuantizeKvQ4(client)],
    ]);
}