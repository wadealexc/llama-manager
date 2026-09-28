// Pure strategy planning for the 24 GiB vs 32 GiB comparison.
export type Quant = "q6" | "q4" | "q3";
export type Kv = "f16" | "q8_0";
export interface Stage { quant: Quant; kv: Kv; ctx: number }
export interface Strategy { id: string; stages: readonly Stage[] }

export const STRATEGIES: readonly Strategy[] = [
    { id: "24g-q3-f16", stages: [{ quant: "q3", kv: "f16", ctx: 196096 }] },
    { id: "24g-q4-q8kv", stages: [{ quant: "q4", kv: "q8_0", ctx: 183296 }] },
    { id: "24g-q6-q4-q3-f16", stages: [
        { quant: "q6", kv: "f16", ctx: 54272 },
        { quant: "q4", kv: "f16", ctx: 109312 },
        { quant: "q3", kv: "f16", ctx: 196096 },
    ] },
    { id: "24g-q6-q4-q4q8kv", stages: [
        { quant: "q6", kv: "f16", ctx: 54272 },
        { quant: "q4", kv: "f16", ctx: 109312 },
        { quant: "q4", kv: "q8_0", ctx: 183296 },
    ] },
    { id: "24g-q6-q6q8-q4-q4q8kv", stages: [
        { quant: "q6", kv: "f16", ctx: 54272 },
        { quant: "q6", kv: "q8_0", ctx: 91136 },
        { quant: "q4", kv: "f16", ctx: 109312 },
        { quant: "q4", kv: "q8_0", ctx: 183296 },
    ] },
    { id: "32g-q6-f16", stages: [{ quant: "q6", kv: "f16", ctx: 175104 }] },
];

// The server needs free space to evaluate the last cached token, plus room for
// checkpoints. Never ask it to decode at its measured hard limit.
export const CTX_MARGIN = 512;
export const CHUNK = 2048;
export function stageCapacity(stage: Stage): number { return stage.ctx - CTX_MARGIN; }
export function isKvReload(from: Stage, to: Stage): boolean {
    return from.quant === to.quant && from.kv !== to.kv;
}
// The fork's slot restore converts non-transposed K/V between these types.
export function canRestoreAcrossSwap(from: Stage, to: Stage): boolean {
    return from.quant !== to.quant &&
        (from.kv === "f16" || from.kv === "q8_0") &&
        (to.kv === "f16" || to.kv === "q8_0");
}
// The fork's POST /reload schema uses snake_case fields, not CLI flags.
export function kvReloadBody(stage: Stage): { n_ctx: number; cache_type_k: Kv; cache_type_v: Kv } {
    return { n_ctx: stage.ctx, cache_type_k: stage.kv, cache_type_v: stage.kv };
}
// During decode the last token sampled by the outgoing model is not yet in
// its KV. Its cache stops one token short of the model's context boundary;
// the receiving model consumes that sampled token from the reference history.
// During mid-prompt handoff, all prefix tokens up to the boundary are evaluated.
export function evaluatedBoundary(stage: Stage, inputLength: number): number {
    return stageCapacity(stage) - (inputLength <= stageCapacity(stage) ? 1 : 0);
}
export function firstPrefillEnd(inputLength: number, stage: Stage): number {
    return Math.min(inputLength, stageCapacity(stage));
}
export function decodeBudget(inputLength: number, generatedLength: number, stage: Stage, chunk = CHUNK): number {
    return Math.max(0, Math.min(chunk, stageCapacity(stage) - inputLength - generatedLength));
}
// llama-server saves only *evaluated* tokens. Its last sampled token has not
// been evaluated and must be included in the next model's prompt on restore.
export function evaluatedTokens(inputPos: number, generated: number): number {
    if (!Number.isSafeInteger(inputPos) || inputPos < 1 || !Number.isSafeInteger(generated) || generated < 0) {
        throw new Error("invalid handoff position");
    }
    return inputPos + Math.max(0, generated - 1);
}
export interface ScorePoint { index: number; label: string }
export function strategyScorePoints(
    referenceLength: number, inputLength: number, strategy: Strategy,
    head = 4, middle = 4, tail = 12, boundaryWidth = 4,
): ScorePoint[] {
    if (referenceLength < 1) throw new Error("empty reference continuation");
    const selected = new Map<number, string>();
    function add(start: number, count: number, label: string) {
        for (let i = Math.max(0, start); i < Math.min(referenceLength, start + count); i++) selected.set(i, label);
    }
    add(0, head, "head");
    for (const [label, fraction] of [["25%", 0.25], ["50%", 0.5], ["75%", 0.75]] as const) {
        add(Math.floor(referenceLength * fraction) - Math.floor(middle / 2), middle, label);
    }
    add(referenceLength - tail, tail, "tail");
    for (const stage of strategy.stages.slice(0, -1)) {
        const index = evaluatedBoundary(stage, inputLength) - inputLength;
        if (index < 0 || index >= referenceLength) continue; // prefill handoff
        add(index - boundaryWidth, 2 * boundaryWidth + 1, `near-${stage.quant}-handoff`);
    }
    return [...selected.entries()].sort(([a], [b]) => a - b).map(([index, label]) => ({ index, label }));
}
export function validateStrategy(strategy: Strategy, promptTokens: number): void {
    if (!strategy.stages.length || promptTokens < 1) throw new Error("strategy has no stages or an empty prompt");
    for (let i = 0; i < strategy.stages.length; i++) {
        if (stageCapacity(strategy.stages[i]) <= 0) throw new Error("stage has no usable context");
        if (i && stageCapacity(strategy.stages[i]) <= stageCapacity(strategy.stages[i - 1])) {
            throw new Error("handoff stage must have strictly more context than its predecessor");
        }
    }
    if (promptTokens >= stageCapacity(strategy.stages.at(-1)!)) {
        throw new Error(`${strategy.id}: prompt does not fit the final stage`);
    }
}
