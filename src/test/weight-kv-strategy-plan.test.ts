// Pure scheduling tests. Run: npm run build && node --test dist/test/weight-kv-strategy-plan.test.js
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { STRATEGIES, canRestoreAcrossSwap, decodeBudget, evaluatedBoundary, evaluatedTokens, firstPrefillEnd, isKvReload, kvReloadBody, stageCapacity, strategyScorePoints, validateStrategy } from "./weight-kv-strategy-plan.js";

const multi = STRATEGIES.find(s => s.id === "24g-q6-q4-q3-f16")!;
test("50k prompt starts Q6 decode, then moves to Q4 and Q3 at context boundaries", () => {
    validateStrategy(multi, 50_000);
    assert.equal(firstPrefillEnd(50_000, multi.stages[0]), 50_000);
    assert.equal(stageCapacity(multi.stages[0]), 53_760);
    assert.equal(decodeBudget(50_000, 0, multi.stages[0]), 2048);
    assert.equal(decodeBudget(50_000, 3759, multi.stages[0]), 1);
    assert.equal(decodeBudget(50_000, 3760, multi.stages[0]), 0);
    assert.ok(decodeBudget(50_000, 3760, multi.stages[1]) > 0);
});

test("75k prompt transfers mid-prefill Q6->Q4, before any generation", () => {
    validateStrategy(multi, 75_000);
    assert.equal(firstPrefillEnd(75_000, multi.stages[0]), 53_760);
    assert.equal(evaluatedTokens(53_760, 0), 53_760);
    assert.equal(firstPrefillEnd(75_000, multi.stages[1]), 75_000);
    assert.equal(decodeBudget(75_000, 0, multi.stages[1]), 2048);
});

test("decoder handoff restores all except the final sampled, unevaluated token", () => {
    assert.equal(evaluatedTokens(50_000, 3760), 53_759);
    assert.equal(evaluatedTokens(50_000, 1), 50_000);
});

test("teacher-forced scoring covers answer tail and the actual decode handoff boundary", () => {
    const start = 50_000, reference = 15_000;
    const boundary = evaluatedBoundary(multi.stages[0], start) - start;
    assert.equal(boundary, 3759);
    const points = strategyScorePoints(reference, start, multi);
    assert.deepEqual(points.slice(0, 4).map(p => p.index), [0, 1, 2, 3]);
    assert.deepEqual(points.slice(-12).map(p => p.index), Array.from({ length: 12 }, (_, i) => reference - 12 + i));
    for (let index = boundary - 4; index <= boundary + 4; index++) {
        assert.equal(points.find(p => p.index === index)?.label, "near-q6-handoff");
    }
    const midPrefill = strategyScorePoints(100, 75_000, multi);
    assert.ok(midPrefill.every(p => p.label !== "near-q6-handoff"));
});

test("new ladder saves f16 Q4 KV, reloads the Q4 context at q8_0, then restores it", () => {
    const ladder = STRATEGIES.find(s => s.id === "24g-q6-q4-q4q8kv")!;
    validateStrategy(ladder, 75_000);
    const [q6, q4F16, q4Q8] = ladder.stages;
    assert.equal(isKvReload(q6, q4F16), false);
    assert.equal(isKvReload(q4F16, q4Q8), true);
    assert.deepEqual(kvReloadBody(q4Q8), { n_ctx: 183296, cache_type_k: "q8_0", cache_type_v: "q8_0" });
    assert.equal(stageCapacity(q4F16), 108_800);
    assert.equal(stageCapacity(q4Q8), 182_784);
    assert.equal(evaluatedTokens(75_000, 33_800), 108_799);
    assert.ok(decodeBudget(75_000, 33_800, q4Q8) > 0);
    assert.equal(decodeBudget(75_000, 182_784 - 75_000, q4Q8), 0);
    const points = strategyScorePoints(40_000, 75_000, ladder);
    assert.ok(points.some(p => p.label === "near-q4-handoff"));
    assert.ok(points.every(p => p.label !== "near-q6-handoff"));
});

test("four-stage ladder reloads Q6 KV, converts q8->f16 when swapping weights, then reloads Q4 KV", () => {
    const ladder = STRATEGIES.find(s => s.id === "24g-q6-q6q8-q4-q4q8kv")!;
    const [q6f16, q6q8, q4f16, q4q8] = ladder.stages;
    assert.deepEqual(ladder.stages.map(s => s.ctx), [54272, 91136, 109312, 183296]);
    assert.equal(isKvReload(q6f16, q6q8), true);
    assert.deepEqual(kvReloadBody(q6q8), { n_ctx: 91136, cache_type_k: "q8_0", cache_type_v: "q8_0" });
    assert.equal(canRestoreAcrossSwap(q6q8, q4f16), true);
    assert.equal(isKvReload(q4f16, q4q8), true);
    assert.equal(stageCapacity(q6q8), 90624);
    validateStrategy(ladder, 75_000);
});
test("baselines are bounded by their explicit context, not arbitrary output limit", () => {
    for (const s of STRATEGIES) {
        validateStrategy(s, 75_000);
        const final = s.stages.at(-1)!;
        const capacity = stageCapacity(final) - 75_000;
        assert.ok(capacity > 0);
        assert.equal(decodeBudget(75_000, capacity, final), 0);
    }
});
