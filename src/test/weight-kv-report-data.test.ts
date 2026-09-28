// Pure aggregation tests. Run: compile test files and node --test the result.
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { aggregate, branchCoverage, inheritedFourStage, isFreeResult, klSummary, metricSummary, paired, runtimeTotals, stats, summarize, type FreeResult, type Row } from "./weight-kv-report-data.js";
const make = (strategy: FreeResult["strategy"], correct: boolean, time: number): FreeResult => ({
    task: "withdrawal_graph-50k-seed50062", strategy, inputTokens: 50000,
    promptSha256: "same", expected: "A", correct, stop: correct ? "eos" : "context_exhausted",
    outputTokens: 100, elapsedMs: time, prefillMs: 10, decodeMs: 80, swapMs: 0, generatedByQuant: {},
});
test("mean and sample SD across completed seeds (including incorrect and exhausted)", () => {
    assert.deepEqual(summarize([]), { n: 0, mean: null, sd: null });
    assert.deepEqual(summarize([10]), { n: 1, mean: 10, sd: null });
    assert.deepEqual(summarize([10, 20, 30]), { n: 3, mean: 20, sd: 10 });
    const rows: Row[] = [10, 20, 30].map((time, i) => ({
        id: `withdrawal_graph-50k-seed5006${i + 2}`, cohort: "new", length: "50k",
        results: { "24g-q3-f16": { ...make("24g-q3-f16", i === 0, time), elapsedMs: time } },
    }));
    assert.deepEqual(metricSummary(rows, "50k", "24g-q3-f16", "elapsedMs"), { n: 3, mean: 20, sd: 10 });
    assert.deepEqual(metricSummary(rows, "75k", "24g-q3-f16", "elapsedMs"), { n: 0, mean: null, sd: null });
});
test("KL aggregation uses only matched task results of one KL type", () => {
    const rows: Row[] = [0.01, 0.03, 0.5].map((value, i) => ({
        id: `withdrawal_graph-25k-seed2506${i}`, length: "25k", cohort: "new",
        results: { "32g-q6-f16": make("32g-q6-f16", true, 1), "24g-q3-f16": make("24g-q3-f16", false, 2) },
        kl: { task: `withdrawal_graph-25k-seed2506${i}`, promptSha256: "same", q6ReferenceTokens: 100,
            klType: i === 2 ? "full-vocab" : "common top-128 + OTHER lower bound", pairedPositions: 20,
            positions: Array.from({ length: 25 }, (_, index) => ({ index, label: "tail" })),
            pairedMeanNats: { "24g-q3-f16": value } },
    }));
    assert.deepEqual(klSummary(rows, "25k", "24g-q3-f16", "common top-128 + OTHER lower bound"),
        { n: 2, mean: 0.02, sd: Math.sqrt(0.0002) });
    assert.equal(klSummary(rows, "25k", "24g-q3-f16", "full-vocab").n, 1);
    assert.equal(klSummary(rows, "10k", "24g-q3-f16", "full-vocab").n, 0);
});
test("report aggregates only completed free runs and keeps paired denominators", () => {
    const r: Row = { id: "withdrawal_graph-50k-seed50062", cohort: "new", length: "50k",
        results: { "24g-q3-f16": make("24g-q3-f16", false, 50), "24g-q6-q4-q3-f16": make("24g-q6-q4-q3-f16", true, 40),
            "32g-q6-f16": make("32g-q6-f16", true, 60) } };
    const data = aggregate([r]);
    assert.equal(stats(data.rows, "50k", "24g-q4-q8kv").completed, 0);
    assert.deepEqual(stats(data.rows, "50k", "24g-q3-f16"), { completed: 1, correct: 0, exhausted: 1 });
    assert.deepEqual(paired(data.rows, "50k", "24g-q6-q4-q3-f16"), { count: 1, fixes: 0, losses: 0, faster: 1, fewerTokens: 0 });
});
test("four-stage ladder inherits only identical short-context Q6-only runs", () => {
    const old = { ...make("24g-q6-q4-q3-f16", true, 100), generatedByQuant: { q6: 100 } };
    const row: Row = { id: old.task, cohort: "new", length: "25k", results: { "24g-q6-q4-q3-f16": old } };
    const inherited = inheritedFourStage(row);
    assert.equal(inherited?.strategy, "24g-q6-q6q8-q4-q4q8kv");
    assert.equal(inherited?.inheritedFrom, "24g-q6-q4-q3-f16");
    assert.equal(inherited?.elapsedMs, old.elapsedMs);
    row.results["24g-q6-q6q8-q4-q4q8kv"] = inherited;
    assert.deepEqual(branchCoverage([row], "25k"), { total: 1, measured: 0, inherited: 1 });
    delete row.results["24g-q6-q6q8-q4-q4q8kv"];
    row.length = "50k";
    assert.equal(inheritedFourStage(row), undefined);
    assert.deepEqual(branchCoverage([row], "50k"), { total: 1, measured: 0, inherited: 0 });
    row.results["24g-q6-q6q8-q4-q4q8kv"] = { ...make("24g-q6-q6q8-q4-q4q8kv", false, 200) };
    assert.deepEqual(branchCoverage([row], "50k"), { total: 1, measured: 1, inherited: 0 });
});
test("cumulative runtime counts completed runs but not inherited aliases", () => {
    const old = { ...make("24g-q6-q4-q3-f16", true, 120_000), generatedByQuant: { q6: 100 } };
    const row: Row = { id: old.task, length: "25k", cohort: "new", results: { "24g-q6-q4-q3-f16": old } };
    row.results["24g-q6-q6q8-q4-q4q8kv"] = inheritedFourStage(row);
    const totals = runtimeTotals([row]);
    assert.deepEqual(totals.byStrategy["24g-q6-q4-q3-f16"], { runs: 1, elapsedMs: 120_000 });
    assert.deepEqual(totals.byStrategy["24g-q6-q6q8-q4-q4q8kv"], { runs: 0, elapsedMs: 0 });
    assert.equal(totals.elapsedMs, 120_000);
    assert.equal(totals.runs, 1);
});
test("report excludes mismatched prompts and KL references rather than pooling them", () => {
    const q6 = make("32g-q6-f16", true, 50);
    const q3 = { ...make("24g-q3-f16", false, 45), promptSha256: "different" };
    const r: Row = { id: q6.task, cohort: "new", length: "50k", results: { "32g-q6-f16": q6, "24g-q3-f16": q3 },
        kl: { task: q6.task, promptSha256: "different", q6ReferenceTokens: 100,
            klType: "common top-128 + OTHER lower bound", pairedPositions: 20, positions: [], pairedMeanNats: {} } };
    const data = aggregate([r]);
    assert.equal(r.results["24g-q3-f16"], undefined);
    assert.equal(r.kl, undefined);
    assert.equal(data.warnings.length, 2);
    assert.ok(isFreeResult(q6, q6.task, "32g-q6-f16"));
});
