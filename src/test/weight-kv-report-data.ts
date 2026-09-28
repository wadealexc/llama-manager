// Pure aggregation for the static strategy report. Never use summary.json:
// the strategy runner overwrites it on each small-batch invocation.
export const STRATEGY_IDS = ["24g-q3-f16", "24g-q4-q8kv", "24g-q6-q4-q3-f16", "32g-q6-f16"] as const;
export const ALT_STRATEGY_ID = "24g-q6-q6q8-q4-q4q8kv" as const;
export type StrategyId = typeof STRATEGY_IDS[number];
export type ReportStrategyId = StrategyId | typeof ALT_STRATEGY_ID;
export const STRATEGY_LABELS: Record<ReportStrategyId, string> = {
    "24g-q3-f16": "24 GiB · Q3 / f16 KV",
    "24g-q4-q8kv": "24 GiB · Q4 / q8 KV",
    "24g-q6-q4-q3-f16": "24 GiB · Q6 → Q4 → Q3 / f16 KV",
    "24g-q6-q6q8-q4-q4q8kv": "24 GiB · Q6/f16→Q6/q8→Q4/f16→Q4/q8",
    "32g-q6-f16": "32 GiB · Q6 / f16 KV",
};
export interface FreeResult {
    task: string; strategy: ReportStrategyId; inputTokens: number; promptSha256: string;
    expected: string; correct: boolean; stop: string; outputTokens: number;
    elapsedMs: number; prefillMs: number; decodeMs: number; swapMs: number;
    generatedByQuant: Record<string, number>;
    inheritedFrom?: StrategyId; // unchanged prefix strategy; no new GPU run
}
export interface KlResult {
    task: string; promptSha256: string; q6ReferenceTokens: number;
    klType: string; pairedPositions: number; positions: Array<{ index: number; label: string }>;
    pairedMeanNats: Record<string, number | null>;
}
export interface Row {
    id: string; length: "10k" | "25k" | "50k" | "75k"; cohort: "new" | "pilot";
    results: Partial<Record<ReportStrategyId, FreeResult>>; kl?: KlResult;
}
export interface ReportData { rows: Row[]; warnings: string[] }
const isObject = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;
export function isFreeResult(v: unknown, task: string, strategy: ReportStrategyId): v is FreeResult {
    if (!isObject(v)) return false;
    return v.task === task && v.strategy === strategy && typeof v.correct === "boolean" &&
        typeof v.promptSha256 === "string" && typeof v.expected === "string" &&
        typeof v.stop === "string" && finite(v.inputTokens) && finite(v.outputTokens) &&
        finite(v.elapsedMs) && finite(v.prefillMs) && finite(v.decodeMs) && finite(v.swapMs) &&
        isObject(v.generatedByQuant);
}
export function isKlResult(v: unknown, task: string): v is KlResult {
    if (!isObject(v)) return false;
    return v.task === task && typeof v.promptSha256 === "string" && typeof v.klType === "string" &&
        finite(v.q6ReferenceTokens) && finite(v.pairedPositions) && Array.isArray(v.positions) &&
        isObject(v.pairedMeanNats);
}
export function aggregate(rows: Row[], warnings: string[] = []): ReportData {
    for (const row of rows) {
        const baseline = Object.values(row.results)[0];
        if (baseline) {
            for (const [id, r] of Object.entries(row.results)) {
                if (r && (r.promptSha256 !== baseline.promptSha256 || r.inputTokens !== baseline.inputTokens || r.expected !== baseline.expected)) {
                    warnings.push(`${row.id}: ${id} has a different prompt, input length or expected answer; excluded`);
                    delete row.results[id as ReportStrategyId];
                }
            }
        }
        const q6 = row.results["32g-q6-f16"];
        if (row.kl && (!q6 || row.kl.promptSha256 !== q6.promptSha256 || row.kl.q6ReferenceTokens !== q6.outputTokens ||
            !row.kl.klType.includes("lower bound") && row.kl.klType !== "full-vocab")) {
            warnings.push(`${row.id}: KL lacks a matching Q6 reference or has an unknown metric type; excluded`);
            delete row.kl;
        }
    }
    return { rows, warnings };
}
// Only short-context runs that finished entirely under Q6/f16 are identical
// to the new ladder. No 50k/75k result can be inherited: both reach Q6/q8.
export function inheritedFourStage(row: Row): FreeResult | undefined {
    if (!(["10k", "25k"] as const).includes(row.length as "10k" | "25k") || row.results[ALT_STRATEGY_ID]) return undefined;
    const old = row.results["24g-q6-q4-q3-f16"];
    if (!old || old.stop !== "eos" || (old.generatedByQuant.q6 ?? 0) !== old.outputTokens ||
        Object.entries(old.generatedByQuant).some(([quant, n]) => quant !== "q6" && n > 0)) return undefined;
    return { ...old, strategy: ALT_STRATEGY_ID, inheritedFrom: "24g-q6-q4-q3-f16" };
}
export function branchCoverage(rows: Row[], length: Row["length"]) {
    const relevant = rows.filter(row => row.length === length && row.results["24g-q6-q4-q3-f16"]);
    return {
        total: relevant.length,
        measured: relevant.filter(row => row.results[ALT_STRATEGY_ID] && !row.results[ALT_STRATEGY_ID]?.inheritedFrom).length,
        inherited: relevant.filter(row => row.results[ALT_STRATEGY_ID]?.inheritedFrom).length,
    };
}
export function summarize(values: number[]): { n: number; mean: number | null; sd: number | null } {
    if (values.some(v => !Number.isFinite(v))) throw new Error("non-finite value in summary");
    const n = values.length;
    if (!n) return { n, mean: null, sd: null };
    const mean = values.reduce((sum, v) => sum + v, 0) / n;
    const sd = n > 1 ? Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1)) : null;
    return { n, mean, sd };
}
export function metricSummary(rows: Row[], length: Row["length"], id: ReportStrategyId, metric: "elapsedMs" | "outputTokens") {
    return summarize(rows.filter(row => row.length === length).flatMap(row => row.results[id] ? [row.results[id]![metric]] : []));
}
// Cumulative inference time across completed, distinct runs shown in this
// cohort. Inherited short-context results are aliases, not extra GPU runs.
export function runtimeTotals(rows: Row[]) {
    const byStrategy = Object.fromEntries(
        [...STRATEGY_IDS, ALT_STRATEGY_ID].map(id => {
            const runs = rows.flatMap(row => row.results[id] && !row.results[id]!.inheritedFrom ? [row.results[id]!] : []);
            return [id, { runs: runs.length, elapsedMs: runs.reduce((sum, run) => sum + run.elapsedMs, 0) }];
        }),
    ) as Record<ReportStrategyId, { runs: number; elapsedMs: number }>;
    return { byStrategy,
        runs: Object.values(byStrategy).reduce((sum, entry) => sum + entry.runs, 0),
        elapsedMs: Object.values(byStrategy).reduce((sum, entry) => sum + entry.elapsedMs, 0) };
}
// Per-task KL has already been averaged over positions valid for all strategies.
// Aggregate only those task-level means, with the same KL type on every row.
export function klSummary(rows: Row[], length: Row["length"], id: StrategyId, klType: string) {
    return summarize(validKl(rows, id).filter(x => x.row.length === length && x.row.kl?.klType === klType).map(x => x.nats));
}
export function stats(rows: Row[], length: Row["length"], id: ReportStrategyId) {
    const samples = rows.filter(row => row.length === length).flatMap(row => row.results[id] ? [row.results[id]!] : []);
    return { completed: samples.length, correct: samples.filter(x => x.correct).length,
        exhausted: samples.filter(x => x.stop === "context_exhausted").length };
}
export function paired(rows: Row[], length: Row["length"], id: ReportStrategyId) {
    const pairs = rows.filter(row => row.length === length && row.results[id] && row.results["32g-q6-f16"])
        .map(row => ({ row, result: row.results[id]!, q6: row.results["32g-q6-f16"]! }));
    return { count: pairs.length,
        fixes: pairs.filter(p => p.result.correct && !p.q6.correct).length,
        losses: pairs.filter(p => !p.result.correct && p.q6.correct).length,
        faster: pairs.filter(p => p.result.elapsedMs < p.q6.elapsedMs).length,
        fewerTokens: pairs.filter(p => p.result.outputTokens < p.q6.outputTokens).length };
}
// The new strategy is not KL-replayed; do not synthesize distributions for it.
export function validKl(rows: Row[], id: StrategyId) {
    return rows.flatMap(row => {
        const v = row.kl?.pairedMeanNats[id];
        return row.kl && row.results[id] && typeof v === "number" && Number.isFinite(v) && v >= 0 && row.kl.pairedPositions > 0
            ? [{ row, nats: v, valid: row.kl.pairedPositions, total: row.kl.positions.length }] : [];
    });
}
