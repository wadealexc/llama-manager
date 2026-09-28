// Offline static report. Reads completed per-task files; does NOT launch models.
// Run: npx tsx src/test/weight-kv-report.ts
// Optional: REPORT_COHORT=all|pilot|new (default all), REPORT_OUT=path/to/report.html
// The main report pools pilot and newer tasks; pilot/new filters are diagnostic.
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { aggregate, ALT_STRATEGY_ID, branchCoverage, inheritedFourStage, isFreeResult, isKlResult, klSummary, metricSummary, paired, runtimeTotals, stats, STRATEGY_IDS, STRATEGY_LABELS, validKl,
    type FreeResult, type KlResult, type ReportData, type Row, type ReportStrategyId, type StrategyId } from "./weight-kv-report-data.js";

const ROOT = resolve(import.meta.dirname, "../..");
const FREE = resolve(process.env.REPORT_FREE_DIR ?? join(ROOT, "logs/weight-kv-strategies"));
const KL = join(ROOT, "logs/weight-kv-strategy-kl");
const COHORT = process.env.REPORT_COHORT ?? "all";
const OUTPUT = resolve(process.env.REPORT_OUT ?? join(ROOT, "logs/weight-kv-report", `report-${COHORT}.html`));
const RESULT_RE = /^(withdrawal_graph-(10|25|50|75)k-seed(\d+))-(24g-q3-f16|24g-q4-q8kv|24g-q6-q4-q3-f16|24g-q6-q6q8-q4-q4q8kv|32g-q6-f16)\.json$/;
const LENGTHS: Row["length"][] = ["10k", "25k", "50k", "75k"];
const ALL_IDS: readonly ReportStrategyId[] = [...STRATEGY_IDS, ALT_STRATEGY_ID];
function cohortFor(length: string, seed: number): Row["cohort"] {
    return length === "10" || length === "25" || seed % 100 >= 62 ? "new" : "pilot";
}
const COLORS: Record<ReportStrategyId, string> = {
    "24g-q3-f16": "#9673e6", "24g-q4-q8kv": "#3aa6c8",
    "24g-q6-q4-q3-f16": "#6848ac", "24g-q6-q6q8-q4-q4q8kv": "#277e9b",
    "32g-q6-f16": "#f5a345",
};
function escapeHtml(s: unknown) {
    return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}
function fmt(n: number | null | undefined, digits = 1) {
    return n === null || n === undefined ? "—" : n.toLocaleString("en-US", { maximumFractionDigits: digits });
}
function duration(ms: number): string {
    const seconds = Math.round(ms / 1000);
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor(seconds / 3600) % 24;
    const minutes = Math.floor(seconds / 60) % 60;
    return [days ? `${days}d` : "", (days || hours) ? `${hours}h` : "", `${minutes}m`].filter(Boolean).join(" ");
}
function readJson(path: string): unknown { return JSON.parse(readFileSync(path, "utf8")) as unknown; }
// Shape encodes the lowest weight precision in each strategy: Q6 circle,
// Q4 square, Q3 triangle. Color continues to distinguish strategies.
function marker(id: ReportStrategyId, x: number, y: number, r = 7): string {
    const fill = COLORS[id];
    const outline = 'stroke="#101821" stroke-width="2"';
    if (id === "32g-q6-f16") return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" ${outline}/>`;
    if (id === "24g-q3-f16" || id === "24g-q6-q4-q3-f16") {
        return `<polygon points="${x},${y - r} ${x - r},${y + r * 0.85} ${x + r},${y + r * 0.85}" fill="${fill}" ${outline}/>`;
    }
    return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" fill="${fill}" ${outline}/>`;
}
function collect(): ReportData {
    if (!existsSync(FREE)) throw new Error(`No results in ${FREE}`);
    if (!["new", "pilot", "all"].includes(COHORT)) throw new Error("REPORT_COHORT must be new, pilot or all");
    const warnings: string[] = [];
    const taskIds = new Map<string, { length: Row["length"]; cohort: Row["cohort"] }>();
    for (const name of readdirSync(FREE)) {
        const match = RESULT_RE.exec(name);
        if (!match) continue; // ignores progress JSON, logs and mutable summary.json
        const cohort = cohortFor(match[2], Number(match[3]));
        if (COHORT !== "all" && COHORT !== cohort) continue;
        taskIds.set(match[1], { length: `${match[2]}k` as Row["length"], cohort });
    }
    const rows: Row[] = [...taskIds.entries()].map(([id, meta]) => {
        const name = `${id}.json`;
        const results: Row["results"] = {};
        for (const strategy of ALL_IDS) {
            const path = join(FREE, `${id}-${strategy}.json`);
            if (!existsSync(path)) continue;
            try {
                const r = readJson(path);
                if (isFreeResult(r, id, strategy)) results[strategy] = r as FreeResult;
                else warnings.push(`${id}/${strategy}: invalid free-run file; excluded`);
            } catch (err) { warnings.push(`${id}/${strategy}: unreadable free-run JSON (${String(err)})`); }
        }
        const path = join(KL, name);
        let kl: KlResult | undefined;
        if (existsSync(path)) {
            try {
                const k = readJson(path);
                if (isKlResult(k, id)) kl = k as KlResult;
                else warnings.push(`${id}: invalid KL result; excluded`);
            } catch (err) { warnings.push(`${id}: unreadable KL JSON (${String(err)})`); }
        }
        return { id, ...meta, results, kl };
    }).filter(row => Object.keys(row.results).length || row.kl);
    rows.sort((a, b) => a.id.localeCompare(b.id));
    const data = aggregate(rows, warnings);
    for (const row of data.rows) {
        const inherited = inheritedFourStage(row);
        if (inherited) row.results[ALT_STRATEGY_ID] = inherited;
    }
    return data;
}
function legend(ids: readonly ReportStrategyId[]) {
    return `<div class="legend">${ids.map(id => `<span><svg viewBox="0 0 20 20" aria-hidden="true">${marker(id, 10, 10, 7)}</svg>${escapeHtml(STRATEGY_LABELS[id])}${id === ALT_STRATEGY_ID ? " (dotted branch)" : ""}</span>`).join("")}</div>`;
}
function inferenceChartLegend(): string {
    const rows: Array<[ReportStrategyId, string, ReportStrategyId | null, string]> = [
        ["24g-q3-f16", "IQ3_S, f16 KV", "24g-q6-q4-q3-f16", "Q6→Q4→Q3, f16 KV"],
        ["24g-q4-q8kv", "Q4_K_XL, q8_0 KV", ALT_STRATEGY_ID, "Q6/f16→Q6/q8→Q4/f16→Q4/q8"],
        ["32g-q6-f16", "Q6_K, f16 KV", null, ""],
    ];
    return `<g class="chart-legend" aria-label="Strategy legend">
        <rect x="207" y="29" width="654" height="80" rx="6" fill="#1c2833" fill-opacity=".94"/>
        <text x="235" y="46" fill="#e7edf4">Legend:</text>
        <path d="M397 51 V103" stroke="#52616f"/>
        ${rows.map(([single, singleText, multi, multiText], i) => {
            const cy = 60 + 19 * i;
            return `${marker(single, 239, cy, 6)}<text x="254" y="${cy + 4}">${escapeHtml(singleText)}</text>` +
                (multi ? `${marker(multi, 416, cy, 6)}<text x="431" y="${cy + 4}">${escapeHtml(multiText)}</text>` : "");
        }).join("")}
    </g>`;
}
function runtime(data: ReportData) {
    const totals = runtimeTotals(data.rows);
    return `<section><h2>Completed benchmark runtime</h2>
        <p><strong>${duration(totals.elapsedMs)}</strong> cumulative inference time across <strong>${totals.runs}</strong> completed, distinct strategy runs in this report.</p>
        <div class="scroll"><table><thead><tr><th>Strategy</th><th>Runs</th><th>Cumulative inference time</th></tr></thead><tbody>
        ${ALL_IDS.map(id => `<tr><td>${escapeHtml(STRATEGY_LABELS[id])}</td><td>${totals.byStrategy[id].runs}</td><td>${duration(totals.byStrategy[id].elapsedMs)}</td></tr>`).join("")}
        </tbody></table></div>
        <p class="muted">Sum of completed runs' elapsedMs, not measured wall-clock time. Includes model loads and swaps; excludes prompt sizing, KL replays, failed attempts and inherited duplicate results. Incomplete runs are not included.</p>
    </section>`;
}
function accuracy(data: ReportData) {
    return `<div class="scroll"><table><thead><tr><th>Input size</th><th>Strategy</th><th>Correct / completed</th><th>Context exhausted</th></tr></thead><tbody>
        ${LENGTHS.flatMap(length => ALL_IDS.map(id => {
            const s = stats(data.rows, length, id);
            return `<tr><td>${length}</td><td>${escapeHtml(STRATEGY_LABELS[id])}</td><td>${s.correct}/${s.completed}</td><td>${s.exhausted}</td></tr>`;
        })).join("")}</tbody></table></div>`;
}
function matrix(data: ReportData) {
    return `<div class="scroll"><table><thead><tr><th>Task</th>${ALL_IDS.map(id => `<th>${escapeHtml(STRATEGY_LABELS[id])}</th>`).join("")}</tr></thead><tbody>
    ${data.rows.map(row => `<tr><td>${escapeHtml(row.id)} <small>${row.cohort}</small></td>${ALL_IDS.map(id => {
        const r = row.results[id];
        return `<td class="${r ? r.correct ? "pass" : "fail" : "missing"}">${!r ? "pending" : `${r.correct ? "✓ correct" : "✗ incorrect"}${r.stop === "context_exhausted" ? " · ctx exhausted" : ""}`}<br><small>${r ? `${fmt(r.outputTokens, 0)} tokens · ${fmt(r.elapsedMs / 1000)}s${r.inheritedFrom ? " · unchanged (inherited)" : ""}` : ""}</small></td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function averageChart(data: ReportData, metric: "elapsedMs" | "outputTokens") {
    // One chart per metric; shapes indicate the lowest weight precision.
    const lengths = LENGTHS;
    const divisor = metric === "elapsedMs" ? 1000 : 1;
    const unit = metric === "elapsedMs" ? "seconds" : "tokens";
    const title = metric === "elapsedMs" ? "Average inference time" : "Average output tokens";
    const summary = lengths.map(length => STRATEGY_IDS.map(id => metricSummary(data.rows, length, id, metric)));
    const alt = lengths.map(length => metricSummary(data.rows, length, ALT_STRATEGY_ID, metric));
    const coverage = lengths.map(length => branchCoverage(data.rows, length));
    // Short prompts never cross the Q6/f16 limit, so inherit the measured
    // original ladder. Draw a dotted divergence only when 50k is measured.
    const showBranch = alt[2].mean !== null && coverage[2].measured > 0;
    const max = Math.max(1, ...summary.flat().map(s => (s.mean ?? 0) / divisor),
        ...(showBranch ? alt.slice(2).map(s => (s.mean ?? 0) / divisor) : []));
    const yMax = max * 1.1;
    // Reserve the chart header for the inset legend so it cannot obscure data.
    const w = 900, h = 440, left = 150, right = 32, top = 125, bottom = 65;
    const y = (v: number) => h - bottom - (Math.max(0, v) / yMax) * (h - bottom - top);
    const xs = lengths.map((_, i) => left + (i + 0.5) * (w - left - right) / lengths.length);
    const guides = xs.map(x => `<path d="M${x} ${top} V${h - bottom}" stroke="#465563" stroke-width="1" stroke-dasharray="2 5"/>`).join("");
    const branchX = (i: number) => xs[i];
    const branch = !showBranch || alt[1].mean === null ? "" : `
        <path d="M${xs[1]} ${y(alt[1].mean / divisor)} L${branchX(2)} ${y(alt[2].mean! / divisor)}${alt[3].mean === null ? "" : ` L${branchX(3)} ${y(alt[3].mean / divisor)}`}" fill="none" stroke="${COLORS[ALT_STRATEGY_ID]}" stroke-width="2.5" stroke-dasharray="4 5"/>
        ${[2, 3].flatMap(i => alt[i].mean === null ? [] : [`<g>${marker(ALT_STRATEGY_ID, branchX(i), y(alt[i].mean! / divisor), 5)}
          <title>${escapeHtml(STRATEGY_LABELS[ALT_STRATEGY_ID])}, ${lengths[i]}: mean ${fmt(alt[i].mean! / divisor)} ${unit}; n=${alt[i].n} measured runs. Includes incorrect/context-exhausted runs.</title>
        </g>`]).join("")}`;
    const paths = STRATEGY_IDS.map((id, j) => {
        const values = lengths.flatMap((_, i) => summary[i][j].mean === null ? [] : [{ x: xs[i], v: summary[i][j].mean! / divisor }]);
        return values.length > 1 ? `<path d="${values.map((v, i) => `${i ? "L" : "M"}${v.x} ${y(v.v)}`).join(" ")}" fill="none" stroke="${COLORS[id]}" stroke-width="2" opacity=".65"/>` : "";
    }).join("");
    const marks = lengths.flatMap((length, i) => STRATEGY_IDS.flatMap((id, j) => {
        const s = summary[i][j];
        if (s.mean === null) return [];
        const cx = xs[i], cy = y(s.mean / divisor);
        return [`<g>${marker(id, cx, cy, 5)}
            <title>${escapeHtml(STRATEGY_LABELS[id])}, ${length}: mean ${fmt(s.mean / divisor)} ${unit}; n=${s.n} completed runs (including incorrect/context-exhausted)</title></g>`];
    })).join("");
    return `<section><h3>${title}</h3><div class="chart-frame"><svg role="img" aria-label="${title} by input size" viewBox="0 0 ${w} ${h}">
        <path d="M${left} ${top} V${h - bottom} H${w - right}" stroke="#71818f" fill="none"/>
        ${[0, 0.25, 0.5, 0.75, 1].map(frac => `<path d="M${left} ${y(yMax * frac)} H${w - right}" stroke="#344654" fill="none"/><text x="${left - 12}" y="${y(yMax * frac) + 4}" text-anchor="end">${fmt(yMax * frac, 0)}</text>`).join("")}
        ${guides}${paths}${marks}${branch}${inferenceChartLegend()}
        ${lengths.map((length, i) => `<text x="${xs[i]}" y="${h - bottom + 23}" text-anchor="middle">${length}</text>`).join("")}
        <text x="${w / 2}" y="${h - 7}" text-anchor="middle">Input context (tokens)</text></svg>
        <span class="axis-unit" style="top:${100 * ((top + h - bottom) / 2) / h}%">${unit}</span></div>
        <p class="muted">Point = arithmetic mean across completed task seeds, including incorrect and context-exhausted runs. The dotted branch starts at the unchanged 25k ladder, then shows measured four-stage results at 50k and 75k. Shorter identical Q6/f16 results are inherited; 50k/75k results are never inferred. Hover for n; missing runs are excluded, not counted as zero.</p></section>`;
}
function klAverageChart(data: ReportData, type: string) {
    const ids = STRATEGY_IDS.slice(0, 3);
    const summaries = LENGTHS.map(length => ids.map(id => klSummary(data.rows, length, id, type)));
    const max = Math.max(0.0001, ...summaries.flat().map(s => s.mean ?? 0));
    const w = 900, h = 360, left = 240, right = 32, top = 45, bottom = 65;
    const yMax = max * 1.12;
    const y = (value: number) => h - bottom - Math.max(0, value) / yMax * (h - bottom - top);
    const xs = LENGTHS.map((_, i) => left + (i + 0.5) * (w - left - right) / LENGTHS.length);
    const guides = xs.map(x => `<path d="M${x} ${top} V${h - bottom}" stroke="#465563" stroke-width="1" stroke-dasharray="2 5"/>`).join("");
    const paths = ids.map((id, j) => {
        const values = LENGTHS.flatMap((_, i) => summaries[i][j].mean === null ? [] : [{ x: xs[i], value: summaries[i][j].mean! }]);
        return values.length > 1 ? `<path d="${values.map((v, i) => `${i ? "L" : "M"}${v.x} ${y(v.value)}`).join(" ")}" fill="none" stroke="${COLORS[id]}" stroke-width="2" opacity=".65"/>` : "";
    }).join("");
    const marks = LENGTHS.flatMap((length, i) => ids.flatMap((id, j) => {
        const s = summaries[i][j]; if (s.mean === null) return [];
        const cx = xs[i], cy = y(s.mean);
        return [`<g>${marker(id, cx, cy, 5)}
            <title>${escapeHtml(STRATEGY_LABELS[id])}, ${length}: mean ${s.mean.toFixed(6)} nats; n=${s.n} matched task replays</title></g>`];
    })).join("");
    return `<section><h3>Average KL(Q6 ‖ strategy) by input size</h3><div class="chart-frame"><svg role="img" aria-label="Average KL divergence by input context in tokens" viewBox="0 0 ${w} ${h}">
        <path d="M${left} ${top} V${h - bottom} H${w - right}" stroke="#71818f" fill="none"/>
        ${[0, 0.25, 0.5, 0.75, 1].map(frac => `<path d="M${left} ${y(yMax * frac)} H${w - right}" stroke="#344654" fill="none"/><text x="${left - 9}" y="${y(yMax * frac) + 4}" text-anchor="end">${(yMax * frac).toFixed(3)}</text>`).join("")}
        ${guides}${paths}${marks}${LENGTHS.map((length, i) => `<text x="${xs[i]}" y="${h - bottom + 23}" text-anchor="middle">${length}</text>`).join("")}
        <text x="${w / 2}" y="${h - 7}" text-anchor="middle">Input context (tokens)</text></svg>
        <span class="axis-unit" style="top:${100 * ((top + h - bottom) / 2) / h}%">${type === "full-vocab" ? "Full-vocabulary KL" : "KL lower bound"} (nats)</span></div>
        ${legend(ids)}
        <p class="muted">Point = mean of per-task KL values. Missing replays are excluded. Hover for n. Q6 reference is zero by definition and is omitted from this plot.</p></section>`;
}
function klPlot(data: ReportData) {
    return `<div class="scroll"><table><thead><tr><th>Task · valid positions</th>${STRATEGY_IDS.slice(0, 3).map(id => `<th>${escapeHtml(STRATEGY_LABELS[id])}</th>`).join("")}</tr></thead><tbody>
        ${data.rows.map(row => `<tr><td>${escapeHtml(row.id)}<br><small>${row.kl ? `${row.kl.pairedPositions}/${row.kl.positions.length} paired` : "KL pending"}</small></td>${STRATEGY_IDS.slice(0, 3).map(id => {
            const v = row.kl?.pairedMeanNats[id];
            const valid = row.results[id] && row.kl && typeof v === "number" && Number.isFinite(v) && v >= 0;
            return `<td>${valid ? `${v.toFixed(6)} nats` : "—"}</td>`;
        }).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function main() {
    const data = collect();
    const taskIds = data.rows.map(row => row.id);
    const full = data.rows.filter(row => STRATEGY_IDS.every(id => row.results[id]));
    const warnings = [...data.warnings];
    if (full.length !== data.rows.length) warnings.push(`${data.rows.length - full.length} of ${data.rows.length} tasks are missing one or more strategy results; accuracy denominators show completed runs only.`);
    const expectedTasks = COHORT === "all" ? 40 : COHORT === "new" ? 36 : 4;
    if (data.rows.length < expectedTasks) warnings.push(`${COHORT} cohort: ${data.rows.length}/${expectedTasks} tasks currently have at least one completed strategy result. Unseen tasks do not appear in plots yet.`);
    const byLength = LENGTHS.flatMap(length => [...STRATEGY_IDS.slice(0, 3), ALT_STRATEGY_ID].map(id => ({ length, id, ...paired(full, length, id) })));
    const klCount = data.rows.filter(row => row.kl).length;
    if (klCount < data.rows.length) warnings.push(`KL replay present for ${klCount}/${data.rows.length} shown tasks; missing KL is not interpreted as zero.`);
    const klTypes = [...new Set(data.rows.flatMap(row => row.kl ? [row.kl.klType] : []))];
    if (klTypes.length > 1) warnings.push(`Multiple KL methods found (${klTypes.join(", ")}); plot uses ${klTypes[0]} only. Per-task table retains all methods.`);
    const pairedTable = `<div class="scroll"><table><thead><tr><th>Length</th><th>24 GiB strategy</th><th>Paired tasks</th><th>Wins vs Q6</th><th>Losses vs Q6</th><th>Faster than Q6</th><th>Fewer output tokens</th></tr></thead><tbody>
        ${byLength.map(s => `<tr><td>${s.length}</td><td>${escapeHtml(STRATEGY_LABELS[s.id])}</td><td>${s.count}</td><td>${s.fixes}</td><td>${s.losses}</td><td>${s.faster}/${s.count}</td><td>${s.fewerTokens}/${s.count}</td></tr>`).join("")}</tbody></table></div>`;
    const branch50 = branchCoverage(data.rows, "50k");
    const branch75 = branchCoverage(data.rows, "75k");
    for (const [length, coverage] of [["50k", branch50], ["75k", branch75]] as const) {
        if (coverage.measured < coverage.total) {
            warnings.push(`Four-stage ladder ${length}: ${coverage.measured}/${coverage.total} measured; branch point is provisional until the remaining tasks finish.`);
        }
    }
    const html = `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weight-KV strategy comparison</title>
    <style>body{background:#101821;color:#e7edf4;font:15px/1.5 system-ui,sans-serif;max-width:1440px;margin:auto;padding:24px}h1,h2,h3{line-height:1.2}h2{margin-top:42px}a{color:#8bc8e2}.muted,small{color:#aabac7}small{font-size:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}section{background:#1c2833;border:1px solid #314352;padding:18px;border-radius:10px;min-width:0}.legend{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}.legend span{white-space:nowrap}.legend svg{width:20px;height:20px;vertical-align:middle;margin-right:5px}.chart-frame{position:relative}.chart-frame>.axis-unit{position:absolute;left:0;transform:translateY(-50%);writing-mode:horizontal-tb;white-space:nowrap;font-size:12px;color:#b9c8d5;pointer-events:none}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:900px;width:100%}th,td{border-bottom:1px solid #344654;text-align:left;padding:9px;vertical-align:top}th{font-size:12px;color:#c9d8e3}td small{white-space:nowrap}.pass{background:#163626}.fail{background:#4a2326}.missing{color:#b2bdc5}svg{width:100%;height:auto}svg text{fill:#b9c8d5;font-size:11px}code{background:#263341;padding:2px 4px;border-radius:3px}details{margin-top:24px}pre{white-space:pre-wrap;word-break:break-all;background:#1c2833;padding:14px;border-radius:8px}@media(max-width:600px){.grid{grid-template-columns:1fr}}</style></head><body>
    <h1>Long-context multi-quant inference</h1><p>Static, offline report · cohort <strong>${escapeHtml(COHORT)}</strong> · ${taskIds.length} tasks with available data. The main (all) view includes the original four pilot seeds and the newer seeds; all used the same methodology. Strategy results: Q3/f16 KV, Q4/q8 KV, Q6→Q4→Q3/f16 KV and Q6/f16→Q6/q8→Q4/f16→Q4/q8 on a 24 GiB budget, plus Q6/f16 KV on 32 GiB. The four-stage strategy inherits only identical short Q6/f16-only runs; 50k and 75k are measured independently.</p>
    ${warnings.length ? `<section><h3>Incomplete or excluded data</h3><ul>${warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("")}</ul></section>` : ""}
    ${runtime(data)}
    <h2>Answer accuracy</h2><p>Exact final 16-character code. Context exhaustion without a correct final answer counts as incorrect; missing runs do not enter denominators. Each length is shown separately.</p>${accuracy(data)}
    <h2>Paired outcomes vs 32 GiB Q6</h2><p>Original rows use tasks with all four original strategy files. The four-stage rows include measured 50k/75k and identical inherited 10k/25k runs, paired to the same task's Q6 result. “Faster” compares elapsed inference time, not task preparation; neither speed nor token count implies correctness.</p>${pairedTable}
    <h2>Time and output cost</h2><p>Input size is on the x-axis; elapsed time is in <strong>seconds</strong>. Points average completed tasks at each size; marker shapes indicate lowest weight precision. Elapsed time includes model loading and swaps, excludes offline prompt sizing. Context-exhausted runs retain their full time and token costs. The four-stage branch appears on these charts only; existing KL results are retained for the four original strategies.</p>${averageChart(data, "elapsedMs")}${averageChart(data, "outputTokens")}
    <h2>KL divergence from Q6</h2><p>Directional KL(Q6 ‖ strategy), on a fixed Q6-generated token history—not the strategies’ divergent free-run outputs. Default top-128 results are <strong>coarse-grained lower bounds</strong>, not exact full-vocabulary KL. Only positions valid for every strategy enter each task’s paired mean. Each plotted point averages those per-task means within one input length, using only the same KL method (${escapeHtml(klTypes[0] ?? "no KL replays yet")}). These are diagnostics, not a ranking of answer quality.</p>${klTypes.length ? klAverageChart(data, klTypes[0]) : `<section><h3>Average KL(Q6 ‖ strategy) by input size</h3><p class="muted">KL replay pending. The graph will appear when a completed per-task KL file is available.</p></section>`}${klPlot(data)}
    <details><summary>KL coverage and provenance</summary><pre>${escapeHtml(JSON.stringify(data.rows.map(row => ({ task: row.id, type: row.kl?.klType ?? null, valid: row.kl?.pairedPositions ?? null, total: row.kl?.positions.length ?? null, strategyKL: Object.fromEntries(STRATEGY_IDS.slice(0, 3).map(id => [id, validKl([row], id)[0]?.nats ?? null])) })), null, 2))}</pre></details>
    <p class="muted">Input files: logs/weight-kv-strategies/&lt;task&gt;-&lt;strategy&gt;.json and logs/weight-kv-strategy-kl/&lt;task&gt;.json. This report does not run models, infer missing outcomes, or use the mutable summary.json.</p></body></html>`;
    mkdirSync(dirname(OUTPUT), { recursive: true });
    writeFileSync(OUTPUT, html);
    const totals = runtimeTotals(data.rows);
    console.log(`Wrote ${OUTPUT} (${data.rows.length} tasks, ${full.length} with all four original strategies, four-stage branch 50k ${branch50.measured}/${branch50.total}, 75k ${branch75.measured}/${branch75.total}, ${warnings.length} warnings)`);
    console.log(`Completed benchmark runtime: ${duration(totals.elapsedMs)} across ${totals.runs} distinct runs (excludes prompt sizing, KL and incomplete runs)`);
}
main();
