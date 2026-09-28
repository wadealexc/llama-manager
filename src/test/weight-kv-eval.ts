// Long-context NIAH evaluation: Q6, Q3, Q6->Q3 (no same-model restore control).
// First review attic/tasks/README.md, then run npx tsx src/test/weight-kv-eval.ts
// Generates no tasks, runs no code from the benchmark repo; tasks are JSON inputs.
// Options: EVAL_TASK=<task-id>, EVAL_LENGTH=75k (default: four withdrawal-graph tasks), EVAL_SUITE=all, EVAL_HEAD=4,
// EVAL_MIDDLE=4 (per quartile), EVAL_TAIL=12, EVAL_TOP_K=128,
// EVAL_MAX_OUTPUT=90000, EVAL_EXACT_KL=1 (large JSON, expensive), LLAMA_BIN=...
// Server stdout: logs/weight-kv-eval-q3/{q6,q3}.log; per-task results: *.json.
// The earlier Q4 comparison in logs/weight-kv-eval/ is left untouched.
// Within this Q3 run, q6.log/q3.log are overwritten for each task.
import { createHash } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type { TaskFile } from "./weight-kv-tasks.js";
import { requestJson } from "./weight-kv-http.js";
import { checkScoreCache, klOnPartition, selectRegions, sharedPartition, type KlEstimate, type Region } from "./weight-kv-metrics.js";

const ROOT = resolve(import.meta.dirname, "../..");
const BIN = process.env.LLAMA_BIN ?? join(ROOT, "llama.cpp/build/bin/llama-server");
const TASK_DIR = join(ROOT, "attic/tasks");
const OUT_DIR = join(ROOT, "logs/weight-kv-eval-q3");
const Q6 = "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q6_K.gguf";
const Q3 = "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-IQ3_S.gguf";
const MAX_OUTPUT = positiveInt("EVAL_MAX_OUTPUT", 90_000);
const HEAD = positiveInt("EVAL_HEAD", 4);
const MIDDLE = positiveInt("EVAL_MIDDLE", 4);
const TAIL = positiveInt("EVAL_TAIL", 12);
const TOP_K = positiveInt("EVAL_TOP_K", 128);
const EXACT_KL = process.env.EVAL_EXACT_KL === "1";
const SLOT = 0;
const MARGIN = 512; // leave room for server bookkeeping, template and outputs
const MIN_HIT = 0.95;
const TIMEOUT_MS = 4 * 60 * 60 * 1000; // upper bound for 90k-token requests

type ModelName = "q6" | "q3";
interface Server { url: string; log: string; proc: ChildProcess; spawnError?: Error; nCtx: number }
interface Completion {
    content: string; tokens: number[]; id_slot: number; tokens_evaluated: number;
    tokens_predicted: number; truncated: boolean; stop_type: string;
    timings: { cache_n: number; prompt_n: number };
    completion_probabilities?: Array<{ id: number; logprob: number; top_logprobs: Array<{ id: number; logprob: number }> }>;
}
interface Prepared { task: TaskFile; hay: string; prompt: string; tokens: number[]; depths: Array<{ target: number; actual: number; index: number }> }
interface Sample { content: string; tokens: number[]; stop: string; final: string | null; correct: boolean; hit: number; capped: boolean }
interface Score { steps: Array<Map<number, number>>; top1: number[]; nll: Array<number | null>; hit: number }
interface ResultRow { id: string; kind: string; target: number; actual: number; expected: string; samples: Record<string, Sample>; stats: Record<string, unknown>; depths: Prepared["depths"] }

function positiveInt(key: string, fallback: number): number {
    const n = Number(process.env[key] ?? fallback);
    if (!Number.isSafeInteger(n) || n < 1) throw new Error(`${key} must be a positive integer`);
    return n;
}
function mean(xs: number[]): number | null { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; }
function sha(s: string): string { return createHash("sha256").update(s).digest("hex"); }
function finalAnswer(text: string): string | null {
    // Grade only the last non-empty line, never a code mentioned in reasoning.
    const last = text.trim().split(/\r?\n/).at(-1)?.trim() ?? "";
    return /^FINAL:\s*([A-Fa-f0-9]{16})\s*$/i.exec(last)?.[1].toUpperCase() ?? null;
}
function hit(r: Completion, label: string, min = MIN_HIT, quiet = false): number {
    const cache = r.timings?.cache_n, processed = r.timings?.prompt_n;
    const total = r.tokens_evaluated;
    if (!Number.isFinite(cache) || !Number.isFinite(processed) || !total) throw new Error(`${label}: no cache metrics`);
    const rate = cache / total;
    if (!quiet) console.log(`  ${label} cache: ${cache}/${total} (${(100 * rate).toFixed(2)}%), processed ${processed}`);
    if (rate < min || cache + processed < total) throw new Error(`${label}: insufficient cache reuse`);
    return rate;
}
async function port(): Promise<number> {
    return new Promise((ok, fail) => {
        const s = net.createServer(); s.once("error", fail);
        s.listen(0, "127.0.0.1", () => {
            const a = s.address();
            if (!a || typeof a === "string") { s.close(); fail(new Error("no port")); return; }
            s.close(() => ok(a.port));
        });
    });
}
async function http<T>(url: string, path: string, body?: unknown, timeout = TIMEOUT_MS): Promise<T> {
    return requestJson<T>(url, path, body, timeout);
}
async function stop(s: Server): Promise<void> {
    if (s.proc.exitCode !== null || s.proc.signalCode !== null || s.spawnError) return;
    const exited = new Promise<void>(done => s.proc.once("exit", () => done()));
    s.proc.kill("SIGTERM");
    if (!await Promise.race([exited.then(() => true), delay(10_000).then(() => false)])) s.proc.kill("SIGKILL");
    await exited;
}
async function launch(name: ModelName, slotDir: string): Promise<Server> {
    const p = await port();
    const log = join(OUT_DIR, `${name}.log`);
    const fd = openSync(log, "w");
    let proc: ChildProcess;
    try {
        proc = spawn(BIN, ["-m", name === "q6" ? Q6 : Q3,
            "--host", "127.0.0.1", "--port", String(p), "-ngl", "99", "--parallel", "1",
            // Let the server fit the largest usable context (as in weight-kv-max-ctx.ts).
            "--slot-save-path", slotDir, "--no-context-shift", "--reasoning", "on",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
        ], { stdio: ["ignore", fd, fd] });
    } finally { closeSync(fd); }
    const s: Server = { url: `http://127.0.0.1:${p}`, log, proc, nCtx: 0 };
    proc.on("error", err => { s.spawnError = err; });
    console.log(`Starting ${name} server; log: ${log}`);
    try {
        const deadline = Date.now() + 300_000;
        while (Date.now() < deadline) {
            if (s.spawnError) throw s.spawnError;
            if (proc.exitCode !== null || proc.signalCode !== null) throw new Error(`server exited (${proc.exitCode}, ${proc.signalCode})`);
            try {
                const slots = await http<Array<{ id: number; n_ctx: number }>>(s.url, "/slots", undefined, 2000);
                if (slots.length === 1 && slots[0].n_ctx > 0) {
                    s.nCtx = slots[0].n_ctx;
                    console.log(`  ${name} context=${s.nCtx}`);
                    return s;
                }
            } catch { /* loading */ }
            await delay(500);
        }
        throw new Error("startup timeout");
    } catch (e) { await stop(s); throw new Error(`${name}: ${String(e)}; see ${log}`); }
}
async function tokenize(s: Server, text: string, addSpecial = false): Promise<number[]> {
    const r = await http<{ tokens: number[] }>(s.url, "/tokenize", { content: text, add_special: addSpecial, parse_special: true });
    return r.tokens;
}
function renderHay(paragraphs: string[], needles: TaskFile["needles"]): { hay: string; chunks: string[]; placements: number[] } {
    const list = paragraphs.slice();
    const placements: number[] = [];
    const planned = needles.map((n, i) => ({ i, n, idx: Math.min(list.length, Math.floor(list.length * n.depth / 100)) })).sort((a, b) => a.idx - b.idx);
    let inserted = 0;
    for (const { i, n, idx } of planned) { list.splice(idx + inserted, 0, n.text); placements[i] = idx + inserted; inserted++; }
    return { hay: list.join("\n\n"), chunks: list, placements };
}
function userPrompt(hay: string, question: string): string {
    return `Read the archive excerpts below. The answer must come from the archive cross-references or memos, not outside knowledge.\n\n${hay}\n\nQuestion: ${question}\n\nYou may reason through the evidence. End with exactly one line in this form: FINAL: <16-character code>.`;
}
async function formatPrompt(s: Server, hay: string, question: string): Promise<string> {
    const r = await http<{ prompt: string }>(s.url, "/apply-template", { messages: [{ role: "user", content: userPrompt(hay, question) }] });
    return r.prompt;
}
async function prepare(s: Server, task: TaskFile): Promise<Prepared> {
    if (sha(task.hayParagraphs.join("\n\n")) !== task.haySha256) throw new Error(`${task.id}: hay checksum mismatch`);
    // Binary search for the number of entire paragraph chunks closest to the
    // target. Needle positions are fractions of the selected hay, not the full
    // source. Count the fully templated prompt in the model's own vocabulary.
    const cache = new Map<number, { prompt: string; count: number }>();
    async function probe(n: number): Promise<{ prompt: string; count: number }> {
        if (!cache.has(n)) {
            const { hay } = renderHay(task.hayParagraphs.slice(0, n), task.needles);
            const prompt = await formatPrompt(s, hay, task.question);
            cache.set(n, { prompt, count: (await tokenize(s, prompt, true)).length });
        }
        return cache.get(n)!;
    }
    let lo = 1, hi = task.hayParagraphs.length;
    if ((await probe(hi)).count < task.targetTokens) throw new Error(`${task.id}: not enough hay for ${task.targetTokens}`);
    while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if ((await probe(mid)).count < task.targetTokens) lo = mid + 1; else hi = mid; }
    const before = await probe(Math.max(1, lo - 1)), after = await probe(lo);
    const chosen = Math.abs(before.count - task.targetTokens) < Math.abs(after.count - task.targetTokens) ? lo - 1 : lo;
    const paragraphs = task.hayParagraphs.slice(0, chosen);
    const { hay, chunks, placements } = renderHay(paragraphs, task.needles);
    const prompt = (await probe(chosen)).prompt;
    const tokens = await tokenize(s, prompt, true);
    if (Math.abs(tokens.length - task.targetTokens) > 400) throw new Error(`${task.id}: token sizing error: ${tokens.length}`);
    const depths = [];
    // Token index within the haystack, before the needle (including any
    // earlier needles). Actual % is approximate relative to templated prompt.
    for (const [i, pos] of placements.entries()) {
        const index = (await tokenize(s, chunks.slice(0, pos).join("\n\n"))).length;
        depths.push({ target: task.needles[i].depth, actual: 100 * index / tokens.length, index });
    }
    console.log(`Prepared ${task.id}: ${tokens.length} tokens (target ${task.targetTokens}); depths ${depths.map(d => d.actual.toFixed(1)).join("%, ")}%`);
    return { task, hay, prompt, tokens, depths };
}
async function complete(s: Server, prefix: number[], n: number, probs = 0, cachePrompt = true): Promise<Completion> {
    const r = await http<Completion>(s.url, "/completion", { prompt: prefix, id_slot: SLOT, cache_prompt: cachePrompt,
        n_predict: n, temperature: 0, n_probs: probs, post_sampling_probs: false,
        return_tokens: true, stream: false });
    if (r.id_slot !== SLOT || r.truncated || !Array.isArray(r.tokens)) throw new Error(`completion invalid/truncated: ${r.id_slot}, ${r.truncated}`);
    return r;
}
async function save(s: Server, file: string): Promise<number> {
    const r = await http<{ n_saved: number; n_written: number }>(s.url, `/slots/${SLOT}?action=save`, { filename: file });
    if (!r.n_saved || !r.n_written) throw new Error(`slot save failed: ${file}`);
    return r.n_saved;
}
async function restore(s: Server, file: string, n: number): Promise<void> {
    const r = await http<{ n_restored: number; n_read: number }>(s.url, `/slots/${SLOT}?action=restore`, { filename: file });
    if (r.n_restored !== n || !r.n_read) throw new Error(`slot restore failed: ${file}, expected ${n}, got ${r.n_restored}`);
}
async function prefill(s: Server, prefix: number[], file?: string): Promise<number> {
    const r = await complete(s, prefix, 0, 0, false); // force independent prefill
    if (r.tokens_evaluated !== prefix.length) throw new Error("prefill prompt count mismatch");
    return file ? save(s, file) : prefix.length;
}
async function sample(s: Server, prefix: number[], cap: number, expected: string, label: string): Promise<Sample> {
    const r = await complete(s, prefix, cap);
    const rate = hit(r, label);
    const final = finalAnswer(r.content);
    const result: Sample = { content: r.content, tokens: r.tokens, stop: r.stop_type, final, correct: final === expected, hit: rate, capped: r.stop_type === "limit" };
    console.log(`  ${label}: ${r.tokens_predicted} output tokens, stop=${r.stop_type}, FINAL=${final ?? "missing"}, correct=${result.correct}`);
    return result;
}
async function score(s: Server, prefix: number[], reference: number[], regions: Region[], nProbs: number, vocab: number, label: string): Promise<Score> {
    const steps: Score["steps"] = [], top1: number[] = [], nll: Array<number | null> = [];
    let first = 0;
    let previousPromptTokens: number | undefined;
    for (const [j, { index: i, label: region }] of regions.entries()) {
        // All models see the SAME Q6 history, including unscored tokens between
        // regions; we never compare distributions after free runs diverge.
        const r = await complete(s, [...prefix, ...reference.slice(0, i)], 1, nProbs);
        const reuse = checkScoreCache(r.tokens_evaluated, r.timings?.cache_n, r.timings?.prompt_n, previousPromptTokens);
        if (j === 0) first = reuse.reusedFraction;
        if (j === 0 || reuse.newlyAdded > 1) {
            console.log(`  ${label} ${region} step ${i}: reused ${reuse.cached}/${previousPromptTokens ?? r.tokens_evaluated} previously available tokens; evaluated ${reuse.newlyAdded} new + ${reuse.replayed} replayed`);
        }
        previousPromptTokens = r.tokens_evaluated;
        const p = r.completion_probabilities?.[0];
        if (r.tokens_predicted !== 1 || !p?.top_logprobs?.length) throw new Error(`${label}: missing probability step ${i}`);
        if (EXACT_KL && p.top_logprobs.length !== vocab) throw new Error(`${label}: full-vocab probabilities unavailable`);
        const top = new Map(p.top_logprobs.map(x => [x.id, x.logprob]));
        if (top.size !== p.top_logprobs.length) throw new Error(`${label}: duplicate token IDs in probabilities`);
        steps.push(top); top1.push(p.top_logprobs[0].id);
        const lp = p.id === reference[i] ? p.logprob : top.get(reference[i]);
        nll.push(lp === undefined ? null : -lp); // missing top-N => censored, not zero
        if ((j + 1) % 8 === 0 || j + 1 === regions.length) console.log(`  ${label} scored ${j + 1}/${regions.length} regions (last index ${i})`);
    }
    return { steps, top1, nll, hit: first };
}
function compare(ref: Score, run: Score, partitions: Array<Set<number>>) {
    const perStepKl = ref.steps.map((p, i) => klOnPartition(p, run.steps[i], partitions[i], EXACT_KL));
    const valid = perStepKl.filter((x): x is KlEstimate => "nats" in x);
    return { klQ6ToRunNats: mean(valid.map(x => x.nats)), klValidPositions: valid.length,
        klTotalPositions: perStepKl.length, klByRegion: perStepKl,
        q6Top1Agreement: mean(ref.top1.map((x, i) => Number(x === run.top1[i]))),
        q6TokenNllKnownNats: mean(run.nll.filter((x): x is number => x !== null)),
        q6TokenNllCoverage: run.nll.filter(x => x !== null).length / run.nll.length,
        nllByRegionNats: run.nll, cacheHit: run.hit };
}
function sharedPartitions(scores: Score[]): Array<Set<number>> {
    return scores[0].steps.map((_, i) => sharedPartition(scores.map(s => s.steps[i]), EXACT_KL));
}
function loadTasks(): TaskFile[] {
    const selected = process.env.EVAL_TASK;
    const length = process.env.EVAL_LENGTH;
    if (length && !/^(50|75)k$/.test(length)) throw new Error("EVAL_LENGTH must be 50k or 75k");
    const files = readdirSync(TASK_DIR).filter(f => f.endsWith(".json") && (
        selected ? f === `${selected}.json` : process.env.EVAL_SUITE === "all" || /^withdrawal_graph-/.test(f)
    ) && (!length || f.includes(`-${length}-`))).sort();
    if (!files.length) throw new Error(`No matching tasks in ${TASK_DIR}; generate via npx tsx src/test/weight-kv-tasks.ts`);
    return files.map(f => JSON.parse(readFileSync(join(TASK_DIR, f), "utf8")) as TaskFile);
}
async function main(): Promise<void> {
    const tasks = loadTasks();
    if (EXACT_KL) console.log("WARNING: EVAL_EXACT_KL=1 transfers entire vocab for every scoring step. Very expensive.");
    mkdirSync(OUT_DIR, { recursive: true });
    const results: ResultRow[] = [];
    // One task per server pair. This bounds scratch disk usage to one long
    // slot file rather than keeping all tasks' KV files simultaneously.
    for (const task of tasks) {
        // Per-task slot files can be several GiB at long context. Temporary
        // Q6 cache files are removed after each task.
        const slotDir = mkdtempSync(join(tmpdir(), "weight-kv-eval-"));
        let s: Server | undefined;
        try {
            s = await launch("q6", slotDir);
            const vocab = (await http<{ data: Array<{ meta: { n_vocab: number } }> }>(s.url, "/v1/models")).data[0].meta.n_vocab;
            if (!Number.isSafeInteger(vocab) || vocab <= 0) throw new Error("Q6 n_vocab unavailable");
            const nProbs = EXACT_KL ? vocab : Math.min(TOP_K, vocab);
            const p = await prepare(s, task);
            const q6Ctx = s.nCtx;
            if (p.tokens.length + MARGIN + 1 >= q6Ctx) throw new Error(`${task.id}: Q6 context too small`);
            console.log(`\n[${task.id}] Q6 independent`);
            const q6File = `${task.id}-q6.bin`;
            const nQ6 = await prefill(s, p.tokens, q6File);
            const q6Cap = Math.min(MAX_OUTPUT, q6Ctx - p.tokens.length - MARGIN);
            const q6Output = await sample(s, p.tokens, q6Cap, task.expected, "Q6");
            const reference = q6Output.tokens;
            if (!reference.length) throw new Error(`${task.id}: Q6 returned no reference tokens`);
            const regions = selectRegions(reference.length, HEAD, MIDDLE, TAIL);
            console.log(`  scoring ${regions.length} positions across ${reference.length} Q6-generated tokens: ${regions.map(r => `${r.index}:${r.label}`).join(", ")}`);
            // Independent reference distribution, without restore.
            await prefill(s, p.tokens);
            // Preserve Q6 output before region scoring; the latter can fail
            // independently of generation or the Q6->Q3 handoff.
            writeFileSync(join(OUT_DIR, `${task.id}-q6-stage.json`), JSON.stringify({
                id: task.id, targetTokens: task.targetTokens, actualTokens: p.tokens.length,
                promptSha256: sha(p.prompt), expected: task.expected, depths: p.depths,
                output: { ...q6Output, tokens: undefined },
            }, null, 2));
            const q6Score = await score(s, p.tokens, reference, regions, nProbs, vocab, "Q6");
            await stop(s); s = undefined;

            s = await launch("q3", slotDir);
            const q3Vocab = (await http<{ data: Array<{ meta: { n_vocab: number } }> }>(s.url, "/v1/models")).data[0].meta.n_vocab;
            if (q3Vocab !== vocab) throw new Error("Q3 vocab size differs");
            if (await formatPrompt(s, p.hay, task.question) !== p.prompt) throw new Error(`${task.id}: Q3/Q6 template mismatch`);
            const actual = await tokenize(s, p.prompt, true);
            if (JSON.stringify(actual) !== JSON.stringify(p.tokens)) throw new Error(`${task.id}: Q3/Q6 tokenizer mismatch`);
            const { tokens } = p;
            if (tokens.length + MARGIN + 1 >= s.nCtx) throw new Error(`${task.id}: Q3 context too small`);
            // All free runs get the same cap; clamp to the smaller model's context.
            const cap = Math.min(q6Cap, s.nCtx - tokens.length - MARGIN);
            if (cap < q6Cap) console.warn(`Q3 output cap (${cap}) is below Q6 cap (${q6Cap}); compare accuracy with this caveat.`);
            console.log(`\n[${task.id}] Q3 independent / Q6->Q3 hybrid (output cap ${cap})`);
            await prefill(s, tokens);
            const q3 = await sample(s, tokens, cap, task.expected, "Q3");
            await prefill(s, tokens); // fresh independent Q3 scoring prefill
            const q3Score = await score(s, tokens, reference, regions, nProbs, vocab, "Q3");
            await restore(s, q6File, nQ6);
            const hybrid = await sample(s, tokens, cap, task.expected, "Q6->Q3");
            await restore(s, q6File, nQ6);
            const hybridScore = await score(s, tokens, reference, regions, nProbs, vocab, "Q6->Q3");
            const partitions = sharedPartitions([q6Score, q3Score, hybridScore]);
            const q3Metrics = compare(q6Score, q3Score, partitions);
            const hybridMetrics = compare(q6Score, hybridScore, partitions);
            const paired = regions.flatMap((r, i) => {
                const a = q3Metrics.klByRegion[i], b = hybridMetrics.klByRegion[i];
                return "nats" in a && "nats" in b ? [{ index: r.index, label: r.label, q3: a.nats, hybrid: b.nats }] : [];
            });
            const row: ResultRow = { id: task.id, kind: task.kind, target: task.targetTokens, actual: tokens.length,
                expected: task.expected, depths: p.depths,
                samples: { q6: q6Output, q3, hybrid },
                stats: { q6: compare(q6Score, q6Score, partitions), q3: q3Metrics, hybrid: hybridMetrics,
                    klPaired: { validPositions: paired.length, q3MeanNats: mean(paired.map(x => x.q3)), hybridMeanNats: mean(paired.map(x => x.hybrid)), positions: paired },
                    regions, klType: EXACT_KL ? "full-vocab" : `lower bound: common top-${nProbs} + OTHER`,
                    reference: "selected start/middle/tail positions on same full Q6-generated continuation" } };
            results.push(row);
            writeFileSync(join(OUT_DIR, `${task.id}.json`), JSON.stringify({ task: { ...task, hayParagraphs: undefined }, promptSha256: sha(p.prompt),
                modelContext: { q6: q6Ctx, q3: s.nCtx }, outputCap: { q6: q6Cap, others: cap }, result: row }, null, 2));
            const klPaired = row.stats.klPaired as { validPositions: number; q3MeanNats: number | null; hybridMeanNats: number | null };
            console.log(`  correct: Q6=${q6Output.correct}, Q3=${q3.correct}, hybrid=${hybrid.correct}`);
            console.log(`  paired KL(Q6 || Q3 / hybrid): ${klPaired.q3MeanNats?.toFixed(5) ?? "n/a"} / ${klPaired.hybridMeanNats?.toFixed(5) ?? "n/a"} nats (lower bounds on ${klPaired.validPositions}/${regions.length} positions)`);
        } finally {
            if (s) await stop(s);
            rmSync(slotDir, { recursive: true, force: true });
        }
    }
    const q3MissHybridHit = results.filter(r => !r.samples.q3.correct && r.samples.hybrid.correct).length;
    const q3HitHybridMiss = results.filter(r => r.samples.q3.correct && !r.samples.hybrid.correct).length;
    const summary = { taskCount: results.length, accuracy: Object.fromEntries(["q6", "q3", "hybrid"].map(k => [k, results.filter(r => r.samples[k].correct).length])),
        paired: { q3MissHybridHit, q3HitHybridMiss }, rows: results.map(r => ({ id: r.id, kind: r.kind, target: r.target, actual: r.actual, depths: r.depths, correct: Object.fromEntries(Object.entries(r.samples).map(([k, v]) => [k, v.correct])) })) };
    const summaryFile = process.env.EVAL_TASK ? `summary-${process.env.EVAL_TASK}.json` : process.env.EVAL_SUITE === "all" ? `summary-all${process.env.EVAL_LENGTH ? `-${process.env.EVAL_LENGTH}` : ""}.json` : `summary-withdrawal-graph${process.env.EVAL_LENGTH ? `-${process.env.EVAL_LENGTH}` : ""}.json`;
    writeFileSync(join(OUT_DIR, summaryFile), JSON.stringify(summary, null, 2));
    console.log(`\nSummary: ${JSON.stringify(summary.accuracy)}, hybrid fixes ${q3MissHybridHit} Q3 misses / introduces ${q3HitHybridMiss}; see ${join(OUT_DIR, summaryFile)}`);
}
main().catch(e => { console.error("evaluation failed:", e); process.exitCode = 1; });
