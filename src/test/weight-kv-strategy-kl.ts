// Teacher-forced KL for the four inference strategies. Uses the exact token
// IDs saved by the 32g-q6-f16 free run, never re-tokenized output text.
// Run *after* generating a 32g-q6-f16 result with outputTokenIds:
//   EVAL_TASK=withdrawal_graph-50k-seed50060 npx tsx src/test/weight-kv-strategy-kl.ts
// Default: new seeds 50062–50069 and 75062–75069. Use EVAL_TASKS=<comma-
// separated IDs> for small batches or EVAL_TASK=<one ID>. Options:
// EVAL_LENGTH=10k|25k|50k|75k, EVAL_HEAD=4, EVAL_MIDDLE=4, EVAL_TAIL=12,
// EVAL_BOUNDARY=4, EVAL_TOP_K=128, EVAL_EXACT_KL=1, LLAMA_BIN=...
// Does NOT rerun free generation. Writes separate files in logs/weight-kv-strategy-kl/.
import { createHash } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type { TaskFile } from "./weight-kv-tasks.js";
import { requestJson } from "./weight-kv-http.js";
import { selectTaskNames } from "./weight-kv-task-selection.js";
import { checkScoreCache, klOnPartition, sharedPartition, type KlEstimate } from "./weight-kv-metrics.js";
import { STRATEGIES, evaluatedBoundary, stageCapacity, strategyScorePoints, validateStrategy, type ScorePoint, type Stage, type Strategy } from "./weight-kv-strategy-plan.js";

const ROOT = resolve(import.meta.dirname, "../..");
const BIN = process.env.LLAMA_BIN ?? join(ROOT, "llama.cpp/build/bin/llama-server");
const TASKS = join(ROOT, "attic/tasks");
const FREE_DIR = join(ROOT, "logs/weight-kv-strategies");
const OUT = join(ROOT, "logs/weight-kv-strategy-kl");
const MODELS: Record<Stage["quant"], string> = {
    q6: "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q6_K.gguf",
    q4: "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    q3: "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-IQ3_S.gguf",
};
const SLOT = 0;
const N_TOP = integer("EVAL_TOP_K", 128);
const EXACT = process.env.EVAL_EXACT_KL === "1";
const HEAD = integer("EVAL_HEAD", 4);
const MIDDLE = integer("EVAL_MIDDLE", 4);
const TAIL = integer("EVAL_TAIL", 12);
const BOUNDARY = integer("EVAL_BOUNDARY", 4);
const DEADLINE_MS = 4 * 60 * 60 * 1000;
interface Server { proc: ChildProcess; url: string; log: string; nCtx: number; error?: Error }
interface Completion { id_slot: number; truncated: boolean; tokens_evaluated: number; tokens_predicted: number;
    timings: { cache_n: number; prompt_n: number };
    completion_probabilities?: Array<{ id: number; logprob: number; top_logprobs: Array<{ id: number; logprob: number }> }> }
interface FreeReference { inputTokens: number; outputTokens: number; outputText: string; outputTokenIds?: number[];
    promptSha256: string; stop: string; strategy: string }
interface Prepared { prompt: string; hay: string; tokens: number[] }
interface Scored { steps: Array<Map<number, number>>; top1: number[]; nll: Array<number | null>;
    stages: string[]; events: Array<{ type: string; quant: string; position: number; cached?: number; processed?: number }> }
function integer(name: string, defaultValue: number) {
    const n = Number(process.env[name] ?? defaultValue);
    if (!Number.isSafeInteger(n) || n < 1) throw new Error(`${name} must be a positive integer`);
    return n;
}
function sha(text: string) { return createHash("sha256").update(text).digest("hex"); }
function mean(values: number[]): number | null { return values.length ? values.reduce((sum, n) => sum + n, 0) / values.length : null; }
function same(a: number[], b: number[]) { return a.length === b.length && a.every((v, i) => v === b[i]); }
function tasks(): TaskFile[] {
    return selectTaskNames(TASKS, process.env).map(f => JSON.parse(readFileSync(join(TASKS, f), "utf8")) as TaskFile);
}
function reference(task: TaskFile): FreeReference & { outputTokenIds: number[] } {
    const path = join(FREE_DIR, `${task.id}-32g-q6-f16.json`);
    let free: FreeReference;
    try { free = JSON.parse(readFileSync(path, "utf8")) as FreeReference; }
    catch { throw new Error(`Missing ${path}. Run EVAL_TASK=${task.id} EVAL_STRATEGY=32g-q6-f16 npx tsx src/test/weight-kv-strategies.ts first.`); }
    if (free.strategy !== "32g-q6-f16" || free.stop !== "eos" ||
        !Array.isArray(free.outputTokenIds) || free.outputTokenIds.length !== free.outputTokens ||
        !free.outputTokenIds.every(x => Number.isSafeInteger(x) && x >= 0)) {
        throw new Error(`No valid exact Q6 reference token IDs in ${path}; rerun only 32g-q6-f16 with updated strategies script.`);
    }
    return free as FreeReference & { outputTokenIds: number[] };
}
function renderHay(paragraphs: string[], needles: TaskFile["needles"]) {
    const list = paragraphs.slice();
    const planned = needles.map((n, i) => ({ n, i, idx: Math.floor(paragraphs.length * n.depth / 100) })).sort((a, b) => a.idx - b.idx);
    for (const [shift, { n, idx }] of planned.entries()) list.splice(idx + shift, 0, n.text);
    return list.join("\n\n");
}
function userPrompt(hay: string, question: string) {
    return `Read the archive excerpts below. The answer must come from the archive cross-references or memos, not outside knowledge.\n\n${hay}\n\nQuestion: ${question}\n\nYou may reason through the evidence. End with exactly one line in this form: FINAL: <16-character code>.`;
}
async function formatted(server: Server, hay: string, task: TaskFile) {
    return (await requestJson<{ prompt: string }>(server.url, "/apply-template", { messages: [{ role: "user", content: userPrompt(hay, task.question) }] })).prompt;
}
async function tokenize(server: Server, prompt: string) {
    return (await requestJson<{ tokens: number[] }>(server.url, "/tokenize", { content: prompt, add_special: true, parse_special: true })).tokens;
}
async function prepare(server: Server, task: TaskFile, free: FreeReference): Promise<Prepared> {
    if (sha(task.hayParagraphs.join("\n\n")) !== task.haySha256) throw new Error(`${task.id}: hay checksum mismatch`);
    const cache = new Map<number, { prompt: string; count: number }>();
    async function probe(n: number) {
        if (!cache.has(n)) {
            const prompt = await formatted(server, renderHay(task.hayParagraphs.slice(0, n), task.needles), task);
            cache.set(n, { prompt, count: (await tokenize(server, prompt)).length });
        }
        return cache.get(n)!;
    }
    let lo = 1, hi = task.hayParagraphs.length;
    if ((await probe(hi)).count < task.targetTokens) throw new Error(`${task.id}: insufficient hay`);
    while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if ((await probe(mid)).count < task.targetTokens) lo = mid + 1; else hi = mid; }
    const prev = Math.max(1, lo - 1);
    const chosen = Math.abs((await probe(prev)).count - task.targetTokens) < Math.abs((await probe(lo)).count - task.targetTokens) ? prev : lo;
    const hay = renderHay(task.hayParagraphs.slice(0, chosen), task.needles);
    const prompt = (await probe(chosen)).prompt;
    const tokens = await tokenize(server, prompt);
    if (tokens.length !== free.inputTokens || sha(prompt) !== free.promptSha256) {
        throw new Error(`${task.id}: reconstructed prompt differs from saved Q6 free run; refusing to compare`);
    }
    return { hay, prompt, tokens };
}
async function port() {
    return new Promise<number>((ok, fail) => {
        const socket = net.createServer(); socket.once("error", fail);
        socket.listen(0, "127.0.0.1", () => {
            const a = socket.address();
            if (!a || typeof a === "string") { socket.close(); fail(new Error("port allocation failed")); return; }
            socket.close(() => ok(a.port));
        });
    });
}
async function stop(s: Server) {
    if (s.error || s.proc.exitCode !== null || s.proc.signalCode !== null) return;
    const done = new Promise<void>(ok => s.proc.once("exit", () => ok()));
    s.proc.kill("SIGTERM");
    if (!await Promise.race([done.then(() => true), delay(10_000).then(() => false)])) s.proc.kill("SIGKILL");
    await done;
}
async function launch(stage: Stage, task: TaskFile, strategy: Strategy, i: number, slotDir: string): Promise<Server> {
    const p = await port();
    const log = join(OUT, `${task.id}-${strategy.id}-${i}-${stage.quant}.log`);
    const fd = openSync(log, "w");
    let proc: ChildProcess;
    try {
        proc = spawn(BIN, ["-m", MODELS[stage.quant], "--host", "127.0.0.1", "--port", String(p), "-ngl", "99", "--parallel", "1",
            "--ctx-size", String(stage.ctx), "--slot-save-path", slotDir, "--no-context-shift", "--reasoning", "on",
            "--cache-type-k", stage.kv, "--cache-type-v", stage.kv], { stdio: ["ignore", fd, fd] });
    } finally { closeSync(fd); }
    const s: Server = { proc, url: `http://127.0.0.1:${p}`, log, nCtx: 0 };
    proc.on("error", e => { s.error = e; });
    try {
        const deadline = Date.now() + 300_000;
        while (Date.now() < deadline) {
            if (s.error) throw s.error;
            if (proc.exitCode !== null || proc.signalCode !== null) throw new Error(`server exited: ${proc.exitCode}/${proc.signalCode}`);
            try {
                const slots = await requestJson<Array<{ n_ctx: number }>>(s.url, "/slots", undefined, 2000);
                if (slots.length === 1 && slots[0].n_ctx > 0) {
                    if (slots[0].n_ctx !== stage.ctx) throw new Error(`ctx mismatch: ${slots[0].n_ctx} != ${stage.ctx}`);
                    s.nCtx = slots[0].n_ctx;
                    return s;
                }
            } catch (e) { if (e instanceof Error && e.message.startsWith("ctx mismatch:")) throw e; }
            await delay(500);
        }
        throw new Error("startup timeout");
    } catch (e) { await stop(s); throw new Error(`${String(e)}; see ${log}`); }
}
async function nextToken(s: Server, prompt: number[], nProbs: number): Promise<Completion> {
    const r = await requestJson<Completion>(s.url, "/completion", { prompt, id_slot: SLOT, cache_prompt: true,
        n_predict: 1, temperature: 0, return_tokens: true, n_probs: nProbs, post_sampling_probs: false, stream: false }, DEADLINE_MS);
    if (r.id_slot !== SLOT || r.truncated || r.tokens_predicted !== 1 || r.tokens_evaluated !== prompt.length) {
        throw new Error(`unexpected scoring completion: slot=${r.id_slot}, truncated=${r.truncated}, n_pred=${r.tokens_predicted}, input=${r.tokens_evaluated}/${prompt.length}`);
    }
    return r;
}
async function evaluate(s: Server, prompt: number[], oldLength: number | undefined, events: Scored["events"], stage: Stage, kind: string, nProbs = 0) {
    const r = await nextToken(s, prompt, nProbs);
    // The very first independent prefill is expected to start from an empty
    // cache. Later scoring requests and cross-model restores must retain the
    // previously evaluated prefix, except a few server-replayed tokens.
    if (oldLength !== undefined) {
        const reuse = checkScoreCache(r.tokens_evaluated, r.timings?.cache_n, r.timings?.prompt_n, oldLength);
        events.push({ type: kind, quant: stage.quant, position: prompt.length, cached: reuse.cached, processed: reuse.processed });
    } else {
        if (r.timings.cache_n !== 0 || r.timings.prompt_n < prompt.length) throw new Error("independent prefill unexpectedly reused state");
        events.push({ type: kind, quant: stage.quant, position: prompt.length, cached: 0, processed: r.timings.prompt_n });
    }
    return r;
}
async function save(s: Server, file: string, expected: number) {
    const r = await requestJson<{ n_saved: number; n_written: number }>(s.url, `/slots/${SLOT}?action=save`, { filename: file });
    if (r.n_saved !== expected || !r.n_written) throw new Error(`state save mismatch: ${r.n_saved} != ${expected}`);
}
async function restore(s: Server, file: string, expected: number) {
    const r = await requestJson<{ n_restored: number; n_read: number }>(s.url, `/slots/${SLOT}?action=restore`, { filename: file });
    if (r.n_restored !== expected || !r.n_read) throw new Error(`state restore mismatch: ${r.n_restored} != ${expected}`);
}
async function scoreStrategy(task: TaskFile, strategy: Strategy, canonical: Prepared, referenceIds: number[], points: ScorePoint[], vocab: number, nProbs: number): Promise<Scored> {
    const dir = mkdtempSync(join(tmpdir(), "kv-strategy-kl-"));
    const steps: Scored["steps"] = [], top1: number[] = [], nll: Array<number | null> = [], stages: string[] = [], events: Scored["events"] = [];
    let s: Server | undefined;
    let known = 0, stageIndex = 0;
    const full = canonical.tokens;
    try {
        validateStrategy(strategy, full.length);
        async function startStage() {
            const stage = strategy.stages[stageIndex];
            s = await launch(stage, task, strategy, stageIndex, dir);
            if (await formatted(s, canonical.hay, task) !== canonical.prompt || !same(await tokenize(s, canonical.prompt), full)) {
                throw new Error(`${strategy.id}: ${stage.quant} template or tokenizer differs from Q6 reference`);
            }
            const currentVocab = (await requestJson<{ data: Array<{ meta: { n_vocab: number } }> }>(s.url, "/v1/models")).data[0].meta.n_vocab;
            if (currentVocab !== vocab) throw new Error(`${strategy.id}: vocab differs from Q6 reference`);
            if (stageIndex) {
                const file = `stage-${stageIndex - 1}.bin`;
                await restore(s, file, known);
                events.push({ type: "restore", quant: stage.quant, position: known });
            }
        }
        async function switchStage() {
            if (!s || stageIndex === strategy.stages.length - 1) throw new Error(`${strategy.id}: cannot hand off further at position ${known}`);
            const prev = strategy.stages[stageIndex];
            const next = strategy.stages[stageIndex + 1];
            if (prev.kv !== next.kv) throw new Error("unsupported KV format switch within multi-quant ladder");
            await save(s, `stage-${stageIndex}.bin`, known);
            events.push({ type: "save", quant: prev.quant, position: known });
            await stop(s); s = undefined;
            stageIndex++;
            await startStage();
        }
        await startStage();
        // At 75k Q6 can only prefill its first ~53k tokens on the 24 GiB
        // strategy. Handoff to Q4 BEFORE scoring any continuation tokens.
        while (full.length > stageCapacity(strategy.stages[stageIndex]) && stageIndex < strategy.stages.length - 1) {
            const stage = strategy.stages[stageIndex];
            const boundary = evaluatedBoundary(stage, full.length);
            if (known < boundary) {
                await evaluate(s!, full.slice(0, boundary), known || undefined, events, stage, "prefill-to-handoff");
                known = boundary;
            }
            await switchStage();
        }
        const history = (n: number) => [...full, ...referenceIds.slice(0, Math.max(0, n - full.length))].slice(0, n);
        for (const { index, label } of points) {
            const wanted = full.length + index;
            // At the boundary, outgoing model samples the last token; its KV
            // contains only the evaluated prefix. The next position belongs
            // to the incoming model, which evaluates that sampled token.
            while (wanted > evaluatedBoundary(strategy.stages[stageIndex], full.length) && stageIndex < strategy.stages.length - 1) {
                const stage = strategy.stages[stageIndex];
                const boundary = evaluatedBoundary(stage, full.length);
                if (known > boundary) throw new Error(`${strategy.id}: already crossed handoff boundary`);
                if (known < boundary) {
                    await evaluate(s!, history(boundary), known || undefined, events, stage, "advance-to-handoff");
                    known = boundary;
                }
                await switchStage();
            }
            const stage = strategy.stages[stageIndex];
            if (wanted > evaluatedBoundary(stage, full.length)) throw new Error(`${strategy.id}: position ${wanted} exceeds final context budget`);
            // n_predict=1 evaluates the exact reference prefix and samples one
            // throwaway token. The next request forces the reference token ID.
            // For large gaps, this also advances KV through the intervening
            // Q6 tokens; their sampled output is never appended to the prompt.
            const r = await evaluate(s!, history(wanted), known || undefined, events, stage, `score:${label}`, nProbs);
            known = wanted;
            const p = r.completion_probabilities?.[0];
            if (!p?.top_logprobs?.length || (EXACT && p.top_logprobs.length !== vocab)) throw new Error(`missing full distribution at ${index}`);
            const map = new Map(p.top_logprobs.map(x => [x.id, x.logprob]));
            if (map.size !== p.top_logprobs.length) throw new Error(`duplicate probability IDs at ${index}`);
            steps.push(map); top1.push(p.top_logprobs[0].id); stages.push(stage.quant);
            const lp = p.id === referenceIds[index] ? p.logprob : map.get(referenceIds[index]);
            nll.push(lp === undefined ? null : -lp);
        }
        return { steps, top1, nll, stages, events };
    } finally { if (s) await stop(s); rmSync(dir, { recursive: true, force: true }); }
}
function compare(ref: Scored, other: Scored, partitions: Array<Set<number>>) {
    const steps = ref.steps.map((p, i) => klOnPartition(p, other.steps[i], partitions[i], EXACT));
    const valid = steps.filter((s): s is KlEstimate => "nats" in s);
    const knownNll = other.nll.filter((n): n is number => n !== null);
    return { klMeanNats: mean(valid.map(v => v.nats)), klValidPositions: valid.length, klTotalPositions: steps.length, klByPosition: steps,
        q6Top1Agreement: mean(ref.top1.map((id, i) => Number(id === other.top1[i]))),
        q6TokenNllKnownNats: mean(knownNll), q6TokenNllCoverage: knownNll.length / steps.length,
        nllByPositionNats: other.nll, quantAtPosition: other.stages, events: other.events };
}
async function run(task: TaskFile) {
    const free = reference(task);
    const multi = STRATEGIES.find(s => s.id === "24g-q6-q4-q3-f16")!;
    const points = strategyScorePoints(free.outputTokenIds.length, free.inputTokens, multi, HEAD, MIDDLE, TAIL, BOUNDARY);
    const strategies = STRATEGIES.filter(s => s.id !== "24g-q6-q4-q4q8kv");
    const all: Record<string, Scored> = {};
    // Rebuild the reference prompt once with Q6. Each scoring stage then
    // verifies its own template and tokenization before using its KV cache.
    const dir = mkdtempSync(join(tmpdir(), "kv-strategy-kl-prepare-"));
    let server: Server | undefined;
    let canonical: Prepared;
    let vocab: number;
    console.log(`[${task.id}] ${points.length} scoring positions across ${free.outputTokens} Q6 reference tokens`);
    try {
        const referenceStrategy = strategies.find(s => s.id === "32g-q6-f16")!;
        server = await launch(referenceStrategy.stages[0], task, referenceStrategy, 0, dir);
        canonical = await prepare(server, task, free);
        vocab = (await requestJson<{ data: Array<{ meta: { n_vocab: number } }> }>(server.url, "/v1/models")).data[0].meta.n_vocab;
        if (!Number.isSafeInteger(vocab) || vocab <= 0) throw new Error("Q6 reference vocab unavailable");
    } finally { if (server) await stop(server); rmSync(dir, { recursive: true, force: true }); }
    const nProbs = EXACT ? vocab : Math.min(vocab, N_TOP);
    for (const strategy of strategies) {
        const scoreFile = join(OUT, `${task.id}-${strategy.id}-scores.json`);
        const referenceHash = sha(JSON.stringify(free.outputTokenIds));
        // Resume a partially completed comparison without recomputing another
        // 50k/75k-token prefill. Incompatible settings cannot be reused.
        try {
            const stored = JSON.parse(readFileSync(scoreFile, "utf8")) as {
                promptSha256: string; referenceTokenIdsSha256: string; positions: ScorePoint[];
                nProbs: number; exactKL: boolean; score: Omit<Scored, "steps"> & { steps: Array<Array<[number, number]>> };
            };
            if (stored.promptSha256 === free.promptSha256 && stored.referenceTokenIdsSha256 === referenceHash &&
                stored.nProbs === nProbs && stored.exactKL === EXACT && JSON.stringify(stored.positions) === JSON.stringify(points) &&
                stored.score.steps.length === points.length && stored.score.nll.length === points.length &&
                stored.score.top1.length === points.length && stored.score.stages.length === points.length &&
                stored.score.steps.every(step => step.length > 0 && step.every(([id, logp]) => Number.isSafeInteger(id) && typeof logp === "number"))) {
                all[strategy.id] = { ...stored.score, steps: stored.score.steps.map(step => new Map(step)) };
                console.log(`  ${strategy.id}: reused ${points.length} saved scoring positions`);
                continue;
            }
        } catch { /* absent or incompatible score file; evaluate again */ }
        all[strategy.id] = await scoreStrategy(task, strategy, canonical, free.outputTokenIds, points, vocab, nProbs);
        // Retain exact distributions if another strategy fails later: this
        // task can take a long time, and top-N scoring cannot be reconstructed
        // from a final answer or an earlier run's KL summary.
        writeFileSync(scoreFile, JSON.stringify({
            task: task.id, strategy: strategy.id, promptSha256: free.promptSha256,
            referenceTokenIdsSha256: referenceHash,
            positions: points, nProbs, exactKL: EXACT,
            score: { ...all[strategy.id], steps: all[strategy.id].steps.map(step => [...step]) },
        }, null, 2));
        console.log(`  ${strategy.id}: scored ${points.length} positions`);
    }
    const q6 = all["32g-q6-f16"];
    const partitions = q6.steps.map((_, i) => sharedPartition(strategies.map(s => all[s.id].steps[i]), EXACT));
    const scores = Object.fromEntries(strategies.map(s => [s.id, compare(q6, all[s.id], partitions)]));
    const other = strategies.filter(s => s.id !== "32g-q6-f16");
    const paired = points.flatMap((point, i) => {
        const vals = other.map(s => scores[s.id].klByPosition[i]);
        return vals.every(v => "nats" in v) ? [{ ...point, values: Object.fromEntries(other.map((s, j) => [s.id, (vals[j] as KlEstimate).nats])) }] : [];
    });
    const result = { task: task.id, q6ReferenceTokens: free.outputTokens, promptSha256: free.promptSha256,
        positions: points, klType: EXACT ? "full-vocab" : `common top-${N_TOP} + OTHER lower bound`, scores,
        pairedPositions: paired.length, pairedMeanNats: Object.fromEntries(other.map(s => [s.id, mean(paired.map(x => x.values[s.id]))])) };
    writeFileSync(join(OUT, `${task.id}.json`), JSON.stringify(result, null, 2));
    console.log(`  comparable positions=${paired.length}/${points.length}; paired KL: ${JSON.stringify(result.pairedMeanNats)}`);
}
async function main() {
    mkdirSync(OUT, { recursive: true });
    for (const task of tasks()) await run(task);
}
main().catch(e => { console.error("strategy KL failed:", e); process.exitCode = 1; });
