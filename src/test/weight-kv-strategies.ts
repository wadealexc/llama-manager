// Compare context-budget strategies on the existing withdrawal-graph tasks.
// From repo root: npx tsx src/test/weight-kv-strategies.ts
// Default: Q6/f16 -> Q6/q8 -> Q4/f16 -> Q4/q8 on all 50k and 75k
// withdrawal-graph tasks (20 runs). Use EVAL_LENGTH=50k or 75k for one
// bracket, or EVAL_TASK/EVAL_TASKS for specific task IDs. The 10k/25k
// trajectories never reach the first boundary and need no rerun.
// EVAL_STRATEGY=<id> and LLAMA_BIN=... are optional.
// Does not use llama-manager; do not run in parallel with another GPU server.
// No arbitrary output-token cap: stop at EOS or the last stage's context limit.
import { createHash } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type { TaskFile } from "./weight-kv-tasks.js";
import { requestJson } from "./weight-kv-http.js";
import { selectTaskNames } from "./weight-kv-task-selection.js";
import { CHUNK, CTX_MARGIN, STRATEGIES, canRestoreAcrossSwap, decodeBudget, evaluatedTokens, firstPrefillEnd, isKvReload, kvReloadBody, validateStrategy, type Stage, type Strategy } from "./weight-kv-strategy-plan.js";

const ROOT = resolve(import.meta.dirname, "../..");
const BIN = process.env.LLAMA_BIN ?? join(ROOT, "llama.cpp/build/bin/llama-server");
const OUT = join(ROOT, "logs/weight-kv-strategies");
const TASK_DIR = join(ROOT, "attic/tasks");
const MODEL: Record<Stage["quant"], string> = {
    q6: "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q6_K.gguf",
    q4: "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    q3: "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-IQ3_S.gguf",
};
const SLOT = 0;
const REQUEST_TIMEOUT_MS = 4 * 60 * 60 * 1000;
interface Server { proc: ChildProcess; url: string; log: string; nCtx: number; spawnError?: Error }
interface Completion {
    content: string; tokens: number[]; id_slot: number; stop_type: string; truncated: boolean;
    tokens_evaluated: number; tokens_predicted: number;
    timings: { cache_n: number; prompt_n: number };
}
interface Prepared { task: TaskFile; prompt: string; tokens: number[]; hay: string }
interface Event { type: string; quant: string; kv: string; atToken: number; durationMs: number; cached?: number; processed?: number; bytes?: number; generated?: number; stop?: string; nCtx?: number }
interface Result {
    task: string; strategy: string; stages: readonly Stage[]; modelPaths: string[];
    ctxMargin: number; chunkTokens: number; promptSha256: string;
    inputTokens: number; expected: string;
    final: string | null; correct: boolean; stop: string; outputTokens: number; outputText: string;
    outputTokenIds?: number[]; // exact reference continuation for teacher-forced KL
    timeToCorrectFinalObservedMs: number | null; tokensWhenCorrectFinalObserved: number | null;
    elapsedMs: number; totalWallMs: number; taskPreparationMs: number;
    prefillMs: number; decodeMs: number; swapMs: number;
    decodeTokensPerSecond: number | null; generatedByQuant: Partial<Record<Stage["quant"], number>>;
    generatedByStage: Array<{ quant: Stage["quant"]; kv: Stage["kv"]; tokens: number }>;
    events: Event[]; contexts: number[];
}
function hash(s: string) { return createHash("sha256").update(s).digest("hex"); }
function finalLine(s: string): string | null {
    const last = s.trim().split(/\r?\n/).at(-1)?.trim() ?? "";
    return /^FINAL:\s*([a-fA-F0-9]{16})\s*$/i.exec(last)?.[1].toUpperCase() ?? null;
}
function selectTasks(): TaskFile[] {
    return selectTaskNames(TASK_DIR, process.env).map(f => JSON.parse(readFileSync(join(TASK_DIR, f), "utf8")) as TaskFile);
}
function selectStrategies(): readonly Strategy[] {
    const selected = process.env.EVAL_STRATEGY ?? "24g-q6-q6q8-q4-q4q8kv";
    const found = STRATEGIES.filter(s => s.id === selected);
    if (!found.length) throw new Error(`Unknown EVAL_STRATEGY: ${selected}`);
    return found;
}
function renderHay(paragraphs: string[], needles: TaskFile["needles"]): string {
    const items = paragraphs.slice();
    const planned = needles.map((n, i) => ({ n, i, idx: Math.floor(paragraphs.length * n.depth / 100) })).sort((a, b) => a.idx - b.idx);
    for (const [shift, { n, idx }] of planned.entries()) items.splice(idx + shift, 0, n.text);
    return items.join("\n\n");
}
function userPrompt(hay: string, question: string): string {
    return `Read the archive excerpts below. The answer must come from the archive cross-references or memos, not outside knowledge.\n\n${hay}\n\nQuestion: ${question}\n\nYou may reason through the evidence. End with exactly one line in this form: FINAL: <16-character code>.`;
}
async function formatted(s: Server, hay: string, task: TaskFile): Promise<string> {
    const r = await requestJson<{ prompt: string }>(s.url, "/apply-template", { messages: [{ role: "user", content: userPrompt(hay, task.question) }] });
    return r.prompt;
}
async function tokenize(s: Server, text: string): Promise<number[]> {
    return (await requestJson<{ tokens: number[] }>(s.url, "/tokenize", { content: text, add_special: true, parse_special: true })).tokens;
}
async function prepare(s: Server, task: TaskFile): Promise<Prepared> {
    if (hash(task.hayParagraphs.join("\n\n")) !== task.haySha256) throw new Error(`${task.id}: hay checksum mismatch`);
    const cache = new Map<number, { prompt: string; count: number }>();
    async function probe(n: number) {
        if (!cache.has(n)) {
            const prompt = await formatted(s, renderHay(task.hayParagraphs.slice(0, n), task.needles), task);
            cache.set(n, { prompt, count: (await tokenize(s, prompt)).length });
        }
        return cache.get(n)!;
    }
    let lo = 1, hi = task.hayParagraphs.length;
    if ((await probe(hi)).count < task.targetTokens) throw new Error(`insufficient hay for ${task.id}`);
    while (lo < hi) { const m = Math.floor((lo + hi) / 2); if ((await probe(m)).count < task.targetTokens) lo = m + 1; else hi = m; }
    const before = Math.max(1, lo - 1);
    const chosen = Math.abs((await probe(before)).count - task.targetTokens) < Math.abs((await probe(lo)).count - task.targetTokens) ? before : lo;
    const hay = renderHay(task.hayParagraphs.slice(0, chosen), task.needles);
    const prompt = (await probe(chosen)).prompt;
    const tokens = await tokenize(s, prompt);
    if (Math.abs(tokens.length - task.targetTokens) > 400) throw new Error(`${task.id}: incorrect prompt length ${tokens.length}`);
    return { task, hay, prompt, tokens };
}
async function freePort(): Promise<number> {
    return new Promise((ok, fail) => {
        const socket = net.createServer(); socket.once("error", fail);
        socket.listen(0, "127.0.0.1", () => {
            const a = socket.address();
            if (!a || typeof a === "string") { socket.close(); fail(new Error("port allocation failed")); return; }
            socket.close(() => ok(a.port));
        });
    });
}
async function stop(s: Server) {
    if (s.spawnError || s.proc.exitCode !== null || s.proc.signalCode !== null) return;
    const done = new Promise<void>(resolveExit => s.proc.once("exit", () => resolveExit()));
    s.proc.kill("SIGTERM");
    if (!await Promise.race([done.then(() => true), delay(10_000).then(() => false)])) s.proc.kill("SIGKILL");
    await done;
}
async function launch(stage: Stage, taskId: string, strategyId: string, index: number, slotDir: string): Promise<Server> {
    const p = await freePort();
    const log = join(OUT, `${taskId}-${strategyId}-${index}-${stage.quant}.log`);
    const fd = openSync(log, "w");
    let proc: ChildProcess;
    try {
        const args = ["-m", MODEL[stage.quant], "--host", "127.0.0.1", "--port", String(p), "-ngl", "99", "--parallel", "1",
            "--ctx-size", String(stage.ctx), "--slot-save-path", slotDir, "--no-context-shift", "--reasoning", "on",
            "--cache-type-k", stage.kv, "--cache-type-v", stage.kv];
        // Slot restore converts KV storage precision, but does not rotate
        // cached vectors into/out of the Hadamard basis used by quantized KV.
        // Force every stage (and subsequent /reload) into the same basis.
        proc = spawn(BIN, args, {
            stdio: ["ignore", fd, fd],
            env: { ...process.env, LLAMA_ATTN_ROT_DISABLE: "1" },
        });
    } finally { closeSync(fd); }
    const s: Server = { url: `http://127.0.0.1:${p}`, log, proc, nCtx: 0 };
    proc.on("error", e => { s.spawnError = e; });
    console.log(`  starting ${stage.quant}/${stage.kv} ctx=${stage.ctx}; log ${log}`);
    try {
        const deadline = Date.now() + 300_000;
        while (Date.now() < deadline) {
            if (s.spawnError) throw s.spawnError;
            if (proc.exitCode !== null || proc.signalCode !== null) throw new Error(`server exited: code=${proc.exitCode}, signal=${proc.signalCode}`);
            try {
                // Explicit short timeout only for readiness; long /completion calls
                // have a separate four-hour deadline and no implicit headers timeout.
                const slots = await requestJson<Array<{ n_ctx: number }>>(s.url, "/slots", undefined, 2000);
                if (slots.length === 1 && slots[0].n_ctx > 0) {
                    s.nCtx = slots[0].n_ctx;
                    if (s.nCtx !== stage.ctx) throw new Error(`expected ctx=${stage.ctx}, server reported ${s.nCtx}; do not silently alter strategy`);
                    return s;
                }
            } catch (e) {
                if (e instanceof Error && e.message.includes("do not silently alter strategy")) throw e;
            }
            await delay(500);
        }
        throw new Error("server readiness timeout");
    } catch (e) { await stop(s); throw new Error(`${String(e)}; see ${s.log}`); }
}
async function completion(s: Server, prompt: number[], nPredict: number, cachePrompt = true): Promise<Completion> {
    const r = await requestJson<Completion>(s.url, "/completion", { prompt, id_slot: SLOT, cache_prompt: cachePrompt,
        n_predict: nPredict, temperature: 0, return_tokens: true, stream: false, n_probs: 0 }, REQUEST_TIMEOUT_MS);
    if (r.id_slot !== SLOT || r.truncated || !Array.isArray(r.tokens) || !["eos", "limit"].includes(r.stop_type)) {
        throw new Error(`completion invalid/truncated: ${JSON.stringify({ slot: r.id_slot, truncated: r.truncated, stop: r.stop_type })}`);
    }
    return r;
}
async function save(s: Server, file: string): Promise<{ n: number; bytes: number }> {
    const r = await requestJson<{ n_saved: number; n_written: number }>(s.url, `/slots/${SLOT}?action=save`, { filename: file });
    if (!r.n_saved || !r.n_written) throw new Error(`save failed: ${JSON.stringify(r)}`);
    return { n: r.n_saved, bytes: r.n_written };
}
async function restore(s: Server, file: string, n: number) {
    const r = await requestJson<{ n_restored: number; n_read: number }>(s.url, `/slots/${SLOT}?action=restore`, { filename: file });
    if (r.n_restored !== n || !r.n_read) throw new Error(`restore mismatch: saved ${n}, restored ${JSON.stringify(r)}`);
    return r.n_read;
}
async function reload(s: Server, next: Stage): Promise<void> {
    const r = await requestJson<{ success: boolean; n_ctx: number; message?: string }>(s.url, "/reload", kvReloadBody(next), REQUEST_TIMEOUT_MS);
    if (!r.success || r.n_ctx !== next.ctx) {
        throw new Error(`reload to ${next.quant}/${next.kv} ctx=${next.ctx} failed: ${JSON.stringify(r)}`);
    }
    const slots = await requestJson<Array<{ n_ctx: number }>>(s.url, "/slots");
    if (slots.length !== 1 || slots[0].n_ctx !== next.ctx) throw new Error(`reload slot ctx mismatch: ${JSON.stringify(slots)}`);
    s.nCtx = next.ctx;
}
function checkReuse(r: Completion, prefixLength: number, label: string, slack = 16) {
    const { cache_n: cached, prompt_n: processed } = r.timings ?? {};
    if (!Number.isSafeInteger(cached) || !Number.isSafeInteger(processed) || cached + processed < r.tokens_evaluated || cached < prefixLength - slack) {
        throw new Error(`${label}: lost prior KV (cached=${cached}, previous=${prefixLength}, processed=${processed}, total=${r.tokens_evaluated})`);
    }
    return { cached, processed };
}
function verifyContinuation(r: Completion, expectedPromptLength: number, label: string) {
    if (r.tokens_evaluated !== expectedPromptLength) throw new Error(`${label}: prompt mismatch ${r.tokens_evaluated} != ${expectedPromptLength}`);
    if (r.tokens_predicted !== r.tokens.length || r.tokens.length === 0) throw new Error(`${label}: generated token count mismatch`);
}

function sameTokens(a: number[], b: number[]): boolean {
    return a.length === b.length && a.every((v, i) => v === b[i]);
}
async function execute(task: TaskFile, strategy: Strategy, verifyPrompt: (p: Prepared) => void): Promise<Result> {
    const slotDir = mkdtempSync(join(tmpdir(), "kv-strategy-"));
    let s: Server | undefined;
    const events: Event[] = [], contexts: number[] = [];
    const started = performance.now();
    let taskPreparationMs = 0;
    const elapsedMs = () => performance.now() - started - taskPreparationMs;
    let input: number[] = [], inputPos = 0, generated: number[] = [], text = "";
    let originalPrompt: Prepared | undefined;
    let firstCorrectMs: number | null = null, firstCorrectTokens: number | null = null;
    let stopReason = "unknown";
    const output = join(OUT, `${task.id}-${strategy.id}.json`);
    if (existsSync(output)) throw new Error(`Refusing to overwrite existing result ${output}`);
    function progress() {
        writeFileSync(join(OUT, `${task.id}-${strategy.id}-progress.json`), JSON.stringify({ task: task.id, strategy: strategy.id,
            inputTokens: input.length, prefilledTokens: inputPos, outputTokens: generated.length,
            elapsedMs: elapsedMs(), taskPreparationMs,
            timeToCorrectFinalObservedMs: firstCorrectMs,
            tokensWhenCorrectFinalObserved: firstCorrectTokens,
            outputText: text, events }, null, 2));
    }
    try {
        for (const [index, stage] of strategy.stages.entries()) {
            const previous = index ? strategy.stages[index - 1] : undefined;
            const kvReload = previous && isKvReload(previous, stage);
            const slotFile = `handoff-${index - 1}.bin`;
            if (kvReload) {
                if (!s) throw new Error(`no live ${stage.quant} server for KV reload`);
                const begin = performance.now();
                await reload(s, stage);
                contexts.push(s.nCtx);
                events.push({ type: "reload", quant: stage.quant, kv: stage.kv, atToken: inputPos + generated.length,
                    nCtx: s.nCtx, durationMs: performance.now() - begin });
            } else {
                const loadStart = performance.now();
                s = await launch(stage, task.id, strategy.id, index, slotDir);
                contexts.push(s.nCtx);
                events.push({ type: "load", quant: stage.quant, kv: stage.kv, atToken: inputPos + generated.length, durationMs: performance.now() - loadStart });
            }
            if (index === 0) {
                const prepareStart = performance.now();
                const prepared = await prepare(s, task);
                verifyPrompt(prepared); // enforce identical template and token IDs across strategies
                taskPreparationMs = performance.now() - prepareStart;
                input = prepared.tokens;
                // Keep the original prompt for cross-stage tokenizer/template checks.
                originalPrompt = prepared;
                validateStrategy(strategy, input.length);
                console.log(`  ${task.id}: actual input ${input.length} tokens, expected ${task.expected}`);
            } else {
                if (!kvReload && !canRestoreAcrossSwap(previous!, stage)) throw new Error("unsupported model/KV format handoff");
                if (!originalPrompt || await formatted(s, originalPrompt.hay, task) !== originalPrompt.prompt ||
                    !sameTokens(await tokenize(s, originalPrompt.prompt), input)) {
                    throw new Error(`${strategy.id}: ${stage.quant} template/tokenizer differs; refusing to restore`);
                }
                // Separate filenames per hop: a stale .ckpt sidecar from an
                // earlier model must never accompany a later stage's slot.
                const begin = performance.now();
                const bytes = await restore(s, slotFile, evaluatedTokens(inputPos, generated.length));
                events.push({ type: "restore", quant: stage.quant, kv: stage.kv, atToken: inputPos + generated.length, bytes, durationMs: performance.now() - begin });
            }
            // For 75k inputs: Q6 prefills its first ~53k tokens, Q4 then
            // processes the remaining input. The discarded n_predict=0 sample
            // is NOT considered generated output and is NOT added to the prompt.
            if (inputPos < input.length) {
                if (generated.length) throw new Error("cannot prefill after decoding has begun");
                const end = firstPrefillEnd(input.length, stage);
                if (end <= inputPos) throw new Error(`${strategy.id}: prefill cannot advance at ${stage.quant}`);
                const begin = performance.now();
                const r = await completion(s, input.slice(0, end), 0, index > 0);
                if (r.tokens_evaluated !== end) throw new Error(`prefill mismatch: ${r.tokens_evaluated} != ${end}`);
                if (index > 0) {
                    const { cached } = checkReuse(r, inputPos, `${stage.quant} prefill handoff`);
                    console.log(`  ${stage.quant} prefill handoff: cached ${cached}/${inputPos}, new ${end - inputPos}`);
                }
                events.push({ type: "prefill", quant: stage.quant, kv: stage.kv, atToken: end, cached: r.timings.cache_n, processed: r.timings.prompt_n, durationMs: performance.now() - begin });
                inputPos = end;
                progress();
            }
            if (inputPos === input.length) {
                if (generated.length && index > 0) {
                    // Slot state contains all but the final sampled token. The
                    // first decode request will evaluate that missing token.
                    console.log(`  ${stage.quant}: resuming at generated token ${generated.length}`);
                }
                while (true) {
                    const n = decodeBudget(input.length, generated.length, stage);
                    if (n === 0) { stopReason = "stage_context_limit"; break; }
                    const prefix = [...input, ...generated];
                    const begin = performance.now();
                    const r = await completion(s, prefix, n);
                    verifyContinuation(r, prefix.length, `${stage.quant} decode`);
                    const { cached, processed } = checkReuse(r, prefix.length - (generated.length ? 1 : 0), `${stage.quant} decode`);
                    generated.push(...r.tokens);
                    text += r.content;
                    events.push({ type: "decode", quant: stage.quant, kv: stage.kv, atToken: input.length + generated.length,
                        generated: r.tokens.length, cached, processed, stop: r.stop_type, durationMs: performance.now() - begin });
                    if (firstCorrectMs === null && finalLine(text) === task.expected) {
                        firstCorrectMs = elapsedMs();
                        firstCorrectTokens = generated.length; // upper bound: chunk end
                    }
                    console.log(`  ${stage.quant}: generated ${generated.length} tokens, ${(elapsedMs() / 1000).toFixed(1)}s elapsed${firstCorrectMs !== null ? " (correct FINAL observed)" : ""}`);
                    progress();
                    if (r.stop_type === "eos") { stopReason = "eos"; break; }
                    if (r.tokens.length !== n) throw new Error(`unexpected early generation stop: ${r.stop_type}, ${r.tokens.length}/${n}`);
                }
            }
            if (stopReason === "eos") break;
            if (index === strategy.stages.length - 1) { stopReason = inputPos < input.length ? "prompt_exceeds_context" : "context_exhausted"; break; }
            const begin = performance.now();
            const saved = await save(s, `handoff-${index}.bin`);
            const expected = evaluatedTokens(inputPos, generated.length);
            if (saved.n !== expected) throw new Error(`saved ${saved.n} != expected evaluated tokens ${expected}`);
            events.push({ type: "save", quant: stage.quant, kv: stage.kv, atToken: inputPos + generated.length, bytes: saved.bytes, durationMs: performance.now() - begin });
            if (!isKvReload(stage, strategy.stages[index + 1])) {
                const unloadBegin = performance.now();
                await stop(s); s = undefined;
                events.push({ type: "unload", quant: stage.quant, kv: stage.kv, atToken: inputPos + generated.length, durationMs: performance.now() - unloadBegin });
            }
            progress();
        }
        const final = finalLine(text);
        const totalMs = (type: string) => events.filter(e => e.type === type).reduce((sum, e) => sum + e.durationMs, 0);
        const decodeMs = totalMs("decode");
        const generatedByQuant: Result["generatedByQuant"] = {};
        const generatedByStage: Result["generatedByStage"] = [];
        for (const stage of strategy.stages) {
            const tokens = events.filter(e => e.type === "decode" && e.quant === stage.quant && e.kv === stage.kv)
                .reduce((sum, e) => sum + (e.generated ?? 0), 0);
            generatedByStage.push({ quant: stage.quant, kv: stage.kv, tokens });
            generatedByQuant[stage.quant] = (generatedByQuant[stage.quant] ?? 0) + tokens;
        }
        // Do not silently replace an existing 50k Q6 reference with a different
        // greedy completion when rerunning solely to recover exact token IDs.
        if (strategy.id === "32g-q6-f16" && existsSync(output)) {
            const previous = JSON.parse(readFileSync(output, "utf8")) as Result;
            if (previous.promptSha256 !== hash(originalPrompt!.prompt) ||
                previous.outputText !== text || previous.outputTokens !== generated.length ||
                (previous.outputTokenIds && !sameTokens(previous.outputTokenIds, generated))) {
                const candidate = output.replace(/\.json$/, "-reference-mismatch.json");
                writeFileSync(candidate, JSON.stringify({ previousTextSha256: hash(previous.outputText), newTextSha256: hash(text),
                    previousPromptSha256: previous.promptSha256, newPromptSha256: hash(originalPrompt!.prompt), generated }, null, 2));
                throw new Error(`Q6 reference differs from prior free run; original result preserved. See ${candidate}`);
            }
        }
        const result: Result = { task: task.id, strategy: strategy.id, stages: strategy.stages,
            modelPaths: strategy.stages.map(x => MODEL[x.quant]), ctxMargin: CTX_MARGIN, chunkTokens: CHUNK,
            promptSha256: hash(originalPrompt!.prompt), inputTokens: input.length,
            expected: task.expected, final, correct: final === task.expected,
            stop: stopReason, outputTokens: generated.length, outputText: text,
            outputTokenIds: strategy.id === "32g-q6-f16" ? generated : undefined,
            timeToCorrectFinalObservedMs: firstCorrectMs,
            tokensWhenCorrectFinalObserved: firstCorrectTokens, elapsedMs: elapsedMs(),
            totalWallMs: performance.now() - started, taskPreparationMs,
            prefillMs: totalMs("prefill"), decodeMs,
            swapMs: totalMs("save") + totalMs("unload") + events.slice(1).filter(e => e.type === "load").reduce((sum, e) => sum + e.durationMs, 0) + totalMs("reload") + totalMs("restore"),
            decodeTokensPerSecond: decodeMs ? generated.length / (decodeMs / 1000) : null, generatedByQuant, generatedByStage,
            events, contexts };
        writeFileSync(output, JSON.stringify(result, null, 2));
        console.log(`  result ${strategy.id}: ${result.stop}, correct=${result.correct}, output=${result.outputTokens} tokens, time=${(result.elapsedMs / 1000).toFixed(1)}s`);
        return result;
    } finally {
        if (s) await stop(s);
        rmSync(slotDir, { recursive: true, force: true });
    }
}
async function main() {
    mkdirSync(OUT, { recursive: true });
    const strategies = selectStrategies();
    let tasks: TaskFile[];
    if (strategies[0].id === "24g-q6-q6q8-q4-q4q8kv" &&
        !process.env.EVAL_TASK && !process.env.EVAL_TASKS &&
        (!process.env.EVAL_LENGTH || ["50k", "75k"].includes(process.env.EVAL_LENGTH))) {
        const lengths = process.env.EVAL_LENGTH ? [process.env.EVAL_LENGTH.slice(0, -1)] : ["50", "75"];
        const ids = lengths.flatMap(length => Array.from({ length: 10 }, (_, seed) => `withdrawal_graph-${length}k-seed${length}06${seed}`));
        tasks = ids.map(id => JSON.parse(readFileSync(join(TASK_DIR, `${id}.json`), "utf8")) as TaskFile);
        console.log(`New four-stage ladder: ${tasks.length} tasks (${ids[0]} … ${ids.at(-1)})`);
    } else if (strategies[0].id === "24g-q6-q4-q4q8kv" &&
        !process.env.EVAL_TASK && !process.env.EVAL_TASKS && !process.env.EVAL_LENGTH) {
        // Look across ALL completed old-ladder runs, including the first two
        // pilot seeds (75060/75061), not just selectTasks()' default cohort.
        const ids = readdirSync(OUT).flatMap(file => {
            const match = /^(withdrawal_graph-(?:10|25|50|75)k-seed\d+)-24g-q6-q4-q3-f16\.json$/.exec(file);
            if (!match) return [];
            try {
                const old = JSON.parse(readFileSync(join(OUT, file), "utf8")) as Result;
                return old.task === match[1] && old.strategy === "24g-q6-q4-q3-f16" && (old.generatedByQuant.q3 ?? 0) > 0 ? [match[1]] : [];
            } catch { return []; }
        }).sort();
        if (!ids.length) throw new Error("no old-ladder runs reached Q3; pass EVAL_TASKS explicitly");
        tasks = ids.map(id => JSON.parse(readFileSync(join(TASK_DIR, `${id}.json`), "utf8")) as TaskFile);
        console.log(`New ladder pilot (old runs reached Q3): ${ids.join(", ")}`);
    } else tasks = selectTasks();
    const results: Array<Pick<Result, "task" | "strategy" | "correct" | "stop" | "inputTokens" | "outputTokens" | "elapsedMs" | "totalWallMs" | "taskPreparationMs" | "timeToCorrectFinalObservedMs" | "tokensWhenCorrectFinalObserved" | "generatedByQuant" | "prefillMs" | "decodeMs" | "swapMs">> = [];
    const errors: Array<{ task: string; strategy: string; error: string }> = [];
    const summaryFile = strategies[0].id === "24g-q6-q6q8-q4-q4q8kv" ? "summary-q6q8-q4q8.json" :
        strategies[0].id === "24g-q6-q4-q4q8kv" ? "summary-q4kv-pilot.json" : "summary.json";
    for (const task of tasks) {
        let canonical: { prompt: string; tokens: number[] } | undefined;
        const verifyPrompt = (p: Prepared) => {
            if (canonical) {
                if (canonical.prompt !== p.prompt || !sameTokens(canonical.tokens, p.tokens)) {
                    throw new Error(`${task.id}: prompt/template/tokenizer mismatch between strategies`);
                }
            } else canonical = { prompt: p.prompt, tokens: p.tokens };
        };
        for (const strategy of strategies) {
            console.log(`\n[${task.id}] ${strategy.id}`);
            try {
                const path = join(OUT, `${task.id}-${strategy.id}.json`);
                if (existsSync(path)) {
                    console.log(`  existing result preserved; skipping ${path}`);
                    continue;
                }
                const r = await execute(task, strategy, verifyPrompt);
                results.push({ task: r.task, strategy: r.strategy, correct: r.correct, stop: r.stop, inputTokens: r.inputTokens,
                    outputTokens: r.outputTokens, elapsedMs: r.elapsedMs, totalWallMs: r.totalWallMs,
                    taskPreparationMs: r.taskPreparationMs, timeToCorrectFinalObservedMs: r.timeToCorrectFinalObservedMs,
                    tokensWhenCorrectFinalObserved: r.tokensWhenCorrectFinalObserved, generatedByQuant: r.generatedByQuant,
                    prefillMs: r.prefillMs, decodeMs: r.decodeMs, swapMs: r.swapMs });
            } catch (err) {
                console.error(`${task.id}/${strategy.id} failed:`, err);
                // Infrastructure errors are not model failures; leave progress and
                // logs intact, then continue to the next independent strategy.
                errors.push({ task: task.id, strategy: strategy.id, error: String(err) });
                process.exitCode = 1;
            }
            const aggregates = Object.fromEntries(strategies.map(({ id }) => {
                const completed = results.filter(r => r.strategy === id);
                return [id, {
                    completed: completed.length,
                    correct: completed.filter(r => r.correct).length,
                    outputTokens: completed.reduce((sum, r) => sum + r.outputTokens, 0),
                    elapsedMs: completed.reduce((sum, r) => sum + r.elapsedMs, 0),
                }];
            }));
            writeFileSync(join(OUT, summaryFile), JSON.stringify({ results, errors, aggregates }, null, 2));
        }
    }
}
main().catch(err => { console.error("strategy benchmark failed:", err); process.exitCode = 1; });
