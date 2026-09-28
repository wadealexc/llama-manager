// Experiment: reuse a Q6 model's slot state with a Q4 quantization of the same model.
//
// Run from the repo root: node --experimental-strip-types src/test/weight-kv-handoff.ts
// (or: npx tsx src/test/weight-kv-handoff.ts)
// Optional: LLAMA_BIN=/path/to/llama-server node --experimental-strip-types ...
// Requires the custom llama.cpp fork, both GGUFs, and enough GPU memory to load each
// model individually. Does not use llama-manager's routing or modify config.yaml.

import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, rmSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

const ROOT = resolve(import.meta.dirname, "../..");
const BIN = process.env.LLAMA_BIN ?? resolve(ROOT, "llama.cpp/build/bin/llama-server");
const Q6_MODEL = "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q6_K.gguf";
const Q4_MODEL = "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q4_K_XL.gguf";
const LOG_DIR = resolve(ROOT, "logs/weight-kv-handoff");
const SLOT_ID = 0;
const Q6_CTX = 2048;
const Q4_CTX = 4096;
const STARTUP_TIMEOUT_MS = 180_000;
const REQUEST_TIMEOUT_MS = 300_000;
const MIN_HIT_RATE = 0.95;

// Keep the prefill long enough that a one-token re-evaluation still yields ~100% hits.
// The codeword also provides an informal check of information retained from the prefix.
const CODEWORD = `cobalt-${Math.floor(Math.random() * 1_000_000)}`;
const USER_PROMPT = `Remember this codeword: ${CODEWORD}. In your answer, include it, then write at least five paragraphs explaining in plain English how an LLM can reuse a KV cache after switching to a lower-precision quantization of the same model. Mention both the benefit and the main caveat.\n\n` +
    "Background: A causal transformer stores keys and values from earlier tokens at each attention layer. A later token can attend to these stored representations without recomputing the entire earlier prefix. Weight quantization changes how the representations are computed, but does not normally change the architecture or the dimensionality of attention. A mixed-precision handoff uses the old cached representations as an approximation, not as the exact cache the lower-precision model would have computed. The answer should distinguish cache compatibility from guaranteed output quality.\n\n".repeat(6);

interface SlotInfo { id: number; n_ctx: number }
interface TokenizeResult { tokens: number[] }
interface TemplateResult { prompt: string }
interface CompletionResult {
    content: string;
    tokens: number[];
    id_slot: number;
    tokens_predicted: number;
    tokens_evaluated: number;
    // This is the count already held in the slot at response time, NOT the cache hit metric.
    tokens_cached: number;
    truncated: boolean;
    stop_type: string;
    timings: { cache_n: number; prompt_n: number };
}
interface SlotActionResult {
    id_slot: number;
    filename: string;
    n_saved?: number;
    n_restored?: number;
    n_written?: number;
    n_read?: number;
}
interface Server { url: string; proc: ChildProcess; log: string }

async function freePort(): Promise<number> {
    return new Promise((ok, fail) => {
        const server = net.createServer();
        server.once("error", fail);
        server.listen(0, "127.0.0.1", () => {
            const address = server.address();
            if (!address || typeof address === "string") {
                server.close();
                fail(new Error("could not allocate a TCP port"));
                return;
            }
            server.close(() => ok(address.port));
        });
    });
}

async function post<T>(url: string, path: string, body: unknown): Promise<T> {
    const response = await fetch(url + path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`POST ${path}: HTTP ${response.status}: ${text.slice(0, 2000)}`);
    return JSON.parse(text) as T;
}

async function startServer(label: string, model: string, ctx: number, slotDir: string): Promise<Server> {
    const port = await freePort();
    const log = join(LOG_DIR, `${label}.log`);
    const fd = openSync(log, "w");
    let proc: ChildProcess;
    try {
        proc = spawn(BIN, [
            "--model", model, "--host", "127.0.0.1", "--port", String(port),
            "--parallel", "1", "--ctx-size", String(ctx), "--n-gpu-layers", "99",
            "--slot-save-path", slotDir, "--no-context-shift", "--reasoning", "off",
            // Deliberately keep KV precision constant across the weight switch.
            "--cache-type-k", "f16", "--cache-type-v", "f16",
        ], { stdio: ["ignore", fd, fd] });
    } finally {
        closeSync(fd);
    }

    let spawnError: Error | undefined;
    proc.on("error", err => { spawnError = err; });
    const url = `http://127.0.0.1:${port}`;
    const started: Server = { url, proc, log };
    console.log(`[${label}] loading ${model} (ctx=${ctx}); log: ${log}`);
    try {
        const deadline = Date.now() + STARTUP_TIMEOUT_MS;
        while (Date.now() < deadline) {
            if (spawnError) throw spawnError;
            if (proc.exitCode !== null || proc.signalCode !== null) {
                throw new Error(`server exited during startup (code=${proc.exitCode}, signal=${proc.signalCode})`);
            }
            try {
                const response = await fetch(url + "/slots", { signal: AbortSignal.timeout(2000) });
                if (response.ok) {
                    const slots = await response.json() as SlotInfo[];
                    if (slots.length === 1 && slots[0].n_ctx > 0) {
                        console.log(`[${label}] ready (slot ctx=${slots[0].n_ctx})`);
                        return started;
                    }
                }
            } catch { /* still loading */ }
            await delay(500);
        }
        throw new Error(`startup timed out after ${STARTUP_TIMEOUT_MS} ms`);
    } catch (err) {
        await stopServer(started);
        throw new Error(`${label}: ${String(err)}; see ${log}`);
    }
}

async function stopServer(server: Server): Promise<void> {
    const { proc } = server;
    if (proc.exitCode !== null || proc.signalCode !== null) return;
    const exited = new Promise<void>(done => proc.once("exit", () => done()));
    proc.kill("SIGTERM");
    if (!await Promise.race([exited.then(() => true), delay(10_000).then(() => false)])) {
        proc.kill("SIGKILL");
    }
    await exited;
}

async function complete(url: string, prompt: number[], nPredict: number, ignoreEos = false): Promise<CompletionResult> {
    const result = await post<CompletionResult>(url, "/completion", {
        prompt,
        id_slot: SLOT_ID,
        cache_prompt: true,
        n_predict: nPredict,
        return_tokens: true,
        stream: false,
        temperature: 0,
        // Keep the Q6 answer unfinished so Q4 can continue in the middle of it.
        ignore_eos: ignoreEos,
    });
    if (result.id_slot !== SLOT_ID || result.truncated) {
        throw new Error(`unexpected completion result: slot=${result.id_slot}, truncated=${result.truncated}`);
    }
    return result;
}

async function save(url: string, filename: string): Promise<number> {
    const result = await post<SlotActionResult>(url, `/slots/${SLOT_ID}?action=save`, { filename });
    if (result.id_slot !== SLOT_ID || !result.n_saved || !result.n_written) {
        throw new Error(`unexpected slot save response: ${JSON.stringify(result)}`);
    }
    console.log(`    saved ${result.n_saved} tokens (${(result.n_written / 1024 ** 2).toFixed(1)} MiB): ${filename}`);
    return result.n_saved;
}

async function restore(url: string, filename: string, nSaved: number): Promise<void> {
    const result = await post<SlotActionResult>(url, `/slots/${SLOT_ID}?action=restore`, { filename });
    console.log(`    restored ${result.n_restored} tokens (${((result.n_read ?? 0) / 1024 ** 2).toFixed(1)} MiB)`);
    if (result.id_slot !== SLOT_ID || result.n_restored !== nSaved || !result.n_read) {
        throw new Error(`restore mismatch: saved=${nSaved}, result=${JSON.stringify(result)}`);
    }
}

// /completion's "tokens_cached" is the slot's size after generation, NOT its
// cache hit count. Use per-request timings (the slot's counters reset on release).
function reportCache(result: CompletionResult, label: string): void {
    const cached = result.timings?.cache_n;
    const processed = result.timings?.prompt_n;
    const total = result.tokens_evaluated;
    if (typeof cached !== "number" || typeof processed !== "number" || total <= 0) {
        throw new Error(`no prompt cache statistics: completion=${JSON.stringify(result)}`);
    }
    const rate = cached / total;
    console.log(`    ${label}: prompt=${total}, cached=${cached}, processed=${processed}, hit=${(rate * 100).toFixed(2)}%`);
    console.log(`    generated=${result.tokens_predicted}, stop=${result.stop_type}; /completion tokens_cached=${result.tokens_cached} (slot size, not hit count)`);
    if (rate < MIN_HIT_RATE || cached + processed < total) {
        throw new Error(`low cache reuse or inconsistent counts: ${label} (${(rate * 100).toFixed(2)}% hit)`);
    }
}

async function main(): Promise<void> {
    mkdirSync(LOG_DIR, { recursive: true });
    const slotDir = mkdtempSync(join(tmpdir(), "llama-weight-kv-"));
    let server: Server | undefined;
    try {
        server = await startServer("q6", Q6_MODEL, Q6_CTX, slotDir);
        const { prompt } = await post<TemplateResult>(server.url, "/apply-template", {
            messages: [{ role: "user", content: USER_PROMPT }],
        });
        const { tokens: prefix } = await post<TokenizeResult>(server.url, "/tokenize", {
            content: prompt, add_special: true, parse_special: true,
        });
        if (prefix.length < 200 || prefix.length + 256 >= Q6_CTX) {
            throw new Error(`prompt has ${prefix.length} tokens; expected 200..${Q6_CTX - 256}. Adjust USER_PROMPT/Q6_CTX.`);
        }
        console.log(`\nprompt: ${prefix.length} token IDs; codeword=${CODEWORD}`);

        // Scenario 1: evaluate the prefix with Q6 and save before Q6 generates
        // any tokens that have been evaluated into KV. This server still samples
        // one token for n_predict=0; that sampled token is not yet in the KV.
        console.log("\n[1] Q6 prefix evaluation (n_predict=0)");
        const prefill = await complete(server.url, prefix, 0);
        console.log(`    Q6 n_predict=0: predicted=${prefill.tokens_predicted}, stop=${prefill.stop_type}`);
        const prefillSaved = await save(server.url, "q6-prefill.bin");

        // Scenario 2: make Q6 generate a partial answer; preserve the *actual*
        // generated token IDs, not a re-tokenized version of its text.
        console.log("\n[2] Q6 generates a partial answer");
        const q6 = await complete(server.url, prefix, 80, true);
        if (!Array.isArray(q6.tokens) || q6.tokens.length < 16 || q6.stop_type !== "limit") {
            throw new Error(`Q6 did not generate an unfinished answer: ${JSON.stringify(q6)}`);
        }
        console.log(`    Q6 output (${q6.tokens.length} tokens):\n${q6.content}`);
        const generatedSaved = await save(server.url, "q6-generated.bin");
        if (generatedSaved < prefillSaved) throw new Error("Q6 generation did not extend the saved slot");

        await stopServer(server); // ensure the Q6 model is unloaded before starting Q4
        server = undefined;

        server = await startServer("q4", Q4_MODEL, Q4_CTX, slotDir);
        const q4Template = await post<TemplateResult>(server.url, "/apply-template", {
            messages: [{ role: "user", content: USER_PROMPT }],
        });
        if (q4Template.prompt !== prompt) {
            throw new Error("Q6 and Q4 have different chat templates; handoff prompt would not match");
        }
        const { tokens: q4Prefix } = await post<TokenizeResult>(server.url, "/tokenize", {
            content: prompt, add_special: true, parse_special: true,
        });
        if (JSON.stringify(q4Prefix) !== JSON.stringify(prefix)) {
            throw new Error("Q6 and Q4 tokenize the prompt differently; cannot measure cache reuse with this prompt");
        }

        console.log("\n[1] Q4 generation from Q6-evaluated prefix");
        await restore(server.url, "q6-prefill.bin", prefillSaved);
        const fromPrefill = await complete(server.url, prefix, 200);
        reportCache(fromPrefill, "Q6-prefill -> Q4");
        console.log(`    Q4 answer:\n${fromPrefill.content}`);

        console.log("\n[2] Q4 continues Q6's partial answer (same slot; replaced by second restore)");
        await restore(server.url, "q6-generated.bin", generatedSaved);
        const fullPrefix = [...prefix, ...q6.tokens];
        // The final Q6-sampled token is normally not evaluated/cached yet.
        // Supplying it here causes Q4 to evaluate that token and continue.
        const continuation = await complete(server.url, fullPrefix, 200);
        reportCache(continuation, "Q6-prefill+generation -> Q4");
        console.log(`    Q6 text:\n${q6.content}`);
        console.log(`    Q4 continuation:\n${continuation.content}`);
        console.log(`    Combined:\n${q6.content}${continuation.content}`);
        console.log("\nInspect both answers for coherence and the codeword. Cache hit PASS for both handoffs.");
    } finally {
        if (server) await stopServer(server);
        rmSync(slotDir, { recursive: true, force: true });
        console.log(`server logs: ${LOG_DIR}`);
    }
}

main().catch(err => {
    console.error("weight KV handoff test failed:", err);
    process.exitCode = 1;
});
