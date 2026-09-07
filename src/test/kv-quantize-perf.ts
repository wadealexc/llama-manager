import { setTimeout as delay } from "node:timers/promises";

const ROUTER_URL = process.env.ROUTER_URL ?? "http://127.0.0.1:10002";
const MODEL = process.env.QUANT_TEST_MODEL ?? "qwen3.8-27b";
const PERF_TOKENS = Number(process.env.PERF_TOKENS ?? 50_000);

const FILLER_UNIT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn, then rests beneath the old oak tree while the sun climbs higher. ";

interface Usage {
    prompt_tokens: number;
    completion_tokens: number;
    prompt_tokens_details?: { cached_tokens?: number };
}

interface ChatResponse {
    choices: { message: { content: string; reasoning_content?: string } }[];
    usage: Usage;
}

interface ModelInfo {
    id: string;
    status: { value: string };
}

interface SlotInfo {
    id: number;
    n_ctx: number;
    n_prompt_tokens?: number;
}

interface ReloadInfo {
    success: boolean;
    n_ctx: number;
    message?: string;
}

interface RestoreInfo {
    id_slot: number;
    filename: string;
    n_restored: number;
    n_read: number;
    timings: { restore_ms: number };
}

interface SaveInfo {
    id_slot: number;
    filename: string;
    n_saved: number;
    n_written: number;
    timings: { save_ms: number };
}

type Precision = "q8_0" | "q4_0";

interface HopStats {
    label: string;
    target: Precision;
    save_ms: number;
    reload_ms: number;
    restore_ms: number;
    n_tokens: number;
    n_bytes_written: number;
    n_bytes_read: number;
    n_ctx: number;
}

async function getJson<T>(route: string): Promise<T> {
    const res = await fetch(ROUTER_URL + route);
    const text = await res.text();
    if (!res.ok) {
        throw new Error(`GET ${route} failed (${res.status}): ${text}`);
    }
    return JSON.parse(text) as T;
}

async function postJson<T>(route: string, body: unknown): Promise<T> {
    const res = await fetch(ROUTER_URL + route, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
    });
    const text = await res.text();
    if (!res.ok) {
        throw new Error(`POST ${route} failed (${res.status}): ${text}`);
    }
    return JSON.parse(text) as T;
}

async function ensureLoaded(): Promise<void> {
    console.log(`loading model ${MODEL}...`);
    await postJson("/models/load", { model: MODEL });

    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
        const models = await getJson<{ data: ModelInfo[] }>("/models");
        const info = models.data.find(m => m.id === MODEL);
        if (info?.status.value === "loaded") {
            console.log(`model ${MODEL} loaded`);
            return;
        }
        await delay(250);
    }
    throw new Error(`model ${MODEL} did not reach 'loaded' in time`);
}

async function countTokens(messages: unknown[]): Promise<number> {
    const res = await postJson<{ input_tokens: number }>("/v1/chat/completions/input_tokens", { model: MODEL, messages });
    return res.input_tokens;
}

async function chat(messages: unknown[]): Promise<{ content: string; reasoning: string; usage: Usage; ms: number }> {
    const t = performance.now();
    const body = await postJson<ChatResponse>("/chat/completions", {
        model: MODEL,
        messages,
        stream: false,
        max_tokens: 64,
        temperature: 0,
    });
    const ms = performance.now() - t;

    const msg = body.choices[0].message;
    return {
        content: msg.content ?? "",
        reasoning: msg.reasoning_content ?? "",
        usage: body.usage,
        ms,
    };
}

function reportUsage(u: Usage): void {
    const cached = u.prompt_tokens_details?.cached_tokens ?? 0;
    const total = u.prompt_tokens;
    const hit = total > 0 ? (cached / total * 100).toFixed(1) : "0.0";
    console.log(`    usage: prompt=${total} cached=${cached} (${hit}% hit) completion=${u.completion_tokens}`);
}

function fmtBytes(bytes: number): string {
    const mib = bytes / (1024 ** 2);
    if (mib < 1024) return `${mib.toFixed(1)} MiB`;
    return `${(mib / 1024).toFixed(2)} GiB`;
}

function fmtSec(ms: number): string {
    return `${(ms / 1000).toFixed(2)}s`;
}

async function hop(label: string, target: Precision, slot_id: number, filename: string): Promise<HopStats> {
    console.log(`[${label}] saving slot...`);
    const t_save = performance.now();
    const save = await postJson<SaveInfo>(`/slots/${slot_id}?action=save`, { filename, model: MODEL });
    const save_ms = performance.now() - t_save;
    console.log(`    saved ${save.n_saved} tokens from slot ${slot_id} (${fmtBytes(save.n_written)} written) [server: ${save.timings.save_ms.toFixed(0)} ms, wall: ${fmtSec(save_ms)}]`);

    console.log(`[hop] reloading model with ${target} kv cache (n_ctx = 0)...`);
    const t_reload = performance.now();
    const reload = await postJson<ReloadInfo>("/reload", {
        model: MODEL,
        n_ctx: 0,
        cache_type_k: target,
        cache_type_v: target,
    });
    const reload_ms = performance.now() - t_reload;
    if (!reload.success) {
        throw new Error(`reload failed: ${reload.message ?? "unknown error"}`);
    }
    console.log(`    reloaded; n_ctx = ${reload.n_ctx} [${fmtSec(reload_ms)}]`);

    console.log(`[hop] restoring slot (server converts to ${target} during restore)...`);
    const t_restore = performance.now();
    const restore = await postJson<RestoreInfo>(`/slots/${slot_id}?action=restore`, { filename, model: MODEL });
    const restore_ms = performance.now() - t_restore;
    console.log(`    restored ${restore.n_restored} tokens from ${filename} (${fmtBytes(restore.n_read)} read) [server: ${restore.timings.restore_ms.toFixed(0)} ms, wall: ${fmtSec(restore_ms)}]`);

    return {
        label,
        target,
        save_ms,
        reload_ms,
        restore_ms,
        n_tokens: restore.n_restored,
        n_bytes_written: save.n_written,
        n_bytes_read: restore.n_read,
        n_ctx: reload.n_ctx,
    };
}

function printSummary(stats: HopStats[]): void {
    console.log("\n[summary]");
    for (const s of stats) {
        console.log(`${s.label} (${s.target}):`);
        console.log(`    save ${fmtSec(s.save_ms)} | reload ${fmtSec(s.reload_ms)} | restore (incl. conversion) ${fmtSec(s.restore_ms)}`);
        console.log(`    tokens: ${s.n_tokens} | state file: ${fmtBytes(s.n_bytes_written)} written, ${fmtBytes(s.n_bytes_read)} read | n_ctx after reload: ${s.n_ctx}`);
    }
}

async function main(): Promise<void> {
    // await ensureLoaded();

    const slots = await getJson<SlotInfo[]>(`/slots?model=${MODEL}`);
    if (slots.length === 0) {
        throw new Error("router returned no slots");
    }
    const n_ctx = Math.max(...slots.map(s => s.n_ctx));
    const target = Math.min(PERF_TOKENS, Math.floor(n_ctx * 0.8));
    if (target < PERF_TOKENS) {
        console.log(`target clamped to ${target} tokens (80% of slot n_ctx ${n_ctx})`);
    }

    const secret = `swordfish-${Math.floor(Math.random() * 1e6)}`;
    console.log(`secret for the conversation: ${secret}`);

    const buildConv = (units: number): unknown[] => [
        { role: "system", content: "You are a helpful assistant. Answer briefly." },
        { role: "user", content: `Remember the codeword "${secret}". Reply with just "ok" and nothing else.\n\n${FILLER_UNIT.repeat(units)}` },
    ];

    console.log(`sizing prompt to ~${target} tokens...`);
    let units = 8;
    let token_count = await countTokens(buildConv(units));
    for (let i = 0; i < 4 && token_count < target; i++) {
        units = Math.max(units + 1, Math.round((units * target) / token_count));
        token_count = await countTokens(buildConv(units));
        console.log(`sizing pass ${i + 1}: ${units} filler repeats -> ${token_count} tokens`);
    }
    console.log(`prompt sized: ${units} filler repeats -> ${token_count} tokens`);

    const conv = buildConv(units);

    console.log(`populating slot (prefill of ${token_count} tokens, may take a while)...`);
    const t1 = await chat(conv);
    console.log(`turn 1 done in ${fmtSec(t1.ms)}`);
    console.log(`    reply: ${JSON.stringify(t1.content)}`);
    reportUsage(t1.usage);
    conv.push({
        role: "assistant",
        content: t1.content,
        reasoning_content: t1.reasoning || undefined,
    });

    const populated = await getJson<SlotInfo[]>(`/slots?model=${MODEL}`);
    const slot = populated.find(s => (s.n_prompt_tokens ?? 0) > 0);
    if (!slot) {
        throw new Error("no slot with prompt tokens found after turn 1");
    }
    const filename = `qperf-${slot.id}.bin`;
    console.log(`slot ${slot.id} populated (${slot.n_prompt_tokens} prompt tokens, slot n_ctx ${slot.n_ctx})`);

    const stats: HopStats[] = [];

    stats.push(await hop("hop 1", "q8_0", slot.id, filename));

    conv.push({ role: "user", content: "What was the codeword I told you at the start? Reply with just the codeword." });
    console.log("[recall] turn 2 (after q8_0)");
    const t2 = await chat(conv);
    console.log(`    reply: ${JSON.stringify(t2.content)} [${fmtSec(t2.ms)}]`);
    reportUsage(t2.usage);
    const ok_q8 = t2.content.toLowerCase().includes(secret);

    stats.push(await hop("hop 2", "q4_0", slot.id, filename));

    console.log("[recall] turn 3 (after q4_0, same conversation)");
    const t3 = await chat(conv);
    console.log(`    reply: ${JSON.stringify(t3.content)} [${fmtSec(t3.ms)}]`);
    reportUsage(t3.usage);
    const ok_q4 = t3.content.toLowerCase().includes(secret);

    printSummary(stats);

    const ok = ok_q8 && ok_q4;
    console.log(`\nrecall after q8_0: ${ok_q8 ? "PASS" : "FAIL"} | recall after q4_0: ${ok_q4 ? "PASS" : "FAIL"}`);
    console.log(`kv quantize perf run ${ok ? "PASS" : "FAIL"}`);
    if (!ok) {
        process.exit(1);
    }
}

main().catch(err => {
    console.error("test error:", err);
    process.exit(1);
});
