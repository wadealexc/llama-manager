import { setTimeout as delay } from "node:timers/promises";

const ROUTER_URL = process.env.ROUTER_URL ?? "http://127.0.0.1:10002";
const MODEL = process.env.QUANT_TEST_MODEL ?? "qwen3.8-27b";

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
    await postJson("/models/load", { model: MODEL });

    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
        const models = await getJson<{ data: ModelInfo[] }>("/models");
        const info = models.data.find(m => m.id === MODEL);
        if (info?.status.value === "loaded") return;
        await delay(250);
    }
    throw new Error(`model ${MODEL} did not reach 'loaded' in time`);
}

async function chat(messages: unknown[]): Promise<{ content: string; reasoning: string; usage: Usage }> {
    const body = await postJson<ChatResponse>("/chat/completions", {
        model: MODEL,
        messages,
        stream: false,
        max_tokens: 5000,
        temperature: 0,
    });

    const msg = body.choices[0].message;
    return {
        content: msg.content ?? "",
        reasoning: msg.reasoning_content ?? "",
        usage: body.usage,
    };
}

async function hop(label: string, target: "q8_0" | "q4_0"): Promise<void> {
    console.log(`[${label}]`);

    const slots = await getJson<SlotInfo[]>(`/slots?model=${MODEL}`);
    const slot = slots.find(s => (s.n_prompt_tokens ?? 0) > 0);
    if (!slot) {
        throw new Error("no slot with prompt tokens found");
    }

    const filename = `qtest-${slot.id}.bin`;
    await postJson(`/slots/${slot.id}?action=save`, { filename, model: MODEL });
    console.log(`    saved slot ${slot.id} to ${filename}`);

    const reload = await postJson<ReloadInfo>("/reload", {
        model: MODEL,
        n_ctx: 0,
        cache_type_k: target,
        cache_type_v: target,
    });
    if (!reload.success) {
        throw new Error(`reload failed: ${reload.message ?? "unknown error"}`);
    }
    console.log(`    reloaded with ${target} kv cache; n_ctx = ${reload.n_ctx}`);

    const restore = await postJson<RestoreInfo>(`/slots/${slot.id}?action=restore`, { filename, model: MODEL });
    console.log(`    restored ${restore.n_restored} tokens from ${filename} [${restore.timings.restore_ms.toFixed(0)} ms]`);
}

function reportUsage(u: Usage): void {
    const cached = u.prompt_tokens_details?.cached_tokens ?? 0;
    const total = u.prompt_tokens;
    const hit = total > 0 ? (cached / total * 100).toFixed(1) : "0.0";
    console.log(`    usage: prompt=${total} cached=${cached} (${hit}% hit) completion=${u.completion_tokens}`);
}

async function main(): Promise<void> {
    await ensureLoaded();

    const secret = `swordfish-${Math.floor(Math.random() * 1e6)}`;
    console.log(`[chat] secret for the conversation: ${secret}`);

    const conv: unknown[] = [
        { role: "system", content: "You are a helpful assistant. Answer briefly." },
        { role: "user", content: `Remember the codeword "${secret}". Reply with just "ok".` },
    ];

    console.log("[chat] turn 1 (codeword set)");
    const t1 = await chat(conv);
    console.log(`    reply: ${JSON.stringify(t1.content)}`);
    reportUsage(t1.usage);
    conv.push({
        role: "assistant",
        content: t1.content,
        reasoning_content: t1.reasoning || undefined,
    });

    await hop("hop 1: f16 -> q8_0", "q8_0");

    conv.push({ role: "user", content: "What was the codeword I told you? Reply with just the codeword." });
    console.log("[chat] turn 2 (recall after q8_0)");
    const t2 = await chat(conv);
    console.log(`    reply: ${JSON.stringify(t2.content)}`);
    reportUsage(t2.usage);
    const ok_q8 = t2.content.toLowerCase().includes(secret);
    console.log(`    recall after q8_0: ${ok_q8 ? "PASS" : "FAIL"}`);

    await hop("hop 2: q8_0 -> q4_0", "q4_0");

    console.log("[chat] turn 3 (recall after q4_0)");
    const t3 = await chat(conv);
    console.log(`    reply: ${JSON.stringify(t3.content)}`);
    reportUsage(t3.usage);
    const ok_q4 = t3.content.toLowerCase().includes(secret);
    console.log(`    recall after q4_0: ${ok_q4 ? "PASS" : "FAIL"}`);

    const ok = ok_q8 && ok_q4;
    console.log(`\nkv quantize restore ${ok ? "PASS" : "FAIL"}`);
    if (!ok) {
        process.exit(1);
    }
}

main().catch(err => {
    console.error("test error:", err);
    process.exit(1);
});
