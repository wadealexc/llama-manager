import { setTimeout as delay } from "node:timers/promises";

const BASE = process.env.MANAGER_URL ?? "http://127.0.0.1:10001";
const MAIN = "qwen3.6-27b";
const TASK = "qwen3.5-9b";

interface Usage {
    prompt_tokens: number;
    completion_tokens: number;
    prompt_tokens_details?: { cached_tokens?: number };
}

interface ChatResponse {
    choices: { message: { content: string; reasoning_content?: string } }[];
    usage: Usage;
}

async function chat(model: string, messages: unknown[], signal?: AbortSignal): Promise<{ content: string; reasoning: string; usage: Usage }> {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
            model,
            messages,
            stream: false,
            max_tokens: 5000,
            temperature: 0,
        }),
        signal,
    });

    if (!res.ok) {
        throw new Error(`${model} chat failed (${res.status}): ${await res.text()}`);
    }

    const body = await res.json() as ChatResponse;
    const msg = body.choices[0].message;
    return {
        content: msg.content,
        reasoning: msg.reasoning_content ?? "",
        usage: body.usage,
    };
}

function fmt_ms(ms: number): string {
    return `${(ms / 1000).toFixed(2)}s`;
}

function truncate(s: string, n: number): string {
    return s.length > n ? s.slice(0, n) + "…" : s;
}

async function timed<T>(label: string, fn: () => Promise<T>): Promise<{ result: T; ms: number }> {
    const t = performance.now();
    const result = await fn();
    const ms = performance.now() - t;
    console.log(`  ${label}: ${fmt_ms(ms)}`);
    return { result, ms };
}

function report_usage(label: string, u: Usage): void {
    const cached = u.prompt_tokens_details?.cached_tokens ?? 0;
    const total = u.prompt_tokens;
    const hit = total > 0 ? (cached / total * 100).toFixed(1) : "0.0";
    console.log(`    usage: prompt=${total} cached=${cached} (${hit}% hit) completion=${u.completion_tokens}`);
}

async function main(): Promise<void> {
    // 1. Start a conversation on the main model. The codeword is only ever
    //    in these messages; the follow-up asks for it without restating it,
    //    so recall depends on the KV cache surviving the swap.
    const secret = `swordfish-${Math.floor(Math.random() * 1e6)}`;
    console.log(`[main] secret for the conversation: ${secret}`);

    const conv: unknown[] = [
        { role: "system", content: "You are a helpful assistant. Answer briefly." },
        { role: "user", content: `Remember the codeword "${secret}". Reply with just "ok".` },
    ];

    console.log("[main] turn 1 (codeword set)");
    const t1 = await timed("chat", () => chat(MAIN, conv));
    console.log(`    reply: ${JSON.stringify(t1.result.content)}`);
    if (t1.result.reasoning) console.log(`    reasoning: ${JSON.stringify(truncate(t1.result.reasoning, 120))}`);
    report_usage("main #1", t1.result.usage);
    conv.push({
        role: "assistant",
        content: t1.result.content,
        reasoning_content: t1.result.reasoning || undefined,
    });

    // 2. Switch to the task model. This forces the planner to stash the
    //    main model's KV cache, load the task model, serve, and on the
    //    next main request restore the main model's KV cache.
    console.log("[task] turn");
    const task_conv: unknown[] = [
        { role: "system", content: "You are a title generator. Reply with a short title." },
        { role: "user", content: "Summarize the conversation so far in 5 words." },
    ];
    const t2 = await timed("chat", () => chat(TASK, task_conv));
    console.log(`    reply: ${JSON.stringify(t2.result.content)}`);
    report_usage("task", t2.result.usage);

    // Let the task request fully settle before the next main request, so the
    // stash-for-task and restore-for-main don't race.
    await delay(500);

    // 3. Return to the main model. Append the new user message to the prior
    //    conversation so the codeword is present in the messages array (so the
    //    model can answer) but the *prior* tokens should be a cache hit if the
    //    stash/restore round-trip preserved the KV.
    console.log("[main] turn 2 (codeword recall)");
    conv.push({ role: "user", content: "What was the codeword I told you? Reply with just the codeword." });
    const t3 = await timed("chat", () => chat(MAIN, conv));
    console.log(`    reply: ${JSON.stringify(t3.result.content)}`);
    if (t3.result.reasoning) console.log(`    reasoning: ${JSON.stringify(truncate(t3.result.reasoning, 120))}`);
    report_usage("main #2", t3.result.usage);

    const ok = t3.result.content.toLowerCase().includes(secret);
    console.log(`\nKV stash/restore ${ok ? "PASS ✓" : "FAIL ✗"}`);
    if (!ok) {
        console.error(`  expected the codeword "${secret}" somewhere in: ${t3.result.content}`);
        process.exit(1);
    }
}

main().catch(err => {
    console.error("test error:", err);
    process.exit(1);
});
