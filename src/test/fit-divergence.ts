import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import * as net from "node:net";
import type { AddressInfo } from "node:net";
import { setTimeout as delay } from "node:timers/promises";
import type { MemoryResponse } from "../client/types.js";

const PROJECT_ROOT = resolve(import.meta.dirname, "../..");
const BIN = process.env.LLAMA_BIN ?? resolve(PROJECT_ROOT, "./llama.cpp/build/bin/llama-server");
const LOG_DIR = resolve(PROJECT_ROOT, "logs/fit-divergence");

const QWEN_MODEL = "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q4_K_XL.gguf";
const QWEN_MMPROJ = "/home/fox/kitsu/models/qwen3.8-dense-mtp/mmproj-BF16.gguf";
const GEMMA_MODEL = "/home/fox/kitsu/models/gemma4-dense/gemma-4-31B-it-Q6_K.gguf";

const BASE_CTX = 1024;
const LOAD_TIMEOUT_MS = 120_000;
const RELOAD_TIMEOUT_MS = 120_000;
const POLL_INTERVAL_MS = 250;

interface Scenario {
    name: string;
    model: string;
    mmproj?: string;
    spec_type?: string;
    spec_draft_n_max?: number;
    spec: boolean;
    mmproj_on: boolean;
    fit_target: number;
}

interface ResultRow {
    name: string;
    spec: boolean;
    mmproj_on: boolean;
    fit_target: number;
    load_ctx?: number;
    reload_ctx?: number;
    load_free?: number;
    reload_free?: number;
    load_err?: string;
    reload_err?: string;
}

function build_scenarios(): Scenario[] {
    const scenarios: Scenario[] = [];

    for (const fit_target of [512, 1024]) {
        for (const spec of [true, false]) {
            for (const mmproj_on of [true, false]) {
                scenarios.push({
                    name: "qwen3.8-27b",
                    model: QWEN_MODEL,
                    mmproj: QWEN_MMPROJ,
                    spec_type: "draft-mtp",
                    spec_draft_n_max: 2,
                    spec,
                    mmproj_on,
                    fit_target,
                });
            }
        }
    }

    for (const fit_target of [512, 1024]) {
        scenarios.push({
            name: "gemma4-31b",
            model: GEMMA_MODEL,
            spec: false,
            mmproj_on: false,
            fit_target,
        });
    }

    return scenarios;
}

function scenario_label(s: Scenario, mode: string): string {
    return `${s.name}-spec${s.spec ? "on" : "off"}-mm${s.mmproj_on ? "on" : "off"}-fit${s.fit_target}-${mode}`
        .replaceAll(".", "_");
}

function scenario_args(s: Scenario): string[] {
    const args = ["-m", s.model, "-ngl", "99", "--fit-target", String(s.fit_target)];

    if (s.mmproj_on && s.mmproj) {
        args.push("--mmproj", s.mmproj);
    }

    if (s.spec && s.spec_type) {
        args.push("--spec-type", s.spec_type);
        if (s.spec_draft_n_max !== undefined) {
            args.push("--spec-draft-n-max", String(s.spec_draft_n_max));
        }
    }

    return args;
}

function find_free_port(): Promise<number> {
    return new Promise((resolve_p, reject) => {
        const server = net.createServer();
        server.once("error", reject);
        server.listen(0, "127.0.0.1", () => {
            const addr = server.address() as AddressInfo;
            server.close(() => resolve_p(addr.port));
        });
    });
}

function err_msg(err: unknown): string {
    const msg = err instanceof Error ? err.message : String(err);
    return msg.replaceAll(/\s+/g, " ").slice(0, 100);
}

async function stop_server(proc: ChildProcess): Promise<void> {
    if (proc.exitCode !== null || proc.signalCode !== null) return;

    proc.kill("SIGTERM");
    const exited = new Promise<void>(r => proc.once("exit", () => r()));
    const graceful = await Promise.race([exited.then(() => true), delay(5000).then(() => false)]);
    if (!graceful) {
        proc.kill("SIGKILL");
    }
    await exited;
}

async function wait_for_slots(port: number, is_exited: () => boolean, log_path: string, timeout_ms: number): Promise<void> {
    const deadline = Date.now() + timeout_ms;

    while (Date.now() < deadline) {
        if (is_exited()) {
            throw new Error(`server exited during startup; see ${log_path}`);
        }

        try {
            const res = await fetch(`http://127.0.0.1:${port}/slots`, { signal: AbortSignal.timeout(1000) });
            if (res.ok) {
                const slots = await res.json() as { n_ctx: number }[];
                if (slots.length > 0 && slots[0].n_ctx > 0) return;
            }
        } catch {
        }

        await delay(POLL_INTERVAL_MS);
    }

    throw new Error(`server did not become ready within ${timeout_ms}ms; see ${log_path}`);
}

async function with_server(args: string[], label: string, fn: (port: number) => Promise<void>): Promise<void> {
    const port = await find_free_port();
    const log_path = join(LOG_DIR, `${label}.log`);
    const log_stream = createWriteStream(log_path);

    const proc = spawn(BIN, ["--host", "127.0.0.1", "--port", String(port), ...args], {
        stdio: ["ignore", "pipe", "pipe"],
    });
    proc.stdout?.pipe(log_stream);
    proc.stderr?.pipe(log_stream);

    let did_exit = false;
    proc.once("exit", (code, signal) => {
        did_exit = true;
        console.log(`    server exited (code=${code} signal=${signal}) log=${log_path}`);
    });

    try {
        await wait_for_slots(port, () => did_exit, log_path, LOAD_TIMEOUT_MS);
        await fn(port);
    } finally {
        log_stream.end();
        await stop_server(proc);
    }
}

async function get_ctx(port: number): Promise<number> {
    const res = await fetch(`http://127.0.0.1:${port}/slots`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) {
        throw new Error(`GET /slots failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
    }

    const slots = await res.json() as { n_ctx: number }[];
    if (slots.length === 0 || slots[0].n_ctx <= 0) {
        throw new Error(`GET /slots returned no usable slots`);
    }

    return slots[0].n_ctx;
}

async function get_free_bytes(port: number): Promise<number> {
    const res = await fetch(`http://127.0.0.1:${port}/memory`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) {
        throw new Error(`GET /memory failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
    }

    const body = await res.json() as MemoryResponse;
    const dev = body.devices.find(d => d.type !== "cpu");
    return dev ? dev.free : 0;
}

async function run_load(s: Scenario): Promise<{ n_ctx: number; free: number }> {
    let n_ctx = 0;
    let free = 0;

    await with_server(scenario_args(s), scenario_label(s, "load"), async (port) => {
        n_ctx = await get_ctx(port);
        free = await get_free_bytes(port);
    });

    return { n_ctx, free };
}

async function run_reload(s: Scenario): Promise<{ n_ctx: number; free: number }> {
    let n_ctx = 0;
    let free = 0;

    await with_server([...scenario_args(s), "--ctx-size", String(BASE_CTX)], scenario_label(s, "reload"), async (port) => {
        const res = await fetch(`http://127.0.0.1:${port}/reload`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ n_ctx: 0 }),
            signal: AbortSignal.timeout(RELOAD_TIMEOUT_MS),
        });

        if (!res.ok) {
            throw new Error(`POST /reload failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
        }

        const body = await res.json() as { success: boolean; n_ctx: number; message?: string };
        if (!body.success) {
            throw new Error(`POST /reload unsuccessful: ${body.message ?? "(no message)"}`);
        }

        n_ctx = body.n_ctx;
        free = await get_free_bytes(port);
    });

    return { n_ctx, free };
}

function print_table(rows: ResultRow[]): void {
    const fmt_ctx = (v: number | undefined, err: string | undefined, width: number): string => {
        if (err !== undefined) return "ERR".padStart(width);
        if (v === undefined) return "-".padStart(width);
        return String(v).padStart(width);
    };

    const fmt_free = (v: number | undefined, width: number): string => {
        if (v === undefined) return "-".padStart(width);
        return `${(v / 1024 ** 3).toFixed(2)}`.padStart(width);
    };

    const header = [
        "model".padEnd(12),
        "spec".padStart(4),
        "mmproj".padStart(6),
        "fit".padStart(5),
        "load_ctx".padStart(9),
        "reload_ctx".padStart(10),
        "delta".padStart(8),
        "load_free".padStart(9),
        "rl_free".padStart(9),
        "note",
    ].join("  ");

    console.log("\n" + "=".repeat(header.length));
    console.log(header);
    console.log("-".repeat(header.length));

    for (const row of rows) {
        const note = row.load_err ?? row.reload_err ?? "";
        const delta = row.load_ctx !== undefined && row.reload_ctx !== undefined
            ? String(row.reload_ctx - row.load_ctx).padStart(8)
            : "".padStart(8);

        console.log([
            row.name.padEnd(12),
            (row.spec ? "on" : "off").padStart(4),
            (row.mmproj_on ? "on" : "off").padStart(6),
            String(row.fit_target).padStart(5),
            fmt_ctx(row.load_ctx, row.load_err, 9),
            fmt_ctx(row.reload_ctx, row.reload_err, 10),
            delta,
            fmt_free(row.load_free, 9),
            fmt_free(row.reload_free, 9),
            note,
        ].join("  "));
    }

    console.log("\n(free columns are device free memory in GiB, queried via GET /memory after fit)");
}

async function main(): Promise<void> {
    if (!existsSync(BIN)) {
        console.error(`llama-server binary not found at '${BIN}' (override with LLAMA_BIN)`);
        process.exit(1);
    }

    mkdirSync(LOG_DIR, { recursive: true });
    console.log(`bin: ${BIN}`);
    console.log(`logs: ${LOG_DIR}\n`);

    const rows: ResultRow[] = [];

    for (const s of build_scenarios()) {
        const row: ResultRow = {
            name: s.name,
            spec: s.spec,
            mmproj_on: s.mmproj_on,
            fit_target: s.fit_target,
        };

        const t = performance.now();
        process.stdout.write(`[${scenario_label(s, "load")}] loading... `);
        try {
            const r = await run_load(s);
            row.load_ctx = r.n_ctx;
            row.load_free = r.free;
            console.log(`n_ctx=${row.load_ctx}`);
        } catch (err) {
            row.load_err = err_msg(err);
            console.log(`FAILED: ${row.load_err}`);
        }

        process.stdout.write(`[${scenario_label(s, "reload")}] loading at ctx ${BASE_CTX}, reloading with n_ctx=0... `);
        try {
            const r = await run_reload(s);
            row.reload_ctx = r.n_ctx;
            row.reload_free = r.free;
            console.log(`n_ctx=${row.reload_ctx}`);
        } catch (err) {
            row.reload_err = err_msg(err);
            console.log(`FAILED: ${row.reload_err}`);
        }

        console.log(`    (${((performance.now() - t) / 1000).toFixed(1)}s)\n`);
        rows.push(row);
    }

    print_table(rows);
}

main().catch(err => {
    console.error("test error:", err);
    process.exit(1);
});
