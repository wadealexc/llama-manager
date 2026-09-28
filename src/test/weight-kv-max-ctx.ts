// Probe the context size llama-server chooses for the Q6 eval configuration.
// Run from the repo root: npx tsx src/test/weight-kv-max-ctx.ts
// Optional: LLAMA_BIN=/path/to/llama-server npx tsx src/test/weight-kv-max-ctx.ts
// Does not request inference; stops the server after /slots reports its context.

import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, rmSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

const ROOT = resolve(import.meta.dirname, "../..");
const BIN = process.env.LLAMA_BIN ?? resolve(ROOT, "llama.cpp/build/bin/llama-server");
const MODEL = "/home/fox/kitsu/models/qwen3.8-dense-mtp/Qwen3.8-27B-UD-Q6_K.gguf";
const LOG_DIR = resolve(ROOT, "logs/weight-kv-max-ctx");
const STARTUP_TIMEOUT_MS = 180_000;

async function freePort(): Promise<number> {
    return new Promise((resolvePort, reject) => {
        const listener = net.createServer();
        listener.once("error", reject);
        listener.listen(0, "127.0.0.1", () => {
            const addr = listener.address();
            if (!addr || typeof addr === "string") {
                listener.close();
                reject(new Error("Unable to allocate a local port"));
                return;
            }
            listener.close(() => resolvePort(addr.port));
        });
    });
}

async function stop(proc: ChildProcess): Promise<void> {
    if (proc.exitCode !== null || proc.signalCode !== null) return;
    const exited = new Promise<void>(resolveExit => proc.once("exit", () => resolveExit()));
    proc.kill("SIGTERM");
    if (!await Promise.race([exited.then(() => true), delay(10_000).then(() => false)])) {
        proc.kill("SIGKILL");
    }
    await exited;
}

async function main(): Promise<void> {
    mkdirSync(LOG_DIR, { recursive: true });
    const log = join(LOG_DIR, "q6.log");
    const slotDir = mkdtempSync(join(tmpdir(), "llama-weight-kv-ctx-"));
    let proc: ChildProcess | undefined;

    try {
        const port = await freePort();
        const fd = openSync(log, "w");
        try {
            // Same Q6 eval settings as weight-kv-eval.ts, EXCEPT no --ctx-size.
            // No --fit-target here either: the eval script uses llama-server's default.
            proc = spawn(BIN, [
                "--model", MODEL, "--host", "127.0.0.1", "--port", String(port),
                "--parallel", "1", "--n-gpu-layers", "99",
                "--slot-save-path", slotDir, "--no-context-shift", "--reasoning", "off",
                "--log-verbosity", "5", // debug; include fit and memory diagnostics in q6.log
                "--cache-type-k", "f16", "--cache-type-v", "f16",
            ], { stdio: ["ignore", fd, fd] });
        } finally {
            closeSync(fd);
        }

        const server = proc;
        let spawnError: Error | undefined;
        server.on("error", err => { spawnError = err; });
        console.log(`Loading Q6 with no --ctx-size: ${MODEL}`);
        console.log(`llama-server log: ${log}`);

        const deadline = Date.now() + STARTUP_TIMEOUT_MS;
        while (Date.now() < deadline) {
            if (spawnError) throw spawnError;
            if (server.exitCode !== null || server.signalCode !== null) {
                throw new Error(`llama-server exited during startup (code=${server.exitCode}, signal=${server.signalCode}); see ${log}`);
            }
            try {
                const response = await fetch(`http://127.0.0.1:${port}/slots`, { signal: AbortSignal.timeout(2000) });
                if (response.ok) {
                    const slots = await response.json() as Array<{ id: number; n_ctx: number }>;
                    if (slots.length === 1 && slots[0].n_ctx > 0) {
                        console.log(`Selected n_ctx per slot: ${slots[0].n_ctx.toLocaleString()} tokens`);
                        console.log(`Inspect ${log} for the fit decision and actual GPU layer offload.`);
                        return;
                    }
                }
            } catch { /* still loading */ }
            await delay(500);
        }
        throw new Error(`llama-server did not become ready within ${STARTUP_TIMEOUT_MS / 1000}s; see ${log}`);
    } finally {
        if (proc) await stop(proc);
        rmSync(slotDir, { recursive: true, force: true });
    }
}

main().catch(err => { console.error("Q6 context probe failed:", err); process.exitCode = 1; });
