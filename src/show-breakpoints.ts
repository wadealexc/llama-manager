import type { LlamaAPI } from "./client/llama-api.js";
import type { ManagerConfig, ModelId } from "./config/types.js";
import type { ModelEntry } from "./planner/model-entry.js";
import type { MemoryResponse } from "./client/types.js";
import { readdir, unlink } from "node:fs/promises";
import { join } from "node:path";
import { logger } from "./logger.js";
import type { ConsolaInstance } from "consola";

const log: ConsolaInstance = logger.withTag('show-breakpoints');

type RungBreakpoint = {
    i: number;
    strategy_name: string;
    n_ctx: number;
    gain: number;
    weight_gib: number;
    context_gib: number;
};

type ModelBreakpoints = {
    name: string;
    device_total_gib: number;
    rungs: RungBreakpoint[];
};

export async function showBreakpoints(
    client: LlamaAPI,
    config: ManagerConfig,
    models: Map<ModelId, ModelEntry>,
    record?: (id: ModelId, ctx: number) => void,
): Promise<void> {
    const ctrl = new AbortController();
    const signal = ctrl.signal;

    let device_total_gib = 0;
    const results: ModelBreakpoints[] = [];

    for (const [name, entry] of models.entries()) {
        try {
            log.info(`loading weights for ${name}`);
            await entry.loadWeights(signal);

            const rungs: RungBreakpoint[] = [];

            let prev = await entry.loadWithKV(null, signal);
            let mem = await client.getMemory(name, signal);
            const { weight_bytes: w0, context_bytes: c0 } = getModelMemory(mem);
            rungs.push({ i: 0, strategy_name: "baseline", n_ctx: prev, gain: 0, weight_gib: w0 / (1024 ** 3), context_gib: c0 / (1024 ** 3) });

            if (device_total_gib === 0) {
                device_total_gib = getDeviceTotalBytes(mem) / (1024 ** 3);
            }

            while (entry.hasNextStrategy()) {
                log.info(`${name}: applying ${entry.ladder[entry.ladder_i + 1].strategy}`);
                const n = await entry.applyNextStrategy(false, signal);
                mem = await client.getMemory(name, signal);
                const { weight_bytes, context_bytes } = getModelMemory(mem);
                rungs.push({
                    i: rungs.length,
                    strategy_name: entry.ladder[entry.ladder_i].strategy as string,
                    n_ctx: n,
                    gain: n - prev,
                    weight_gib: weight_bytes / (1024 ** 3),
                    context_gib: context_bytes / (1024 ** 3),
                });
                prev = n;
            }

            await entry.unloadHard();
            results.push({ name, device_total_gib, rungs });
            record?.(name, rungs.at(-1)!.n_ctx);
            log.info(`${name}: done`);
        } catch (err) {
            log.error(`error processing model ${name}: ${err}`);
            try { await entry.unloadHard(); } catch { }
        }
    }

    await cleanupSlots(config.router.slot_save_path).catch(err => {
        log.warn(`cleanupSlots error: ${err}`);
    });

    printBreakpoints(results);
}

function getModelMemory(mem: MemoryResponse): { weight_bytes: number; context_bytes: number } {
    let weight_bytes = 0;
    let context_bytes = 0;
    for (const dev of mem.devices) {
        if (dev.type === "cpu") continue;
        for (const name of ["main", "spec", "mmproj"] as const) {
            const c = dev.components[name];
            if (!c) continue;
            weight_bytes += c.model;
            context_bytes += c.context + c.compute;
        }
    }
    return { weight_bytes, context_bytes };
}

function getDeviceTotalBytes(mem: MemoryResponse): number {
    for (const dev of mem.devices) {
        if (dev.type !== "cpu") return dev.total;
    }
    return 0;
}

function printBreakpoints(results: ModelBreakpoints[]): void {
    if (results.length === 0) {
        log.info("no breakpoints to show");
        return;
    }

    const PAD = " ";

    let out = "";

    for (const model of results) {
        const baseline = model.rungs[0].n_ctx;
        const baseline_str = fmtNum(baseline);
        const device_str = `device: ${model.device_total_gib.toFixed(0)} GiB`;

        const rows = [PAD + "i".padStart(2) + "  strategy        ctx (tokens)    gain (tokens)       weights / ctx GiB"];
        for (const rung of model.rungs) {
            const i_str = String(rung.i).padStart(2);
            const strat = rung.strategy_name.padEnd(15);
            const ctx = fmtNum(rung.n_ctx).padStart(12);
            const gain = rung.gain > 0 ? `(+${fmtNum(rung.gain)})` : "";
            const gain_padded = gain.padStart(15);
            const wc = `${rung.weight_gib.toFixed(2)} / ${rung.context_gib.toFixed(2)}`;
            const wc_padded = wc.padStart(17);
            rows.push(PAD + i_str + "  " + strat + " " + ctx + "  " + gain_padded + "       " + wc_padded);
        }
        const final = model.rungs.at(-1)!.n_ctx;
        rows.push(PAD + "final ctx: " + fmtNum(final).padStart(12) + " tokens");

        // Rule extends 1 char past the rightmost table value; the bar overhangs the rule by 1.
        const tableW = Math.max(...rows.map(r => r.length));
        const rule = PAD + "─".repeat(tableW - PAD.length + 1);
        const bar = "═".repeat(rule.length + 1);

        // Banner content sits inside the bar with a 2-char inset on the right.
        const lhs = PAD + model.name.padEnd(16) + "   baseline: " + baseline_str + " tokens";
        const rhs = device_str;
        const gap = Math.max(1, bar.length - 2 - lhs.length - rhs.length);

        out += "\n";
        out += bar + "\n";
        out += lhs + " ".repeat(gap) + rhs + "\n";
        out += bar + "\n";
        out += rows[0] + "\n";
        out += rule + "\n";
        for (const row of rows.slice(1, -1)) {
            out += row + "\n";
        }
        out += rule + "\n";
        out += rows.at(-1) + "\n";
    }

    console.log(out);
}

function fmtNum(n: number): string {
    return n.toLocaleString("en-US");
}

async function cleanupSlots(dir: string): Promise<void> {
    let names: string[];
    try {
        names = await readdir(dir);
    } catch {
        return;
    }

    const targets = names.filter(n => n.endsWith('.bin') || n.endsWith('.ckpt'));
    await Promise.allSettled(targets.map(n => unlink(join(dir, n))));
}