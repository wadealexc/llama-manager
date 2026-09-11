import type { LlamaAPI } from "./client/llama-api.js";
import type { ManagerConfig, ModelId } from "./config/types.js";
import type { ModelEntry } from "./planner/model-entry.js";
import type { MemoryResponse } from "./client/types.js";
import { readdir, unlink } from "node:fs/promises";
import { join } from "node:path";
import { logger } from "./logger.js";
import type { ConsolaInstance } from "consola";
import type { Timer } from "./planner/timer.js";

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

export async function walkBreakpoints(
    client: LlamaAPI,
    entry: ModelEntry,
    signal: AbortSignal,
    t?: Timer,
): Promise<ModelBreakpoints> {
    log.info(`loading weights for ${entry.name}`);
    await entry.loadWeights(signal, t);

    const rungs: RungBreakpoint[] = [];

    let prev = await entry.loadWithKV(null, signal, t);
    let mem = await client.getMemory(entry.name, signal);
    const { weight_bytes: w0, context_bytes: c0 } = getModelMemory(mem);
    rungs.push({ i: 0, strategy_name: "baseline", n_ctx: prev, gain: 0, weight_gib: w0 / (1024 ** 3), context_gib: c0 / (1024 ** 3) });

    const device_total_gib = getDeviceTotalBytes(mem) / (1024 ** 3);

    const infos = await client.getModels(signal);
    const n_ctx_train = infos.find(info => info.id === entry.name)?.meta?.n_ctx_train;

    const get_n_ctx_seq = async (): Promise<number> => {
        const slots = await client.getSlots(entry.name, signal).catch(() => undefined);
        return slots?.[0]?.n_ctx ?? 0;
    };

    if (n_ctx_train !== undefined) {
        const n_ctx_seq = await get_n_ctx_seq();
        if (n_ctx_seq >= n_ctx_train) {
            const addtl = entry.ladder.length - 1 - entry.ladder_i;
            log.info(`max ctx per seq reached for ${entry.name} (${n_ctx_train}); no strategies needed`);

            entry.ladder.splice(1);
        }
    }

    while (entry.hasNextStrategy()) {
        log.info(`${entry.name}: applying ${entry.ladder[entry.ladder_i + 1].strategy}`);
        const n_ctx = await entry.applyNextStrategy(false, signal, t);
        mem = await client.getMemory(entry.name, signal);
        const { weight_bytes, context_bytes } = getModelMemory(mem);
        rungs.push({
            i: rungs.length,
            strategy_name: entry.ladder[entry.ladder_i].strategy as string,
            n_ctx: n_ctx,
            gain: n_ctx - prev,
            weight_gib: weight_bytes / (1024 ** 3),
            context_gib: context_bytes / (1024 ** 3),
        });
        prev = n_ctx;

        const n_ctx_seq = await get_n_ctx_seq();

        if (n_ctx_train !== undefined && n_ctx_seq >= n_ctx_train) {
            const addtl = entry.ladder.length - 1 - entry.ladder_i;
            if (addtl !== 0) {
                log.info(`max ctx per seq reached for ${entry.name} (${n_ctx_train}); ignoring ${addtl} additional strategies`);
            } else {
                log.info(`max ctx per seq reached for ${entry.name} (${n_ctx_train})`);
            }

            entry.ladder.splice(entry.ladder_i + 1);
            break;
        }
    }

    await entry.unloadHard(t);
    return { name: entry.name, device_total_gib, rungs };
}

export async function showBreakpoints(
    client: LlamaAPI,
    config: ManagerConfig,
    models: Map<ModelId, ModelEntry>,
    record?: (id: ModelId, ctx: number) => void,
): Promise<void> {
    const ctrl = new AbortController();
    const signal = ctrl.signal;

    const results: ModelBreakpoints[] = [];

    for (const [name, entry] of models.entries()) {
        try {
            const result = await walkBreakpoints(client, entry, signal);
            results.push(result);
            record?.(name, result.rungs.at(-1)!.n_ctx);
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

    console.log(results.map(printModelBreakpoints).join("\n"));
}


/*

Stringify a model's strategy ladder for printing:

════════════════════════════════════════════════════════════════════════════
 qwen3.8-27b        baseline: 150,784 tokens                device: 31 GiB
════════════════════════════════════════════════════════════════════════════
  i  strategy        ctx (tokens)    gain (tokens)       weights / ctx GiB
 ──────────────────────────────────────────────────────────────────────────
  0  baseline             150,784                            17.13 / 12.01
  1  disable-spec         182,784        (+32,000)           17.13 / 12.03
  2  mmproj-to-cpu        200,960        (+18,176)           16.02 / 13.16
  3  quantize-kv-q8       262,144        (+61,184)           16.02 / 10.40
 ──────────────────────────────────────────────────────────────────────────
 final ctx:      262,144 tokens

*/
export function printModelBreakpoints(model: ModelBreakpoints): string {
    const PAD = " ";

    let out = "";

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

    return out;
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
