import type { LlamaAPI } from "./client/llama-api.js";
import { LoadStatus, type ManagerConfig, type ModelId } from "./config/types.js";
import type { ModelEntry } from "./planner/model-entry.js";
import type { MemoryResponse } from "./client/types.js";
import { logger } from "./logger.js";
import type { ConsolaInstance } from "consola";
import type { Timer } from "./planner/timer.js";

const log: ConsolaInstance = logger.withTag('show-breakpoints');

type DeviceMemory = {
    weight_gib: number;
    context_gib: number;
};

type RungBreakpoint = {
    i: number;
    strategy_name: string;
    n_ctx: number;
    gain: number;
    gpu: DeviceMemory;
    cpu: DeviceMemory;
};

type ModelBreakpoints = {
    name: string;
    gpu_total_gib: number;
    cpu_total_gib: number;
    rungs: RungBreakpoint[];
};

export async function walkBreakpoints(
    client: LlamaAPI,
    entry: ModelEntry,
    signal: AbortSignal,
    t?: Timer,
): Promise<ModelBreakpoints> {
    const rungs: RungBreakpoint[] = [];

    await entry.loadWeights(signal, t);
    let prev = await entry.applyRung(0, signal, t);
    let mem = await entry.getMemory(signal);
    const { gpu: gpu0, cpu: cpu0 } = getModelMemory(mem);
    const w0 = gpu0.weight_bytes;
    const c0 = gpu0.context_bytes;
    rungs.push({ 
        i: 0, 
        strategy_name: "baseline", 
        n_ctx: prev, 
        gain: 0, 
        gpu: toGiB(gpu0),
        cpu: toGiB(cpu0),
    });

    const { gpu: gpu_total, cpu: cpu_total } = getDeviceTotalBytes(mem);

    const infos = await client.getModels(signal);
    const n_ctx_train = infos.find(info => info.id === entry.name)?.meta?.n_ctx_train;

    const getNCtxSeq = async (): Promise<number> => {
        const slots = await entry.getSlots(signal).catch(() => undefined);
        return slots?.[0]?.n_ctx ?? -1;
    };

    let idx = 0;
    // set rung capacity to slot ctx
    let prev_seq = await getNCtxSeq();
    entry.ladder[idx]!.n_ctx_cap = prev_seq;
    entry.ladder[idx]!.bytes_needed = w0 + c0;
    idx++;

    if (n_ctx_train !== undefined && prev_seq >= n_ctx_train) {
        log.info(`max ctx per seq reached for ${entry.name} (${n_ctx_train}); no strategies needed`);

        entry.ladder.splice(1);
    }

    while (entry.hasNextRung()) {
        log.info(`${entry.name}: applying ${entry.rungLabel(entry.ladder_i + 1)}`);
        const n_ctx = await entry.applyNextRung(signal, t);
        mem = await entry.getMemory(signal);
        const { gpu, cpu } = getModelMemory(mem);
        const { weight_bytes, context_bytes } = gpu;
        rungs.push({
            i: rungs.length,
            strategy_name: entry.rungLabel(entry.ladder_i),
            n_ctx: n_ctx,
            gain: n_ctx - prev,
            gpu: toGiB(gpu),
            cpu: toGiB(cpu),
        });
        prev = n_ctx;

        // set rung capacity to slot ctx
        const n_ctx_seq = await getNCtxSeq();
        entry.ladder[idx]!.n_ctx_cap = n_ctx_seq;
        entry.ladder[idx]!.bytes_needed = weight_bytes + context_bytes;

        if (n_ctx_seq <= prev_seq) {
            throw new Error(
                `walkBreakpoints: model ${entry.name}: rung ${idx} ('${entry.rungLabel(idx)}' | variant ${entry.curVariant()}) ` +
                `measured ${n_ctx_seq} ctx, which does not exceed the previous rung's ${prev_seq}`
            );
        }

        prev_seq = n_ctx_seq;
        idx++;

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

    return { 
        name: entry.name, 
        gpu_total_gib: gpu_total / (1024 ** 3),
        cpu_total_gib: cpu_total / (1024 ** 3),
        rungs 
    };
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

    const ordered = [...models.entries()].sort(([a], [b]) =>
        Number(a === config.default_model) - Number(b === config.default_model));

    for (const [name, entry] of ordered) {
        try {
            const result = await walkBreakpoints(client, entry, signal);
            results.push(result);
            record?.(name, result.rungs.at(-1)!.n_ctx);
            log.info(`${name}: done`);
        } catch (err) {
            try { await entry.unloadHard(); } catch { }
            throw err;
        }

        if (name !== config.default_model) {
            try { await entry.unloadHard(); } catch { }
        }
    }

    printBreakpoints(results);
}

type DeviceBytes = { weight_bytes: number; context_bytes: number };

function getModelMemory(mem: MemoryResponse): { gpu: DeviceBytes; cpu: DeviceBytes } {
    const gpu: DeviceBytes = { weight_bytes: 0, context_bytes: 0 };
    const cpu: DeviceBytes = { weight_bytes: 0, context_bytes: 0 };
    for (const dev of mem.devices) {
        const out = dev.type === "cpu" ? cpu : gpu;
        for (const name of ["main", "spec", "mmproj"] as const) {
            const c = dev.components[name];
            if (!c) continue;
            out.weight_bytes += c.model;
            out.context_bytes += c.context + c.compute;
        }
    }
    return { gpu, cpu };
}

function toGiB(bytes: DeviceBytes): DeviceMemory {
    return {
        weight_gib: bytes.weight_bytes / (1024 ** 3),
        context_gib: bytes.context_bytes / (1024 ** 3),
    };
}

function getDeviceTotalBytes(mem: MemoryResponse): { gpu: number; cpu: number } {
    const gpu = mem.devices.find(dev => dev.type !== "cpu")?.total ?? 0;
    const cpu = mem.devices.find(dev => dev.type === "cpu")?.total ?? 0;
    return { gpu, cpu };
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

═══════════════════════════════════════════════════════════════╤═══════════════╤═══════════════
 qwen3.8-flash-next                                            │   gpu: 31 GiB │  cpu: 115 GiB
═══════════════════════════════════════════════════════════════╪═══════════════╪═══════════════
  i  strategy                 ctx (tokens)    gain (tokens)    │ weights / ctx │ weights / ctx
 ──────────────────────────────────────────────────────────────┼───────────────┼───────────────
  0  baseline                       90,624                     │  24.12 / 5.62 │  86.43 / 0.31
  1  disable-spec                  246,784       (+156,160)    │  20.91 / 9.22 │  86.43 / 0.30
  2  quantize-kv-q8                262,144        (+15,360)    │  20.91 / 6.79 │  86.43 / 0.30
 ──────────────────────────────────────────────────────────────┴───────────────┴───────────────
 final ctx:      262,144 tokens

*/
export function printModelBreakpoints(model: ModelBreakpoints): string {
    const PAD = " ";
    const strategy_width = 24;
    const mem_header = "weights / ctx";

    const devices = [
        { label: `gpu: ${model.gpu_total_gib.toFixed(0)} GiB`, cells: model.rungs.map(rung => fmtMem(rung.gpu)) },
        { label: `cpu: ${model.cpu_total_gib.toFixed(0)} GiB`, cells: model.rungs.map(rung => fmtMem(rung.cpu)) },
    ];
    const col_width = Math.max(mem_header.length, ...devices.flatMap(dev => [dev.label.length, ...dev.cells.map(cell => cell.length)]));

    const left_header = PAD + "i".padStart(2) + "  " + "strategy".padEnd(strategy_width) + " " + "ctx (tokens)".padStart(12) + "  " + "gain (tokens)".padStart(15) + "    ";
    const left_width = left_header.length;
    const withDevices = (left: string, cells: string[]): string =>
        (left.padEnd(left_width) + cells.map(cell => "│ " + cell.padStart(col_width) + " ").join("")).trimEnd();

    const name_width = left_width - PAD.length - 1;
    const name_chars = Array.from(model.name);
    const name = name_chars.length > name_width
        ? name_chars.slice(0, name_width - 1).join('') + '…'
        : model.name;

    const banner = withDevices(PAD + name, devices.map(dev => dev.label));
    const header = withDevices(left_header, devices.map(() => mem_header));

    const rows: string[] = [];
    for (const [idx, rung] of model.rungs.entries()) {
        const i_str = String(rung.i).padStart(2);
        const label = Array.from(rung.strategy_name);
        const strat = (label.length > strategy_width
            ? label.slice(0, strategy_width - 1).join('') + '…'
            : rung.strategy_name).padEnd(strategy_width);
        const ctx = fmtNum(rung.n_ctx).padStart(12);
        const gain = rung.gain > 0 ? `(+${fmtNum(rung.gain)})` : "";
        const gain_padded = gain.padStart(15);
        rows.push(withDevices(PAD + i_str + "  " + strat + " " + ctx + "  " + gain_padded, devices.map(dev => dev.cells[idx])));
    }
    const final = model.rungs.at(-1)!.n_ctx;
    const footer = PAD + "final ctx: " + fmtNum(final).padStart(12) + " tokens";

    // Rule extends 1 char past the rightmost table value; the bar overhangs the rule by 1 on the left.
    const width = left_width + devices.length * (col_width + 3);
    const dividers = devices.map((_, i) => left_width + i * (col_width + 3));
    const line = (fill: string, junction: string, first: string): string =>
        Array.from({ length: width }, (_, i) => dividers.includes(i) ? junction : i === 0 ? first : fill).join("");

    // Banner shows each device's capacity above its column.
    const out = [
        line("═", "╤", "═"),
        banner,
        line("═", "╪", "═"),
        header,
        line("─", "┼", PAD),
        ...rows,
        line("─", "┴", PAD),
        footer,
    ];

    return out.join("\n") + "\n";
}

function fmtMem(mem: DeviceMemory): string {
    return `${mem.weight_gib.toFixed(2)} / ${mem.context_gib.toFixed(2)}`;
}

function fmtNum(n: number): string {
    return n.toLocaleString("en-US");
}
