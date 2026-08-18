import type { ConsolaInstance } from "consola";
import type { ModelConfig, ModelRole } from "../config/types.js";
import type { LlamaAPI } from "../client/llama-api.js";
import type { ModelId } from "../types.js";
import type { MemoryResponse } from "../client/types.js";
import type { StrategyId } from "./types.js";

const GIB = 1024 ** 3;

// Bar width
const BAR_W = 40;

// Print memory allocation for models as a bar
export class PrintMemory {

    log: ConsolaInstance;
    client: LlamaAPI;

    constructor(log: ConsolaInstance, client: LlamaAPI) {
        this.log = log;
        this.client = client;
    }

    async print(
        title: string,
        models: Partial<Record<ModelRole, ModelConfig>>,
        strategies?: StrategyId[]
    ): Promise<void> {
        const model_main = models['main']!;
        const model_task = models['task'];

        const models_mem: Map<ModelId, MemoryResponse> = new Map();

        try {
            const res = await this.client.getMemory(model_main.name);
            models_mem.set(model_main.name, res);
        } catch (err) {
            // Assume model is not loaded, and zero out mem
            models_mem.set(model_main.name, { devices: [] });
        }

        if (model_task) {
            try {
                const res = await this.client.getMemory(model_task.name);
                models_mem.set(model_task.name, res);
            } catch (err) {
                // Assume model is not loaded, and zero out mem
                models_mem.set(model_task.name, { devices: [] });
            }
        }

        const fmt = (b: number): string => b <= 0 ? "0.00 GiB" : `${(b / GIB).toFixed(2)} GiB`;

        type Component = { model: number; context: number; compute: number };

        // Aggregate per device across all models' responses
        const devices = new Map<string, { name: string; type: string; total: number; free: number; models: Map<ModelId, Record<string, Component>> }>();
        for (const [model_id, mem] of models_mem) {
            for (const dev of mem.devices) {
                let entry = devices.get(dev.name);
                if (!entry) {
                    entry = { name: dev.name, type: dev.type, total: dev.total, free: dev.free, models: new Map() };
                    devices.set(dev.name, entry);
                }
                entry.models.set(model_id, dev.components as Record<string, Component>);
            }
        }

        const lines: string[] = ["\n".padEnd(50, "="), title];
        if (strategies !== undefined) {
            if (strategies.length === 0) {
                lines.push("(no strategies applied)");
            } else {
                for (const s of strategies) lines.push(`- ${s}`);
            }
        }
        lines.push("");

        // headline per device
        for (const dev of devices.values()) {
            const used = dev.total - dev.free;
            const label = `${dev.name} (${dev.type})`;
            lines.push(`${label.padEnd(12)}: (${fmt(used)} used)`);
        }
        lines.push("");

        // bar + line items for non-CPU devices only
        for (const dev of devices.values()) {
            if (dev.type === "cpu") continue;
            lines.push(`Usage (${dev.name}):`);

            // per-model static/dynamic totals
            const model_totals: { id: ModelId; is_task: boolean; st: number; dyn: number }[] = [];
            for (const [model_id, comps] of dev.models) {
                let st = 0, dyn = 0;
                for (const c of Object.values(comps)) {
                    if (!c) continue;
                    st += c.model;
                    dyn += c.context + c.compute;
                }
                model_totals.push({ id: model_id, is_task: model_id === models['task']?.name, st, dyn });
            }
            const main_m = model_totals.find(m => !m.is_task) ?? null;
            const task_m = model_totals.find(m => m.is_task) ?? null;

            lines.push(buildBar({
                dev_total: dev.total,
                main: main_m ? { static: main_m.st, dynamic: main_m.dyn } : null,
                task: task_m ? { static: task_m.st, dynamic: task_m.dyn } : null,
            }));
            lines.push("");

            let attributed = 0;
            for (const mt of model_totals) {
                const comps = dev.models.get(mt.id)!;
                const gpu_total = mt.st + mt.dyn;
                lines.push(`- ${mt.id}: (${fmt(gpu_total)} gpu)`);
                for (const name of ["main", "spec", "mmproj"] as const) {
                    const c = comps[name] as Component | undefined;
                    if (!c) continue;
                    if (name === "mmproj") {
                        lines.push(`  - mmproj: ${fmt(c.model)}`);
                    } else {
                        lines.push(`  - ${name.padEnd(4)} (weights | ctx | cmp): (${fmt(c.model)} | ${fmt(c.context)} | ${fmt(c.compute)})`);
                    }
                }
                attributed += gpu_total;
            }

            const free_b = dev.free;
            const unaccounted = Math.max(0, (dev.total - dev.free) - attributed);
            const free_part = `free: ${fmt(free_b)}`;
            const unacc_part = unaccounted > 0 ? ` | unaccounted: ${fmt(unaccounted)}` : "";
            lines.push(`- other: (${free_part}${unacc_part})`);
        }
        lines.push("".padEnd(50, "="));
        this.log.info(lines.join("\n"));
    }
}

type Memory = {
    dev_total: number;
    main: {
        static: number;
        dynamic: number;
    } | null;
    task: {
        static: number;
        dynamic: number;
    } | null;
}

/**
 * Construct a printable bar that displays the proportion of the device occupied by
 * two models. The main model is denoted by 'X', and the task model by 'O'.
 * 
 * Static vs dynamic (weights vs cache/compute) are denoted by upper/lowercase.
 * Free space is denoted by '.'
 * 
 * Bar segments are laid out in this order:
 * [main, static][main, dynamic][free space][task, dynamic][task, static]
 * 
 * E.g a device with main and task model might look like this:
 * [XXXXXXXXXXXXxxxxx.......oooooOOOOOOO]
 * 
 */
function buildBar(mem: Memory): string {
    // Bytes as fraction of bar width
    const f = (b: number) => mem.dev_total > 0 ? (b / mem.dev_total) * BAR_W : 0;

    let pos = 0;
    const seg = (end: number, ch: string) => {
        const start = pos;
        pos = end;
        return ch.repeat(Math.max(0, Math.round(end) - Math.round(start)));
    };

    const parts: string[] = [];
    let used = 0;

    if (mem.main) {
        parts.push(seg(f(mem.main.static), "X"));
        parts.push(seg(f(mem.main.static + mem.main.dynamic), "x"));
        used += mem.main.static + mem.main.dynamic;
    }

    const free_b = Math.max(0, mem.dev_total - used - (mem.task ? mem.task.static + mem.task.dynamic : 0));
    if (mem.task) {
        parts.push(seg(f(mem.main ? mem.main.static + mem.main.dynamic + free_b : free_b), "."));
        parts.push(seg(f((mem.main ? mem.main.static + mem.main.dynamic : 0) + free_b + mem.task.dynamic), "o"));
        parts.push(seg(f(mem.dev_total), "O"));
    } else {
        parts.push(seg(f(used + free_b), "."));
        parts.push(seg(f(mem.dev_total), "."));
    }

    // pad to BAR_W in case of rounding drift
    let bar = parts.join("");
    if (bar.length < BAR_W) bar += ".".repeat(BAR_W - bar.length);
    if (bar.length > BAR_W) bar = bar.slice(0, BAR_W);
    return `[${bar}]`;
}