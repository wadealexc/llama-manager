// Shared, model-free selection for strategy free runs and KL replay.
import { readdirSync } from "node:fs";

export function selectTaskNames(dir: string, env: Record<string, string | undefined>): string[] {
    const single = env.EVAL_TASK?.trim();
    const list = env.EVAL_TASKS?.trim();
    const length = env.EVAL_LENGTH?.trim();
    if (single && list) throw new Error("Use EVAL_TASK or EVAL_TASKS, not both");
    if (length && !/^(10|25|50|75)k$/.test(length)) throw new Error("EVAL_LENGTH must be 10k, 25k, 50k or 75k");
    const available = readdirSync(dir).filter(f => /^withdrawal_graph-(10|25|50|75)k-seed\d+\.json$/.test(f));
    const requested = single ? [single] : list ? list.split(",").map(s => s.trim()) : null;
    if (requested) {
        if (requested.some(id => !/^withdrawal_graph-(10|25|50|75)k-seed\d+$/.test(id))) {
            throw new Error("EVAL_TASK/EVAL_TASKS requires comma-separated withdrawal_graph task IDs (without .json)");
        }
        if (new Set(requested).size !== requested.length) throw new Error("Duplicate task ID in EVAL_TASKS");
        for (const id of requested) {
            if (!available.includes(`${id}.json`)) throw new Error(`Unknown task ${id} in ${dir}`);
            if (length && !id.includes(`-${length}-`)) throw new Error(`Task ${id} conflicts with EVAL_LENGTH=${length}`);
        }
    }
    // Do not launch 36 tasks accidentally: a default run selects the new
    // seeds at the specified length. Existing 50k/75k default remains the
    // previously announced eight-seed cohort; 10k/25k use ten seeds each.
    const names = available.filter(f =>
        (requested ? requested.includes(f.slice(0, -5)) :
            length ? f.includes(`-${length}-`) && (/^(?:10|25)k$/.test(length)
                ? /seed(?:100|250)6\d\.json$/.test(f)
                : /seed(?:500|750)6[2-9]\.json$/.test(f))
                : /seed(?:500|750)6[2-9]\.json$/.test(f)) &&
        (!length || f.includes(`-${length}-`)),
    ).sort();
    if (!names.length) throw new Error(`No matching tasks in ${dir}`);
    return names;
}
