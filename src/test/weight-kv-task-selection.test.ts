// Model-free selection tests. Run via isolated TypeScript compile + node --test.
import { strict as assert } from "node:assert";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { selectTaskNames } from "./weight-kv-task-selection.js";

test("select only new seeded tasks by default, allow two-at-a-time batches", () => {
    const dir = mkdtempSync(join(tmpdir(), "kv-task-selection-"));
    const name = (length: string, seed: number) => `withdrawal_graph-${length}-seed${seed}`;
    try {
        for (const l of ["50k", "75k"]) for (const offset of [60, 61, 62, 63, 69]) {
            writeFileSync(join(dir, `${name(l, Number(l.slice(0, -1)) * 1000 + offset)}.json`), "{}");
        }
        for (const [length, seed] of [["10k", 10060], ["10k", 10069], ["25k", 25060], ["25k", 25069]] as const) {
            writeFileSync(join(dir, `${name(length, seed)}.json`), "{}");
        }
        const all = selectTaskNames(dir, {});
        assert.equal(all.length, 6);
        assert.ok(all.every(f => /seed(?:500|750)6[2-9]\.json$/.test(f)));
        assert.equal(selectTaskNames(dir, { EVAL_LENGTH: "75k" }).length, 3);
        assert.deepEqual(selectTaskNames(dir, { EVAL_LENGTH: "10k" }),
            [name("10k", 10060), name("10k", 10069)].map(n => `${n}.json`));
        assert.deepEqual(selectTaskNames(dir, { EVAL_LENGTH: "25k" }),
            [name("25k", 25060), name("25k", 25069)].map(n => `${n}.json`));
        const two = [name("50k", 50062), name("50k", 50063)];
        assert.deepEqual(selectTaskNames(dir, { EVAL_TASKS: two.join(",") }), two.map(n => `${n}.json`));
        assert.deepEqual(selectTaskNames(dir, { EVAL_TASK: name("50k", 50060) }), [`${name("50k", 50060)}.json`]);
        assert.throws(() => selectTaskNames(dir, { EVAL_TASKS: two[0], EVAL_TASK: two[1] }), /not both/);
        assert.throws(() => selectTaskNames(dir, { EVAL_TASKS: `${two[0]},${two[0]}` }), /Duplicate/);
        assert.throws(() => selectTaskNames(dir, { EVAL_TASK: two[0], EVAL_LENGTH: "75k" }), /conflicts/);
        assert.throws(() => selectTaskNames(dir, { EVAL_TASK: name("50k", 50064) }), /Unknown/);
        assert.throws(() => selectTaskNames(dir, { EVAL_LENGTH: "20k" }), /EVAL_LENGTH/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});
