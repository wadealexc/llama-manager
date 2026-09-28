// Pure-math tests; no server or GPU.
// Run: npm run build && node --test dist/test/weight-kv-metrics.test.js
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { checkScoreCache, klOnPartition, selectRegions, sharedPartition } from "./weight-kv-metrics.js";

const dist = (entries: Array<[number, number]>) => new Map(entries.map(([id, p]) => [id, Math.log(p)]));

test("scoring jump distinguishes new reference tokens from lost cached prefix", () => {
    assert.ok(checkScoreCache(49979, 49975, 4).reusedFraction > 0.99);
    // The previous prompt was 49,982 tokens. Advancing the reference by 3,234
    // requires evaluating them, and should not count as cache eviction.
    const jump = checkScoreCache(53216, 49982, 3234, 49982);
    assert.equal(jump.newlyAdded, 3234);
    assert.equal(jump.replayed, 0);
    assert.equal(jump.reusedFraction, 1);
    assert.throws(() => checkScoreCache(53216, 100, 53116, 49982), /lost previous prompt cache/);
    assert.throws(() => checkScoreCache(53216, 49000, 4216, 49982), /lost previous prompt cache/);
});

test("region selection spans beginning, quartiles and final answer without duplicates", () => {
    const selected = selectRegions(200, 4, 4, 12);
    assert.equal(selected.length, 28);
    assert.deepEqual(selected.slice(0, 4).map(r => r.index), [0, 1, 2, 3]);
    assert.deepEqual(selected.slice(-12).map(r => r.index), Array.from({ length: 12 }, (_, i) => 188 + i));
    assert.deepEqual(selectRegions(3).map(r => r.index), [0, 1, 2]);
});

test("exact KL matches analytic two-token calculation and is directional", () => {
    const p = dist([[1, 0.75], [2, 0.25]]);
    const q = dist([[1, 0.50], [2, 0.50]]);
    const result = klOnPartition(p, q, new Set([1, 2]), true);
    assert.ok("nats" in result);
    assert.ok(Math.abs(result.nats - (0.75 * Math.log(1.5) + 0.25 * Math.log(0.5))) < 1e-12);
    assert.equal("nats" in klOnPartition(p, p, new Set([1, 2]), true), true);
});

test("top-N coarse KL is a lower bound and common across candidate runs", () => {
    const p = dist([[1, 0.4], [2, 0.3], [3, 0.2], [4, 0.1]]);
    const q = dist([[1, 0.3], [2, 0.2], [3, 0.1], [4, 0.4]]);
    const h = dist([[1, 0.35], [2, 0.25], [3, 0.15], [4, 0.25]]);
    const ids = sharedPartition([new Map([...p].slice(0, 2)), new Map([...q].slice(0, 3)), new Map([...h].slice(0, 2))], false);
    assert.deepEqual([...ids], [1, 2]);
    const coarse = klOnPartition(p, q, ids);
    const full = klOnPartition(p, q, new Set([1, 2, 3, 4]), true);
    assert.ok("nats" in coarse && "nats" in full);
    assert.ok(coarse.nats >= 0 && coarse.nats <= full.nats + 1e-12);
    assert.ok("nats" in klOnPartition(p, h, ids));
});

test("rounded-away OTHER mass drops shared IDs or explicitly reports unresolvable", () => {
    const p = dist([[1, 0.7], [2, 0.3]]);
    const q = dist([[1, 0.6], [2, 0.4]]);
    const ids = sharedPartition([p, q], false);
    assert.equal(ids.size, 1);
    const result = klOnPartition(p, q, ids);
    assert.ok("nats" in result && Number.isFinite(result.nats));
    assert.ok("unavailable" in klOnPartition(p, q, new Set([1, 2])));
    const concentrated = dist([[1, 1 - 1e-12], [2, 1e-12]]);
    assert.ok("unavailable" in klOnPartition(concentrated, concentrated, sharedPartition([concentrated], false)));
});
