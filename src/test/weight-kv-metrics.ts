// Pure, model-independent scoring helpers for weight-kv-eval.ts.
export type RegionLabel = "head" | "25%" | "50%" | "75%" | "tail";
export interface Region { index: number; label: RegionLabel }
export interface KlEstimate { nats: number; retainedIds: number; otherP: number; otherQ: number }
export type KlResult = KlEstimate | { unavailable: string };

/**
 * Scoring jumps forward through a fixed reference continuation. Only the
 * previous prompt should already be cached; the newly added reference tokens
 * MUST be evaluated. Consequently cached/currentTotal is not a cache-hit
 * rate for jumps. Check retention of the previous prompt instead.
 *
 * The server may re-evaluate a few tokens to obtain logits (and to handle
 * checkpoints), so tolerate a small amount of replay. Any substantial loss
 * of previously computed state is still an error.
 */
export function checkScoreCache(
    total: number,
    cached: number,
    processed: number,
    previousPromptTokens?: number,
    minInitialHit = 0.95,
    replayTolerance = 16,
): { cached: number; processed: number; reusedFraction: number; newlyAdded: number; replayed: number } {
    if (![total, cached, processed].every(Number.isSafeInteger) || total <= 0 || cached < 0 || cached > total || processed < 0 || cached + processed < total) {
        throw new Error(`invalid cache counts: total=${total}, cached=${cached}, processed=${processed}`);
    }
    if (previousPromptTokens === undefined) {
        if (cached / total < minInitialHit) throw new Error(`initial cache reuse too low: ${cached}/${total}`);
        return { cached, processed, reusedFraction: cached / total, newlyAdded: 0, replayed: total - cached };
    }
    if (!Number.isSafeInteger(previousPromptTokens) || previousPromptTokens <= 0 || previousPromptTokens > total) {
        throw new Error(`invalid previous prompt length: ${previousPromptTokens} > ${total}`);
    }
    const replayed = Math.max(0, previousPromptTokens - cached);
    if (replayed > replayTolerance) {
        throw new Error(`lost previous prompt cache: retained ${cached}/${previousPromptTokens}, replayed at least ${replayed} old tokens (allowed ${replayTolerance}); total=${total}, processed=${processed}`);
    }
    return {
        cached, processed,
        reusedFraction: Math.min(cached, previousPromptTokens) / previousPromptTokens,
        newlyAdded: total - previousPromptTokens,
        replayed,
    };
}

export function selectRegions(length: number, head = 4, middle = 4, tail = 12): Region[] {
    if (length < 1) throw new Error("cannot score empty reference");
    const selected = new Map<number, RegionLabel>();
    const add = (start: number, count: number, label: RegionLabel) => {
        for (let i = Math.max(0, start); i < Math.min(length, start + count); i++) selected.set(i, label);
    };
    add(0, head, "head");
    for (const [label, fraction] of [["25%", 0.25], ["50%", 0.5], ["75%", 0.75]] as const) {
        add(Math.floor(length * fraction) - Math.floor(middle / 2), middle, label);
    }
    add(length - tail, tail, "tail");
    return [...selected.entries()].sort(([a], [b]) => a - b).map(([index, label]) => ({ index, label }));
}

// The same token-ID partition must be used for Q4 and hybrid at each step.
// Start with the intersection across all runs, and (for top-N mode) drop the
// least probable IDs until each distribution has a resolvable OTHER bucket.
// If an empty partition would be necessary, report unavailable instead of 0.
export function sharedPartition(distributions: Array<ReadonlyMap<number, number>>, exact: boolean): Set<number> {
    if (!distributions.length) throw new Error("no distributions");
    const ids = new Set([...distributions[0].keys()].filter(id => distributions.every(d => d.has(id))));
    if (exact) return ids;
    const mass = (d: ReadonlyMap<number, number>) => {
        let sum = 0;
        for (const id of ids) sum += Math.exp(d.get(id)!);
        return sum;
    };
    while (ids.size > 1 && distributions.some(d => 1 - mass(d) <= 1e-8)) {
        // Drop the smallest *largest* mass across distributions: a token with
        // low probability everywhere, not an important token in another run.
        const worst = [...ids].reduce((a, b) =>
            Math.max(...distributions.map(d => d.get(a)!)) < Math.max(...distributions.map(d => d.get(b)!)) ? a : b);
        ids.delete(worst);
    }
    return ids;
}

// Common-ID + OTHER is a coarse-graining of both full distributions, hence
// D_KL on it is a lower bound. Do not renormalize top-N or substitute epsilon
// for an unresolvable float-rounded tail. Exact mode requires full vocab.
export function klOnPartition(p: ReadonlyMap<number, number>, q: ReadonlyMap<number, number>, ids: ReadonlySet<number>, exact = false): KlResult {
    if (!ids.size) return { unavailable: "no shared token IDs" };
    if (exact && (ids.size !== p.size || ids.size !== q.size)) return { unavailable: "full-vocab token IDs differ" };
    let massP = 0, massQ = 0, sum = 0;
    for (const id of ids) {
        const lp = p.get(id), lq = q.get(id);
        if (lp === undefined || lq === undefined || !Number.isFinite(lp) || !Number.isFinite(lq)) return { unavailable: `missing or nonfinite token ${id}` };
        const a = Math.exp(lp), b = Math.exp(lq);
        massP += a; massQ += b;
        sum += a * (lp - lq); // log-space ratio, avoiding tiny-probability underflow
    }
    const otherP = 1 - massP, otherQ = 1 - massQ;
    if (!exact) {
        if (otherP <= 1e-8 || otherQ <= 1e-8) return { unavailable: `OTHER mass unresolved (P=${otherP}, Q=${otherQ})` };
        sum += otherP * Math.log(otherP / otherQ);
    }
    if (!Number.isFinite(sum) || sum < -1e-7) return { unavailable: `invalid KL ${sum}` };
    return { nats: Math.max(0, sum), retainedIds: ids.size, otherP, otherQ };
}
