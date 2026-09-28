// Offline, deterministic NIAH task builder. Does NOT execute code from the
// reference repo or contact a model. Generates reviewable JSON + a short index
// under attic/tasks/. The essays remain under attic/; see attribution there.
// Run from repo root: npx tsx src/test/weight-kv-tasks.ts
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

export type Kind = "single" | "disambiguation" | "chain" | "long_chain" | "revision_chain" | "withdrawal_graph";
export interface Needle { depth: number; text: string }
export interface GraphStep { source: string; winningRevision: number; next: string; withdrawnRevisions: number[] }
export interface GraphRecipe { policy: string; steps: GraphStep[]; decoyTerminals: string[] }
export interface TaskFile {
    id: string;
    kind: Kind;
    seed: number;
    targetTokens: number;
    source: string;
    sourceCommit: string;
    // Long enough to let the evaluator trim to exact Qwen-token length.
    hayParagraphs: string[];
    needles: Needle[];
    question: string;
    expected: string;
    chain?: string[];
    graph?: GraphRecipe; // offline-verifiable answer path for withdrawal_graph
    haySha256: string;
}
const ROOT = resolve(import.meta.dirname, "../..");
const ESSAYS = join(ROOT, "attic/needle-in-a-haystack/needlehaystack/PaulGrahamEssays");
const OUT = join(ROOT, "attic/tasks");
const COMMIT = "021385d68d3202e37893e9d3cd29011c569abe30";
function rng(seed: number): () => number {
    let x = seed >>> 0;
    return () => { x = (x + 0x6d2b79f5) >>> 0; let t = Math.imul(x ^ (x >>> 15), 1 | x); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };
}
function shuffle<T>(values: T[], random: () => number): T[] {
    for (let i = values.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [values[i], values[j]] = [values[j], values[i]]; }
    return values;
}
function code(seed: number, i: number): string {
    return createHash("sha256").update(`weight-kv-needle:v1:${seed}:${i}`).digest("hex").slice(0, 16).toUpperCase();
}
// Each text slice ends at a word boundary. No model tokenizer is used offline;
// the evaluator sizes the final prompt using the running Qwen model's /tokenize.
function hay(seed: number): { paragraphs: string[]; source: string } {
    const random = rng(seed);
    const files = shuffle(readdirSync(ESSAYS).filter(f => f.endsWith(".txt")).sort(), random);
    const paragraphs: string[] = [];
    for (const file of files) {
        const text = readFileSync(join(ESSAYS, file), "utf8").replaceAll(/\s+/g, " ").trim();
        // Chunks of 70 words keep the length adjustment granularity small.
        const words = text.split(" ");
        for (let i = 0; i < words.length; i += 70) {
            const chunk = words.slice(i, i + 70).join(" ").trim();
            if (chunk.length > 100) paragraphs.push(chunk);
        }
    }
    return { paragraphs, source: `gkamradt/needle-in-a-haystack: needlehaystack/PaulGrahamEssays/*.txt (shuffled with seed ${seed})` };
}
function make(kind: Kind, targetTokens: number, seed: number, depths: readonly number[]): TaskFile {
    const { paragraphs, source } = hay(seed);
    // Need enough hay for tokenizers that use substantially fewer than four
    // characters per token. Real target is verified by evaluator at runtime.
    if (paragraphs.join("\n\n").length < 5 * targetTokens) throw new Error("not enough hay for requested target");
    const entity = `Archive entry ${code(seed, 0)}`;
    const values = Array.from({ length: 10 }, (_, i) => code(seed, i + 1));
    let needles: Needle[];
    let expected: string;
    let question: string;
    let chain: string[] | undefined;
    if (kind === "single") {
        needles = [{ depth: depths[0], text: `Archive memo: The verification code for ${entity} is ${values[0]}.` }];
        expected = values[0];
        question = `According to the archive memo, what is the verification code for ${entity}?`;
    } else if (kind === "disambiguation") {
        needles = depths.map((depth, i) => ({ depth, text: `Archive memo: The verification code for Archive entry ${code(seed, i)} is ${values[i]}.` }));
        expected = values[0];
        question = `According to the archive memos, what is the verification code for ${entity}? Beware similar entries.`;
    } else {
        chain = values.slice(0, 4);
        // A -> B -> C -> D. The answer is never in the question or in the
        // initial link; the link facts are spread and must be combined.
        needles = depths.map((depth, i) => ({ depth, text: `Archive cross-reference: entry ${chain![i]} points to entry ${chain![i + 1]}.` }));
        expected = chain[3];
        question = `Follow the archive cross-references starting at entry ${chain[0]} until you reach an entry with no further cross-reference. What is its entry code?`;
    }
    const id = `${kind}-${targetTokens / 1000}k-seed${seed}`;
    return { id, kind, seed, targetTokens, source, sourceCommit: COMMIT, hayParagraphs: paragraphs,
        needles, question, expected, chain, haySha256: createHash("sha256").update(paragraphs.join("\n\n")).digest("hex") };
}
// Harder tasks: eight hops in deliberately scrambled document order. The
// revision variant adds a competing obsolete link for six of the sources.
function makeHard(kind: "long_chain" | "revision_chain", targetTokens: number, seed: number): TaskFile {
    const { paragraphs, source } = hay(seed);
    if (paragraphs.join("\n\n").length < 5 * targetTokens) throw new Error("not enough hay for requested target");
    const chain = Array.from({ length: 9 }, (_, i) => code(seed, i));
    const random = rng(seed + 88);
    const facts: string[] = [];
    for (let i = 0; i < chain.length - 1; i++) {
        facts.push(`Archive cross-reference, revision 2 (current): entry ${chain[i]} points to entry ${chain[i + 1]}.`);
        if (kind === "revision_chain" && i < 6) {
            facts.push(`Archive cross-reference, revision 1 (obsolete): entry ${chain[i]} points to entry ${code(seed, 100 + i)}.`);
        }
    }
    // Place irrelevant links as extra plausible-looking records.
    for (let i = 0; i < 4; i++) facts.push(`Archive cross-reference, revision 2 (current): entry ${code(seed, 200 + i)} points to entry ${code(seed, 300 + i)}.`);
    shuffle(facts, random); // makes document order independent of traversal order
    const needles = facts.map((text, i) => ({ depth: 4 + 92 * (i + 0.5) / facts.length, text }));
    const question = `Starting at entry ${chain[0]}, follow the CURRENT (highest revision) archive cross-reference from each entry until there is no further cross-reference. What is the final entry code? Ignore obsolete revisions and unrelated entries.`;
    return { id: `${kind}-${targetTokens / 1000}k-seed${seed}`, kind, seed, targetTokens, source, sourceCommit: COMMIT,
        hayParagraphs: paragraphs, needles, question, expected: chain.at(-1)!, chain,
        haySha256: createHash("sha256").update(paragraphs.join("\n\n")).digest("hex") };
}
// Three revisions for every source. The highest published revision is often
// WRONG: separate withdrawal notices invalidate it. Every wrong branch leads
// to its own plausible terminal code, rather than simply ending prematurely.
// Unlike the previous suite, none of the link statements marks the right edge
// "current"; the solver must find the policy and join it with withdrawal facts.
function makeWithdrawalGraph(targetTokens: number, seed: number): TaskFile {
    const { paragraphs, source } = hay(seed);
    if (paragraphs.join("\n\n").length < 5 * targetTokens) throw new Error("not enough hay for requested target");
    const random = rng(seed + 881);
    const chain = Array.from({ length: 11 }, (_, i) => code(seed, i));
    const policy = "Archive policy for cross-references: for each source entry, use the highest-numbered published revision that has NOT been withdrawn. A withdrawal applies only to that source and revision, regardless of where its notice appears. Follow the selected destination and repeat until that destination has no published cross-references. Do not treat publication order or a higher withdrawn revision as authoritative.";
    const records: string[] = [policy];
    const steps: GraphStep[] = [];
    const decoyTerminals: string[] = [];
    for (let i = 0; i < chain.length - 1; i++) {
        const winningRevision = [1, 3, 2, 1, 2, 3, 1, 2, 3, 1][(i + (seed % 3)) % 10];
        const withdrawnRevisions = [1, 2, 3].filter(rev => rev > winningRevision);
        const choices = [1, 2, 3].map(rev => {
            const dest = rev === winningRevision ? chain[i + 1] : code(seed, 100 + 3 * i + rev);
            if (rev !== winningRevision) {
                const terminal = code(seed, 500 + 3 * i + rev);
                decoyTerminals.push(terminal);
                records.push(`Archive cross-reference for source ${dest}, revision 1: destination ${terminal}.`);
            }
            return { rev, dest };
        });
        for (const { rev, dest } of choices) {
            records.push(`Archive cross-reference for source ${chain[i]}, revision ${rev}: destination ${dest}.`);
        }
        for (const rev of withdrawnRevisions) {
            records.push(`Withdrawal notice: cross-reference revision ${rev} for source ${chain[i]} is withdrawn. Its destination must not be used.`);
        }
        steps.push({ source: chain[i], winningRevision, next: chain[i + 1], withdrawnRevisions });
    }
    // More unrelated records; none share a source with the answer path.
    for (let i = 0; i < 14; i++) records.push(`Archive cross-reference for source ${code(seed, 900 + i)}, revision 1: destination ${code(seed, 950 + i)}.`);
    const hayHash = createHash("sha256").update(paragraphs.join("\n\n")).digest("hex");
    shuffle(records, random);
    const needles = records.map((text, i) => ({ depth: 3 + 94 * (i + 0.5) / records.length, text }));
    const question = `Following the archive's cross-reference policy, start at source entry ${chain[0]} and follow selected destinations until there is no published cross-reference for the destination. What is that final entry code?`;
    const task: TaskFile = { id: `withdrawal_graph-${targetTokens / 1000}k-seed${seed}`, kind: "withdrawal_graph", seed,
        targetTokens, source, sourceCommit: COMMIT, hayParagraphs: paragraphs, needles, question,
        expected: chain.at(-1)!, chain, graph: { policy, steps, decoyTerminals }, haySha256: hayHash };
    validateWithdrawalGraph(task);
    return task;
}

export function validateWithdrawalGraph(task: TaskFile): void {
    const recipe = task.graph;
    if (!recipe || !task.chain || !recipe.steps.length) throw new Error(`missing graph recipe: ${task.id}`);
    const records = new Map<string, Map<number, string>>();
    const withdrawn = new Map<string, Set<number>>();
    for (const { text } of task.needles) {
        const edge = /^Archive cross-reference for source ([A-F0-9]{16}), revision ([1-3]): destination ([A-F0-9]{16})\.$/.exec(text);
        if (edge) {
            const [, src, rawRev, dest] = edge;
            const byRevision = records.get(src) ?? new Map<number, string>();
            if (byRevision.has(Number(rawRev))) throw new Error(`duplicate link ${src} revision ${rawRev}`);
            byRevision.set(Number(rawRev), dest);
            records.set(src, byRevision);
        }
        const notice = /^Withdrawal notice: cross-reference revision ([1-3]) for source ([A-F0-9]{16}) is withdrawn\./.exec(text);
        if (notice) {
            const [, rawRev, src] = notice;
            const set = withdrawn.get(src) ?? new Set<number>();
            set.add(Number(rawRev)); withdrawn.set(src, set);
        }
    }
    let current = task.chain[0];
    const visited = new Set<string>();
    for (const step of recipe.steps) {
        if (visited.has(current) || current !== step.source) throw new Error(`broken graph path at ${current}`);
        visited.add(current);
        const choices = [...(records.get(current) ?? [])].filter(([rev]) => !withdrawn.get(current)?.has(rev));
        if (!choices.length) throw new Error(`no live link for ${current}`);
        const [rev, dest] = choices.sort((a, b) => b[0] - a[0])[0];
        if (rev !== step.winningRevision || dest !== step.next ||
            JSON.stringify([...(withdrawn.get(current) ?? [])].sort()) !== JSON.stringify(step.withdrawnRevisions)) {
            throw new Error(`ambiguous or wrong path at ${current}`);
        }
        current = dest;
    }
    if (records.has(current) || current !== task.expected || recipe.decoyTerminals.includes(current) || task.question.includes(current)) {
        throw new Error(`invalid terminal for ${task.id}`);
    }
}

function main() {
    mkdirSync(OUT, { recursive: true });
    const tasks: TaskFile[] = [];
    for (const length of [50_000, 75_000]) {
        // Two deterministic seeds/depth rotations at each length. Each task
        // gets a distinct hay permutation and new answer codes.
        for (const [offset, depths] of [[0, [12, 38, 72, 89]], [1, [24, 52, 81, 94]]] as const) {
            const seed = length + offset;
            tasks.push(make("single", length, seed + 10, [depths[0]]));
            tasks.push(make("disambiguation", length, seed + 20, depths));
            tasks.push(make("chain", length, seed + 30, depths.slice(0, 3)));
        }
    }
    // Same graph-generation rules and seeds-per-length as the existing cohort;
    // only the amount of hay changes. No task is Qwen-tokenized offline.
    for (const length of [10_000, 25_000]) {
        for (let offset = 60; offset <= 69; offset++) {
            tasks.push(makeWithdrawalGraph(length, length + offset));
        }
    }
    // Freeze withdrawal_graph task design, lengths and generation logic. The
    // additional eight seeds per length extend the sample without changing
    // its difficulty parameters. Preserve the original four seeds as-is.
    for (const length of [50_000, 75_000]) {
        tasks.push(makeHard("long_chain", length, length + 40));
        tasks.push(makeHard("revision_chain", length, length + 41));
        for (let offset = 60; offset <= 69; offset++) {
            tasks.push(makeWithdrawalGraph(length, length + offset));
        }
    }
    const rows: string[] = ["# Weight-KV long-context tasks", "", "Generated by `npx tsx src/test/weight-kv-tasks.ts` (offline).",
        "Hay: gkamradt/needle-in-a-haystack, commit " + COMMIT + ", bundled Paul Graham essays.",
        "This file previews needles/questions/answers. JSON files include the full shuffled hay; the evaluator trims it using Qwen /tokenize to hit the target length.",
        "Do not copy the essays into tracked files without checking their separate redistribution rights.",
        "New short-context withdrawal_graph tasks: seeds 10060–10069 (10k) and 25060–25069 (25k), ten each. Graph rules are identical to the 50k/75k cohort; the needles are denser in shorter hay.",
        "New long-context seeds: 50062–50069 and 75062–75069 (eight per length). Previous seeds 50060–61 and 75060–61 are retained unchanged.", ""];
    for (const task of tasks) {
        writeFileSync(join(OUT, `${task.id}.json`), JSON.stringify(task, null, 2) + "\n");
        rows.push(`## ${task.id} (${task.kind}, target ${task.targetTokens} tokens)`, "", `Question: ${task.question}`, `Expected FINAL: ${task.expected}`, "",
            ...(task.graph ? [`Ground-truth path: ${task.graph.steps.map(s => `${s.source} [r${s.winningRevision}]`).join(" → ")} → ${task.expected}`, ""] : []),
            ...task.needles.map(n => `- ~${n.depth.toFixed(1)}%: ${n.text}`), "");
    }
    writeFileSync(join(OUT, "README.md"), rows.join("\n"));
    console.log(`Wrote ${tasks.length} tasks and README.md to ${OUT}`);
}
main();
