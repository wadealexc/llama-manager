type Segment = {
    label: string;
    duration_ms: number;
};

type TimerEntry =
    | { kind: 'leaf'; segment: Segment }
    | { kind: 'child'; timer: Timer };

export class Timer {

    label: string;
    entries: TimerEntry[] = [];
    cur: { label: string; start: number } | undefined;

    constructor(label: string) {
        this.label = label;
    }

    start(label: string) {
        this.cur = { label, start: performance.now() };
    }

    stop() {
        if (!this.cur) return;
        this.entries.push({
            kind: 'leaf',
            segment: {
                label: this.cur.label,
                duration_ms: performance.now() - this.cur.start,
            },
        });
        this.cur = undefined;
    }

    child(label: string): Timer {
        const c = new Timer(label);
        this.entries.push({ kind: 'child', timer: c });
        return c;
    }

    totalMs(): number {
        let total = 0;
        for (const e of this.entries) {
            if (e.kind === 'leaf') total += e.segment.duration_ms;
            else total += e.timer.totalMs();
        }
        return total;
    }

    fmt(indent: number = 0): string {
        const pad = '    '.repeat(indent);
        const lines: string[] = [];

        let header = `${pad}${this.label}`;
        if (this.entries.length > 0) {
            header += ` (total elapsed: ${fmtMs(this.totalMs())})`;
        }
        lines.push(header);

        let i = 0;
        while (i < this.entries.length) {
            const entry = this.entries[i];
            if (entry.kind === 'child') {
                lines.push(entry.timer.fmt(indent + 1));
                i++;
            } else {
                const group: Segment[] = [];
                while (i < this.entries.length && this.entries[i].kind === 'leaf') {
                    group.push((this.entries[i] as { kind: 'leaf'; segment: Segment }).segment);
                    i++;
                }
                const parts = group.map(s => `${s.label}: ${fmtMs(s.duration_ms)}`);
                lines.push(`${pad}    [${parts.join(', ')}]`);
            }
        }

        return lines.join('\n');
    }
}

function fmtMs(ms: number): string {
    return `${(ms / 1000).toFixed(2)}s`;
}