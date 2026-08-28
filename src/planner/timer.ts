type StartTime = {
    label: string;
    start: number;
}

type Time = {
    label: string;
    duration_ms: number;
}

export class Timer {

    cur: StartTime | undefined;
    segments: Time[] = [];

    start(label: string) {
        this.cur = { label, start: performance.now() };
    }

    stop() {
        if (!this.cur) return;

        const ms = performance.now() - this.cur.start;

        this.segments.push({
            label: this.cur.label,
            duration_ms: ms,
        });
    }

    fmtTotal(): string {
        let total_ms = 0;
        for (const t of this.segments) {
            total_ms += t.duration_ms;
        }

        return fmt(total_ms);
    }

    fmtSegments(): string {
        let parts = [];
        for (const t of this.segments) {
            parts.push(`${t.label}: ${fmt(t.duration_ms)}`);
        }

        return `[` + parts.join(`, `) + `]`;
    }
}

function fmt(ms: number): string {
    return `${(ms / 1000).toFixed(2)}s`;
}