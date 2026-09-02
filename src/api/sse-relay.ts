export interface SSEFrame {
    done: boolean;
    data?: unknown;
}

export class SSERelay implements AsyncIterable<SSEFrame> {

    reader: ReadableStreamDefaultReader<Uint8Array>;
    decoder = new TextDecoder();
    buf = "";

    constructor(body: ReadableStream<Uint8Array>) {
        this.reader = body.getReader();
    }

    async *[Symbol.asyncIterator](): AsyncIterator<SSEFrame> {
        try {
            while (true) {
                const { done, value } = await this.reader.read();
                if (done) {
                    if (this.buf.trim()) {
                        const frame = this.#parseEvent(this.buf);
                        if (frame) yield frame;
                    }
                    break;
                }

                this.buf += this.decoder.decode(value, { stream: true });
                const events = this.buf.split("\n\n");
                this.buf = events.pop() ?? "";

                for (const event of events) {
                    const frame = this.#parseEvent(event);
                    if (frame) yield frame;
                }
            }
        } finally {
            this.reader.releaseLock();
        }
    }

    #parseEvent(event: string): SSEFrame | null {
        if (!event.trim()) return null;

        const lines = event.split("\n");
        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);

            if (data === "[DONE]") {
                return { done: true };
            }

            try {
                return { done: false, data: JSON.parse(data) };
            } catch {
                continue;
            }
        }

        return null;
    }
}