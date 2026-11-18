export async function captureOutput(fn: () => Promise<void>): Promise<string> {
    const chunks: string[] = [];
    const streams = [process.stdout, process.stderr];
    const writes = streams.map(stream => stream.write);
    for (const stream of streams) {
        stream.write = ((chunk: string | Uint8Array) => {
            chunks.push(String(chunk));
            return true;
        }) as typeof stream.write;
    }
    try {
        await fn();
    } finally {
        streams.forEach((stream, i) => { stream.write = writes[i]; });
    }
    return chunks.join('');
}
