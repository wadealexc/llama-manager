// llama-server can take more than Node fetch/undici's default 300s headers
// timeout before returning a non-streaming completion. node:http does not have
// that implicit deadline; our explicit AbortSignal covers the whole request.
import { request } from "node:http";

export async function requestJson<T>(baseUrl: string, path: string, body?: unknown, timeoutMs = 4 * 60 * 60 * 1000): Promise<T> {
    const url = new URL(path, baseUrl);
    const data = body === undefined ? undefined : Buffer.from(JSON.stringify(body));
    return new Promise<T>((resolve, reject) => {
        const req = request(url, {
            method: data === undefined ? "GET" : "POST",
            headers: data === undefined ? undefined : {
                "content-type": "application/json",
                "content-length": data.length,
            },
            signal: AbortSignal.timeout(timeoutMs),
        }, res => {
            const chunks: Buffer[] = [];
            res.on("data", (chunk: Buffer) => chunks.push(chunk));
            res.on("error", reject);
            res.on("end", () => {
                const text = Buffer.concat(chunks).toString("utf8");
                if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
                    reject(new Error(`${path}: HTTP ${res.statusCode}: ${text.slice(0, 1500)}`));
                    return;
                }
                try { resolve(JSON.parse(text) as T); }
                catch (err) { reject(new Error(`${path}: invalid JSON response: ${String(err)}`)); }
            });
        });
        req.on("error", reject);
        req.end(data);
    });
}
