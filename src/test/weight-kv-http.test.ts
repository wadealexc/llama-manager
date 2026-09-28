// Model-free test: slow headers must not trigger an implicit 300s fetch limit.
// Run: npm run build && node --test dist/test/weight-kv-http.test.js
import { strict as assert } from "node:assert";
import { createServer } from "node:http";
import { test } from "node:test";
import { requestJson } from "./weight-kv-http.js";

test("GET and POST JSON tolerate delayed response headers", async () => {
    const server = createServer(async (req, res) => {
        await new Promise(resolve => setTimeout(resolve, 50));
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ method: req.method, path: req.url }));
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    try {
        const address = server.address();
        if (!address || typeof address === "string") throw new Error("server unavailable");
        const url = `http://127.0.0.1:${address.port}`;
        assert.deepEqual(await requestJson(url, "/slots", undefined, 1000), { method: "GET", path: "/slots" });
        assert.deepEqual(await requestJson(url, "/completion", { prompt: "x" }, 1000), { method: "POST", path: "/completion" });
    } finally {
        server.closeAllConnections();
        await new Promise<void>(resolve => server.close(() => resolve()));
    }
});

test("request deadline still aborts a stalled server", async () => {
    const server = createServer((_req, _res) => {});
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    try {
        const address = server.address();
        if (!address || typeof address === "string") throw new Error("server unavailable");
        await assert.rejects(requestJson(`http://127.0.0.1:${address.port}`, "/completion", {}, 100));
    } finally {
        server.closeAllConnections();
        await new Promise<void>(resolve => server.close(() => resolve()));
    }
});
