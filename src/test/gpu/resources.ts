import type { ApiServer } from '../../api/server.js';
import type { RouterProcess } from '../../client/router-process.js';
import type { Planner } from '../../planner/planner.js';
import type { PromptCache } from '../../planner/prompt-cache.js';

const resources = new Set<GpuTestResources>();
let stopping = false;

async function stopTests(event: string, exit_code: number, err?: unknown): Promise<void> {
    if (stopping) return;
    stopping = true;
    console.error(`GPU tests shutting down: ${event}`);
    if (err !== undefined) console.error(err);
    const results = await Promise.allSettled([...resources].map(resource => resource.shutdown()));
    for (const result of results) {
        if (result.status === 'rejected') console.error(result.reason);
    }
    process.exit(exit_code);
}

for (const [event, exit_code] of [['SIGINT', 130], ['SIGTERM', 143], ['SIGHUP', 129]] as const) {
    process.on(event, () => { void stopTests(event, exit_code); });
}
for (const event of ['uncaughtException', 'unhandledRejection'] as const) {
    process.on(event, err => { void stopTests(event, 1, err); });
}

export class GpuTestResources {
    directory: string;
    #controller = new AbortController();
    #cleanups: (() => Promise<void>)[] = [];
    #shutdown_promise?: Promise<void>;

    constructor(directory: string) {
        if (stopping) throw new Error('GPU tests are shutting down');
        this.directory = directory;
        resources.add(this);
    }

    get signal(): AbortSignal {
        return this.#controller.signal;
    }

    trackRouter(router: RouterProcess): void {
        this.signal.throwIfAborted();
        this.#cleanups.push(() => router.shutdown());
    }

    trackPlanner(planner: Planner): void {
        this.signal.throwIfAborted();
        this.#cleanups.push(() => planner.shutdown());
    }

    trackApi(api: ApiServer): void {
        this.signal.throwIfAborted();
        this.#cleanups.push(async () => {
            api.server?.closeAllConnections();
            await api.shutdown();
        });
    }

    trackCache(cache: PromptCache): void {
        this.signal.throwIfAborted();
        this.#cleanups.push(() => cache.shutdown());
    }

    shutdown(): Promise<void> {
        this.#shutdown_promise ??= this.#shutdown();
        return this.#shutdown_promise;
    }

    async #shutdown(): Promise<void> {
        this.#controller.abort('GPU test shutdown');
        try {
            const results = await Promise.allSettled(this.#cleanups.map(cleanup => Promise.resolve().then(cleanup)));
            const errors = results.flatMap(result => result.status === 'rejected' ? [result.reason] : []);
            if (errors.length) throw new AggregateError(errors, 'GPU test cleanup failed');
        } finally {
            resources.delete(this);
            if (stopping) console.error(`GPU test artifacts preserved: ${this.directory}`);
        }
    }
}
