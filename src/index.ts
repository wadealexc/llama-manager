import loadConfig from "./config/loader.js";
import { RouterProcess } from "./client/router-process.js";
import type { ConsolaInstance } from "consola";
import { logger } from "./logger.js";
import { Planner } from "./planner/planner.js";
import { ApiServer } from "./api/server.js";
import { ModelEntry } from "./planner/model-entry.js";
import type { Rung } from "./planner/model-entry.js";
import type { ModelId, ModelState, StrategyId } from "./config/types.js";
import type { Strategy } from "./planner/types.js";
import { createStrategies } from "./planner/strategies/index.js";

const log: ConsolaInstance = logger.withTag('main');

const CONFIG_PATH = process.env.MANAGER_CONFIG ?? "./config.yaml";
const [config, preset_path] = await loadConfig(CONFIG_PATH);

const router = new RouterProcess(config.router, config.model_load);

// Shutdown if we receive an interrupt or any uncaught errors
for (const evt of ['SIGINT', 'SIGTERM', 'SIGHUP', 'uncaughtException', 'unhandledRejection'] as const) {
    process.once(evt, async (err?: unknown) => {
        if (err instanceof Error) {
            log.error(`${evt}: ${err.message}`);
            if (err.stack) log.error(err.stack);
        } else if (err !== undefined) {
            log.error(`${evt}: ${String(err)}`);
        }
        await shutdown(evt);
        process.exit();
    });
}

const llama_api = await router.start(preset_path);

const strategies = createStrategies(llama_api);

const models = new Map<ModelId, ModelEntry>();
for (const [name, cfg] of Object.entries(config.models)) {
    const rungs = buildLadder(cfg.initial_state, cfg.ladder, strategies);
    const entry = new ModelEntry(llama_api, name, rungs);
    models.set(name, entry);
}

if (models.size === 0) {
    throw new Error('no models configured');
}

const planner = new Planner(llama_api, config, models);

const api = new ApiServer(planner, config);

try {
    await planner.serveDefault();
    await api.start();
} catch (err) {
    log.error(`startup error: ${err}`);
    await shutdown('error');
}

/* -------------------- STOP SERVER -------------------- */

async function shutdown(event: string) {
    log.info(`shutdown: ${event}`);

    await Promise.allSettled([
        planner?.shutdown().catch(err => log.warn(`planner shutdown error: ${err}`)),
        router?.shutdown().catch(err => log.warn(`router shutdown error: ${err}`)),
        api?.shutdown().catch(err => log.warn(`api shutdown error: ${err}`)),
    ]);

    log.info('goodbye!');
}

function buildLadder(initial: ModelState, ids: StrategyId[], strats: Map<StrategyId, Strategy>): Rung[] {
    const rungs: Rung[] = [{ strategy: 'none', impl: null!, state: initial }];
    let state = initial;
    for (const id of ids) {
        const s = strats.get(id);
        if (!s) continue;
        if (!s.canApply(state)) continue;
        state = s.getNewState(state);
        rungs.push({ strategy: id, impl: s, state });
    }
    return rungs;
}