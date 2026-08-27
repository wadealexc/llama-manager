import loadConfig from "./config/loader.js";
import { RouterProcess } from "./client/router-process.js";
import type { ConsolaInstance } from "consola";
import { logger } from "./logger.js";
import { Planner } from "./planner/planner.js";
import { ApiServer } from "./api/server.js";

const log: ConsolaInstance = logger.withTag('main');

const CONFIG_PATH = process.env.MANAGER_CONFIG ?? "./config.yaml";
const FORCE_COST_MODEL = process.argv.includes('--build-cost-model');
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

const planner = new Planner(llama_api, config);
const api = new ApiServer(planner, config);

try {
    await planner.initCostModel(CONFIG_PATH, FORCE_COST_MODEL);
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

