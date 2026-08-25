import loadConfig from "./config/loader.js";
import { RouterProcess } from "./client/router-process.js";
import type { ConsolaInstance } from "consola";
import { logger } from "./logger.js";
import { Planner } from "./planner/planner.js";
import { ApiServer } from "./api/server.js";

const log: ConsolaInstance = logger.withTag('main');

const CONFIG_PATH = process.env.MANAGER_CONFIG ?? "./config.yaml";
const [config, preset_path] = await loadConfig(CONFIG_PATH);

const router = new RouterProcess(config.router, config.model_load);

// Shutdown if we receive an interrupt or any uncaught errors
for (const evt of ['SIGINT', 'SIGTERM', 'SIGHUP', 'uncaughtException', 'unhandledRejection'] as const) {
    process.once(evt, async () => {
        await shutdown(evt);
        process.exit();
    });
}

const llama_api = await router.start(preset_path);

const planner = new Planner(llama_api, config);
try {
    await planner.buildCostModel();
} catch (err) {
    log.error(`buildCostModel error: ${err}`);
    await shutdown('error');
}

const api = new ApiServer(planner, config);
await api.start();

await shutdown('debuggin');
log.info('done!');

/* -------------------- STOP SERVER -------------------- */

async function shutdown(event: string) {
    log.info(`shutdown: ${event}`);

    await Promise.allSettled([
        api?.shutdown(),
        router?.shutdown(),
        planner?.shutdown(),
    ]);

    log.info('goodbye!');
}

