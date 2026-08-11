import loadConfig from "./config/loader.js";
import { RouterProcess } from "./client/router-process.js";

const CONFIG_PATH = process.env.MANAGER_CONFIG ?? "./config.yaml";

const config = await loadConfig(CONFIG_PATH);

const router = new RouterProcess(config.router);

/* -------------------- STOP SERVER -------------------- */

async function shutdown(event: string) {
    console.log(`shutdown: ${event}`);

    await Promise.allSettled([
        router.shutdown(),
    ]);

    console.log('goodbye!');
}

// Shutdown if we receive an interrupt or any uncaught errors
for (const evt of ['SIGINT', 'SIGTERM', 'SIGHUP', 'uncaughtException', 'unhandledRejection'] as const) {
    process.once(evt, async () => {
        await shutdown(evt);
        process.exit();
    });
}
