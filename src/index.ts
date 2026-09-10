import { join, resolve } from "node:path";
import { showBreakpoints } from "./show-breakpoints.js";
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
import { parseArgvAndConfig } from "./cli.js";

const log: ConsolaInstance = logger.withTag('main');

const PROJECT_ROOT = resolve(import.meta.dirname, '..');
const PRESET_PATH = join(PROJECT_ROOT, 'generated-preset.ini');
const DEFAULT_CONFIG_PATH = resolve(PROJECT_ROOT, 'config.yaml');

let router: RouterProcess | undefined;
let planner: Planner | undefined;
let api: ApiServer | undefined;

createShutdownHandlers();

const [parsed, config] = await parseArgvAndConfig(
    PROJECT_ROOT, 
    PRESET_PATH, 
    DEFAULT_CONFIG_PATH
);

// TODO: temporarily disabling hadamard rotation to simplify strategy implementation
const has_kv_quantize_strat = Object.values(config.models).some(m => m.ladder.some(id => ['quantize-kv-q8', 'quantize-kv-q4'].includes(id)));
if (has_kv_quantize_strat) {
    process.env.LLAMA_ATTN_ROT_DISABLE = '1';
}

// start llama-server in router mode
router = new RouterProcess(config.router, config.model_load);
const llama_api = await router.start(PRESET_PATH);

const strategies = createStrategies(llama_api);

// define models from config
const models = new Map<ModelId, ModelEntry>();
for (const [name, cfg] of Object.entries(config.models)) {
    const rungs = buildLadder(cfg.initial_state, cfg.ladder, strategies);
    const entry = new ModelEntry(llama_api, name, cfg.aliases, rungs);
    models.set(name, entry);
}

if (models.size === 0) {
    throw new Error('no models configured');
}

planner = new Planner(llama_api, config, models);
api = new ApiServer(planner, config);

// if `--calc-breakpoints`, load each model and pretty-print strategy breakpoints on startup
if (parsed.calc_breakpoints) {
    try {
        await showBreakpoints(llama_api, config, models, (id: ModelId, ctx: number) => {
            planner.max_ctx.set(id, ctx);
        });
    } catch (err) {
        log.error(`startup error (showBreakpoints): ${err}`);
        await shutdown('error');
        process.exit(1);
    }
}

// start HTTP API and serve default model (unless `--idle` is set)
try {
    if (!parsed.idle) {
        await planner.serveDefault();
    }
    await api.start();

    log.box([
        `llama-manager is running`,
        ``,
        `  api:    http://${config.host}:${config.port}`,
        `  models: ${[...models.keys()].join(', ')}`,
    ].join('\n'));
} catch (err) {
    log.error(`startup error: ${err}`);
    await shutdown('error');
    process.exit(1);
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

/* -------------------- STOP SERVER -------------------- */

// Shutdown if we receive an interrupt or any uncaught errors
function createShutdownHandlers() {
    for (const evt of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
        process.once(evt, async () => {
            await shutdown(evt);
            process.exit(0);
        });
    }

    for (const evt of ['uncaughtException', 'unhandledRejection'] as const) {
        process.once(evt, async (err?: unknown) => {
            if (err instanceof Error) {
                log.error(`${evt}: ${err.message}`);
                if (err.stack) log.error(err.stack);
            } else if (err !== undefined) {
                log.error(`${evt}: ${String(err)}`);
            }
            await shutdown(evt);
            process.exit(1);
        });
    }
}

async function shutdown(event: string) {
    log.info(`shutdown: ${event}`);

    await Promise.allSettled([
        router?.shutdown().catch(err => log.warn(`router shutdown error: ${err}`)),
        planner?.shutdown().catch(err => log.warn(`planner shutdown error: ${err}`)),
        api?.shutdown().catch(err => log.warn(`api shutdown error: ${err}`)),
    ]);

    log.info('goodbye!');
}