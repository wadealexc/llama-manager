import { join, resolve } from "node:path";
import { readFileSync } from "node:fs";
import { showBreakpoints } from "./show-breakpoints.js";
import { parseArgs, parseRouterConfig } from "./config/parser.js";
import type { ParsedArgs } from "./config/parser.js";
import { buildConfig } from "./config/build.js";
import { RouterProcess } from "./client/router-process.js";
import type { ConsolaInstance } from "consola";
import { logger } from "./logger.js";
import { Planner } from "./planner/planner.js";
import { ApiServer } from "./api/server.js";
import { ModelEntry } from "./planner/model-entry.js";
import type { Rung } from "./planner/model-entry.js";
import type { ManagerConfig, ModelId, ModelState, StrategyId } from "./config/types.js";
import type { Strategy } from "./planner/types.js";
import { createStrategies } from "./planner/strategies/index.js";

const log: ConsolaInstance = logger.withTag('main');

const PROJECT_ROOT = resolve(import.meta.dirname, '..');
const PRESET_PATH = join(PROJECT_ROOT, 'generated-preset.ini');
const DEFAULT_CONFIG_PATH = resolve(PROJECT_ROOT, 'config.yaml');

const USAGE = `Usage: llama-manager [manager-flags] [llama-server-flags]

llama-manager reactively manages model context windows. Pass your existing 
llama-server invocation directly to run in server mode:

  llama-manager --model ./model.gguf --mmproj ./mmproj.gguf -ngl 999

Pass a config file to run in router mode:

  llama-manager --config ./config.yaml

Flags (router mode):
  --config <path>            run in router mode with the given config file
  --show-breakpoints         print per-rung context capacity, then exit
  --serve                    load default models at startup

Flags (server mode):
  --host <host>              external listen host (default: 127.0.0.1)
  --port <port>              external listen port (default: 8080)
  --sleep-idle-seconds <n>   unload all models after n idle seconds (default: disabled)
  --ladder <id,...>          override the default strategy ladder
    (default: [disable-spec, mmproj-to-cpu, quantize-kv-q8, quantize-kv-q4])

Flags (both):
  --calc-max-ctx             calculate max ctx for GET /v1/models
  --version                  print version
  --help                     print this message

All other arguments are passed through to llama-server.`;

let parsed: ParsedArgs;
let config: ManagerConfig;
try {
    parsed = parseArgs(process.argv.slice(2));

    if (parsed.help) {
        await flushOutput(USAGE + '\n');
        process.exit(0);
    }

    if (parsed.version) {
        await flushOutput(`llama-manager ${getVersion()}\n`);
        process.exit(0);
    }

    if (parsed.mode === 'server') {
        const env_path = process.env.MANAGER_CONFIG;
        if (env_path !== undefined && resolve(env_path) !== DEFAULT_CONFIG_PATH) {
            throw new Error(`model source cannot be combined with manager config path (MANAGER_CONFIG=${env_path})`);
        }
    }

    const config_path = parsed.mode === 'server'
        ? undefined
        : (parsed.config_path ?? process.env.MANAGER_CONFIG ?? DEFAULT_CONFIG_PATH);

    const source = parsed.mode === 'server' ? parsed.source : parseRouterConfig(config_path!);
    config = await buildConfig(source, PROJECT_ROOT, PRESET_PATH);
} catch (err) {
    log.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
}

// TODO: temporarily disabling hadamard rotation to simplify strategy implementation
const has_kv_quantize_strat = Object.values(config.models).some(m => m.ladder.some(id => ['quantize-kv-q8', 'quantize-kv-q4'].includes(id)));
if (has_kv_quantize_strat) {
    process.env.LLAMA_ATTN_ROT_DISABLE = '1';
}

const router = new RouterProcess(config.router, config.model_load);

// Shutdown if we receive an interrupt or any uncaught errors
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

const llama_api = await router.start(PRESET_PATH);

const strategies = createStrategies(llama_api);

const models = new Map<ModelId, ModelEntry>();
for (const [name, cfg] of Object.entries(config.models)) {
    const rungs = buildLadder(cfg.initial_state, cfg.ladder, strategies);
    const entry = new ModelEntry(llama_api, name, cfg.aliases, rungs);
    models.set(name, entry);
}

if (models.size === 0) {
    throw new Error('no models configured');
}

const planner = new Planner(llama_api, config, models);
const api = new ApiServer(planner, config);

if (parsed.show_breakpoints) {
    try {
        await showBreakpoints(llama_api, config, models, (id: ModelId, ctx: number) => {
            planner.max_ctx.set(id, ctx);
        });
    } catch (err) {
        log.error(`startup error (showBreakpoints): ${err}`);
        await shutdown('error');
        process.exit(1);
    }

    // in router mode, shut down after --show-breakpoints finishes
    if (parsed.mode === 'router') {
        await shutdown('show-breakpoints finished');
        process.exit(0);
    }
} else if (parsed.calc_max_ctx) {
    try {
        // when loading a model for the first time, this runs through all strategies to calc a max
        // ctx. (enables `/v1/models` to return the model's max ctx after all strategies are applied)
        await planner.calcMaxCtx();
    } catch (err) {
        log.error(`startup error (calcMaxCtx): ${err}`);
        await shutdown('error');
        process.exit(1);
    }
}

try {
    if (parsed.mode === 'server' || process.argv.includes('--serve')) {
        await planner.serveDefault();
    }
    await api.start();
} catch (err) {
    log.error(`startup error: ${err}`);
    await shutdown('error');
    process.exit(1);
}

async function flushOutput(text: string): Promise<void> {
    await new Promise<void>((resolve) => {
        process.stdout.write(text, () => resolve());
    });
}

function getVersion(): string {
    const pkg = JSON.parse(readFileSync(join(PROJECT_ROOT, 'package.json'), 'utf8')) as { version: string };
    return pkg.version;
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

async function shutdown(event: string) {
    log.info(`shutdown: ${event}`);

    await Promise.allSettled([
        router?.shutdown().catch(err => log.warn(`router shutdown error: ${err}`)),
        planner?.shutdown().catch(err => log.warn(`planner shutdown error: ${err}`)),
        api?.shutdown().catch(err => log.warn(`api shutdown error: ${err}`)),
    ]);

    log.info('goodbye!');
}