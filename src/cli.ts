import { readFileSync } from "node:fs";
import { parseArgs, type ParsedArgs } from "./config/parser.js";
import type { ManagerConfig } from "./config/types.js";
import type { ConsolaInstance } from "consola";
import { logger } from "./logger.js";
import { buildConfig } from "./config/build.js";
import { join } from "node:path";

const log: ConsolaInstance = logger.withTag('cli');

const USAGE = `Usage: llama-manager [manager-flags] [llama-server-flags]

llama-manager reactively manages model context windows. Pass your existing 
llama-server invocation directly to run a single model:

  llama-manager --model ./model.gguf --mmproj ./mmproj.gguf -ngl 999

Pass a config file to define multiple models to serve (models are swapped
in response to requests):

  llama-manager --config ./config.yaml

A model's breakpoint table (context capacity at each strategy) is printed the
first time the model is loaded.

Flags:
  --calc-breakpoints         calculate per-rung context capacity on startup rather
                             than on first load.
  --idle                     start without loading any models
  --host <host>              external listen host (default: 127.0.0.1)
  --port <port>              external listen port (default: 8080)
  --sleep-idle-seconds <n>   unload all models after n idle seconds (default: 600)
  --ladder <id,...>          override the derived strategy ladder
    (default: [disable-spec, mmproj-to-cpu, quantize-kv-q8, quantize-kv-q4])
  --slot-save-path <path>    directory for kvcache save/restore files
  --bin <path>               path to the llama-server binary
  --config <path>            supply a config file. router mode reads models from it;
                             single model mode takes all settings but ignores model entries
  --version                  print version
  --help                     print this message

Environment variables:
- MANAGER_CONFIG=<path>      supply a YAML config
- LLAMA_BIN=<path>           path to llama-server binary

All other arguments are passed through to llama-server.`;

export async function parseArgvAndConfig(
    project_root: string,
    preset_path: string,
    default_config_path: string,
): Promise<[ParsedArgs, ManagerConfig]> {
    let parsed: ParsedArgs;
    let config: ManagerConfig;
    try {
        parsed = parseArgs(process.argv.slice(2));
    
        if (parsed.help) {
            await flushOutput(USAGE + '\n');
            process.exit(0);
        }
    
        if (parsed.version) {
            await flushOutput(`llama-manager ${getVersion(project_root)}\n`);
            process.exit(0);
        }
    
        config = await buildConfig(parsed, project_root, preset_path, default_config_path);
    } catch (err) {
        log.error(err instanceof Error ? err.message : String(err));
        process.exit(1);
    }

    return [parsed, config];
}

function getVersion(project_root: string): string {
    const pkg = JSON.parse(readFileSync(join(project_root, 'package.json'), 'utf8')) as { version: string };
    return pkg.version;
}

async function flushOutput(text: string): Promise<void> {
    await new Promise<void>((resolve) => {
        process.stdout.write(text, () => resolve());
    });
}