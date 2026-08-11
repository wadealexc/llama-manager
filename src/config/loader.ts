import { mkdirSync, readFileSync } from "node:fs";
import { load as yamlLoad } from "js-yaml";
import type { ManagerConfig } from "./types.js";
import { ConfigError } from "./types.js";

export class ConfigLoader {
    
    // Load and validate the config file at `path`
    async load(path: string): Promise<ManagerConfig> {
        let raw: unknown;
        try {
            raw = yamlLoad(readFileSync(path, "utf8"));
        } catch (e) {
            throw new ConfigError(
                `failed to parse config: ${(e as Error).message}`,
                path,
            );
        }

        // TODO: structural validation
        const config = raw as ManagerConfig;

        // create log directory if needed
        mkdirSync(config.router.llama_log_dir, { recursive: true });

        return config;
    }
}

export default async function loadConfig(path: string): Promise<ManagerConfig> {
    const loader = new ConfigLoader();
    return await loader.load(path);
}
