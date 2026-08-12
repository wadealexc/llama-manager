import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { load as yamlLoad } from "js-yaml";
import type { ManagerConfig, ModelRole } from "./types.js";

// Manager fields on a model entry; everything else is treated as a
// router preset field and passed to llama-server via a generated .ini.
const MANAGER_MODEL_FIELDS = new Set([
    "name",
    "expected_response_tokens",
    "ladder",
    "c",
    "ctx-size",
]);

const MIN_CTX = 1024;
const CTX_KEY = "ctx-size";

const ROLES: ModelRole[] = ["main", "task"];

const PRESET_OUT_PATH = "./generated-preset.ini";

export class ConfigLoader {

    async load(path: string): Promise<[ManagerConfig, string]> {
        let raw: unknown;
        try {
            raw = yamlLoad(readFileSync(path, "utf8"));
        } catch (e) {
            throw new Error(`failed to parse config: ${(e as Error).message}`);
        }

        const config = raw as ManagerConfig;
        mkdirSync(config.router.llama_log_dir, { recursive: true });

        writeFileSync(PRESET_OUT_PATH, this.#buildIni(config), "utf8");

        return [config, PRESET_OUT_PATH];
    }

    // Strip manager fields from each model entry and write the rest to a
    // .ini for llama-server's router mode
    #buildIni(config: ManagerConfig): string {
        const models = config.models;
        const sections: string[] = [];

        for (const role of ROLES) {
            const entry = models[role];
            if (!entry) continue;

            const name = entry.name ?? role;
            const lines = [`[${name}]`];
            // Generate model config with bare minimum allowed ctx
            lines.push(`${CTX_KEY} = ${MIN_CTX}`);

            for (const [key, value] of Object.entries(entry)) {
                if (MANAGER_MODEL_FIELDS.has(key)) continue;
                lines.push(`${key} = ${this.#iniValue(value)}`);
            }
            sections.push(lines.join("\n"));
        }

        return sections.join("\n\n") + "\n";
    }

    #iniValue(v: unknown): string {
        if (typeof v === "string") return v;
        if (typeof v === "number" || typeof v === "boolean") return String(v);
        if (Array.isArray(v)) return v.map(String).join(", ");
        return JSON.stringify(v);
    }
}

export default async function loadConfig(path: string): Promise<[ManagerConfig, string]> {
    const loader = new ConfigLoader();
    return await loader.load(path);
}
