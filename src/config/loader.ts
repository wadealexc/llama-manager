import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { load as yamlLoad } from "js-yaml";
import type { ManagerConfig, ModelConfig, ModelRole } from "./types.js";
import { DEFAULT_FIT_OVERHEAD_MIB } from "../llama-cpp-constants.js";

// Manager fields on a model entry; everything else is treated as a
// router preset field and passed to llama-server via a generated .ini.
const MANAGER_MODEL_FIELDS = new Set([
    "name",
    "expected_response_tokens",
    "ladder",
    "c",
    "ctx-size",
    "has_spec",
    "has_mmproj",
    "fit_target_mib",
    "kv_unified",
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
        this.#deriveModelFields(config);
        mkdirSync(config.router.llama_log_dir, { recursive: true });

        writeFileSync(PRESET_OUT_PATH, this.#buildIni(config), "utf8");

        return [config, PRESET_OUT_PATH];
    }

    #deriveModelFields(config: ManagerConfig): void {
        for (const role of ROLES) {
            const entry = config.models[role] as (ModelConfig & Record<string, unknown>) | undefined;
            if (!entry) continue;

            entry.has_spec = getHasSpec(entry);
            entry.has_mmproj = getHasMmproj(entry);
            entry.kv_unified = getKvUnified(entry);

            const fit = entry['fit-target'] ?? entry['fitt'];
            entry.fit_target_mib = typeof fit === 'number' ? fit : DEFAULT_FIT_OVERHEAD_MIB;
        }
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

function getHasSpec(entry: Record<string, unknown>): boolean {
    const spec = entry['spec-type'];
    let types: string[];
    if (typeof spec === 'string') {
        types = spec.split(',').map(t => t.trim().toLowerCase()).filter(t => t !== '');
    } else if (Array.isArray(spec)) {
        types = spec.map(t => String(t).trim().toLowerCase()).filter(t => t !== '');
    } else {
        return false;
    }
    return types.length > 0 && !types.includes('none');
}

// TODO: missing `no-mmproj`
// TODO: assumes mmproj is on GPU
function getHasMmproj(entry: Record<string, unknown>): boolean {
    const path = entry['mmproj'] ?? entry['mm'];
    const url = entry['mmproj-url'] ?? entry['mmu'];
    const auto = entry['mmproj-auto'];

    return (typeof path === 'string' && path.length > 0)
        || (typeof url === 'string' && url.length > 0)
        || (typeof auto === 'boolean' && auto);
}

function getKvUnified(entry: Record<string, unknown>): boolean {
    const pos = entry['kv-unified'] ?? entry['kvu'];
    const neg = entry['no-kv-unified'] ?? entry['no-kvu'];

    if (typeof pos === 'boolean') return pos;
    if (typeof neg === 'boolean') return !neg;

    const par = entry['parallel'] ?? entry['np'];
    if (typeof par === 'number') return par === -1;

    return true;
}

export default async function loadConfig(path: string): Promise<[ManagerConfig, string]> {
    const loader = new ConfigLoader();
    return await loader.load(path);
}
