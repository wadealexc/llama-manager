import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { load as yamlLoad } from "js-yaml";
import { type ManagerConfig, type ModelConfig, type ModelState, type StrategyId } from "./types.js";
import { DEFAULT_KV_PRECISION, MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";

const DEFAULT_LOG_DIR = './logs/';
const DEFAULT_SLOT_SAVE_DIR = './slots/';

const CTX_KEY = "ctx-size";

const MANAGER_FIELDS = new Set([
    "ladder",
    "ctx-size",
    "c",
    "cache-type-k",
    "ctk",
    "cache-type-v",
    "ctv",
    "slot-save-path",
]);

type RawModel = Record<string, unknown>;

export class ConfigLoader {

    async load(path: string, project_root: string, preset_out_path: string): Promise<[ManagerConfig, string]> {
        let raw: Record<string, unknown>;
        try {
            raw = yamlLoad(readFileSync(path, "utf8")) as Record<string, unknown>;
        } catch (e) {
            throw new Error(`failed to parse config: ${(e as Error).message}`);
        }

        const raw_models: Record<string, RawModel> = (raw.models as Record<string, RawModel>) ?? {};
        const raw_router = (raw.router ?? {}) as Record<string, unknown>;
        const raw_model_load = (raw['model-load'] ?? {}) as Record<string, unknown>;

        const llama_log_dir = normalizeDir((raw_router['llama-log-dir'] as string) ?? DEFAULT_LOG_DIR);
        const slot_save_path = normalizeDir((raw_router['slot-save-path'] as string) ?? DEFAULT_SLOT_SAVE_DIR);

        const models: Record<string, ModelConfig> = {};
        if (Object.keys(raw_models).length === 0) {
            throw new Error(`no models found in config`);
        }

        for (const [key, raw_entry] of Object.entries(raw_models)) {
            models[key] = this.#buildEntry(key, raw_entry);
        }

        const raw_defaults = raw['default-models'] as string[] | undefined;
        const model_keys = Object.keys(models);
        const default_models = (raw_defaults && raw_defaults.length > 0)
            ? raw_defaults
            : [model_keys[0]];

        for (const id of default_models) {
            if (!models[id]) {
                throw new Error(`default-model '${id}' not found in models`);
            }
        }

        const config: ManagerConfig = {
            router: {
                bin: resolve(project_root, (raw_router['bin'] as string) ?? './llama.cpp/build/bin/llama-server'),
                llama_log_dir: resolve(project_root, llama_log_dir),
                slot_save_path: resolve(project_root, slot_save_path),
                listen: (raw_router['listen'] as string) ?? '127.0.0.1:10000',
                poll_interval_ms: (raw_router['poll-interval-ms'] as number) ?? 50,
                poll_timeout_ms: (raw_router['poll-timeout-ms'] as number) ?? 10000,
                shutdown_grace_period_ms: (raw_router['shutdown-grace-period-ms'] as number) ?? 10000,
            },
            listen: (raw.listen as string) ?? '127.0.0.1:10001',
            idle_timeout: (raw['idle-timeout'] as number) ?? 300,
            model_load: {
                poll_interval_ms: (raw_model_load['poll-interval-ms'] as number) ?? 100,
                poll_timeout_ms: (raw_model_load['poll-timeout-ms'] as number) ?? 50000,
            },
            models,
            default_models,
        };

        mkdirSync(config.router.llama_log_dir, { recursive: true });
        mkdirSync(config.router.slot_save_path, { recursive: true });
        writeFileSync(preset_out_path, this.#buildIni(raw_models, config.router.slot_save_path), "utf8");

        return [config, preset_out_path];
    }

    #buildEntry(name: string, raw: RawModel): ModelConfig {
        const initial_state: ModelState = {
            mmproj_loaded: getHasMmproj(raw),
            spec_loaded: getHasSpec(raw),
            kv_unified: getKvUnified(raw),
            cache_type_k: DEFAULT_KV_PRECISION,
            cache_type_v: DEFAULT_KV_PRECISION,
        };

        return {
            name,
            ladder: (raw['ladder'] as StrategyId[]) ?? [],
            initial_state,
        };
    }

    #buildIni(raw: Record<string, RawModel>, slot_save_path: string): string {
        const sections: string[] = [];

        for (const [name, entry] of Object.entries(raw)) {
            const lines = [
                `[${name}]`,
                `${CTX_KEY} = ${MIN_ALLOWED_CTX}`,
                `cache-type-k = ${DEFAULT_KV_PRECISION}`,
                `cache-type-v = ${DEFAULT_KV_PRECISION}`,
                `slot-save-path = ${slot_save_path}`,
            ];

            for (const [key, value] of Object.entries(entry)) {
                if (MANAGER_FIELDS.has(key)) continue;
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

function normalizeDir(p: string): string {
    return p.endsWith('/') ? p : p + '/';
}

function getHasSpec(entry: RawModel): boolean {
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
function getHasMmproj(entry: RawModel): boolean {
    const path = entry['mmproj'] ?? entry['mm'];
    const url = entry['mmproj-url'] ?? entry['mmu'];
    const auto = entry['mmproj-auto'];

    return (typeof path === 'string' && path.length > 0)
        || (typeof url === 'string' && url.length > 0)
        || (typeof auto === 'boolean' && auto);
}

function getKvUnified(entry: RawModel): boolean {
    const pos = entry['kv-unified'] ?? entry['kvu'];
    const neg = entry['no-kv-unified'] ?? entry['no-kvu'];

    if (typeof pos === 'boolean') return pos;
    if (typeof neg === 'boolean') return !neg;

    const par = entry['parallel'] ?? entry['np'];
    if (typeof par === 'number') return par === -1;

    return true;
}

export default async function loadConfig(path: string, project_root: string, preset_out_path: string): Promise<[ManagerConfig, string]> {
    const loader = new ConfigLoader();
    return await loader.load(path, project_root, preset_out_path);
}