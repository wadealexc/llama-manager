import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { load as yamlLoad } from "js-yaml";
import { LoadStatus, type ManagerConfig, type ModelEntry, type ModelRole, type ModelState, type StrategyId } from "./types.js";
import { DEFAULT_FIT_OVERHEAD_MIB, DEFAULT_KV_PRECISION, MIN_ALLOWED_CTX } from "../llama-cpp-constants.js";

const DEFAULT_LOG_DIR = './logs/';
const DEFAULT_SLOT_SAVE_DIR = './slots/';

const CTX_KEY = "ctx-size";

const MANAGER_FIELDS = new Set([
    "name",
    "expected_response_tokens",
    "ladder",
    "ctx-size",
    "c",
    "cache-type-k",
    "ctk",
    "cache-type-v",
    "ctv",
    "slot-save-path",
]);

const ROLES: ModelRole[] = ["main", "task"];

const PRESET_OUT_PATH = "./generated-preset.ini";

type RawModel = Record<string, unknown>;
type RawModels = Partial<Record<ModelRole, RawModel>>;

export class ConfigLoader {

    async load(path: string): Promise<[ManagerConfig, string]> {
        let raw: Record<string, unknown>;
        try {
            raw = yamlLoad(readFileSync(path, "utf8")) as Record<string, unknown>;
        } catch (e) {
            throw new Error(`failed to parse config: ${(e as Error).message}`);
        }

        const raw_models: RawModels = (raw.models as RawModels) ?? {};
        const raw_router = (raw.router ?? {}) as Record<string, unknown>;
        const llama_log_dir = normalizeDir((raw_router['llama_log_dir'] as string) ?? DEFAULT_LOG_DIR);
        const slot_save_path = normalizeDir((raw_router['slot-save-path'] as string) ?? DEFAULT_SLOT_SAVE_DIR);

        const config: ManagerConfig = {
            router: {
                bin: raw_router['bin'] as string,
                llama_log_dir,
                slot_save_path,
                listen: raw_router['listen'] as string,
                poll_interval_ms: raw_router['poll_interval_ms'] as number,
                poll_timeout_ms: raw_router['poll_timeout_ms'] as number,
                shutdown_grace_period_ms: raw_router['shutdown_grace_period_ms'] as number,
            },
            listen: raw.listen as string,
            idle_timeout: raw.idle_timeout as number,
            model_load: raw.model_load as ManagerConfig["model_load"],
            models: this.#buildEntries(raw_models),
        };

        mkdirSync(config.router.llama_log_dir, { recursive: true });
        mkdirSync(config.router.slot_save_path, { recursive: true });
        writeFileSync(PRESET_OUT_PATH, this.#buildIni(raw_models, config.router.slot_save_path), "utf8");

        return [config, PRESET_OUT_PATH];
    }

    #buildEntries(raw: RawModels): Partial<Record<ModelRole, ModelEntry>> {
        const entries: Partial<Record<ModelRole, ModelEntry>> = {};
        for (const role of ROLES) {
            const r = raw[role];
            if (!r) continue;
            entries[role] = this.#buildEntry(role, r);
        }
        return entries;
    }

    #buildEntry(role: ModelRole, raw: RawModel): ModelEntry {
        const name = typeof raw['name'] === 'string' ? raw['name'] : role;

        const initial_state: ModelState = {
            n_ctx: MIN_ALLOWED_CTX,
            mmproj_loaded: getHasMmproj(raw),
            spec_loaded: getHasSpec(raw),
            kv_unified: getKvUnified(raw),
            cache_type_k: DEFAULT_KV_PRECISION,
            cache_type_v: DEFAULT_KV_PRECISION,
        };

        return {
            role,
            name,
            expected_response_tokens: raw['expected_response_tokens'] as number,
            fit_target_mib: getFitTarget(raw),
            status: LoadStatus.UNLOADED,
            ladder: (raw['ladder'] as StrategyId[]) ?? [],
            applied: [],
            initial_state,
            current_state: { ...initial_state },
        };
    }

    #buildIni(raw: RawModels, slot_save_path: string): string {
        const sections: string[] = [];

        for (const role of ROLES) {
            const entry = raw[role];
            if (!entry) continue;

            const name = typeof entry['name'] === 'string' ? entry['name'] : role;
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

function getFitTarget(entry: RawModel): number {
    const fit = entry['fit-target'] ?? entry['fitt'];
    return typeof fit === 'number' ? fit : DEFAULT_FIT_OVERHEAD_MIB;
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

export default async function loadConfig(path: string): Promise<[ManagerConfig, string]> {
    const loader = new ConfigLoader();
    return await loader.load(path);
}
