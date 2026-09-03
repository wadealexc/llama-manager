export interface InputTokensResponse {
    input_tokens: number;
    object: string;
}

export type SpeculativeType = 
    "none" | "draft-simple" | "draft-eagle3" | "draft-mtp" | "draft-dflash" | (string & {});

// TODO: more exhaustive
export type KVPrecision = 
    "f16" | "q8_0" | "q4_0";

export interface ReloadParams {
    // sizing / batch
    n_ctx?: number;
    n_batch?: number;
    n_ubatch?: number;

    // KV cache precision
    cache_type_k?: KVPrecision;
    cache_type_v?: KVPrecision;

    // boolean flags
    offload_kqv?: boolean;
    op_offload?: boolean;
    swa_full?: boolean;
    kv_unified?: boolean;

    // multimodal
    mmproj?: {
        path?: string;  // empty string unloads mmproj
        mmproj_offload?: boolean;
        image_min_tokens?: number;
        image_max_tokens?: number;
        mtmd_batch_max_tokens?: number;
    };

    // speculative decoder
    spec?: {
        types?: SpeculativeType[]; // passing "none" disables
        draft?: {
            path: string;
            n_max?: number;
            n_min?: number;
            p_split?: number;
            p_min?: number;
            n_gpu_layers?: number;
        }
    };
}

// Responses

export type RouterModelStatus = 
    | 'downloading' 
    | 'downloaded' 
    | 'unloaded' 
    | 'loading'
    | 'loaded'
    | 'sleeping'
    | 'unknown'

export interface ModelInfo {
    id: string;
    aliases: string[];
    tags: string[];
    object: string;
    owned_by: string;
    created: number;
    status: {
        value: RouterModelStatus;
        args: string[];
        preset?: unknown;
        exit_code?: number;
        failed?: boolean;
    };
    architecture: {
        input_modalities: string[];
        output_modalities: string[];
    };
    source: string;
    can_remove: boolean;

    // Child supplies this to router once running
    meta?: {
        vocab_type: number;
        n_vocab: number;
        n_ctx: number;
        n_ctx_train: number;
        n_embd: number;
        n_params: number;
        size: number;
        ftype: string;
    }
}

export interface Slot {
    id: number;
    n_ctx: number;
    speculative: boolean;
    is_processing: boolean;
    id_task?: number;
    n_prompt_tokens?: number;
    n_prompt_tokens_processed?: number;
    n_prompt_tokens_cache?: number;
    params?: unknown;
    next_token?: {
        has_next_token: boolean;
        has_new_line: boolean;
        n_remain: number;
        n_decoded: number;
    }
}

export interface SlotSave {
    id_slot: number;
    filename: string;
    n_saved: number;
    n_written: number;
    timings: {
        save_ms: number;
    }
}

export interface SlotRestore {
    id_slot: number;
    filename: string;
    n_restored: number;
    n_read: number;
    timings: {
        restore_ms: number;
    }
}

export interface StatusResponse {
    success: boolean;
    message?: string;
}

export interface ReloadResponse {
    success: boolean;
    n_ctx: number;
    message?: string;
}

export interface HealthResponse {
    status: string;
}

export interface MemoryResponse {
    devices: {
        name: string;
        type: string;
        total: number;
        free: number;
        components: {
            main?: {
                model: number;
                context: number;
                compute: number;
            },
            spec?: {
                model: number;
                context: number;
                compute: number;
            },
            mmproj?: {
                model: number;
                context: number;
                compute: number;
            },
        }
    }[]
}

// Errors

export class HttpError extends Error {
    constructor(
        route: string,
        message: string,
        public statusCode: number,
    ) {
        super(`${route} failed with: ${message}`);
        this.name = 'HttpError';
    }
}