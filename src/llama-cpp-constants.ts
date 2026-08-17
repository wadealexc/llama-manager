
// when --fit-target is not specified, fit ensures this amount
// of space remains free on the device
export const DEFAULT_FIT_OVERHEAD_MIB = 1024;

// kvcaches cannot be created with less than this ctx
// TODO: patch llama.cpp so we can fully unload kvcache
export const MIN_ALLOWED_CTX = 1024;
