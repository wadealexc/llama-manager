#!/usr/bin/env bash
#
# build-server.sh — build llama-server from the llama.cpp submodule.
#
# Usage:
#   ./scripts/build-server.sh                 # CPU build → llama.cpp/build
#   ./scripts/build-server.sh build-cuda      # named build dir
#   ./scripts/build-server.sh build-cuda --gpu
#   ./scripts/build-server.sh --gpu           # → llama.cpp/build (CUDA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"

BUILD_NAME="build"
USE_GPU=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) USE_GPU=true; shift ;;
        *) BUILD_NAME="$1"; shift ;;
    esac
done

BUILD_DIR="$LLAMA_DIR/$BUILD_NAME"
BIN="$BUILD_DIR/bin/llama-server"

CMAKE_ARGS=(-S "$LLAMA_DIR" -B "$BUILD_DIR" -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF)
if $USE_GPU; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release)
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j10 --target llama-server

echo "==> build complete: $BIN"
