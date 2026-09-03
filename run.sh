#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.nvm/nvm.sh"

cd "$(dirname "$0")"
exec node dist/index.js