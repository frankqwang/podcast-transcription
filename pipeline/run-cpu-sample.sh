#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="$PWD/cache/numba"
exec .venv/bin/python scripts/transcribe.py --device cpu --start 188 --duration 60 --output outputs/cpu-sample "$@"
