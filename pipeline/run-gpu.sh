#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="$PWD/cache/numba"
.venv/bin/python scripts/check_env.py --require-gpu
.venv/bin/python scripts/transcribe.py --device cuda --output outputs/full "$@"
.venv/bin/python scripts/align.py --device cuda --output outputs/full
