#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export UV_CACHE_DIR="$PWD/cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/tools/python"
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4
uv_bin="$PWD/tools/uv-x86_64-unknown-linux-gnu/uv"
if [[ ! -x "$uv_bin" ]]; then
  echo 'Missing bundled uv executable; see tools/uv.tar.gz.' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  "$uv_bin" venv --python 3.12 .venv
fi
"$uv_bin" pip install --python .venv/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$uv_bin" pip install --python .venv/bin/python -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
.venv/bin/python scripts/download_models.py
"$uv_bin" pip check --python .venv/bin/python
"$uv_bin" pip freeze --python .venv/bin/python > requirements.lock.txt
.venv/bin/python scripts/check_env.py > logs/environment.json
