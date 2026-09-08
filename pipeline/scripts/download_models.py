"""Download public Qwen weights into this workspace, without API credentials."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "cache/modelscope"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "8")
from modelscope import snapshot_download

for name in ("Qwen3-ASR-1.7B", "Qwen3-ForcedAligner-0.6B"):
    path = ROOT / "models" / name
    print(f"Downloading Qwen/{name} -> {path}", flush=True)
    snapshot_download(f"Qwen/{name}", local_dir=str(path))
    with (path / "config.json").open() as f:
        config = json.load(f)
    print(f"Ready: {name}; model_type={config.get('model_type')}", flush=True)
