import argparse
import importlib.metadata as md
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--require-gpu", action="store_true")
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
          "cuda_available": torch.cuda.is_available(),
          "compiled_arches": torch._C._cuda_getArchFlags() if hasattr(torch._C, "_cuda_getArchFlags") else None}
for package in ("qwen-asr", "transformers", "av", "imageio-ffmpeg"):
    result[package] = md.version(package)
for name in ("Qwen3-ASR-1.7B", "Qwen3-ForcedAligner-0.6B"):
    path = root / "models" / name
    result[name] = {"config_exists": (path / "config.json").exists(),
                    "weights_bytes": sum(f.stat().st_size for f in path.glob("*.safetensors"))}
if torch.cuda.is_available():
    result["gpu"] = torch.cuda.get_device_name()
    result["capability"] = torch.cuda.get_device_capability()
    result["vram_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    result["fp16_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
print(json.dumps(result, indent=2))
if args.require_gpu and not result["cuda_available"]:
    raise SystemExit("GPU unavailable. Check Windows NVIDIA driver and WSL GPU access.")
