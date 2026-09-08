# Local pipeline reference

The repository includes the reusable implementation under `pipeline/`. In this workspace it is also available at `/home/wq/dev/podcast-transcription`. It includes an isolated Python 3.12 environment, Qwen model download, resumable ASR, forced alignment, environment checks, and GPU/CPU launchers.

## Commands

```bash
cd /path/to/podcast-transcription/pipeline
./run-cpu-sample.sh
./run-gpu-sample.sh
./run-gpu.sh
```

The default input is the current podcast file. For another file, pass `--audio /absolute/path/file.m4a` to `scripts/transcribe.py`; choose a new `--output` directory for a different input or settings. The GPU launcher first checks CUDA, then writes `outputs/full/transcript.md`, `segments.jsonl`, `metrics.json`, and alignment output.

The pipeline uses Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B from ModelScope. The ASR output is raw model output; the Markdown transcript explicitly says it is not human-audited. `segments.jsonl` is the resume checkpoint. Never run two jobs against the same output directory.

If the environment is not prepared, run `setup.sh`. It installs PyTorch 2.8 CUDA 12.6 wheels and the Qwen package, downloads models, and writes `requirements.lock.txt`. Installation may require network access. After a Windows NVIDIA driver is installed, verify WSL GPU access with `./.venv/bin/python scripts/check_env.py` before running the GPU launcher.

## Review checklist

Check proper names, company names, English terms, numbers, dates, overlapping speech, repeated phrases, and low-confidence or silent sections. Use a separate text model only after the raw transcript is saved, and tell the user when text has been normalized or summarized.
