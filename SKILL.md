---
name: podcast-transcription
description: "Transcribe user-provided local audio, especially long Chinese podcasts and interviews, with resumable ASR, timestamps, and optional speaker labeling; do not use for generic audio editing or remote copyrighted audio that the user has not supplied."
---

# Podcast transcription

Use this skill when the user provides a local audio file or a local path and asks for a transcript, timestamps, speaker separation, or a summary. Treat the supplied file as the source of truth. Keep the machine transcript and the edited reading version separate.

## Workflow

1. Confirm the input path is readable. On WSL, translate a Windows path such as `C:\Users\Name\Downloads\x.m4a` to `/mnt/c/Users/Name/Downloads/x.m4a`; do not claim the file is inaccessible before checking `/mnt/c`.
2. Inspect duration, channels, sample rate, size, and available CPU/GPU. Do not install a Linux NVIDIA driver inside WSL; WSL uses the Windows driver.
3. Prefer the local Qwen3-ASR pipeline in [references/local-pipeline.md](references/local-pipeline.md) for user-provided Chinese audio. Use the 1.7B model for quality when a compatible GPU is available; use the 0.6B model or CPU implementation when resources are limited. Use a cloud service only when the user chooses it or local setup is unavailable.
4. Run a short representative sample before the full file. Measure real-time factor and inspect names, numbers, English terms, rapid overlap, silence, and hallucinations. Do not promise a completion time from benchmark numbers alone.
5. Run long audio in resumable chunks. Preserve the original recognition text, source time range, model/device, and per-chunk timing. A killed process must be safe to resume without duplicating completed chunks.
6. Run forced alignment in a separate process if word/character timestamps are requested; release the ASR model first to protect GPU memory. Alignment locates the generated text and does not correct recognition errors.
7. Add speaker labels only with an actual diarization pipeline. Segment boundaries alone are not speaker turns. Mark uncertain speaker assignment and unclear words for review.
8. Deliver: raw transcript, timestamped transcript, a cleaned reading version only when requested, a concise summary, and a short list of uncertain terms. State what was machine-generated and what was manually checked.

## Model and hardware rules

- For an RTX 2070 Super 8GB, use CUDA PyTorch with FP16, SDPA, batch size 1, and process ASR and alignment in separate processes. Verify `torch.cuda.is_available()` and an FP16 matrix operation before starting.
- Do not load the ASR and forced-aligner models together on an 8GB card.
- Never silently fall back from requested GPU to CPU; report the fallback and ask whether to continue, unless the user explicitly asked for automatic fallback.
- Use Chinese as the forced language for a Chinese podcast only when the sample confirms it; allow automatic detection for mixed-language or uncertain audio.
- Keep external downloads and cloud uploads explicit. Local processing is the default for a local file.

## Output and copyright boundary

When the user supplies an audio file, process it as requested. When only a remote podcast URL is supplied, do not download and return a full verbatim transcript of the copyrighted episode; provide a summary or ask for a user-provided file instead. Never present a generated transcript as human-verified verbatim text.

For the concrete resumable implementation used in this workspace, read [references/local-pipeline.md](references/local-pipeline.md) and use `/home/wq/dev/podcast-transcription` when it exists.
