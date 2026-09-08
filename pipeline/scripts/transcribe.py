"""Resumable local ASR; CPU float32 or Turing-compatible CUDA float16.

ASR output is preserved verbatim. Segment timestamps are audio boundaries,
not word alignment. A separate process adds word alignment without keeping
both models on the GPU. No speaker identity is inferred from the text.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "cache/numba"))


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def stamp(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def prepare_audio(source):
    import imageio_ffmpeg
    import soundfile as sf
    stat = source.stat()
    identity = {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    target = ROOT / "cache" / f"audio-{key}.wav"
    if not target.exists():
        temp = target.with_suffix(".tmp.wav")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                        "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(temp)], check=True)
        temp.replace(target)
    audio, sr = sf.read(target, dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    return audio, sr, identity


def boundaries(audio, sr, start, stop, chunk_seconds):
    """Contiguous chunks; move cuts to lowest-energy 100ms near target.

    No audio is omitted, including silence. These are not VAD speaker turns.
    """
    import numpy as np
    pos = start
    while stop - pos > chunk_seconds + 4:
        target = pos + chunk_seconds
        candidates = np.arange(target - 3, target + 3, 0.1)
        energies = [float(np.mean(audio[max(0, int((t-.05)*sr)):int((t+.05)*sr)] ** 2))
                    for t in candidates]
        cut = round(float(candidates[int(np.argmin(energies))]), 3)
        yield pos, cut
        pos = cut
    if stop > pos:
        yield pos, stop


def render(out, rows):
    text = "# 自动转写初稿\n\n未经人工听校；时间为处理片段边界，未区分说话人。原始识别文字未润色。\n\n"
    for row in rows:
        text += f"[{stamp(row['start'])} → {stamp(row['end'])}]\n\n{row['text']}\n\n"
    (out / "transcript.md").write_text(text, encoding="utf-8")
    (out / "transcript.txt").write_text("\n\n".join(r["text"] for r in rows), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default=os.environ.get("PODCAST_AUDIO"), help="Path to the user-provided local audio file; may also be set with PODCAST_AUDIO")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=0, help="0 = remainder of audio")
    p.add_argument("--chunk-seconds", type=float, default=25)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()
    if not args.audio:
        p.error("--audio or PODCAST_AUDIO is required")
    if args.start < 0 or args.duration < 0 or not 10 <= args.chunk_seconds <= 60:
        p.error("start/duration must be nonnegative; chunk-seconds must be 10..60")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(4, args.threads))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install Windows NVIDIA driver and verify WSL GPU access. No automatic CPU fallback.")
    from qwen_asr import Qwen3ASRModel
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    audio, sr, identity = prepare_audio(Path(args.audio).resolve())
    end = min(len(audio)/sr, args.start + args.duration) if args.duration else len(audio)/sr
    if end <= args.start:
        p.error("start must be earlier than end of audio")
    model_path = ROOT / "models/Qwen3-ASR-1.7B"
    chunks = list(boundaries(audio, sr, args.start, end, args.chunk_seconds))
    config = {"audio": identity, "model": str(model_path), "start": args.start,
              "end": end, "chunks": chunks, "language": "Chinese", "max_new_tokens": args.max_new_tokens}
    # Device is intentionally excluded: CPU checkpoint can resume on GPU.
    config = json.loads(json.dumps(config))
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit("Output directory belongs to different audio/settings. Choose a new output directory.")
    atomic_json(config_path, config)
    rows = []
    jsonl = out / "segments.jsonl"
    if jsonl.exists():
        # A killed process may leave an incomplete final line; only recover that line.
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise
        for i, row in enumerate(rows):
            if row["index"] != i or (row["start"], row["end"]) != tuple(chunks[i]):
                raise SystemExit("Invalid resume checkpoint")
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    if len(rows) == len(chunks):
        render(out, rows)
        print("Already complete", flush=True)
        return
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"Loading {model_path.name}: {args.device}, {dtype}, {args.threads} CPU threads", flush=True)
    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(str(model_path), dtype=dtype,
        device_map="cuda:0" if args.device == "cuda" else "cpu",
        attn_implementation="sdpa", max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens)
    load_seconds = time.perf_counter() - t0
    print(f"Model loaded in {load_seconds:.1f}s; {len(chunks)} chunks, resuming at {len(rows)}", flush=True)
    with jsonl.open("a", encoding="utf-8") as f:
        for i in range(len(rows), len(chunks)):
            a, b = chunks[i]
            tic = time.perf_counter()
            with torch.inference_mode():
                result = model.transcribe(audio=(audio[round(a*sr):round(b*sr)], sr), language="Chinese")[0]
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tic
            row = {"index": i, "start": a, "end": b, "text": result.text,
                   "language": result.language, "device": args.device,
                   "seconds": elapsed, "rtf": elapsed/(b-a)}
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
            f.flush()
            os.fsync(f.fileno())
            rows.append(row)
            render(out, rows)
            print(f"[{i+1}/{len(chunks)}] {a:.1f}-{b:.1f}s; took {elapsed:.1f}s; RTF={row['rtf']:.3f}\n{row['text']}", flush=True)
    total = sum(r["seconds"] for r in rows)
    summary = {"audio_seconds": end-args.start, "asr_seconds": total,
               "rtf": total/(end-args.start), "realtime_speed": (end-args.start)/total,
               "this_run_model_load_seconds": load_seconds, "torch": torch.__version__,
               "threads": args.threads, "segments": len(rows),
               "note": "Machine transcript; not human-audited. Timing excludes alignment and diarization."}
    if args.device == "cuda":
        summary["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated()/1024**3
    atomic_json(out / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
