"""Align a completed transcript in a separate process; preserve raw ASR text."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from transcribe import ROOT, atomic_json, prepare_audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    import torch
    from qwen_asr import Qwen3ForcedAligner
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    out = Path(args.output).resolve()
    cfg = json.loads((out / "config.json").read_text())
    rows = [json.loads(s) for s in (out / "segments.jsonl").read_text().splitlines()]
    if len(rows) != len(cfg["chunks"]):
        raise SystemExit("ASR incomplete; finish transcription before alignment")
    audio, sr, identity = prepare_audio(Path(cfg["audio"]["path"]))
    if identity != cfg["audio"]:
        raise SystemExit("Source audio changed")
    model = Qwen3ForcedAligner.from_pretrained(str(ROOT / "models/Qwen3-ForcedAligner-0.6B"),
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="cuda:0" if args.device == "cuda" else "cpu", attn_implementation="sdpa")
    target = out / "aligned"
    target.mkdir(exist_ok=True)
    combined = []
    for row in rows:
        path = target / f"{row['index']:05d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("text") == row["text"]:
                combined.append(saved)
                continue
        a, b = row["start"], row["end"]
        tic = time.perf_counter()
        with torch.inference_mode():
            result = model.align(audio=(audio[round(a*sr):round(b*sr)], sr),
                                 text=row["text"], language="Chinese")[0] if row["text"].strip() else []
        words = []
        for item in result:
            d = asdict(item)
            d["start_time"] += a
            d["end_time"] += a
            words.append(d)
        saved = {**row, "alignment_seconds": time.perf_counter()-tic, "words": words}
        atomic_json(path, saved)
        combined.append(saved)
        print(f"Aligned {row['index']+1}/{len(rows)}", flush=True)
    atomic_json(out / "aligned.json", combined)


if __name__ == "__main__":
    main()
