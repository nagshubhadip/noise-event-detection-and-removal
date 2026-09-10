"""End-to-end orchestration for the v-3 WavLM SED baseline.

Run order mirrors the original notebook:
  1. Config + budget resolution + seeding.
  2. Stream the dataset and decode/rasterise once into `rows`.
  3. Bucket by tier, build train/val loaders.
  4. Two-stage mean-teacher training (checkpoint saved to CKPT).
  5. Post-processing sweep on cached val posteriors -> best (thr, med, min_dur, gate).
  6. Optional: write the Track 1 submission ZIP from a test-audio folder.

Import errors for torch/datasets/etc. on a local machine are expected - this is
designed to run in the Kaggle GPU environment where those packages exist.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch

from config import (
    BUDGET, CFG, CKPT, WORK, budget_settings, seed_everything,
)
import data
from model import WavLMSED
from train import train
from evaluate import cache_posteriors, sweep_postprocessing
from inference import export_track1


AUD = (".wav", ".flac", ".mp3", ".ogg")


def read_audio(p, sr=CFG["sr"]):
    """Load an audio file to mono float32 at the model sample rate."""
    import soundfile as sf
    import librosa
    w, s = sf.read(str(p), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if s != sr:
        w = librosa.resample(w, orig_sr=s, target_sr=sr)
    return w


def write_submission(model, device, thr, med, min_dur, gate,
                     test_dir=Path("/kaggle/input/indoml-track1-test")):
    """Write predictions.jsonl (+ zip) for every clip under `test_dir`.

    Official format: ZIP with predictions.jsonl at the ROOT; every eval clip appears
    exactly once; clips with nothing detected get an empty events list, not a missing
    line.
    """
    from tqdm.auto import tqdm
    jsonl, zip_path = WORK / "predictions.jsonl", WORK / "submission_track1.zip"
    if not test_dir.exists():
        print(f"{test_dir} not found - attach the Track 1 test set and re-run.")
        return
    files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    print(len(files), "test clips")
    with open(jsonl, "w", encoding="utf-8") as f:
        for p in tqdm(files):
            rec = export_track1(p.stem, read_audio(p), model, device,
                                thr, med, min_dur, gate)
            f.write(json.dumps({"clip_id": p.stem, "events": rec["events"]},
                               ensure_ascii=False) + "\n")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, "predictions.jsonl")           # at ZIP root, no enclosing folder
    n = sum(1 for _ in open(jsonl))
    empty = sum(1 for l in open(jsonl) if not json.loads(l)["events"])
    print(f"{n} lines ({empty} with no events) -> {zip_path}")


def run(budget=BUDGET, hf_token=None, do_submission=False):
    """Full pipeline. Returns (best_state, best_postproc_params)."""
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data.seed_torch()

    max_per_quality, stage1_epochs, stage2_epochs, t1_bs, train_limit = \
        budget_settings(budget)

    # --- data ---
    raw = data.load_stream(hf_token)
    rows = data.build_rows(raw, max_per_quality)
    gold, silver, bronze = data.bucket_by_tier(rows)
    train_dl, val_dl = data.split_and_load(gold, silver, bronze, t1_bs, train_limit)

    # --- train ---
    state = train(train_dl, val_dl, device, stage1_epochs, stage2_epochs)

    # --- post-processing sweep on the best checkpoint ---
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = WavLMSED().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    cache, ref = cache_posteriors(model, val_dl, device)
    (thr, med, min_dur, gate), _ = sweep_postprocessing(cache, ref)
    ck.update(thr=float(thr), med=int(med), min_dur=float(min_dur), gate=float(gate))
    torch.save(ck, CKPT)
    print(f"best post-proc: thr={thr} med={med} min_dur={min_dur} gate={gate}")

    # --- optional submission ---
    if do_submission:
        write_submission(model, device, thr, med, min_dur, gate)

    return state, (thr, med, min_dur, gate)


if __name__ == "__main__":
    run()
