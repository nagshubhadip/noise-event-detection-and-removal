"""End-to-end orchestration for the v-1 Track 2 (noise removal) baseline.

Run order mirrors the original notebook:
  1. Config + budget + seeding + dir creation.
  2. Resolve the validation set (folders -> metrics).
  3. Build clean/noise banks from streamed gold clips.
  4. Simulate mixtures -> train/val loaders.
  5. Build the encoder + mask net; train on simulated mixtures (SI-SDR loss).
  6. Load the mandated ASR; score the validation set and sweep the blend.
  7. Fine-tune on real validation pairs; re-sweep the blend on the held-out slice.
  8. Enhance the test set, transcribe with the mandated ASR, package the submission.

Import errors for torch/datasets/jiwer/etc. locally are expected - designed for the
Kaggle GPU environment.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from config import (
    BUDGET, CFG, CKPT, WORK, AUD, TRANSCRIPT_NAME, ID_KEY, TEXT_KEY, WAV_SUBTYPE,
    BLEND, budget_sizes, ensure_dirs, seed_everything,
)
import data
import model as model_mod
from train import train_on_sim, finetune_on_real
from inference import enhance, check_output, Transcriber
from evaluate import si_sdr, normalize, delta_wer


def _biggest_audio_dir(exclude=()):
    """Largest audio directory under /kaggle/input excluding given dirs (test fallback)."""
    root = Path("/kaggle/input")
    best = None
    for d in (sorted(root.iterdir()) if root.exists() else []):
        if d in exclude:
            continue
        n = sum(1 for p in d.rglob("*") if p.suffix.lower() in AUD)
        if n and (best is None or n > best[1]):
            best = (d, n)
    return best[0] if best else None


def sweep_blend(model, transcriber, subset, gt_map, noisy_map, ids, cache, refcache,
                blends=(0.3, 0.5, 0.7, 0.85, 1.0)):
    """Sweep the blend factor and return the one maximising Combined = SI-SDR + 100*dWER.

    SI-SDR is computed on the synthetic subset (those with a clean ref); dWER over all
    clips with a ground-truth transcript. Reference/noisy transcripts are cached by the
    caller since only the enhanced side changes with the blend.
    """
    def score_val(blend):
        enh, sis = {}, []
        for v in subset:
            y = enhance(model, cache[v["id"]], blend)
            if v["id"] in gt_map:
                enh[v["id"]] = transcriber.transcribe(y)
            if v["id"] in refcache:
                sis.append(si_sdr(refcache[v["id"]], y))
        dw = delta_wer(gt_map, noisy_map, enh, ids) if ids else float("nan")
        return (float(np.mean(sis)) if sis else float("nan")), dw

    rows = []
    for bl in blends:
        s, dw = score_val(bl)
        comb = (0.0 if math.isnan(s) else s) + (0.0 if math.isnan(dw) else 100 * dw)
        rows.append((bl, s, dw, comb))
        print(f"  blend {bl:.2f}: SI-SDR {s:+7.2f}  dWER {100*dw:+6.2f}%  Combined {comb:+7.2f}")
    valid = [r for r in rows if not (math.isnan(r[1]) and math.isnan(r[2]))]
    assert valid, "neither metric could be computed"
    best = max(valid, key=lambda r: r[3])[0]
    print(f"best blend {best}")
    return best


def _prepare_subset(transcriber, subset):
    """Decode clips + compute reference/noisy transcripts once for a scoring subset."""
    gt_map, noisy_map, ids, cache, refcache = {}, {}, [], {}, {}
    for v in subset:
        cache[v["id"]] = data.read_audio(v["path"])
        if v["ref"] is not None:
            refcache[v["id"]] = data.read_audio(v["ref"])
        g = v["gt"] or (transcriber.transcribe(refcache[v["id"]]) if v["id"] in refcache else None)
        if not g:
            continue
        ids.append(v["id"])
        gt_map[v["id"]] = g
        noisy_map[v["id"]] = transcriber.transcribe(cache[v["id"]])
    return gt_map, noisy_map, ids, cache, refcache


def write_submission(model, transcriber, best_blend, test_dir=None, val_dir=None):
    """Enhance the test set, transcribe, and package predictions into the ZIP.

    The ZIP must contain transcripts.jsonl plus every enhanced WAV at the ROOT, one
    line per clip (empty text allowed but heavily penalised), all 16 kHz mono PCM-16.
    """
    if test_dir is None:
        cand = Path("/kaggle/input/indoml-track2-test")
        test_dir = cand if cand.exists() else _biggest_audio_dir(
            exclude=(val_dir,) if val_dir else ())
    assert test_dir is not None, "attach the Track 2 test audio"
    print("TEST_DIR =", test_dir)

    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    stems = [p.stem for p in test_files]
    dupes = [x for x, c in Counter(stems).items() if c > 1]
    assert not dupes, f"duplicate stems: {dupes[:5]}"

    # --- enhance every clip ---
    sub_dir = WORK / "t2_submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for p in test_files:
        dst = sub_dir / f"{p.stem}.wav"
        w = data.read_audio(p)
        if dst.exists():
            y, _ = sf.read(str(dst), dtype="float32")
        else:
            y = enhance(model, w, best_blend)
            sf.write(dst, y, CFG["sr"], subtype=WAV_SUBTYPE)
        stats.append(check_output(w, y))
    n_bad = sum(1 for s in stats if not s["ok"])
    assert n_bad == 0, "outputs failed the audio checks"
    assert len(list(sub_dir.glob("*.wav"))) == len(test_files), "MISSING WAV = -50 dB"

    # --- transcribe every enhanced clip with the mandated ASR ---
    tj = WORK / TRANSCRIPT_NAME
    done = {}
    batch = 8
    for i in range(0, len(test_files), batch):
        chunk = test_files[i:i + batch]
        texts = transcriber.transcribe_paths([str(sub_dir / f"{p.stem}.wav") for p in chunk])
        for p, t in zip(chunk, texts):
            done[p.stem] = (t or "").strip()
    with open(tj, "w", encoding="utf-8") as f:
        for s_ in stems:
            f.write(json.dumps({ID_KEY: s_, TEXT_KEY: done.get(s_, "")},
                               ensure_ascii=False) + "\n")

    # --- package the ZIP (transcripts + WAVs at the root) ---
    zp = WORK / "submission_track2.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tj, TRANSCRIPT_NAME)
        for f_ in sorted(sub_dir.glob("*.wav")):
            z.write(f_, f_.name)
    names = zipfile.ZipFile(zp).namelist()
    assert TRANSCRIPT_NAME in names, "transcripts.jsonl missing"
    assert all("/" not in n for n in names), "files must be at the ZIP root"
    print(f"READY: {zp} ({len(names)} entries)")


def run(budget=BUDGET, hf_token=None, do_finetune=True, do_submission=False):
    """Full Track 2 pipeline."""
    seed_everything()
    ensure_dirs()
    data.seed_torch()
    sizes = budget_sizes(budget)

    # --- validation + banks + simulation ---
    val_dir, val_items = data.resolve_validation()
    clean_idx, noise_idx = data.build_banks(sizes["MAX_GOLD"], hf_token)
    records = data.simulate_dataset(clean_idx, noise_idx, sizes["N_MIX"])
    train_dl, val_dl, _, sim_val_recs = data.build_dataloaders(records, sizes["BS"])

    # --- model + training ---
    model, encoder, _ = model_mod.core_model(hf_token)
    train_on_sim(model, encoder, train_dl, val_dl, sizes["EPOCHS"])
    model.load_state_dict(__import__("torch").load(CKPT, map_location=data.device,
                                                   weights_only=False)["model"])
    model.eval()

    # --- ASR + validation scoring + blend sweep ---
    transcriber = Transcriber(hf_token)
    syn_pool = [v for v in val_items if v["syn"] and v["ref"] is not None]
    nat_pool = [v for v in val_items if not (v["syn"] and v["ref"] is not None)]
    syn_pool.sort(key=lambda v: v["gt"] is None)
    nat_pool.sort(key=lambda v: v["gt"] is None)
    subset = syn_pool[:60] + nat_pool[:60]
    gt_map, noisy_map, ids, cache, refcache = _prepare_subset(transcriber, subset)
    best_blend = sweep_blend(model, transcriber, subset, gt_map, noisy_map, ids, cache, refcache)

    # --- fine-tune on real pairs + re-sweep on held-out slice ---
    if do_finetune:
        hold_items = finetune_on_real(model, encoder, val_items, sizes["BS"])
        if hold_items:
            gt_map, noisy_map, ids, cache, refcache = _prepare_subset(transcriber, hold_items)
            best_blend = sweep_blend(model, transcriber, hold_items,
                                     gt_map, noisy_map, ids, cache, refcache)

    # --- submission ---
    if do_submission:
        write_submission(model, transcriber, best_blend, val_dir=val_dir)

    return best_blend


if __name__ == "__main__":
    run()
