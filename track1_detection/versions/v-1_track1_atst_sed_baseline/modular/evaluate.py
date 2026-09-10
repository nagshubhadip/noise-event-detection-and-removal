"""Evaluation + decoding-parameter sweep for v-1.

Two things live here:

1. ``evaluate_local_folder`` — score predictions against a ``reference.jsonl``
   using the competition-style Event-F1 (micro) + temporal Dice (macro).
2. ``sweep_decoding_params`` — cache each clip's posterior ONCE, then re-decode
   across a grid of post-processing knobs. High Dice + low Event-F1 means the
   acoustics are fine but the segmentation is off, which these knobs fix without
   retraining.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm.auto import tqdm

from config import CFG
from inference import (compute_posterior_for_waveform,
                       decode_posterior_to_events)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def read_audio_file_local(p):
    """Read an audio file to mono float32 at the config sample rate."""
    p = str(p)
    try:
        w, sr = sf.read(p, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
    except Exception:
        w, sr = librosa.load(p, sr=None, mono=True)
    if sr != CFG["sr"]:
        w = librosa.resample(w, orig_sr=sr, target_sr=CFG["sr"])
    return np.asarray(w, dtype=np.float32)


def load_reference_jsonl(p):
    """Load ground-truth events keyed by ``clip_id`` -> sorted ``[(on, off)]``."""
    gt = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get("clip_id", ""))
            clean = []
            for e in (rec.get("events", []) or []):
                try:
                    a, b = float(e["onset"]), float(e["offset"])
                except Exception:
                    continue
                if b > a:
                    clean.append((a, b))
            gt[cid] = sorted(clean)
    return gt


# ---------------------------------------------------------------------------
# Interval math (for Dice) and event matching (for Event-F1)
# ---------------------------------------------------------------------------
def merge_intervals(intervals):
    """Merge overlapping intervals into a disjoint, sorted list."""
    if not intervals:
        return []
    itv = sorted(intervals)
    out = [list(itv[0])]
    for a, b in itv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_duration(intervals):
    return float(sum(max(0.0, b - a) for a, b in intervals))


def intersection_duration(a_list, b_list):
    """Total overlap duration between two interval lists (both merged first)."""
    a_list, b_list = merge_intervals(a_list), merge_intervals(b_list)
    i = j = 0
    inter = 0.0
    while i < len(a_list) and j < len(b_list):
        a1, a2 = a_list[i]
        b1, b2 = b_list[j]
        lo, hi = max(a1, b1), min(a2, b2)
        if hi > lo:
            inter += hi - lo
        if a2 < b2:
            i += 1
        else:
            j += 1
    return inter


def event_match(g, p):
    """Collar match: onset & offset within max(20% of GT duration, 50 ms)."""
    dur = max(1e-6, g[1] - g[0])
    tol = max(0.2 * dur, 0.05)
    return abs(g[0] - p[0]) <= tol and abs(g[1] - p[1]) <= tol


def event_f1(gt_events, pr_events):
    """Greedy 1-1 event matching -> ``(f1, tp, fp, fn)``."""
    matched, tp = set(), 0
    for g in gt_events:
        best_j, best_err = -1, 1e18
        for j, p in enumerate(pr_events):
            if j in matched or not event_match(g, p):
                continue
            err = abs(g[0] - p[0]) + abs(g[1] - p[1])
            if err < best_err:
                best_err, best_j = err, j
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
    fp, fn = len(pr_events) - tp, len(gt_events) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9), tp, fp, fn


def _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores):
    """Combine pooled TP/FP/FN + per-clip Dice into the summary triple."""
    p = sum_tp / (sum_tp + sum_fp + 1e-9)
    r = sum_tp / (sum_tp + sum_fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    return f1, dice, f1 + dice


# ---------------------------------------------------------------------------
# Folder evaluation
# ---------------------------------------------------------------------------
def evaluate_local_folder(audio_dir, ref_jsonl, predict_fn, thr=0.5, med=7,
                          exts=(".wav", ".flac", ".mp3", ".ogg"), progress_every=100):
    """Score every audio file that has a GT entry; print + return the metrics.

    ``predict_fn`` maps a waveform -> event dicts (usually
    ``inference.predict_events_for_waveform``).
    """
    gt_map = load_reference_jsonl(ref_jsonl)
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in exts)
    print(f"Found {len(files)} files | GT entries: {len(gt_map)}")

    sum_tp = sum_fp = sum_fn = 0
    dice_scores, per_clip, missing = [], [], 0
    for p in tqdm(files, desc="local-val"):
        cid = p.stem
        if cid not in gt_map:
            missing += 1
            continue
        gt_events = gt_map[cid]
        pred = predict_fn(read_audio_file_local(p), thr=thr, med=med)
        pr_events = [(float(x["onset"]), float(x["offset"])) for x in pred]

        _, tp, fp, fn = event_f1(gt_events, pr_events)
        sum_tp += tp; sum_fp += fp; sum_fn += fn
        inter = intersection_duration(gt_events, pr_events)
        gt_d = total_duration(merge_intervals(gt_events))
        pr_d = total_duration(merge_intervals(pr_events))
        dice_scores.append((2.0 * inter) / (gt_d + pr_d + 1e-9))
        per_clip.append(cid)

        if progress_every and len(per_clip) % progress_every == 0:
            f1_p, dice_p, comb_p = _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores)
            print(f"[prefix {len(per_clip)}] F1={f1_p:.4f} Dice={dice_p:.4f} Comb={comb_p:.4f}")

    f1, dice, comb = _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores)
    print(f"\nLocal val | files={len(per_clip)} missing={missing}")
    print(f"  Event F1 (micro) : {f1:.4f}")
    print(f"  Dice (macro)     : {dice:.4f}")
    print(f"  Combined         : {comb:.4f}")
    return dict(event_f1_micro=f1, dice_macro=dice, combined=comb,
                files_evaluated=len(per_clip), missing_gt=missing)


# ---------------------------------------------------------------------------
# Decoding-parameter sweep
# ---------------------------------------------------------------------------
SWEEP_THR = [0.35, 0.45, 0.5, 0.55, 0.65]
SWEEP_MED = [1, 3, 5, 7]
SWEEP_MERGE_GAP = [0.0, 0.03, 0.05, 0.10]
SWEEP_MIN_DUR = [0.05, 0.10]


def sweep_decoding_params(audio_dir, ref_jsonl, exts=(".wav", ".flac", ".mp3", ".ogg"),
                          max_clips=800, optimize="event_f1"):
    """Cache posteriors once, then grid-search the decode knobs.

    Returns ``(best_config, all_results_sorted)``.
    """
    gt_map = load_reference_jsonl(ref_jsonl)
    files = [p for p in sorted(Path(audio_dir).rglob("*"))
             if p.suffix.lower() in exts and p.stem in gt_map]
    if max_clips is not None:
        files = files[:max_clips]
    print(f"Sweep: caching {len(files)} posteriors ...")

    cache = []
    for p in tqdm(files, desc="sweep-cache"):
        p_any, true_sec = compute_posterior_for_waveform(read_audio_file_local(p))
        cache.append((gt_map[p.stem], p_any, true_sec))

    grid = [(t, m, g, d) for t in SWEEP_THR for m in SWEEP_MED
            for g in SWEEP_MERGE_GAP for d in SWEEP_MIN_DUR]
    print(f"Sweep: evaluating {len(grid)} configs ...")

    best, results = None, []
    for thr, med, merge_gap, min_dur in tqdm(grid, desc="sweep-grid"):
        sum_tp = sum_fp = sum_fn = 0
        dice_scores = []
        for gt_events, p_any, true_sec in cache:
            pred = decode_posterior_to_events(p_any, true_sec, thr=thr, med=med,
                                              min_dur=min_dur, merge_gap=merge_gap)
            pr_events = [(float(x["onset"]), float(x["offset"])) for x in pred]
            _, tp, fp, fn = event_f1(gt_events, pr_events)
            sum_tp += tp; sum_fp += fp; sum_fn += fn
            inter = intersection_duration(gt_events, pr_events)
            gt_d = total_duration(merge_intervals(gt_events))
            pr_d = total_duration(merge_intervals(pr_events))
            dice_scores.append((2.0 * inter) / (gt_d + pr_d + 1e-9))
        f1_m, dice_m, comb = _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores)
        score = comb if optimize == "combined" else f1_m
        row = dict(thr=thr, med=med, merge_gap=merge_gap, min_dur=min_dur,
                   event_f1=f1_m, dice=dice_m, combined=comb, score=score)
        results.append(row)
        if best is None or score > best["score"]:
            best = row

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\nBest: {best}")
    return best, results
