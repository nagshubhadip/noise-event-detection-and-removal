"""Evaluation for v-2: competition-style Event-F1 (micro) + temporal Dice (macro).

Two scorers:
- ``test_saved_model_on_rows``   — score the in-memory labelled val rows.
- ``test_saved_model_on_folder`` — score external audio against a reference.jsonl.

Both reuse the same interval-math + event-matching helpers as v-1.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm.auto import tqdm

from config import CFG
import inference


# ---------------------------------------------------------------------------
# Interval math + event matching (identical rules to the competition scorer)
# ---------------------------------------------------------------------------
def merge_intervals(intervals):
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


def total_dur(intervals):
    return float(sum(max(0.0, b - a) for a, b in intervals))


def inter_dur(a_list, b_list):
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
    """Collar match: within max(20% of GT duration, 50 ms) on both edges."""
    dur = max(1e-6, g[1] - g[0])
    tol = max(0.2 * dur, 0.05)
    return abs(g[0] - p[0]) <= tol and abs(g[1] - p[1]) <= tol


def event_counts(gt_events, pr_events):
    """Greedy 1-1 matching -> ``(tp, fp, fn)``."""
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
            matched.add(best_j); tp += 1
    return tp, len(pr_events) - tp, len(gt_events) - tp


def events_from_label(lab_row, frames_per_sec=100.0):
    """Reconstruct GT ``(onset, offset)`` spans from a rasterised any-noise row."""
    b = (np.asarray(lab_row) >= 0.5).astype(np.uint8)
    events, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            events.append((start / frames_per_sec, i / frames_per_sec)); start = None
    if start is not None:
        events.append((start / frames_per_sec, len(b) / frames_per_sec))
    return events


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def read_audio_file(p):
    """Read audio to mono float32 at the config sample rate."""
    try:
        w, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
    except Exception:
        w, sr = librosa.load(str(p), sr=None, mono=True)
    if sr != CFG["sr"]:
        w = librosa.resample(w, orig_sr=sr, target_sr=CFG["sr"])
    return np.asarray(w, dtype=np.float32)


def load_reference_jsonl(p):
    """Ground-truth events keyed by clip_id -> sorted ``[(on, off)]``."""
    gt = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            clean = []
            for e in (rec.get("events", []) or []):
                try:
                    a, b = float(e["onset"]), float(e["offset"])
                except Exception:
                    continue
                if b > a:
                    clean.append((a, b))
            gt[str(rec.get("clip_id", ""))] = sorted(clean)
    return gt


def _summary(sum_tp, sum_fp, sum_fn, dice_scores, tag):
    p = sum_tp / (sum_tp + sum_fp + 1e-9)
    r = sum_tp / (sum_tp + sum_fn + 1e-9)
    event_f1 = 2 * p * r / (p + r + 1e-9)
    dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    print(f"\n[{tag}] Event F1={event_f1:.4f} | Dice={dice:.4f} | "
          f"Combined={event_f1 + dice:.4f} (TP/FP/FN={sum_tp}/{sum_fp}/{sum_fn})")
    return dict(event_f1=event_f1, dice=dice, combined=event_f1 + dice,
                tp=sum_tp, fp=sum_fp, fn=sum_fn)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
def test_saved_model_on_rows(rows, thr=0.5, med=5, tag="val"):
    """Event-F1 + Dice on in-memory rows that carry strong labels."""
    rows = [r for r in rows if r.get("has_strong", False)]
    print(f"[{tag}] scoring {len(rows)} labelled clips (thr={thr}, med={med})")
    sum_tp = sum_fp = sum_fn = 0
    dice_scores = []
    for r in tqdm(rows, desc=f"{tag}-test", leave=False):
        gt = events_from_label(r["lab"][0])
        pred = inference.predict_events_for_waveform(
            r["wav"].astype(np.float32) if not isinstance(r["wav"], np.ndarray)
            else r["wav"], thr=thr, med=med)
        pr = [(x["onset"], x["offset"]) for x in pred]
        tp, fp, fn = event_counts(gt, pr)
        sum_tp += tp; sum_fp += fp; sum_fn += fn
        inter = inter_dur(gt, pr)
        dice_scores.append((2 * inter) /
                           (total_dur(merge_intervals(gt)) + total_dur(merge_intervals(pr)) + 1e-9))
    return _summary(sum_tp, sum_fp, sum_fn, dice_scores, tag)


def test_saved_model_on_folder(audio_dir, ref_jsonl, thr=0.5, med=5,
                               exts=(".wav", ".flac", ".mp3", ".ogg"), tag="local-test"):
    """Event-F1 + Dice on external audio vs. a reference.jsonl."""
    gt_map = load_reference_jsonl(ref_jsonl)
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in exts)
    print(f"[{tag}] {len(files)} files | GT entries {len(gt_map)} (thr={thr}, med={med})")
    sum_tp = sum_fp = sum_fn = 0
    dice_scores, missing = [], 0
    for p in tqdm(files, desc=tag, leave=False):
        if p.stem not in gt_map:
            missing += 1
            continue
        gt = gt_map[p.stem]
        pred = inference.predict_events_for_waveform(read_audio_file(p), thr=thr, med=med)
        pr = [(x["onset"], x["offset"]) for x in pred]
        tp, fp, fn = event_counts(gt, pr)
        sum_tp += tp; sum_fp += fp; sum_fn += fn
        inter = inter_dur(gt, pr)
        dice_scores.append((2 * inter) /
                           (total_dur(merge_intervals(gt)) + total_dur(merge_intervals(pr)) + 1e-9))
    print(f"[{tag}] missing GT for {missing} files")
    return _summary(sum_tp, sum_fp, sum_fn, dice_scores, tag)
