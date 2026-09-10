"""Official scorer metrics and cached evaluation for the v-3 WavLM baseline.

These functions are copied verbatim from the Track 1 Evaluation page so that local
scores match the leaderboard exactly - do NOT "improve" them.

Key scoring subtleties that drive tuning:
  * event-match tolerance is `max(0.20 * duration, 0.05)` - a 50 ms floor.
  * matching is global closest-first (not first-fit).
  * Segment Dice is macro per clip, and a clip with no reference AND no prediction
    scores 1.0 - so one false alarm on a clean clip costs that clip's whole point.

Combined = Event F1 + Segment Dice (max 2.0).

`cache_posteriors` runs the model once; `score_cached` then scores any number of
post-processing configs on the cached probabilities without re-running the network.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import median_filter

from config import FRAME_SEC


# ---------------------------------------------------------------------------
# Event-based F1
# ---------------------------------------------------------------------------
def match_events(ref_events, pred_events, tolerance_frac=0.20):
    """Global closest-first matching of predicted to reference events.

    A prediction matches a reference if both its onset and offset are within
    `max(tolerance_frac * ref_duration, 0.05)` seconds. Among all valid candidate
    pairs, the closest are matched first (greedy over the sorted distance list).
    Returns (true_positives, false_positives, false_negatives).
    """
    matched_ref, matched_pred, candidates = set(), set(), []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp


def event_based_f1(ref_data, pred_data):
    """Micro-averaged event F1 across all clips. Predicted events for clip_ids not in
    the reference count as pure false positives. Returns (f1, prec, rec, TP, FP, FN)."""
    TP = FP = FN = 0
    for cid, ref_events in ref_data.items():
        tp, fp, fn = match_events(ref_events, pred_data.get(cid, []))
        TP += tp; FP += fp; FN += fn
    for cid in pred_data:
        if cid not in ref_data:
            FP += len(pred_data[cid])
    p = TP / (TP + FP) if (TP + FP) else 0.0
    r = TP / (TP + FN) if (TP + FN) else 0.0
    return ((2 * p * r / (p + r)) if (p + r) else 0.0), p, r, TP, FP, FN


# ---------------------------------------------------------------------------
# Segment Dice
# ---------------------------------------------------------------------------
def events_to_frames(events, max_time, frame_len=0.01):
    """Rasterise a list of (onset, offset) events to a binary 10 ms occupancy mask."""
    n = int(max_time / frame_len) + 1
    mask = [0] * n
    for on, off in events:
        for i in range(int(on / frame_len), min(int(off / frame_len) + 1, n)):
            mask[i] = 1
    return mask


def segment_dice(ref_data, pred_data):
    """Macro-averaged per-clip Dice over 10 ms frames. A clip with no reference and
    no prediction scores 1.0 (which is why the clip gate is valuable)."""
    scores = []
    for cid, ref_events in ref_data.items():
        pred_events = pred_data.get(cid, [])
        all_ev = ref_events + pred_events
        if not all_ev:
            scores.append(1.0)
            continue
        max_time = max(off for _, off in all_ev) + 0.5
        rm = events_to_frames(ref_events, max_time)
        pm = events_to_frames(pred_events, max_time)
        inter = sum(r & p for r, p in zip(rm, pm))
        total = sum(rm) + sum(pm)
        scores.append(1.0 if total == 0 else 2.0 * inter / total)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Posterior -> events + cached evaluation
# ---------------------------------------------------------------------------
def prob_to_events(p, thr=0.5, med=7, frame_sec=FRAME_SEC, min_dur=0.05, merge_gap=0.10):
    """Turn a 1-D probability track into (onset, offset) events.

    Steps: threshold -> optional median filter (de-flicker) -> extract runs ->
    merge events closer than `merge_gap` -> drop events shorter than `min_dur`.
    """
    b_ = (np.asarray(p) >= thr).astype(np.uint8)
    if med > 1:
        b_ = median_filter(b_, size=med, mode="nearest")
    ev, start = [], None
    for i, v in enumerate(b_):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ev.append([start * frame_sec, i * frame_sec])
            start = None
    if start is not None:
        ev.append([start * frame_sec, len(b_) * frame_sec])
    out = []
    for e in ev:
        if out and e[0] - out[-1][1] <= merge_gap:
            out[-1][1] = e[1]                      # merge across a short gap
        else:
            out.append(e)
    return [e for e in out if e[1] - e[0] >= min_dur]


@torch.no_grad()
def cache_posteriors(model, dl, device):
    """Run the model ONCE over a loader and cache (pred_prob, gt, clip_prob) per clip.

    Posteriors do not change when sweeping post-processing, so caching lets us score
    hundreds of configs cheaply. Also returns the ground-truth `ref` event dict.
    """
    model.eval()
    cache = []
    for batch in dl:
        frame, clip = model(batch["wav"].to(device), batch["mask"].to(device))
        for p, cp, g, msk in zip(frame[:, 0].float().cpu().numpy(),
                                 clip[:, 0].float().cpu().numpy(),
                                 batch["lab"][:, 0].numpy(),
                                 batch["mask"].numpy()):
            nv = int(msk.sum())                    # keep only valid (non-pad) frames
            cache.append((p[:nv].copy(), g[:nv].copy(), float(cp)))
    ref = {f"c{i}": [tuple(e) for e in prob_to_events(g, 0.5, 1, min_dur=0.0, merge_gap=0.0)]
           for i, (_, g, _) in enumerate(cache)}
    return cache, ref


def score_cached(cache, ref, thr=0.5, med=7, min_dur=0.05, merge_gap=0.10, clip_gate=0.0):
    """Score one post-processing config against the cached posteriors.

    `clip_gate`: if the attention-pooled clip probability is below this, emit NO
    events for that clip. Because a clean clip with no prediction scores a full Dice
    point, staying silent when confident buys more than suppressing a real event
    costs - so the gate is swept alongside thr/med/min_dur.
    """
    pred = {}
    for i, (p, _, cp) in enumerate(cache):
        pred[f"c{i}"] = ([] if cp < clip_gate else
                         [tuple(e) for e in prob_to_events(p, thr, med, min_dur=min_dur,
                                                           merge_gap=merge_gap)])
    f1, prec, rec, TP, FP, FN = event_based_f1(ref, pred)
    dice = segment_dice(ref, pred)
    return dict(f1=f1, dice=dice, score=f1 + dice, precision=prec, recall=rec,
                tp=TP, fp=FP, fn=FN)


def evaluate(model, dl, device, **kw):
    """Convenience: cache posteriors then score with one config."""
    cache, ref = cache_posteriors(model, dl, device)
    return score_cached(cache, ref, **kw)


def sweep_postprocessing(cache, ref, verbose=True):
    """Grid-search thr / median / min_dur / clip_gate over cached posteriors.

    Returns (best_params, sorted_results) where best_params is
    (thr, med, min_dur, gate).
    """
    results = []
    for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        for med in [1, 3, 5, 9, 15, 21]:
            for min_dur in [0.05, 0.12, 0.25]:
                for gate in [0.0, 0.3, 0.5, 0.7]:
                    m = score_cached(cache, ref, thr=thr, med=med,
                                     min_dur=min_dur, clip_gate=gate)
                    results.append((m["score"], m["f1"], m["dice"], thr, med, min_dur,
                                    gate, m["precision"], m["recall"]))
    results.sort(reverse=True)
    if verbose:
        print(f"{'Comb':>7}{'F1':>7}{'Dice':>7}{'thr':>6}{'med':>5}"
              f"{'mind':>7}{'gate':>6}{'P':>7}{'R':>7}")
        for r in results[:12]:
            print(f"{r[0]:7.4f}{r[1]:7.3f}{r[2]:7.3f}{r[3]:6.2f}{r[4]:5d}{r[5]:7.2f}"
                  f"{r[6]:6.2f}{r[7]:7.3f}{r[8]:7.3f}")
    best = (results[0][3], results[0][4], results[0][5], results[0][6])
    return best, results
