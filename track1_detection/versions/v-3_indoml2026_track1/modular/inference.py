"""Whole-clip inference and Track 1 submission export for the v-3 WavLM baseline.

`t1_posteriors` runs a whole clip through the model. The conv front-end and GRU take
any length, so no windowing is needed below MAX_ONESHOT_SEC - which also avoids a
train/inference mismatch since normalisation is computed over the whole clip in both
places. Longer clips are processed in overlapping 30 s windows and averaged.

`export_track1` turns posteriors into onset/offset events using the swept
post-processing params (threshold, median, min duration, clip gate).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from config import CFG, FRAME_SEC, MIN_SAMPLES, MAX_ONESHOT_SEC, N_OUT
from data import prep_wav
from evaluate import prob_to_events


@torch.no_grad()
def t1_posteriors(wav, model, device, sr=CFG["sr"]):
    """Return (post (N_OUT, T20), clip_prob_any_noise) for a whole clip.

    Short clips are padded to MIN_SAMPLES. Clips up to MAX_ONESHOT_SEC go through in a
    single pass; longer ones use overlapping 30 s windows (2 s hop) whose posteriors
    are averaged frame-wise, with the clip prob taken as the max across windows.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - len(wav)))
    dur = len(wav) / sr
    n_out = max(1, int(math.ceil(dur / FRAME_SEC)))

    # --- single-pass path ---
    if dur <= MAX_ONESHOT_SEC:
        x = torch.from_numpy(prep_wav(wav)).unsqueeze(0).to(device)
        fr, cl = model(x, None)
        p = fr[0].float().cpu().numpy()
        cp = float(cl[0, 0])
        if p.shape[1] != n_out:                       # align to the clip's frame count
            p = np.stack([np.interp(np.linspace(0, 1, n_out),
                                    np.linspace(0, 1, p.shape[1]), r) for r in p])
        return p, cp

    # --- windowed path for long clips ---
    win, hp = int(30 * sr), int(28 * sr)
    starts = list(range(0, max(1, len(wav) - win + hp), hp)) or [0]
    acc = np.zeros((N_OUT, n_out)); cnt = np.zeros(n_out) + 1e-9; cps = []
    for s0 in starts:
        seg = wav[s0:s0 + win]
        if len(seg) < MIN_SAMPLES:
            continue
        x = torch.from_numpy(prep_wav(seg)).unsqueeze(0).to(device)
        fr, cl = model(x, None)
        p = fr[0].float().cpu().numpy(); cps.append(float(cl[0, 0]))
        off = int(round(s0 / sr / FRAME_SEC))         # frame offset of this window
        for j in range(p.shape[1]):
            k = off + j
            if 0 <= k < n_out:
                acc[:, k] += p[:, j]; cnt[k] += 1
    return acc / cnt, (max(cps) if cps else 0.0)


def export_track1(clip_id, wav, model, device, thr, med, min_dur, gate, sr=CFG["sr"]):
    """Produce the submission record for one clip: {clip_id, events:[{onset,offset}]}.

    The clip gate short-circuits to no events when the model believes the clip is
    clean (see evaluate.score_cached for why that pays off).
    """
    post, clip_p = t1_posteriors(wav, model, device, sr)
    dur = max(len(wav) / sr, MIN_SAMPLES / sr)
    spans = []
    events_iter = [] if clip_p < gate else prob_to_events(post[0], thr, med, min_dur=min_dur)
    for on, off in events_iter:
        on, off = float(max(0.0, on)), float(min(dur, off))
        if off - on < 0.05:                           # drop sub-50 ms slivers
            continue
        spans.append([on, off])
    return {"clip_id": clip_id,
            "events": [{"onset": round(float(a), 3), "offset": round(float(b), 3)}
                       for a, b in spans]}
