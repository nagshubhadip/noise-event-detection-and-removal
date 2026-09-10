"""Inference: turn a waveform into event ``(onset, offset)`` predictions for v-1.

The chain is deliberately split so a parameter sweep can re-decode cheaply:

    compute_posterior_for_waveform()   # run the model once -> frame posterior
        -> decode_posterior_to_events()   # threshold + clean up -> events
             prob_to_events()             # binarise + merge
             split_on_valleys()           # undo over-merge at posterior dips
             _trim_events_to_duration()   # drop fabricated events in padding

Long clips are processed with overlapping windows and the posteriors averaged.
"""

import math

import numpy as np
import torch
from scipy.ndimage import median_filter

from config import CFG, device
from data import prep_wav


# The frame constants and model are injected at runtime (they depend on the
# finalised clip length and the trained model), so callers set these once.
FRAME_SEC = None
FRAMES_OUT = None
N_SAMP = None
_model = None


def bind(model, frame_dims):
    """Wire the trained model + frame constants into this module."""
    global _model, N_SAMP, FRAMES_OUT, FRAME_SEC
    _model = model
    N_SAMP, _, FRAMES_OUT, FRAME_SEC = frame_dims


# ---------------------------------------------------------------------------
# Posterior -> events
# ---------------------------------------------------------------------------
def prob_to_events(p, thr=0.5, med=7, frame_sec=None, min_dur=0.05, merge_gap=0.10):
    """Binarise a 1-D posterior and return merged ``[start, end]`` events (sec)."""
    frame_sec = frame_sec or FRAME_SEC
    b = (np.asarray(p) >= thr).astype(np.uint8)
    if med > 1:
        b = median_filter(b, size=med, mode="nearest")  # kill 1-frame flicker

    events, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            events.append([start * frame_sec, i * frame_sec])
            start = None
    if start is not None:
        events.append([start * frame_sec, len(b) * frame_sec])

    # Merge events separated by a gap <= merge_gap.
    out = []
    for e in events:
        if out and e[0] - out[-1][1] <= merge_gap:
            out[-1][1] = e[1]
        else:
            out.append(e)
    return [e for e in out if e[1] - e[0] >= min_dur]


def split_on_valleys(p_any, events, frame_sec=None, valley_drop=0.3, min_dur=0.10):
    """Split a region into several events where the posterior dips (a valley).

    This undoes over-merge — the second biggest Event-F1 error source.
    """
    frame_sec = frame_sec or FRAME_SEC
    p_any = np.asarray(p_any)
    out = []
    for a, b in events:
        i0, i1 = int(round(a / frame_sec)), int(round(b / frame_sec))
        seg = p_any[i0:i1]
        if len(seg) < 3:
            out.append((a, b)); continue
        peak = float(seg.max())
        cut = [k for k in range(1, len(seg) - 1)
               if seg[k] < peak - valley_drop and seg[k] <= seg[k - 1] and seg[k] <= seg[k + 1]]
        if not cut:
            out.append((a, b)); continue
        prev = 0
        for c in cut + [len(seg)]:
            s, e = (i0 + prev) * frame_sec, (i0 + c) * frame_sec
            if e - s >= min_dur:
                out.append((s, e))
            prev = c
    return out


def _trim_events_to_duration(events, true_sec):
    """Clamp/drop events outside the clip's real duration (padding is garbage)."""
    out = []
    for a, b in events:
        a, b = max(0.0, float(a)), min(float(true_sec), float(b))
        if b - a >= 0.05:
            out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Model -> posterior
# ---------------------------------------------------------------------------
@torch.no_grad()
def _infer_chunk_any_prob(wav_chunk):
    """Run the model on one fixed-size chunk; return the any-noise posterior."""
    x = torch.from_numpy(prep_wav(wav_chunk)).unsqueeze(0).to(device)
    m = torch.ones(1, FRAMES_OUT, device=device)
    frame_logits, _ = _model(x, m)
    return torch.sigmoid(frame_logits[0, 0]).float().cpu().numpy()


@torch.no_grad()
def compute_posterior_for_waveform(wav, overlap=0.5):
    """Return ``(p_any, true_sec)`` — the any-noise posterior on the 20 ms grid.

    Short clips are padded and run once; long clips are tiled with overlapping
    windows and the per-frame posteriors averaged.
    """
    wav = wav.astype(np.float32)
    true_sec = len(wav) / CFG["sr"]

    if len(wav) <= N_SAMP:
        if len(wav) < N_SAMP:
            wav = np.pad(wav, (0, N_SAMP - len(wav)))
        return _infer_chunk_any_prob(wav), true_sec

    frame_samp = CFG["hop"] * CFG["time_pool"]         # samples per output frame
    total_frames = int(math.ceil(len(wav) / frame_samp))
    acc = np.zeros(total_frames, dtype=np.float32)
    cnt = np.zeros(total_frames, dtype=np.float32)

    step = max(frame_samp, int(N_SAMP * (1.0 - overlap)))
    starts = list(range(0, max(1, len(wav) - N_SAMP + 1), step))
    if starts[-1] != len(wav) - N_SAMP:
        starts.append(max(0, len(wav) - N_SAMP))

    for s in starts:
        ch = wav[s:s + N_SAMP]
        if len(ch) < N_SAMP:
            ch = np.pad(ch, (0, N_SAMP - len(ch)))
        p = _infer_chunk_any_prob(ch)
        g0 = int(round(s / frame_samp))
        g1 = min(total_frames, g0 + len(p))
        L = max(0, g1 - g0)
        if L > 0:
            acc[g0:g1] += p[:L]
            cnt[g0:g1] += 1.0
    return acc / np.maximum(cnt, 1e-6), true_sec


def decode_posterior_to_events(p_any, true_sec, thr=0.5, med=7,
                               min_dur=0.05, merge_gap=0.10, split_valleys=True):
    """Full decode: threshold -> merge -> valley-split -> trim -> dict list."""
    events = prob_to_events(p_any, thr=thr, med=med, min_dur=min_dur, merge_gap=merge_gap)
    if split_valleys:
        events = split_on_valleys(p_any, events, min_dur=min_dur)
    events = _trim_events_to_duration(events, true_sec)
    return [{"onset": round(float(a), 3), "offset": round(float(b), 3)} for a, b in events]


@torch.no_grad()
def predict_events_for_waveform(wav, thr=0.5, med=7, overlap=0.5,
                                min_dur=0.05, merge_gap=0.10):
    """Convenience: waveform -> decoded event dicts in one call."""
    p_any, true_sec = compute_posterior_for_waveform(wav, overlap=overlap)
    return decode_posterior_to_events(p_any, true_sec, thr=thr, med=med,
                                      min_dur=min_dur, merge_gap=merge_gap)
