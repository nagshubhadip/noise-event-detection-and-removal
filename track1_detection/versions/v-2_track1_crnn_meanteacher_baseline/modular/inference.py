"""Inference + event decoding for v-2 (uses the EMA teacher at test time).

Mirrors the v-1 decoder but the model consumes BOTH waveform and log-mel. The
trained teacher and the frame constants are injected via ``bind``.
"""

import math

import numpy as np
import torch
from scipy.ndimage import median_filter

from config import CFG, device
from data import prep_wav, logmel


FRAME_SEC = None
FRAMES_OUT = None
N_SAMP = None
_model = None


def bind(model, frame_dims):
    """Wire the (teacher) model + frame constants into this module."""
    global _model, N_SAMP, FRAMES_OUT, FRAME_SEC
    _model = model
    N_SAMP, _, FRAMES_OUT, FRAME_SEC = frame_dims


def prob_to_events(p, thr=0.5, med=5, frame_sec=None, min_dur=0.05, merge_gap=0.0):
    """Binarise a posterior, median-filter, and merge into events (seconds)."""
    frame_sec = frame_sec or FRAME_SEC
    b = (np.asarray(p) >= thr).astype(np.uint8)
    if med > 1:
        b = median_filter(b, size=med, mode="nearest")
    events, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            events.append([start * frame_sec, i * frame_sec]); start = None
    if start is not None:
        events.append([start * frame_sec, len(b) * frame_sec])
    out = []
    for e in events:
        if out and e[0] - out[-1][1] <= merge_gap:
            out[-1][1] = e[1]
        else:
            out.append(e)
    return [e for e in out if e[1] - e[0] >= min_dur]


@torch.no_grad()
def _chunk_prob(wav_chunk):
    """Any-noise posterior for one fixed-size chunk (wav + mel inputs)."""
    wav_chunk = prep_wav(wav_chunk.astype(np.float32))
    x = torch.from_numpy(wav_chunk).unsqueeze(0).to(device)
    mel = torch.from_numpy(logmel(wav_chunk)).unsqueeze(0).to(device)
    m = torch.ones(1, FRAMES_OUT, device=device)
    _model.eval()
    logits, _ = _model(x, mel, m)
    return torch.sigmoid(logits[0, 0]).float().cpu().numpy()


def _trim(events, true_sec):
    """Clamp/drop events outside the real clip duration."""
    out = []
    for a, b in events:
        a, b = max(0.0, float(a)), min(float(true_sec), float(b))
        if b - a >= 0.05:
            out.append((a, b))
    return out


@torch.no_grad()
def predict_events_for_waveform(wav, thr=0.5, med=5, overlap=0.5):
    """Waveform -> event dict list. Long clips use overlapping windows."""
    wav = wav.astype(np.float32)
    true_sec = len(wav) / CFG["sr"]

    if len(wav) <= N_SAMP:
        w = np.pad(wav, (0, N_SAMP - len(wav))) if len(wav) < N_SAMP else wav
        p = _chunk_prob(w)
        ev = prob_to_events(p, thr=thr, med=med)
        return [{"onset": round(a, 3), "offset": round(b, 3)}
                for a, b in _trim(ev, true_sec)]

    frame_samp = CFG["hop"] * CFG["time_pool"]
    total = int(math.ceil(len(wav) / frame_samp))
    acc = np.zeros(total, np.float32)
    cnt = np.zeros(total, np.float32)
    step = max(frame_samp, int(N_SAMP * (1.0 - overlap)))
    starts = list(range(0, max(1, len(wav) - N_SAMP + 1), step))
    if starts[-1] != len(wav) - N_SAMP:
        starts.append(max(0, len(wav) - N_SAMP))

    for s in starts:
        ch = wav[s:s + N_SAMP]
        if len(ch) < N_SAMP:
            ch = np.pad(ch, (0, N_SAMP - len(ch)))
        p = _chunk_prob(ch)
        g0 = int(round(s / frame_samp))
        g1 = min(total, g0 + len(p))
        L = max(0, g1 - g0)
        if L > 0:
            acc[g0:g1] += p[:L]
            cnt[g0:g1] += 1.0

    ev = prob_to_events(acc / np.maximum(cnt, 1e-6), thr=thr, med=med)
    return [{"onset": round(a, 3), "offset": round(b, 3)}
            for a, b in _trim(ev, true_sec)]
