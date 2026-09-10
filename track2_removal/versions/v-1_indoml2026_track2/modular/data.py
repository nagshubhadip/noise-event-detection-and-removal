"""Data pipeline for the v-1 Track 2 baseline.

Covers, in order:
  * Audio IO   - `read_audio`, `decode_audio` (version-proof HF decoding).
  * VAD/spans  - `spans_from`, `complement`, `speech_ratio`.
  * Validation - `resolve_validation()` maps the three indoml-validation folders onto
    the two metrics and builds `val_items`.
  * Banks      - `build_banks()` carves clean-speech and isolated-noise spans from gold
    clips (streamed).
  * Simulation - `simulate_dataset()` mixes one clean span with 1-2 noise events at a
    sampled event-local SNR.
  * Dataset    - `T2Data` returns (mix, clean) pairs; `stft`/`istft` helpers.
"""

from __future__ import annotations

import io
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    CFG, N_FFT, HOP, N_SAMP, SIM, MAX_SPEECH_RATIO, AUD, REPO,
    CLEAN_DIR, NOISE_DIR, MIX_DIR, REF_DIR, MANIFEST, WORK,
)
from evaluate import si_sdr


device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_torch(seed: int = CFG["seed"]):
    """Seed torch (CPU + CUDA) for reproducible model init and sampling."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Audio IO
# ---------------------------------------------------------------------------
def read_audio(p):
    """Load a file to mono float32 at the config sample rate."""
    w, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != CFG["sr"]:
        w = librosa.resample(w, orig_sr=sr, target_sr=CFG["sr"])
    return np.ascontiguousarray(w, dtype=np.float32)


def decode_audio(a, target_sr=CFG["sr"]):
    """Decode one HF `audio` cell (dict/bytes/path/AudioDecoder) to mono float32.

    We always decode ourselves because the `datasets` Audio decoder has changed
    repeatedly; raw bytes are the only version-proof interface.
    """
    w = sr = None
    if isinstance(a, dict):
        if a.get("array") is not None:
            w, sr = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
        elif a.get("bytes"):
            w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        elif a.get("path"):
            w, sr = sf.read(a["path"], dtype="float32", always_2d=False)
    elif hasattr(a, "get_all_samples"):
        s = a.get_all_samples()
        w, sr = s.data.numpy().astype(np.float32), int(s.sample_rate)
    if w is None:
        raise ValueError(f"cannot decode {type(a)}")
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=0) if w.shape[0] < w.shape[1] else w.mean(axis=1)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(w, dtype=np.float32)


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def spans_from(ex):
    """Sorted (start, end) noise-event spans from a gold clip's timestamp field."""
    out = []
    for s in (ex.get("NoiseSubCategoryTimeStamp") or []):
        st, en = _f(s.get("start")), _f(s.get("end"))
        if st is not None and en is not None and en > st:
            out.append((st, en))
    return sorted(out)


def complement(spans, dur, pad=0.05):
    """Return the gaps *between* event spans (padded), i.e. the clean-speech regions."""
    free, cur = [], 0.0
    for st, en in spans:
        st, en = max(0.0, st - pad), min(dur, en + pad)
        if st > cur:
            free.append((cur, st))
        cur = max(cur, en)
    if cur < dur:
        free.append((cur, dur))
    return free


# Optional WebRTC VAD; falls back to a speech-band energy ratio if unavailable.
try:
    import webrtcvad
    _vad = webrtcvad.Vad(2)
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False


def speech_ratio(w, sr=CFG["sr"]):
    """Fraction of a span that looks like speech - used to reject speechy noise spans.

    Uses WebRTC VAD when installed; otherwise the energy fraction in the 300-3400 Hz
    speech band.
    """
    if HAVE_VAD:
        pcm = np.clip(w * 32767, -32768, 32767).astype(np.int16).tobytes()
        n = int(sr * 0.03) * 2
        fr = [pcm[i:i + n] for i in range(0, len(pcm) - n + 1, n)]
        return float(np.mean([_vad.is_speech(f, sr) for f in fr])) if fr else 1.0
    S = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    f = np.fft.rfftfreq(len(w), 1 / sr)
    return float(S[(f >= 300) & (f <= 3400)].sum() / (S.sum() + 1e-9))


# ---------------------------------------------------------------------------
# Validation-set resolver
# ---------------------------------------------------------------------------
def resolve_validation():
    """Locate the indoml-validation folders and build `val_items`.

    Returns (VAL_DIR, val_items) where each item is a dict with:
      id, path, syn (bool), ref (clean reference path or None), gt (transcript or None).
    Synthetic clips carry both SI-SDR and dWER; natural clips carry dWER only.
    """
    def _find_val_root():
        for c in [Path("/kaggle/input/indoml-validation"),
                  Path("/kaggle/input/datasets/srinjoy3222/indoml-validation")]:
            if c.exists():
                return c
        root = Path("/kaggle/input")
        for d in (sorted(root.iterdir()) if root.exists() else []):
            if any("validation" in q.name.lower() for q in d.rglob("*") if q.is_dir()):
                return d
        return None

    VAL_DIR = _find_val_root()
    assert VAL_DIR is not None, "indoml-validation not found under /kaggle/input"

    def _dir_like(*keys):
        for d in VAL_DIR.rglob("*"):
            if d.is_dir() and all(k.lower() in d.name.lower() for k in keys):
                return d
        return None

    NAT_DIR = _dir_like("natural", "noisy")
    SYN_DIR = _dir_like("synthetic", "noise")
    CLEAN_REF = _dir_like("synthetic", "clean")

    # --- metadata (keyed by clip id) ---
    val_meta = {}
    mp = next((p for p in VAL_DIR.rglob("*.json") if "metadata" in p.name.lower()), None)
    if mp is None:
        mp = next(iter(VAL_DIR.rglob("*.json")), None)
    if mp is not None:
        obj = json.loads(mp.read_text(encoding="utf-8", errors="replace"))

        def _strip(k):
            k = str(k)
            return k[:-4] if k.lower().endswith(".wav") else k

        if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
            val_meta = {_strip(k): v for k, v in obj.items()}
        else:
            rows = obj if isinstance(obj, list) else [obj]
            for key in ("clips", "data", "items", "records", "validation"):
                if isinstance(obj, dict) and isinstance(obj.get(key), list):
                    rows = obj[key]
                    break
            for r in rows:
                if not isinstance(r, dict):
                    continue
                cid = (r.get("clip_id") or r.get("id") or r.get("filename")
                       or r.get("audio") or r.get("file"))
                if cid is not None:
                    val_meta[_strip(cid)] = r

    def _pick(r, *names):
        for n in names:
            v = r.get(n)
            if v not in (None, "", [], {}):
                return v
        return None

    def _ref_for(stem):
        # Match a synthetic clip to its clean reference by stem (with a few suffix fallbacks).
        if CLEAN_REF is None:
            return None
        for q in CLEAN_REF.rglob(f"{stem}.*"):
            if q.suffix.lower() in AUD:
                return q
        base = re.sub(r"(_noisy|_noise|_mix|_mixed)$", "", stem, flags=re.I)
        for q in CLEAN_REF.rglob(f"{base}.*"):
            if q.suffix.lower() in AUD:
                return q
        return None

    val_items = []
    for d, syn in [(NAT_DIR, False), (SYN_DIR, True)]:
        if d is None:
            continue
        for p in sorted(q for q in d.rglob("*") if q.suffix.lower() in AUD):
            r = val_meta.get(p.stem, {})
            val_items.append(dict(
                id=p.stem, path=p, syn=syn,
                ref=(_ref_for(p.stem) if syn else None),
                gt=_pick(r, "transcript", "text", "ground_truth", "reference_text", "sentence"),
            ))
    print(f"VAL_DIR={VAL_DIR} | {len(val_items)} clips")
    return VAL_DIR, val_items


# ---------------------------------------------------------------------------
# Clean/noise bank building (streamed)
# ---------------------------------------------------------------------------
def build_banks(max_gold, hf_token=None, verbose=True):
    """Stream gold clips and carve clean-speech + isolated-noise spans to disk.

    Clean spans come from the complement of the event timestamps (>= min_clean_sec);
    noise spans come from the event timestamps themselves (filtered by duration and by
    `speech_ratio` so speechy events are rejected). Returns (clean_idx, noise_idx).
    """
    from datasets import load_dataset, Audio
    raw = load_dataset(REPO, split="train", streaming=True, token=hf_token)
    raw = raw.cast_column("audio", Audio(decode=False))

    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k):
            return x

    clean_idx, noise_idx, n_gold = [], [], 0
    t0 = time.time()
    for i, ex in enumerate(tqdm(raw, desc="banks")):
        if ex["annotationQuality"] != "verified_timestamps":
            continue
        if n_gold >= max_gold:
            break
        try:
            wav = decode_audio(ex["audio"])
        except Exception:
            continue
        if wav is None or len(wav) < CFG["sr"] * 0.5:
            continue
        n_gold += 1
        dur = len(wav) / CFG["sr"]
        sp = spans_from(ex)
        # clean spans = gaps between events
        for k, (a, b) in enumerate(complement(sp, dur)):
            if b - a < SIM["min_clean_sec"]:
                continue
            seg = wav[int(a * CFG["sr"]):int(b * CFG["sr"])]
            if np.abs(seg).max() < 1e-3:
                continue
            p = CLEAN_DIR / f"c{i:07d}_{k}.wav"
            sf.write(p, seg, CFG["sr"])
            clean_idx.append({"path": str(p), "dur": len(seg) / CFG["sr"]})
        # noise spans = the events themselves (reject speechy ones)
        for k, (a, b) in enumerate(sp):
            d = min(b, dur) - max(0.0, a)
            if not (SIM["min_noise_sec"] <= d <= SIM["max_noise_sec"]):
                continue
            seg = wav[int(a * CFG["sr"]):int(b * CFG["sr"])]
            if np.abs(seg).max() < 1e-3 or speech_ratio(seg) > MAX_SPEECH_RATIO:
                continue
            p = NOISE_DIR / f"n{i:07d}_{k}.wav"
            sf.write(p, seg, CFG["sr"])
            noise_idx.append({"path": str(p), "dur": len(seg) / CFG["sr"]})

    json.dump(clean_idx, open(WORK / "clean_index.json", "w"))
    json.dump(noise_idx, open(WORK / "noise_index.json", "w"))
    if verbose:
        print(f"{n_gold} gold clips in {(time.time()-t0)/60:.1f} min")
        print(f"clean {len(clean_idx)} | noise {len(noise_idx)}")
    assert clean_idx and noise_idx, "empty bank"
    return clean_idx, noise_idx


# ---------------------------------------------------------------------------
# Mixture simulation
# ---------------------------------------------------------------------------
def _make_rng():
    return np.random.default_rng(SIM["seed"])


def _fit(w, n, rng):
    """Crop or tile a noise waveform to exactly n samples."""
    if len(w) >= n:
        s = rng.integers(0, len(w) - n + 1)
        return w[s:s + n]
    return np.tile(w, int(np.ceil(n / max(1, len(w)))))[:n]


def _build_clean(clean_idx, rng, tries=30):
    """Pick a clean span long enough for a training window, else the longest tried."""
    best = None
    for _ in range(tries):
        c = clean_idx[rng.integers(0, len(clean_idx))]
        w, _ = sf.read(c["path"], dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        if best is None or len(w) > len(best):
            best = w
        if len(w) >= SIM["min_clean_sec"] * CFG["sr"]:
            return w[:N_SAMP]
    return best[:N_SAMP] if best is not None else None


def simulate_one(clean_idx, noise_idx, rng):
    """Simulate one (mix, clean, events) triple.

    One clean span from ONE source clip (splicing different clips would change speaker
    and channel). Events are scaled to a sampled event-local SNR, floored against the
    clip's overall power so an event landing on a pause is not scaled to silence.
    """
    clean = _build_clean(clean_idx, rng)
    if clean is None or len(clean) < int(SIM["min_clean_sec"] * CFG["sr"]):
        return None, None, []
    n = len(clean)
    pk = np.abs(clean).max()
    if pk > 0:
        clean = clean / pk * rng.uniform(0.3, 0.85)
    mix = clean.copy()
    ref_pow = float(np.mean(clean ** 2))
    placed, occ = [], []
    for _ in range(int(rng.integers(SIM["events"][0], SIM["events"][1] + 1))):
        nb = noise_idx[rng.integers(0, len(noise_idx))]
        w, _ = sf.read(nb["path"], dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        L = int(min(len(w) / CFG["sr"], SIM["max_noise_sec"]) * CFG["sr"])
        if L < int(SIM["min_noise_sec"] * CFG["sr"]) or L >= n:
            continue
        # find a non-overlapping placement
        s = None
        for _t in range(20):
            cand = int(rng.integers(0, n - L))
            if all(cand + L <= a or cand >= b for a, b in occ):
                s = cand
                break
        if s is None:
            continue
        snr = float(rng.uniform(*SIM["snr_db"]))
        seg = _fit(w, L, rng)
        pc = max(float(np.mean(clean[s:s + L] ** 2)), 0.1 * ref_pow, 1e-10)
        pn = float(np.mean(seg ** 2)) + 1e-10
        seg = seg * math.sqrt(pc / (pn * 10 ** (snr / 10.0)))
        f = min(int(0.02 * CFG["sr"]), L // 2)     # 20 ms fades to avoid clicks
        if f > 0:
            seg[:f] *= np.linspace(0, 1, f)
            seg[-f:] *= np.linspace(1, 0, f)
        mix[s:s + L] += seg
        occ.append((s, s + L))
        placed.append({"onset": round(s / CFG["sr"], 3),
                       "offset": round((s + L) / CFG["sr"], 3),
                       "snr_db": round(snr, 2)})
    m = np.abs(mix).max()
    if m > 0.99:                                   # avoid clipping
        mix, clean = mix / m * 0.99, clean / m * 0.99
    return mix.astype(np.float32), clean.astype(np.float32), placed


def simulate_dataset(clean_idx, noise_idx, n_mix, verbose=True):
    """Generate `n_mix` mixtures, write mix/clean wavs, and return the manifest records.

    Verifies `mix == clean + noise` outside the event spans (self-consistency check).
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k):
            return x

    rng = _make_rng()
    records = []
    for i in tqdm(range(n_mix), desc="simulate"):
        cid = f"sim{i:06d}"
        mix, clean, ev = simulate_one(clean_idx, noise_idx, rng)
        if mix is None or not ev:
            continue
        sf.write(MIX_DIR / f"{cid}.wav", mix, CFG["sr"])
        sf.write(REF_DIR / f"{cid}.wav", clean, CFG["sr"])
        records.append({"id": cid, "mix": str(MIX_DIR / f"{cid}.wav"),
                        "clean": str(REF_DIR / f"{cid}.wav"), "events": ev})
    assert records, "no mixtures produced"
    with open(MANIFEST, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if verbose:
        resid, sis = [], []
        for r in records[:200]:
            mix, _ = sf.read(r["mix"], dtype="float32")
            cl, _ = sf.read(r["clean"], dtype="float32")
            sis.append(si_sdr(cl, mix))
            inside = np.zeros(len(mix), bool)
            for e in r["events"]:
                a = max(0, int(e["onset"] * CFG["sr"]) - 16)
                b = min(len(mix), int(e["offset"] * CFG["sr"]) + 16)
                inside[a:b] = True
            if (~inside).any():
                resid.append(np.abs((mix - cl)[~inside]).max())
        print(f"{len(records)} mixtures | residual outside events {max(resid):.2e} "
              f"| input SI-SDR {np.mean(sis):+.2f} dB")
        assert max(resid) < 1e-2
    return records


# ---------------------------------------------------------------------------
# STFT helpers + torch dataset
# ---------------------------------------------------------------------------
_win = torch.hann_window(N_FFT).to(device)


def stft(x):
    return torch.stft(x, N_FFT, HOP, window=_win, return_complex=True)


def istft(X, n):
    return torch.istft(X, N_FFT, HOP, window=_win, length=n)


class T2Data(Dataset):
    """Returns fixed-length (mix, clean) waveform pairs from the manifest records."""

    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        mix, _ = sf.read(r["mix"], dtype="float32")
        clean, _ = sf.read(r["clean"], dtype="float32")
        mix, clean = mix[:N_SAMP], clean[:N_SAMP]
        if len(mix) < N_SAMP:
            mix = np.pad(mix, (0, N_SAMP - len(mix)))
            clean = np.pad(clean, (0, N_SAMP - len(clean)))
        return dict(mix=torch.from_numpy(mix.astype(np.float32)),
                    clean=torch.from_numpy(clean.astype(np.float32)))


def build_dataloaders(records, bs):
    """95/5 train/val split over the simulated records."""
    split = int(0.95 * len(records))
    train_recs, sim_val_recs = records[:split], records[split:]
    train_dl = DataLoader(T2Data(train_recs), batch_size=bs, shuffle=True,
                          num_workers=2, drop_last=True)
    val_dl = DataLoader(T2Data(sim_val_recs), batch_size=bs, shuffle=False, num_workers=2)
    return train_dl, val_dl, train_recs, sim_val_recs
