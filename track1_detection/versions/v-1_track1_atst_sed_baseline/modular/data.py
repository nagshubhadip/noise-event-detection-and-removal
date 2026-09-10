"""Data reading, label rasterisation and the PyTorch dataset for v-1.

The flow is:

    load_train_subset()      # HF stream -> in-memory rows (wav + labels)
        -> finalize_clip_length()   # pick clip_sec from the data
        -> Track1Dataset / build_dataloaders()   # windowed tensors
        -> estimate_pos_weight()    # class imbalance for the loss

Audio is always decoded manually (``Audio(decode=False)``) because the
``datasets`` Audio feature changed decoders twice; handing back raw bytes we
decode ourselves is the only version-proof path.
"""

import io
import math
import random
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, Audio

from config import CFG, CAT2IDX, N_OUT, REPO, HF_TOKEN, SEED


# ---------------------------------------------------------------------------
# 1. Low-level audio decode
# ---------------------------------------------------------------------------
def decode_audio(a, target_sr=CFG["sr"]):
    """Decode one ``datasets`` audio payload into a mono float32 waveform.

    Handles every shape the Audio(decode=False) feature can hand back:
    an in-memory ``array``, raw ``bytes``, or a ``path`` — plus bare path
    strings. Falls back from ``soundfile`` to ``librosa`` when a container is
    awkward, downmixes to mono, and resamples to ``target_sr``.
    """
    w = sr = None

    if isinstance(a, dict):
        if a.get("array") is not None:
            w, sr = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
        elif a.get("bytes") is not None:
            b = a["bytes"]
            try:
                w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            except Exception:
                # Some codecs need a real file handle; spill to a temp file.
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
                    tmp.write(b)
                    tmp.flush()
                    w, sr = librosa.load(tmp.name, sr=None, mono=True)
        elif a.get("path"):
            p = a["path"]
            try:
                w, sr = sf.read(p, dtype="float32", always_2d=False)
            except Exception:
                w, sr = librosa.load(p, sr=None, mono=True)
    elif isinstance(a, (str, Path)) and str(a):
        p = str(a)
        try:
            w, sr = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            w, sr = librosa.load(p, sr=None, mono=True)

    if w is None:
        raise ValueError(f"cannot decode audio payload type={type(a)}")

    # Downmix stereo -> mono (guessing the channel axis from the shape).
    if w.ndim > 1:
        w = w.mean(axis=0) if w.shape[0] < w.shape[1] else w.mean(axis=1)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(w, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Annotation parsing + label rasterisation
# ---------------------------------------------------------------------------
def spans_from(ex):
    """Extract sorted ``(start, end, category)`` spans from one example.

    Silently drops malformed / zero-length spans.
    """
    spans = []
    for s in (ex.get("NoiseSubCategoryTimeStamp") or []):
        try:
            st, en = float(s.get("start")), float(s.get("end"))
        except Exception:
            continue
        if en > st:
            spans.append((st, en, s.get("category")))
    return sorted(spans)


def build_labels(spans, n_frames_100hz, has_strong=True):
    """Rasterise event spans to a ``(N_OUT, T)`` uint8 grid at 100 Hz.

    Channel 0 = any-noise; channel ``1 + cat_idx`` = that category. Overlapping
    events are naturally allowed. For weak (bronze) clips ``has_strong=False``
    yields an all-zero grid — those rows are dropped from strong training.
    """
    y = np.zeros((N_OUT, n_frames_100hz), dtype=np.uint8)
    if has_strong:
        for st, en, cat in spans:
            a = max(0, int(round(st * 100)))
            b = min(n_frames_100hz, int(round(en * 100)))
            if b <= a:
                continue
            y[0, a:b] = 1
            ci = CAT2IDX.get(cat)
            if ci is not None:
                y[1 + ci, a:b] = 1
    return y


# ---------------------------------------------------------------------------
# 3. Load a subset of the dataset into memory
# ---------------------------------------------------------------------------
def load_train_subset(max_rows=4000):
    """Stream the HF dataset and return a list of in-memory row dicts.

    Each row = ``{wav, lab, quality, has_strong}``. Rows whose audio fails to
    decode are skipped (and counted). ``max_rows`` caps the pass so a smoke run
    stays quick.
    """
    ds = load_dataset(REPO, split="train", token=HF_TOKEN)
    ds = ds.cast_column("audio", Audio(decode=False))
    print("audio decode mode: datasets.Audio decode=False (manual decode)")

    rows, skipped = [], 0
    for i, ex in enumerate(ds):
        if i >= max_rows:
            break
        q = ex.get("annotationQuality")
        has_strong = q in ("verified_timestamps", "unverified_timestamps")
        try:
            wav = decode_audio(ex["audio"])
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"skipping sample {i} due to audio decode error: {e}")
            continue
        spans = spans_from(ex)
        nf = 1 + len(wav) // CFG["hop"]
        lab = build_labels(spans, nf, has_strong=has_strong)
        rows.append(dict(wav=wav, lab=lab, quality=q, has_strong=has_strong))

    if not rows:
        raise RuntimeError("No training rows decoded. Check HF token/access.")
    print("loaded", len(rows), "rows | skipped", skipped,
          "| mix=", Counter([r["quality"] for r in rows]))
    return rows


# ---------------------------------------------------------------------------
# 4. Choose the clip length from the data, then derive frame constants
# ---------------------------------------------------------------------------
# AST's feature extractor pads/truncates to a FIXED window, so a larger clip_sec
# spreads the same fixed tokens over more time => coarser boundaries. Prefer p95
# (robust to a few long outliers) and keep the window short.
AST_MAX_CLIP_SEC = 10.0
PAD_EXTRA_SEC = 2.0


def finalize_clip_length(rows, auto=True):
    """Optionally set ``CFG['clip_sec']`` from the length distribution, then
    compute and return the derived frame constants used everywhere else.

    Returns ``(N_SAMP, FRAMES_100HZ, FRAMES_OUT, FRAME_SEC)``.
    """
    dur_sec = np.asarray([len(r["wav"]) / CFG["sr"] for r in rows], dtype=np.float32)
    mx = float(dur_sec.max()) if len(dur_sec) else CFG["clip_sec"]
    p95 = float(np.percentile(dur_sec, 95)) if len(dur_sec) else CFG["clip_sec"]
    med = float(np.median(dur_sec)) if len(dur_sec) else CFG["clip_sec"]
    print(f"audio duration (sec) | median={med:.2f} | p95={p95:.2f} | max={mx:.2f}")

    if auto:
        base = float(min(p95, mx))
        proposed = float(math.ceil(base + PAD_EXTRA_SEC))
        CFG["clip_sec"] = float(min(proposed, AST_MAX_CLIP_SEC))
        print(f"clip_sec set to {CFG['clip_sec']}s (capped at {AST_MAX_CLIP_SEC}s)")

    N_SAMP = int(CFG["clip_sec"] * CFG["sr"])
    FRAMES_100HZ = int(CFG["clip_sec"] * CFG["sr"] / CFG["hop"])
    FRAMES_OUT = FRAMES_100HZ // CFG["time_pool"]
    FRAME_SEC = CFG["hop"] * CFG["time_pool"] / CFG["sr"]
    print(f"N_SAMP={N_SAMP} | FRAMES_OUT={FRAMES_OUT} | FRAME_SEC={FRAME_SEC:.4f}s")
    return N_SAMP, FRAMES_100HZ, FRAMES_OUT, FRAME_SEC


# ---------------------------------------------------------------------------
# 5. Per-sample loss weighting by annotation tier
# ---------------------------------------------------------------------------
QUALITY_WEIGHTS = {
    "verified_timestamps": 1.00,    # gold
    "unverified_timestamps": 0.75,  # silver
    "no_timestamps": 0.25,          # bronze (only if strong-only is off)
}


def prep_wav(w):
    """Per-clip standardisation (zero mean, unit std) the encoder expects."""
    return (w - w.mean()) / (w.std() + 1e-5)


def is_clean_clip_row(r):
    """True if the clip has no timestamped noise (any-noise channel is empty)."""
    return float(np.asarray(r["lab"])[0].sum()) == 0.0


class Track1Dataset(Dataset):
    """Yields fixed-length windows + rasterised 20 ms labels + a valid mask.

    A random crop is taken during training (deterministic first-window crop at
    eval). Labels are cropped to match, then max-pooled 10 ms -> 20 ms. The
    ``mask`` marks frames that correspond to real (non-padded) audio so the loss
    ignores padding.
    """

    def __init__(self, rows, frame_dims, train=True):
        # frame_dims = (N_SAMP, FRAMES_100HZ, FRAMES_OUT, FRAME_SEC)
        self.rows = rows
        self.train = train
        self.N_SAMP, self.FRAMES_100HZ, self.FRAMES_OUT, self.FRAME_SEC = frame_dims

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, lab = r["wav"], r["lab"].astype(np.float32)
        N_SAMP, F100, FOUT = self.N_SAMP, self.FRAMES_100HZ, self.FRAMES_OUT

        # --- crop or pad the waveform to the fixed window ---
        if len(wav) > N_SAMP:
            s = random.randint(0, len(wav) - N_SAMP) if self.train else 0
            wav = wav[s:s + N_SAMP]
            lab = lab[:, s // CFG["hop"]: s // CFG["hop"] + F100]
            valid = N_SAMP
        else:
            valid = len(wav)
            wav = np.pad(wav, (0, N_SAMP - len(wav)))

        # --- align label width to the window ---
        if lab.shape[1] < F100:
            lab = np.pad(lab, ((0, 0), (0, F100 - lab.shape[1])))
        lab = lab[:, :F100]

        # 100 Hz -> 50 Hz (20 ms) via max-pool so a positive frame survives.
        lab_20ms = lab.reshape(lab.shape[0], FOUT, CFG["time_pool"]).max(axis=2)

        # valid mask on the 20 ms grid
        m = np.zeros(FOUT, dtype=np.float32)
        n_valid = max(1, int(math.ceil(valid / CFG["sr"] / self.FRAME_SEC)))
        m[:min(FOUT, n_valid)] = 1.0

        sw = float(QUALITY_WEIGHTS.get(r.get("quality"), 1.0))
        return dict(
            wav=torch.from_numpy(prep_wav(wav).astype(np.float32)),
            lab=torch.from_numpy(lab_20ms.astype(np.float32)),
            mask=torch.from_numpy(m),
            sw=torch.tensor(sw, dtype=torch.float32),
        )


def split_rows(rows, use_strong_only=True, val_clean_only=False):
    """Filter + train/val split the rows.

    - ``use_strong_only``: drop bronze (weak) clips from frame training.
    - ``val_clean_only``: force the val split to clean clips (rarely useful —
      an all-zero val target is trivially minimised, so default False).
    Returns ``(train_rows, val_rows)``.
    """
    n_all = len(rows)
    n_strong = sum(int(r.get("has_strong", False)) for r in rows)
    print(f"rows: total={n_all} | strong={n_strong} | bronze={n_all - n_strong}")

    if use_strong_only:
        rows = [r for r in rows if r.get("has_strong", False)]
        print(f"using strong-only rows: {len(rows)}")

    random.Random(SEED).shuffle(rows)
    n_val = max(1, int(len(rows) * CFG["val_frac"]))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    if val_clean_only:
        clean = [r for r in val_rows if is_clean_clip_row(r)]
        if clean:
            val_rows = clean
            print(f"val forced clean-only: {len(val_rows)}")
    return train_rows, val_rows


def estimate_pos_weight(train_rows, frame_dims):
    """Estimate per-channel BCE ``pos_weight`` = neg/pos, clipped to [1, 50].

    Rare noise frames would otherwise be swamped by the negative majority.
    """
    N_SAMP, F100, FOUT, FRAME_SEC = frame_dims
    pos = np.zeros(N_OUT, dtype=np.float64)
    total = np.zeros(N_OUT, dtype=np.float64)
    for r in train_rows:
        lab = r["lab"].astype(np.float32)
        if lab.shape[1] < F100:
            lab = np.pad(lab, ((0, 0), (0, F100 - lab.shape[1])))
        lab = lab[:, :F100]
        lab_20ms = lab.reshape(lab.shape[0], FOUT, CFG["time_pool"]).max(axis=2)
        n_valid = min(FOUT, max(1, int(math.ceil(
            min(len(r["wav"]), N_SAMP) / CFG["sr"] / FRAME_SEC))))
        y = lab_20ms[:, :n_valid]
        pos += y.sum(axis=1)
        total += n_valid
    pw = np.clip(np.maximum(total - pos, 1.0) / np.maximum(pos, 1.0), 1.0, 50.0)
    return pw.astype(np.float32)


def build_dataloaders(train_rows, val_rows, frame_dims):
    """Wrap the split rows in DataLoaders (train shuffled, val not)."""
    train_dl = DataLoader(Track1Dataset(train_rows, frame_dims, train=True),
                          batch_size=CFG["batch_size"], shuffle=True, num_workers=2)
    val_dl = DataLoader(Track1Dataset(val_rows, frame_dims, train=False),
                        batch_size=CFG["batch_size"], shuffle=False, num_workers=2)
    return train_dl, val_dl
