"""Data reading, feature extraction and datasets for v-2.

Mirrors the v-1 reader (manual decode, span parsing, label rasterisation) and
adds:
- ``logmel`` — the log-mel spectrogram the CNN branch consumes,
- a labelled/unlabelled split (strong tiers vs. bronze) for Mean-Teacher,
- ``LabeledDataset`` / ``UnlabeledDataset`` returning both waveform and mel.

Frame constants (N_SAMP, FRAMES_*, FRAME_SEC) depend on the finalised clip
length, so they are computed by ``finalize_clip_length`` and passed around.
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

from config import CFG, CAT2IDX, N_OUT, REPO, HF_TOKEN, SEED, QUALITY_WEIGHTS


# ---------------------------------------------------------------------------
# 1. Decode + annotation parsing (identical logic to v-1)
# ---------------------------------------------------------------------------
def decode_audio(a, target_sr=CFG["sr"]):
    """Decode any datasets audio payload into mono float32 at ``target_sr``."""
    w = sr = None
    if isinstance(a, dict):
        if a.get("array") is not None:
            w, sr = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
        elif a.get("bytes") is not None:
            b = a["bytes"]
            try:
                w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
                    tmp.write(b); tmp.flush()
                    w, sr = librosa.load(tmp.name, sr=None, mono=True)
        elif a.get("path"):
            try:
                w, sr = sf.read(a["path"], dtype="float32", always_2d=False)
            except Exception:
                w, sr = librosa.load(a["path"], sr=None, mono=True)
    elif isinstance(a, (str, Path)) and str(a):
        try:
            w, sr = sf.read(str(a), dtype="float32", always_2d=False)
        except Exception:
            w, sr = librosa.load(str(a), sr=None, mono=True)
    if w is None:
        raise ValueError(f"cannot decode audio payload type={type(a)}")
    if w.ndim > 1:
        w = w.mean(axis=0) if w.shape[0] < w.shape[1] else w.mean(axis=1)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(w, dtype=np.float32)


def spans_from(ex):
    """Sorted ``(start, end, category)`` spans; malformed entries dropped."""
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
    """Rasterise spans to ``(N_OUT, T)`` at 100 Hz (ch 0 = any-noise)."""
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


def load_train_subset(max_rows=4000):
    """Stream the dataset into in-memory rows (wav + labels + tier flags)."""
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
                print(f"skip {i}: {e}")
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
# 2. Clip length + frame constants
# ---------------------------------------------------------------------------
PAD_EXTRA_SEC = 2.0
AST_MAX_CLIP_SEC = 10.0


def finalize_clip_length(rows):
    """Set ``CFG['clip_sec']`` from the p95 length and return frame constants.

    Returns ``(N_SAMP, FRAMES_100HZ, FRAMES_OUT, FRAME_SEC)``.
    """
    dur = np.asarray([len(r["wav"]) / CFG["sr"] for r in rows], dtype=np.float32)
    mx = float(dur.max()) if len(dur) else CFG["clip_sec"]
    p95 = float(np.percentile(dur, 95)) if len(dur) else CFG["clip_sec"]
    print(f"audio duration (sec) | p95={p95:.2f} | max={mx:.2f}")
    CFG["clip_sec"] = float(min(math.ceil(min(p95, mx) + PAD_EXTRA_SEC), AST_MAX_CLIP_SEC))

    N_SAMP = int(CFG["clip_sec"] * CFG["sr"])
    FRAMES_100HZ = int(CFG["clip_sec"] * CFG["sr"] / CFG["hop"])
    FRAMES_OUT = FRAMES_100HZ // CFG["time_pool"]
    FRAME_SEC = CFG["hop"] * CFG["time_pool"] / CFG["sr"]
    print(f"clip_sec={CFG['clip_sec']} | N_SAMP={N_SAMP} | "
          f"FRAMES_OUT={FRAMES_OUT} | FRAME_SEC={FRAME_SEC:.4f}s")
    return N_SAMP, FRAMES_100HZ, FRAMES_OUT, FRAME_SEC


# ---------------------------------------------------------------------------
# 3. Feature helpers
# ---------------------------------------------------------------------------
def prep_wav(w):
    """Standardise a waveform (zero mean, unit std)."""
    return (w - w.mean()) / (w.std() + 1e-5)


def logmel(wav):
    """``(n_mels, T)`` standardised log-mel on the 10 ms grid for the CNN branch."""
    m = librosa.feature.melspectrogram(
        y=wav, sr=CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop"],
        n_mels=CFG["n_mels"], power=2.0, center=True)
    m = librosa.power_to_db(m, ref=np.max).astype(np.float32)
    return (m - m.mean()) / (m.std() + 1e-5)


# ---------------------------------------------------------------------------
# 4. Datasets (labelled + unlabelled) — bound to frame constants
# ---------------------------------------------------------------------------
class _Framing:
    """Small holder so the datasets can share the frame constants."""
    def __init__(self, frame_dims):
        self.N_SAMP, self.FRAMES_100HZ, self.FRAMES_OUT, self.FRAME_SEC = frame_dims

    def fit_wav(self, wav, train):
        """Crop (random in train, head at eval) or pad to the fixed window.

        Returns ``(wav, start_sample, valid_samples)``.
        """
        if len(wav) > self.N_SAMP:
            s = random.randint(0, len(wav) - self.N_SAMP) if train else 0
            return wav[s:s + self.N_SAMP], s, self.N_SAMP
        return np.pad(wav, (0, self.N_SAMP - len(wav))), 0, len(wav)

    def labels_20ms(self, lab, s):
        """Crop labels to the window and max-pool 10 ms -> 40 ms output grid."""
        F100, FOUT = self.FRAMES_100HZ, self.FRAMES_OUT
        lab = lab[:, s // CFG["hop"]: s // CFG["hop"] + F100]
        if lab.shape[1] < F100:
            lab = np.pad(lab, ((0, 0), (0, F100 - lab.shape[1])))
        lab = lab[:, :F100]
        return lab.reshape(lab.shape[0], FOUT, CFG["time_pool"]).max(axis=2)

    def valid_mask(self, valid):
        m = np.zeros(self.FRAMES_OUT, dtype=np.float32)
        n_valid = min(self.FRAMES_OUT,
                      max(1, int(math.ceil(valid / CFG["sr"] / self.FRAME_SEC))))
        m[:n_valid] = 1.0
        return m


class LabeledDataset(Dataset):
    """Strong (gold/silver) clips: returns wav + mel + labels + mask + weight."""
    def __init__(self, rows, framing: _Framing, train=True):
        self.rows, self.fr, self.train = rows, framing, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, s, valid = self.fr.fit_wav(r["wav"], self.train)
        lab20 = self.fr.labels_20ms(r["lab"].astype(np.float32), s)
        m = self.fr.valid_mask(valid)
        wav = prep_wav(wav.astype(np.float32))
        return dict(
            wav=torch.from_numpy(wav),
            mel=torch.from_numpy(logmel(wav)),
            lab=torch.from_numpy(lab20.astype(np.float32)),
            mask=torch.from_numpy(m),
            sw=torch.tensor(float(QUALITY_WEIGHTS.get(r.get("quality"), 1.0)),
                            dtype=torch.float32),
        )


class UnlabeledDataset(Dataset):
    """Bronze (no-timestamp) clips: wav + mel + mask only (Mean-Teacher input)."""
    def __init__(self, rows, framing: _Framing):
        self.rows, self.fr = rows, framing

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wav, s, valid = self.fr.fit_wav(self.rows[i]["wav"], True)
        m = self.fr.valid_mask(valid)
        wav = prep_wav(wav.astype(np.float32))
        return dict(
            wav=torch.from_numpy(wav),
            mel=torch.from_numpy(logmel(wav)),
            mask=torch.from_numpy(m),
        )


# ---------------------------------------------------------------------------
# 5. Split, pos_weight, dataloaders
# ---------------------------------------------------------------------------
def split_labeled_unlabeled(rows):
    """Return ``(labeled_rows, unlabeled_rows)`` by annotation tier."""
    labeled = [r for r in rows if r.get("has_strong", False)]
    unlabeled = [r for r in rows if not r.get("has_strong", False)]
    print(f"labeled(strong)={len(labeled)} | unlabeled(bronze)={len(unlabeled)}")
    return labeled, unlabeled


def estimate_pos_weight(train_rows, framing: _Framing):
    """Per-channel BCE pos_weight = neg/pos clipped to [1, 50]."""
    pos = np.zeros(N_OUT)
    total = np.zeros(N_OUT)
    for r in train_rows:
        lab20 = framing.labels_20ms(r["lab"].astype(np.float32), 0)
        n_valid = min(framing.FRAMES_OUT, max(1, int(math.ceil(
            min(len(r["wav"]), framing.N_SAMP) / CFG["sr"] / framing.FRAME_SEC))))
        y = lab20[:, :n_valid]
        pos += y.sum(axis=1)
        total += n_valid
    pw = np.clip(np.maximum(total - pos, 1.0) / np.maximum(pos, 1.0), 1.0, 50.0)
    return pw.astype(np.float32)


def build_dataloaders(labeled_rows, unlabeled_rows, framing: _Framing):
    """Split labelled into train/val and build all three loaders.

    Returns ``(train_dl, val_dl, unlab_dl, train_rows, val_rows)``.
    """
    random.Random(SEED).shuffle(labeled_rows)
    n_val = max(1, int(len(labeled_rows) * CFG["val_frac"]))
    val_rows, train_rows = labeled_rows[:n_val], labeled_rows[n_val:]

    unlab_bs = max(1, int(math.ceil(CFG["batch_size"] * CFG["unlab_batch_frac"])))
    train_dl = DataLoader(LabeledDataset(train_rows, framing, True),
                          batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], drop_last=True)
    val_dl = DataLoader(LabeledDataset(val_rows, framing, False),
                        batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"])
    unlab_dl = (DataLoader(UnlabeledDataset(unlabeled_rows, framing),
                           batch_size=unlab_bs, shuffle=True,
                           num_workers=CFG["num_workers"], drop_last=True)
                if unlabeled_rows else None)
    print(f"train={len(train_rows)} | val={len(val_rows)} | "
          f"unlab={len(unlabeled_rows)} | batch={CFG['batch_size']} accum={CFG['accum_steps']}")
    return train_dl, val_dl, unlab_dl, train_rows, val_rows


def make_framing(frame_dims):
    """Public constructor for the frame-constant holder."""
    return _Framing(frame_dims)
