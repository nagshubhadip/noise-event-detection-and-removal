"""Data loading, label rasterisation and the SED dataset for the v-3 WavLM baseline.

The flow is:
  1. `load_stream()` opens the HF dataset as a stream with `decode=False`.
  2. `build_rows()` makes a single pass over the stream, decoding each kept clip to
     an int16 waveform in memory and rasterising its labels to a 100 Hz grid.
  3. `bucket_by_tier()` splits those rows into gold/silver/bronze.
  4. `SEDData` crops fixed 5 s windows and returns training tensors.
  5. `build_dataloaders()` produces the train/val loaders (val is gold-only).

Everything stays in memory - there is no disk cache - which matches the reference
script and keeps the pipeline simple.
"""

from __future__ import annotations

import io
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from config import (
    CFG, CAT2IDX, N_OUT, N_SAMP, FRAMES_100HZ, FRAMES_OUT, FRAME_SEC, REPO,
)


# ---------------------------------------------------------------------------
# Torch seeding. random/np are seeded in config; torch seeds model init and the
# DataLoader sampler, which otherwise start from OS entropy.
# ---------------------------------------------------------------------------
def seed_torch(seed: int = CFG["seed"]):
    """Seed torch (CPU + CUDA). DataLoader workers inherit from this seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------
def decode_audio(a, target_sr: int = CFG["sr"]):
    """Decode one `audio` cell to a mono float32 waveform at `target_sr`.

    Handles every shape the `datasets` Audio feature has ever returned: a dict with
    `array`/`bytes`/`path`, a torchcodec `AudioDecoder`, or an object with `.path`.
    We always decode ourselves (config passes decode=False) because the library's
    decoder has changed twice and only raw bytes are version-proof.
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
    elif hasattr(a, "path"):
        w, sr = sf.read(a.path, dtype="float32", always_2d=False)
    if w is None:
        raise ValueError(f"cannot decode {type(a)}")
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:                                  # downmix stereo -> mono
        w = w.mean(axis=0) if w.shape[0] < w.shape[1] else w.mean(axis=1)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(w, dtype=np.float32)


def _f(x):
    """Best-effort float parse; returns None on failure (bad timestamp strings)."""
    try:
        return float(x)
    except Exception:
        return None


def spans_from(ex):
    """Extract sorted (start, end, category) spans from a row's timestamp field.

    Bronze clips legitimately have an empty list here - that is correct, not a
    parsing failure.
    """
    out = []
    for s in (ex.get("NoiseSubCategoryTimeStamp") or []):
        st, en = _f(s.get("start")), _f(s.get("end"))
        if st is not None and en is not None and en > st:
            out.append((st, en, s.get("category")))
    return sorted(out)


def build_labels(spans, n_frames, has_strong):
    """Rasterise timestamp spans onto a 100 Hz `(N_OUT, n_frames)` label grid.

    Channel 0 = "any noise", channels 1..7 = the per-category masks. When
    `has_strong` is False (Bronze) the array stays all zeros because there is no
    timing to rasterise.
    """
    lab = np.zeros((N_OUT, n_frames), dtype=np.uint8)
    if has_strong:
        for st, en, cat in spans:
            a, b = max(0, int(round(st * 100))), min(n_frames, int(round(en * 100)))
            if b <= a:
                continue
            lab[0, a:b] = 1
            ci = CAT2IDX.get(cat)
            if ci is not None:
                lab[1 + ci, a:b] = 1
    return lab


def clip_tags(ex):
    """Build the clip-level `(N_OUT,)` tag vector from `NoiseCategory`.

    Present on every tier including Bronze; channel 0 flips on when any category is
    present. This is the only supervision Bronze clips carry.
    """
    y = np.zeros(N_OUT, dtype=np.float32)
    cats = ex.get("NoiseCategory") or []
    if len(cats):
        y[0] = 1.0
    for c in cats:
        ci = CAT2IDX.get(c)
        if ci is not None:
            y[1 + ci] = 1.0
    return y


# ---------------------------------------------------------------------------
# Streaming load + single decode pass
# ---------------------------------------------------------------------------
def load_stream(hf_token: str | None = None):
    """Open the dataset as a decode=False stream so we pull rows lazily and stop as
    soon as the per-tier caps are met (never downloading shards we won't use)."""
    from datasets import load_dataset, Audio
    raw = load_dataset(REPO, split="train", streaming=True, token=hf_token)
    raw = raw.cast_column("audio", Audio(decode=False))
    return raw


def build_rows(raw, max_per_quality, verbose: bool = True):
    """Single pass over the stream: decode each kept clip once and rasterise labels.

    Returns a list of dict rows, each with:
      wav      - int16 waveform (keeps RAM small; WavLM consumes raw audio)
      lab      - (N_OUT, T) 100 Hz label mask
      tags     - (N_OUT,) clip-level tags
      quality  - annotation tier string
      has_strong - True for gold/silver (timestamped), False for bronze
    """
    from tqdm.auto import tqdm

    counts_done = {k: 0 for k in max_per_quality}

    def done():
        # Finished once every capped tier has hit its cap (None means "no cap").
        for q, cap in max_per_quality.items():
            if cap is None:
                return False
            if cap and counts_done[q] < cap:
                return False
        return True

    rows, t0 = [], time.time()
    for i, ex in enumerate(tqdm(raw, desc="loading")):
        q = ex["annotationQuality"]
        cap = max_per_quality.get(q, 0)
        if cap == 0:
            continue
        if cap is not None and counts_done[q] >= cap:
            if done():
                break
            continue
        try:
            wav = decode_audio(ex["audio"])
        except Exception as e:
            if i < 5:
                print("decode failed row", i, e)
            continue
        if wav is None or len(wav) < CFG["sr"] * 0.2:   # skip <200 ms junk
            continue
        spans = spans_from(ex)
        has_strong = q in ("verified_timestamps", "unverified_timestamps")
        nf = 1 + len(wav) // CFG["hop"]
        rows.append(dict(
            wav=(np.clip(wav, -1, 1) * 32767).astype(np.int16),
            lab=build_labels(spans, nf, has_strong),
            tags=clip_tags(ex),
            quality=q, has_strong=bool(has_strong),
        ))
        counts_done[q] += 1

    if verbose:
        print(counts_done, f"| {len(rows)} rows | {(time.time()-t0)/60:.1f} min")
        if counts_done.get("verified_timestamps", 0) == 0:
            print("!! ZERO gold - detection will be weak and the Track 2 banks thin")
    return rows


# ---------------------------------------------------------------------------
# Tier bucketing + per-tier loss weights
# ---------------------------------------------------------------------------
# Per-sample loss weight by tier (gold full, silver down-weighted, bronze tiny).
QW = {"verified_timestamps": 1.0,
      "unverified_timestamps": CFG["silver_weight"],
      "no_timestamps": 0.2}


def bucket_by_tier(rows, verbose: bool = True):
    """Split the in-memory rows into (gold, silver, bronze) lists."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["quality"]].append(r)
    gold = by_q["verified_timestamps"]      # trustworthy onset/offset
    silver = by_q["unverified_timestamps"]  # timestamps without agreement
    bronze = by_q["no_timestamps"]          # clip-level tags only
    if verbose:
        print(f"gold {len(gold)} | silver {len(silver)} | bronze {len(bronze)}")
    return gold, silver, bronze


# ---------------------------------------------------------------------------
# Windowing helpers used by the dataset
# ---------------------------------------------------------------------------
def prep_wav(w):
    """Per-clip standardisation (zero mean, unit std). Applied identically at train
    and inference time so the encoder sees the same distribution in both."""
    w = np.asarray(w, dtype=np.float32)
    return (w - w.mean()) / (w.std() + 1e-5)


def _labels_20ms(lab, s):
    """Crop the 100 Hz label grid to the chosen window then max-pool 10 ms -> 20 ms
    so the label rate matches the model's FRAMES_OUT output rate."""
    lab = lab[:, s // CFG["hop"]: s // CFG["hop"] + FRAMES_100HZ]
    if lab.shape[1] < FRAMES_100HZ:
        lab = np.pad(lab, ((0, 0), (0, FRAMES_100HZ - lab.shape[1])))
    lab = lab[:, :FRAMES_100HZ]
    return lab.reshape(lab.shape[0], FRAMES_OUT, CFG["time_pool"]).max(axis=2)


class SEDData(Dataset):
    """Fixed 5 s window dataset over the in-memory rows.

    Each item returns wav/lab/tags/mask/strong/w tensors. A subtle correctness fix:
    a random crop can cut the event out of the window while the stored clip tag still
    says it is present. For timestamped clips we recompute the tag from the cropped
    labels so the weak loss is not poisoned by that contradiction.
    """

    def __init__(self, rows, train=True):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        m = self.rows[i]
        wav = m["wav"].astype(np.float32) / 32767.0
        lab = m["lab"].astype(np.float32)          # (N_OUT, T100)
        tags = m["tags"].astype(np.float32)        # (N_OUT,)

        # Crop (train: random, val: from 0) or pad to a fixed 5 s window.
        if len(wav) > N_SAMP:
            s = random.randint(0, len(wav) - N_SAMP) if self.train else 0
            wav = wav[s:s + N_SAMP]
            valid = N_SAMP
        else:
            s = 0
            valid = len(wav)
            wav = np.pad(wav, (0, N_SAMP - len(wav)))

        lab20 = _labels_20ms(lab, s)               # (N_OUT, FRAMES_OUT)
        strong = 1.0 if m["has_strong"] else 0.0
        # Recompute clip tag from the cropped labels for timestamped clips.
        if m["has_strong"]:
            tags = (lab20.max(axis=1) > 0).astype(np.float32)

        # Frame validity mask (1 for real audio frames, 0 for padding).
        mask = np.zeros(FRAMES_OUT, dtype=np.float32)
        n_valid = min(FRAMES_OUT, max(1, int(math.ceil(valid / CFG["sr"] / FRAME_SEC))))
        mask[:n_valid] = 1.0

        return dict(
            wav=torch.from_numpy(prep_wav(wav)),
            lab=torch.from_numpy(lab20.astype(np.float32)),
            tags=torch.from_numpy(tags),
            mask=torch.from_numpy(mask),
            strong=torch.tensor(strong, dtype=torch.float32),
            w=torch.tensor(float(QW.get(m["quality"], 1.0)), dtype=torch.float32),
        )


def split_and_load(gold, silver, bronze, t1_bs, train_limit=None, verbose=True):
    """Build the train/val DataLoaders.

    The validation set is gold-only (the only trustworthy strong labels). Everything
    else (leftover gold + all silver + all bronze) trains. `train_limit` optionally
    caps the training set for smoke runs.
    """
    random.Random(CFG["seed"]).shuffle(gold)
    n_val = max(1, int(len(gold) * CFG["val_frac"]))
    val_rows = gold[:n_val]
    train_rows = gold[n_val:] + silver + bronze
    random.Random(CFG["seed"]).shuffle(train_rows)
    if train_limit is not None:
        train_rows = train_rows[:train_limit]
    if verbose:
        print(f"train {len(train_rows)} | val {len(val_rows)} (gold only)")

    train_dl = DataLoader(SEDData(train_rows, True), batch_size=t1_bs, shuffle=True,
                          num_workers=CFG["num_workers"], drop_last=True, pin_memory=True)
    val_dl = DataLoader(SEDData(val_rows, False), batch_size=t1_bs, shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)
    return train_dl, val_dl
