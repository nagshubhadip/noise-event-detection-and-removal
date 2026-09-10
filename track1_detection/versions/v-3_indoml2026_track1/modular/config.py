"""Configuration and global constants for the v-3 WavLM SED baseline.

This is the single source of truth for every tunable in the pipeline: the budget
switch, the audio/training hyper-parameters, the category vocabulary, the dataset
location and the derived frame-grid constants that the rest of the package imports.

Nothing here trains or touches the network - it only defines constants and a couple
of tiny seeding helpers, so it is safe to import from anywhere without side effects
beyond RNG seeding when you call `seed_everything()`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Budget switch - the one knob you actually touch.
#   "fast" : ~3 h smoke/dev run with small per-tier caps and few epochs.
#   "full" : ~7 h full run using far more clips and more epochs.
# ---------------------------------------------------------------------------
BUDGET = "fast"


# ---------------------------------------------------------------------------
# Core audio + training config. Consumed all over the package via CFG[...].
# ---------------------------------------------------------------------------
CFG = dict(
    sr=16000,             # WavLM expects 16 kHz mono audio
    hop=160,              # 160 / 16000 = 10 ms label grid
    clip_sec=5.0,         # 5 s window: halves the WavLM sequence, doubles the batch
    time_pool=2,          # pool 10 ms -> 20 ms, matching WavLM's native 50 Hz output
    n_cat=7,              # number of noise categories (channel 0 is "any noise")
    silver_weight=0.4,    # Silver-tier labels count 40% of Gold
    ema_decay=0.999,      # mean-teacher EMA decay
    max_cons_w=2.0,       # max consistency weight after ramp-up
    rampup_epochs=5,      # consistency ramp-up length in epochs
    pos_weight=4.0,       # up-weight rare noise frames in BCE
    mixup_prob=0.5,       # probability of applying mixup to a batch
    mixup_alpha=0.2,      # Beta(alpha, alpha) mixing coefficient
    val_frac=0.1,         # 10% of gold clips held out for validation
    weight_decay=1e-2,
    num_workers=2,
    seed=42,
)


# ---------------------------------------------------------------------------
# Categories. Predicted internally (channels 1..7); Track 1 submits only the
# class-agnostic "any noise" channel 0, so these are for training signal only.
# ---------------------------------------------------------------------------
CATS = [
    "animal", "vehicle_traffic", "baby_child", "singing_music",
    "phone_signal_alarm", "appliance_machine", "human_non_speech",
]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + CFG["n_cat"]   # 8 output channels: [any-noise, cat0..cat6]


# ---------------------------------------------------------------------------
# Dataset + output locations (Kaggle-style paths).
# ---------------------------------------------------------------------------
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
WORK = Path("/kaggle/working")
CKPT = WORK / "t1_wavlm.pt"


# ---------------------------------------------------------------------------
# Encoder settings.
#   DROP_LAST_LAYERS: Schmid et al. (ICASSP 2025) found the final WavLM layers
#   overfit the pretraining task while earlier layers transfer better, so we drop
#   the last two transformer blocks.
# ---------------------------------------------------------------------------
ENCODER_NAME = "microsoft/wavlm-base-plus"
DROP_LAST_LAYERS = 2
LR_ENC, LR_HEAD = 5e-5, 1e-3


# ---------------------------------------------------------------------------
# Derived constants used everywhere else (window / frame sizes).
# ---------------------------------------------------------------------------
N_SAMP = int(CFG["clip_sec"] * CFG["sr"])                 # samples in a 5 s window
FRAMES_100HZ = int(CFG["clip_sec"] * CFG["sr"] / CFG["hop"])   # 10 ms frames per window
FRAMES_OUT = FRAMES_100HZ // CFG["time_pool"]             # 20 ms output frames per window
FRAME_SEC = CFG["hop"] * CFG["time_pool"] / CFG["sr"]     # seconds per (20 ms) output frame

# WavLM conv front-end needs >= ~400 samples; keep a margin for very short clips.
MIN_SAMPLES = 640
# Longest clip we run through the model in a single forward pass. Above this we
# window the clip to bound memory (see inference.t1_posteriors).
MAX_ONESHOT_SEC = 120.0


def budget_settings(budget: str = BUDGET, train_limit: int | None = None):
    """Return the (MAX_PER_QUALITY, STAGE1_EPOCHS, STAGE2_EPOCHS, T1_BS, TRAIN_LIMIT)
    tuple for the requested budget.

    `MAX_PER_QUALITY` caps how many clips of each annotation tier we decode. When
    `train_limit` is small, decoding thousands of clips is wasteful, so the caps are
    tightened to just above the limit (plus headroom for the gold-only val split).
    """
    if budget == "fast":
        max_per_quality = {"verified_timestamps": 6000, "unverified_timestamps": 6000,
                           "no_timestamps": 2000}
        stage1_epochs, stage2_epochs, t1_bs = 3, 8, 16
        default_limit = 100          # smoke-run cap on training clips
    else:
        max_per_quality = {"verified_timestamps": None, "unverified_timestamps": 20000,
                           "no_timestamps": 8000}
        stage1_epochs, stage2_epochs, t1_bs = 6, 16, 16
        default_limit = None         # use everything

    limit = default_limit if train_limit is None else train_limit
    if limit is not None:
        cap = limit + 32             # headroom for the gold-only validation split
        max_per_quality = {k: (min(v, cap) if v is not None else cap)
                           for k, v in max_per_quality.items()}
    return max_per_quality, stage1_epochs, stage2_epochs, t1_bs, limit


def seed_everything(seed: int = CFG["seed"]):
    """Seed Python and NumPy RNGs. Torch is seeded separately in data.py once torch
    is imported, so this module stays import-light."""
    random.seed(seed)
    np.random.seed(seed)


# Seed the light RNGs at import time so constant-time helpers are reproducible.
seed_everything()
