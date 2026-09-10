"""Configuration and global setup for the ATST-style SED baseline (v-1).

This module centralises every knob and constant the rest of the package reads.
Keeping it in one place means the data, model, training and inference modules
never hard-code a hyper-parameter — they import it from here.

Nothing in this file has side effects beyond seeding RNGs and reading the
environment, so it is safe to import from anywhere.
"""

import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (incl. CUDA) so runs are comparable.

    Called once from ``main``; also safe to call again before a fresh split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device / hardware
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_AMP = torch.cuda.is_available()      # automatic mixed precision on GPU
USE_DP = N_GPU > 1                        # wrap model in nn.DataParallel


# ---------------------------------------------------------------------------
# Core training / audio configuration
# ---------------------------------------------------------------------------
# NOTE: ``clip_sec`` is only a default — the data pipeline can overwrite it from
# the observed audio-length distribution (see data.finalize_clip_length).
CFG = dict(
    sr=16000,              # sample rate WavLM/AST expect
    hop=160,               # 160/16000 = 10 ms → the 100 Hz label grid
    clip_sec=5.0,          # fixed training-window length (seconds)
    time_pool=2,           # pool 10 ms → 20 ms output frames
    val_frac=0.1,          # fraction of clips held out for validation
    batch_size=16 if N_GPU >= 2 else 8,
    stage1_epochs=5,       # frozen-backbone warmup
    stage2_epochs=15,      # fine-tune backbone
    lr_head=1e-3,
    lr_backbone=5e-5,
    weight_decay=1e-2,
)

# Backbone checkpoint. Point ATST_BACKBONE at the official ATST weights if you
# have them; the AST default keeps the code runnable out of the box.
ATST_BACKBONE = os.environ.get(
    "ATST_BACKBONE", "MIT/ast-finetuned-audioset-10-10-0.4593"
)

# HF dataset repo + access token (accept the dataset terms first).
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# Channel 0 of every label tensor is the class-agnostic "any-noise" target that
# Track 1 actually submits; channels 1..7 are the individual categories, trained
# as auxiliary supervision only.
CATS = [
    "animal", "vehicle_traffic", "baby_child", "singing_music",
    "phone_signal_alarm", "appliance_machine", "human_non_speech",
]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + len(CATS)      # 8 output channels


def describe_runtime() -> str:
    """Return a one-line summary of the detected hardware/config."""
    return (f"device={device} | gpus={N_GPU} | amp={USE_AMP} | "
            f"data_parallel={USE_DP} | batch_size={CFG['batch_size']}")
