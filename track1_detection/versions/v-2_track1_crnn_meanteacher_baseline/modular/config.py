"""Configuration + global setup for the CRNN + Mean-Teacher baseline (v-2).

v-2 improves on v-1 by adding (1) a fine-resolution CNN branch for sharp event
boundaries and (2) a Mean-Teacher consistency loss that turns the previously
discarded bronze clips into unlabelled training data.

This module holds every constant the other modules import.
"""

import os
import gc
import random

import numpy as np
import torch

# Reduce CUDA fragmentation (mitigates the Stage-2 OOM on 16 GB GPUs).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def free_memory():
    """Force a GC + CUDA cache flush between heavy epochs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Reproducibility + hardware
# ---------------------------------------------------------------------------
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_AMP = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
CATS = [
    "animal", "vehicle_traffic", "baby_child", "singing_music",
    "phone_signal_alarm", "appliance_machine", "human_non_speech",
]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + len(CATS)      # channel 0 = any-noise, 1..7 = categories


# ---------------------------------------------------------------------------
# Training / audio configuration
# ---------------------------------------------------------------------------
CFG = dict(
    sr=16000,
    n_mels=128,
    n_fft=1024,
    hop=160,               # 10 ms mel frames
    time_pool=4,           # CNN pools time by 4 -> 40 ms output frames
    clip_sec=10.0,         # re-finalised from the data (AST window cap = 10 s)
    val_frac=0.1,
    batch_size=4 if N_GPU < 2 else 12,   # low: Stage-2 runs extra fwd passes
    accum_steps=2,                        # effective batch = batch_size * accum
    unlab_batch_frac=1.0,                 # unlabeled batch = ceil(bs * this)
    num_workers=2,
    stage1_epochs=6,       # frozen backbone, train CRNN head
    stage2_epochs=20,      # unfreeze + heavy Mean-Teacher
    lr_head=1e-3,
    lr_backbone=5e-5,
    weight_decay=1e-2,
    # ---- Mean-Teacher ----
    ema_decay=0.999,
    mt_max_weight=30.0,    # consistency-weight ramp-up target
    mt_rampup_epochs=8,
)

# Quality -> per-sample loss weight (annotation tier reliability).
QUALITY_WEIGHTS = {
    "verified_timestamps": 1.00,
    "unverified_timestamps": 0.75,
    "no_timestamps": 0.25,
}

# ---------------------------------------------------------------------------
# Backbone selection
# ---------------------------------------------------------------------------
# Real ATST-Frame is optional (needs an external checkpoint + package). When
# enabled it REPLACES the AST stream; otherwise AST is used.
USE_ATST_FRAME = os.environ.get("USE_ATST_FRAME", "0") == "1"
ATST_FRAME_CKPT = os.environ.get("ATST_FRAME_CKPT", "")
ATST_FRAME_REPO = os.environ.get("ATST_FRAME_REPO", "")
AST_BACKBONE = os.environ.get("AST_BACKBONE", "MIT/ast-finetuned-audioset-10-10-0.4593")

# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def describe_runtime() -> str:
    return (f"device={device} | gpus={N_GPU} | amp={USE_AMP} | "
            f"use_atst_frame={USE_ATST_FRAME}")
