"""Configuration and constants for the v-1 Track 2 (noise removal) baseline.

Single source of truth for the budget switch, audio/STFT settings, the exact
submission spec (transcript filename, keys, WAV subtype, mandated ASR repo), the
output-stage knobs (blend + RMS restore), the encoder choice, the mixture-simulation
parameters and all working-directory paths.

Import-light: only defines constants and seeds the Python/NumPy RNGs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Budget switch - the one knob you actually touch.
#   "tiny" : ~100-clip smoke test.
#   "fast" : ~1 h 30 run.
#   "full" : ~3 h run.
# ---------------------------------------------------------------------------
BUDGET = "tiny"

CFG = dict(sr=16000, clip_sec=5.0, seed=42)
N_FFT, HOP = 512, 128                          # STFT for the mask network


# ---------------------------------------------------------------------------
# Exact submission spec (from the Evaluation page). Getting any of these wrong
# fails the audit or the format check.
# ---------------------------------------------------------------------------
TRANSCRIPT_NAME = "transcripts.jsonl"
ID_KEY, TEXT_KEY = "clip_id", "text"           # clip_id = filename WITHOUT .wav
WAV_SUBTYPE = "PCM_16"                          # WAV, PCM 16-bit signed
ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"         # mandated; the audit checks this


# ---------------------------------------------------------------------------
# Output stage.
# ---------------------------------------------------------------------------
BLEND = 0.85                                   # y = BLEND*est + (1-BLEND)*mix
MATCH_RMS = True                               # restore level; free under SI-SDR, vital for ASR


# ---------------------------------------------------------------------------
# Encoder choice for the mask network conditioning.
# ---------------------------------------------------------------------------
T2_ENCODER = "sravaani"                         # "sravaani" | "wavlm" | "none"
WAVLM_NAME = "microsoft/wavlm-base-plus"
ENC_PROJ = 128                                  # encoder projection dim before FiLM


# ---------------------------------------------------------------------------
# Budget-dependent sizes.
#   MAX_GOLD : gold clips scanned to build the clean/noise banks.
#   N_MIX    : simulated mixtures to generate.
#   EPOCHS   : simulation-training epochs. BS : batch size.
# ---------------------------------------------------------------------------
def budget_sizes(budget: str = BUDGET):
    if budget == "tiny":
        return dict(MAX_GOLD=100, N_MIX=100, EPOCHS=3, BS=8)
    if budget == "fast":
        return dict(MAX_GOLD=6000, N_MIX=4000, EPOCHS=25, BS=8)
    return dict(MAX_GOLD=12000, N_MIX=8000, EPOCHS=40, BS=8)


# ---------------------------------------------------------------------------
# Dataset + working-directory layout.
# ---------------------------------------------------------------------------
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
WORK = Path("/kaggle/working")

CLEAN_DIR = WORK / "bank/clean"                 # clean speech spans (no annotated event)
NOISE_DIR = WORK / "bank/noise"                 # isolated noise-event spans
MIX_DIR = WORK / "t2/mix"                        # simulated noisy mixtures
REF_DIR = WORK / "t2/clean"                      # matching clean references
MANIFEST = WORK / "t2_manifest.jsonl"
CKPT = WORK / "t2_model.pt"


def ensure_dirs():
    """Create the bank/mix/ref working directories (idempotent)."""
    for d in [CLEAN_DIR, NOISE_DIR, MIX_DIR, REF_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mixture-simulation parameters.
#   Events are placed into a single clean span at a sampled event-local SNR.
# ---------------------------------------------------------------------------
SIM = dict(dur_sec=CFG["clip_sec"], min_clean_sec=2.5, min_noise_sec=0.3,
           max_noise_sec=3.0, events=(1, 2), snr_db=(-5.0, 12.0), seed=1234)
MAX_SPEECH_RATIO = 0.35                          # reject "noise" spans that are mostly speech
N_SAMP = int(CFG["clip_sec"] * CFG["sr"])        # samples in a training window

AUD = (".wav", ".flac", ".mp3", ".ogg")


def seed_everything(seed: int = CFG["seed"]):
    """Seed the Python and NumPy RNGs (torch is seeded in data.seed_torch)."""
    random.seed(seed)
    np.random.seed(seed)


seed_everything()
