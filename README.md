# Noise Event Detection and Removal — IndoML 2026 Datathon Baselines

Baselines for the two tracks of the IndoML 2026 Datathon, organised into one
subfolder per track.

## Repository layout

```
.
├── track1_detection/        # Track 1 — Noise Event Detection (onset/offset localisation)
│   └── versions/
│       ├── v-1_track1_atst_sed_baseline/          # ATST-Frame SED
│       ├── v-2_track1_crnn_meanteacher_baseline/  # CRNN mean-teacher SSL
│       └── v-3_indoml2026_track1/                 # WavLM SED (mean-teacher, MIL)
└── track2_removal/          # Track 2 — Noise Event Removal (enhancement + ASR)
    └── versions/
        └── v-1_indoml2026_track2/                 # STFT mask net + SraVaani/WavLM FiLM
```

Each version folder contains:

- the **original** baseline (`.ipynb` / `.py`) — kept untouched, and
- a **`modular/`** subpackage — the same logic split into focused, well-commented
  modules (`config.py`, `data.py`, `model.py`, `losses.py`, `train.py`,
  `inference.py`, `evaluate.py`, `main.py`) plus a `README.md` with a module map and
  data-flow diagram.

## Tracks

### Track 1 — Detection
Predict onset/offset timestamps of noise events in speech clips. Scored by
Event F1 + Segment Dice (max 2.0). Track 1 submissions are class-agnostic.

### Track 2 — Removal
Enhance noisy clips and submit both the enhanced audio and its transcript (from the
mandated SraVaani-1.0 ASR). Scored by SI-SDR + 100 × dWER.

## Running

The baselines target the Kaggle GPU environment (they depend on `torch`,
`datasets`, `transformers`, `soundfile`, `librosa`, `jiwer`, etc.). Each modular
package exposes a `run()` entry point in its `main.py`.
