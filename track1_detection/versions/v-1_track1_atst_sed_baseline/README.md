# Track 1 — v-1: ATST-style SED Baseline

**Files**
- [v-1_track1_atst_sed_baseline.py](v-1_track1_atst_sed_baseline.py) — runnable script
- [v-1_track1_atst_sed_baseline.ipynb](v-1_track1_atst_sed_baseline.ipynb) — notebook counterpart

## What this is

The first Track 1 (Noise Event Detection) baseline. It fine-tunes a patch-wise
**AST** backbone (`MIT/ast-finetuned-audioset-10-10-0.4593`, a stand-in for the
official ATST checkpoint) into a frame-level Sound Event Detection (SED) head.

## Pipeline

1. **Data** — streams `ARTPARK-IISc/Vaani-Noise-Event-Dataset`, decodes audio
   manually (`Audio(decode=False)`), and rasterises timestamp spans to a 100 Hz
   `(8, T)` label grid (channel 0 = any-noise, channels 1–7 = categories).
2. **Model** — AST backbone → drop `[CLS]`/`[distill]` tokens → max-pool over
   frequency → BiGRU → framewise `strong` head + attention-pooled `clip` head.
3. **Loss** — focal BCE + a boundary-aware edge term (targets the Event-F1
   onset/offset tolerance).
4. **Training** — two stages: (1) frozen backbone, train head; (2) unfreeze
   backbone at a lower LR. Early stops on rising val loss.
5. **Inference** — sliding-window posterior, median filtering, valley-splitting,
   then a decoding-parameter sweep (threshold / median / merge-gap / min-dur).

## Annotation tiers

| tier | field | use |
|---|---|---|
| Gold | `verified_timestamps` | trustworthy onset/offset (weight 1.00) |
| Silver | `unverified_timestamps` | timestamps without agreement (weight 0.75) |
| Bronze | `no_timestamps` | clip-level tags only; dropped by `USE_STRONG_ONLY` |

## Known limitation

AST produces ~100 ms time tokens, so onset/offset edges physically cannot land
inside the 50–80 ms Event-F1 tolerance — Event-F1 is capped around ~0.25 while
Dice reaches ~0.66. This is the motivation for **v-2** (CRNN + Mean-Teacher).

## Run

```bash
python v-1_track1_atst_sed_baseline.py
```

Set `HF_TOKEN` (accept the dataset terms first) and, optionally, `ATST_BACKBONE`.
