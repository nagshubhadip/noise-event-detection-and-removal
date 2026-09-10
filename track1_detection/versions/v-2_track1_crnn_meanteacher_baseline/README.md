# Track 1 — v-2: CRNN + Mean-Teacher Baseline

**File**
- [v-2_track1_crnn_meanteacher_baseline.py](v-2_track1_crnn_meanteacher_baseline.py)

## What this is

The second Track 1 baseline. It fixes the two weaknesses of **v-1** (the AST-only
SED model):

1. **Boundary resolution** — adds a parallel **CNN branch** on a log-mel
   spectrogram at a 40 ms output grid (ATST-Frame-like), fused with the AST
   stream. The CNN supplies the local temporal detail that sharpens onset/offset
   edges so they can land inside the Event-F1 tolerance.
2. **Unlabelled data** — adds a **Mean-Teacher** (EMA teacher) consistency loss so
   the Bronze / no-timestamp clips that v-1 discarded are used as unlabelled data.

## Pipeline

1. **Data** — same reader as v-1 (`load_train_subset` builds in-memory rows).
   Labelled = strong (Gold/Silver); unlabelled = Bronze.
2. **Model** — `CRNN_SED`: AST (or optional real ATST-Frame) sequence stream +
   CNN branch → FiLM/merge → BiGRU → framewise + attention heads.
3. **Losses** — focal BCE (labelled) + MSE consistency between student and EMA
   teacher frame probabilities (all data), with a sigmoid ramp-up.
4. **Training** — Stage 1: frozen backbone, train CRNN head. Stage 2: unfreeze +
   heavy Mean-Teacher on unlabelled data. The EMA teacher is used at inference.
5. **Evaluation** — Event-F1 (micro) + temporal Dice (macro), mirroring the
   competition scorer, plus a folder-based scorer against `reference.jsonl`.

## Optional real ATST-Frame

Set `USE_ATST_FRAME=1` with `ATST_FRAME_REPO` and `ATST_FRAME_CKPT` to swap the
AST stream for a genuine ATST-Frame backbone (needs the Audio-WestlakeU code +
`atst_as2M.ckpt`; not pip-installable). Disabled by default.

## Run

```bash
python v-2_track1_crnn_meanteacher_baseline.py
```

Set `HF_TOKEN` before running.
