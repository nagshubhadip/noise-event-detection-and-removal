# Track 1 — v-3: WavLM SED Notebook

**File**
- [v-3_indoml2026_track1.ipynb](v-3_indoml2026_track1.ipynb)

## What this is

The end-to-end, fully documented Track 1 notebook. It fine-tunes a **WavLM**
(`microsoft/wavlm-base-plus`) encoder into a frame-level SED model, and is the
version written to be read top-to-bottom (each section has a markdown explainer).

## Pipeline

1. **Config** — one `BUDGET` switch (`fast` / `full`) sets per-tier caps,
   epochs and batch size; derived window/frame constants follow.
2. **Data** — streams `ARTPARK-IISc/Vaani-Noise-Event-Dataset`
   (`streaming=True`, `Audio(decode=False)`), decodes each clip once and keeps an
   in-memory `rows` list (int16 waveform + 100 Hz `(8, T)` labels + clip tags).
   Streaming stops as soon as the per-tier caps are met, so no unused shards are
   pulled.
3. **Dataset** — fixed 5 s windows; a random crop recomputes the clip tag from
   the cropped labels so the weak loss is never poisoned. Gold gives the
   validation split; Gold + Silver + Bronze train.
4. **Model / training** — WavLM encoder + BiGRU SED head, mean-teacher style
   consistency, focal BCE with class `pos_weight`, mixup.
5. **Submission** — decodes posteriors to events and writes
   `predictions.jsonl` + `submission_track1.zip`.

## Annotation tiers

| tier | field | role |
|---|---|---|
| Gold | `verified_timestamps` | validation + strong supervision |
| Silver | `unverified_timestamps` | down-weighted strong supervision |
| Bronze | `no_timestamps` | clip-level tags only (weak / attention head) |

## Run

Designed for a Kaggle GPU session: set `HF_TOKEN` in Kaggle Secrets, turn
Internet + GPU on, and run the cells in order.
