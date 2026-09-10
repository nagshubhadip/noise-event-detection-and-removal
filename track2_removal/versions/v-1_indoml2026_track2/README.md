# Track 2 — v-1: Noise Event Removal Notebook

**File**
- [v-1_indoml2026_track2.ipynb](v-1_indoml2026_track2.ipynb)

## What this is

The Track 2 (Noise Event Removal) baseline. It trains a **BiGRU magnitude-mask**
denoiser on the STFT (reconstructed with the mixture phase) and submits enhanced
audio plus transcripts. The submission is `submission_track2.zip`.

## Scoring

```
Combined = SI-SDR(synthetic subset only) + 100 x dWER(all clips)
```

- **SI-SDR** — absolute, on the synthetic clips that have a clean reference.
- **dWER** — from your submitted transcripts vs. private ground truth; an audit
  re-runs the mandated ASR (`ARTPARK-IISc/SraVaani-1.0`) on your WAVs.

Combined is dominated by SI-SDR, so dWER is treated as a penalty to keep ≥ 0.

## Pipeline

1. **Config** — one `BUDGET` switch (`tiny` / `fast` / `full`).
2. **Scorer** — the official SI-SDR + pooled dWER metric, copied verbatim.
3. **Validation** — resolves the `indoml-validation` folder layout (natural /
   synthetic / clean-ref) and builds the paired item list.
4. **Data banks** — streams `ARTPARK-IISc/Vaani-Noise-Event-Dataset`
   (`streaming=True`, `Audio(decode=False)`) and, from Gold clips, cuts a
   **clean** bank (speech between events) and a VAD-filtered **noise** bank. It
   iterates rows and stops once `MAX_GOLD` is reached.
5. **Simulate** — mixes clean + scaled noise at sampled event-local SNRs to make
   paired training data.
6. **Model** — `MaskNet`: STFT input + optional **SraVaani** / WavLM encoder via
   FiLM at two depths → BiGRU → magnitude mask → iSTFT. Loss is SI-SDR.
7. **Fine-tune** — on the real validation pairs (holds out a slice for selection)
   to close the sim→real gap.
8. **Output** — blend with the mixture + RMS restore, transcribe with the
   mandated ASR, and package the ZIP (WAVs + `transcripts.jsonl`, root-level).

## Run

Designed for a Kaggle GPU session: accept the Vaani **and** SraVaani-1.0 terms,
set `HF_TOKEN` in Kaggle Secrets, attach `indoml-validation` + the Track 2 test
audio, and run the cells in order.
