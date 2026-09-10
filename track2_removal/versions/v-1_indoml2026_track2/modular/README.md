# v-1 Track 2 (noise removal) — modular package

A modular, well-commented rewrite of the v-1 Track 2 notebook. The original notebook
(`../v-1_indoml2026_track2.ipynb`) is **unchanged**; this package splits the same
logic into focused modules.

> Import errors for `torch`, `datasets`, `transformers`, `soundfile`, `librosa`,
> `jiwer`, `huggingface_hub`, `webrtcvad`, `tqdm` are expected on a local machine —
> this targets the Kaggle GPU environment.

## Module map

| File | Responsibility |
|------|----------------|
| [config.py](config.py) | Budget switch, audio/STFT settings, exact submission spec, output-stage knobs, encoder choice, simulation params, working-directory paths. |
| [evaluate.py](evaluate.py) | Official scorer verbatim — `si_sdr`, `normalize`, pooled `delta_wer`, `combined`. |
| [data.py](data.py) | Audio IO, VAD/spans, validation resolver, clean/noise bank building, mixture simulation, `T2Data` dataset, STFT helpers. |
| [model.py](model.py) | `SraVaaniEnc` / `WavLMEnc` / `build_encoder`, `MaskNet` (BiGRU + two-depth FiLM), `enhance_batch`. |
| [losses.py](losses.py) | `si_sdr_loss` (time-domain, the metric itself). |
| [train.py](train.py) | Simulation training + fine-tuning on real validation pairs. |
| [inference.py](inference.py) | Output stage (`enhance` + blend + RMS restore), `check_output`, `Transcriber` (mandated ASR). |
| [main.py](main.py) | End-to-end orchestration, blend sweep, submission packaging. |

## Data flow

```
resolve_validation ─────────────┐  (folders -> SI-SDR / dWER roles)
                                 ▼
stream gold clips ─► build_banks ─► clean bank + noise bank
                                 │
                                 ▼
                simulate_dataset ─► mix/clean pairs + manifest
                                 │
                                 ▼
        build_dataloaders ─► train_dl, val_dl
                                 │
   build_encoder ─► MaskNet ─────┤
                                 ▼
            train_on_sim (SI-SDR loss) ─► CKPT
                                 │
        Transcriber (mandated ASR) + sweep_blend
                                 │
        finetune_on_real (held-out slice) ─► re-sweep blend
                                 │
                                 ▼
        write_submission ─► enhance -> transcribe -> submission_track2.zip
```

## Scoring reminders

- **Combined = SI-SDR(synthetic) + 100 × dWER(fraction)**, dominated by SI-SDR.
- dWER is **pooled** (total edits / total ref words), not a mean of ratios.
- `normalize()` strips inline tags (`<horn>…</horn>`) and punctuation.
- The submission ZIP needs `transcripts.jsonl` + every enhanced WAV at the **root**,
  16 kHz mono PCM-16, one JSONL line per clip.

## Run

```python
from main import run
run(budget="tiny", hf_token="hf_...", do_finetune=True, do_submission=False)
```
