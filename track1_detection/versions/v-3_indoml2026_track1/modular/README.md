# v-3 WavLM SED — modular package

A modular, well-commented rewrite of the v-3 WavLM Sound-Event-Detection notebook.
The original notebook (`../v-3_indoml2026_track1.ipynb`) is **unchanged**; this
package splits the same logic into focused modules for readability and reuse.

> Import errors for `torch`, `datasets`, `transformers`, `librosa`, `soundfile`,
> `scipy`, `tqdm` are expected on a local machine — this code targets the Kaggle GPU
> environment where those packages exist.

## Module map

| File | Responsibility |
|------|----------------|
| [config.py](config.py) | Budget switch, `CFG` hyper-parameters, categories, dataset/output paths, derived frame-grid constants, seeding, `budget_settings()`. |
| [data.py](data.py) | Streaming load (`decode=False`), one-pass decode + label rasterisation, tier bucketing, `SEDData` dataset (5 s windows), DataLoader builder. |
| [model.py](model.py) | `WavLMSED` — WavLM-base+ encoder (last 2 layers dropped) + BiGRU + frame head + attention-pooled clip head (MIL). |
| [losses.py](losses.py) | Positive-weighted BCE, consistency ramp-up, wav augmentation, union-label mixup, EMA teacher update. |
| [evaluate.py](evaluate.py) | Official scorer (Event F1 + Segment Dice), posterior→events, cached scoring, post-processing sweep. |
| [train.py](train.py) | Two-stage mean-teacher training loop (frozen → unfrozen encoder). |
| [inference.py](inference.py) | Whole-clip / windowed posteriors, `export_track1` event export. |
| [main.py](main.py) | End-to-end orchestration + Track 1 submission writer. |

## Data flow

```
stream dataset (decode=False)
        │  data.build_rows  (decode once, rasterise 100 Hz labels, keep int16 in RAM)
        ▼
   rows ──► data.bucket_by_tier ──► gold / silver / bronze
        │                                   │
        │  data.split_and_load (val = gold only)
        ▼
 train_dl, val_dl ──► train.train (2-stage mean teacher) ──► CKPT
        │
        ▼
 evaluate.cache_posteriors (run once) ──► evaluate.sweep_postprocessing
        │                                   └─► best (thr, med, min_dur, gate)
        ▼
 inference.export_track1 ──► main.write_submission ──► predictions.jsonl + zip
```

## Tier supervision (why three annotation qualities matter)

| tier | `quality` | frame label | clip tag | loss path |
|------|-----------|-------------|----------|-----------|
| Gold | `verified_timestamps` | ✅ trustworthy | ✅ | frame (w=1.0) + weak + consistency |
| Silver | `unverified_timestamps` | ✅ no agreement | ✅ | frame (w=0.4) + weak + consistency |
| Bronze | `no_timestamps` | ❌ all zeros | ✅ | weak (MIL) + consistency only |

## Run

```python
from main import run
run(budget="fast", hf_token="hf_...", do_submission=False)
```
