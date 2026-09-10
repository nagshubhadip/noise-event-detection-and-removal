# v-2 CRNN + Mean-Teacher — Modular version

A modularised, heavily-commented rewrite of
[../v-2_track1_crnn_meanteacher_baseline.py](../v-2_track1_crnn_meanteacher_baseline.py).
The original single-file script is kept untouched; this package splits the same
logic into focused modules.

## Why v-2 exists

v-1 (AST-only) produces ~100 ms time tokens, so onset/offset edges can't land
inside the 50–80 ms Event-F1 tolerance. v-2 fixes this with:
1. a **CNN branch** on log-mel at a 40 ms grid (sharp boundaries), fused with AST;
2. a **Mean-Teacher** consistency loss so the bronze / no-timestamp clips that v-1
   discarded become unlabelled training data.

## Module map

| file | responsibility |
|---|---|
| [config.py](config.py) | hyper-parameters, device/AMP, `free_memory`, backbone selection, tiers |
| [data.py](data.py) | decode, labels, `logmel`, labelled/unlabelled split, datasets, loaders |
| [model.py](model.py) | `ASTBackbone`, `ATSTFrameBackbone`, `CNNBranch`, `CRNN_SED` |
| [losses.py](losses.py) | focal BCE, `consistency_mse`, `sigmoid_rampup`, `ema_update` |
| [train.py](train.py) | Stage 1 (frozen) → Stage 2 (unfreeze + Mean-Teacher) with EMA teacher |
| [inference.py](inference.py) | posterior + event decoding (teacher model, wav+mel inputs) |
| [evaluate.py](evaluate.py) | Event-F1/Dice scorers on rows and on a folder |
| [main.py](main.py) | wires everything end-to-end |

## Data flow

```
load_train_subset ─► finalize_clip_length ─► split_labeled_unlabeled
                                                   │
                                         build_dataloaders (train/val/unlab)
                                                   ▼
              student = CRNN_SED ;  teacher = EMA copy
                                                   ▼
     train:  Stage 1 (head only) ─► Stage 2 (unfreeze + MT, EMA updates)
                                                   ▼
             inference.bind(teacher) ─► predict_events_for_waveform
                                                   ▼
            test_saved_model_on_rows / _on_folder  (Event-F1 + Dice)
```

## Run

```bash
export HF_TOKEN=hf_xxx
# optional real ATST-Frame backbone:
#   export USE_ATST_FRAME=1 ATST_FRAME_REPO=/path ATST_FRAME_CKPT=/path/atst_as2M.ckpt
python main.py
```

Behaviour matches the original script; only the layout and comments differ.
