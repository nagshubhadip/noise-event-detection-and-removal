# v-1 ATST SED — Modular version

This folder is a **modularised, heavily-commented** rewrite of
[../v-1_track1_atst_sed_baseline.py](../v-1_track1_atst_sed_baseline.py). The
original single-file script (and its notebook) are kept untouched; this package
splits the exact same logic into focused modules so each piece can be read,
tested and reused on its own.

## Module map

| file | responsibility |
|---|---|
| [config.py](config.py) | all hyper-parameters, device/AMP flags, label taxonomy, seeding |
| [data.py](data.py) | audio decode, span parsing, label rasterisation, `load_train_subset`, `Track1Dataset`, splits, `pos_weight`, dataloaders |
| [model.py](model.py) | `ATSTSED` (AST/ATST backbone + BiGRU + strong/attention heads) |
| [losses.py](losses.py) | focal BCE + boundary-aware edge loss = `combined_loss` |
| [train.py](train.py) | two-stage training loop with shared early-stopping + AMP |
| [inference.py](inference.py) | posterior computation + event decoding (merge, valley-split, trim) |
| [evaluate.py](evaluate.py) | Event-F1/Dice folder scorer + decoding-parameter sweep |
| [main.py](main.py) | wires everything together end-to-end |

## Data flow

```
load_train_subset ─► finalize_clip_length ─► split_rows ─► build_dataloaders
                                                   │
                                          estimate_pos_weight
                                                   ▼
                        ATSTSED ─► train (stage 1 frozen ─► stage 2 fine-tune)
                                                   ▼
                    inference.bind ─► predict_events_for_waveform
                                                   ▼
                 evaluate_local_folder / sweep_decoding_params
```

## Run

```bash
export HF_TOKEN=hf_xxx          # accept the Vaani dataset terms first
export ATST_BACKBONE=...        # optional: real ATST checkpoint
python main.py
```

The behaviour is identical to the original script; only the file layout and the
comments differ.
