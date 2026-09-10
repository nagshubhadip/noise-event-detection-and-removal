"""End-to-end entrypoint for the modular v-1 ATST SED baseline.

Run order mirrors the original single-file script:

    config -> data -> model -> train -> inference -> evaluate

Usage:
    export HF_TOKEN=hf_xxx            # accept the dataset terms first
    python main.py

Point ``VAL_AUDIO_DIR`` / ``VAL_REF_JSONL`` at your local validation folder to
run scoring + the decoding sweep after training.
"""

from pathlib import Path

import torch
import torch.nn as nn

import config
import data
import inference
import evaluate
from config import CFG, device, USE_DP, N_GPU, describe_runtime
from model import ATSTSED, core_model
from train import train


# Local validation data (external audio + reference.jsonl). Adjust as needed.
VAL_AUDIO_DIR = Path("/kaggle/input/datasets/shubhadipnag/input-audio/audio")
VAL_REF_JSONL = Path("/kaggle/input/datasets/shubhadipnag/input-data/reference.jsonl")
DO_DECODING_SWEEP = True


def main():
    config.seed_everything()
    print(describe_runtime())

    # --- 1. Data: load rows, finalise clip length, split, weight, load ---
    rows = data.load_train_subset(max_rows=4000)
    frame_dims = data.finalize_clip_length(rows, auto=True)
    train_rows, val_rows = data.split_rows(rows, use_strong_only=True)
    pos_weight_np = data.estimate_pos_weight(train_rows, frame_dims)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print("pos_weight:", [round(float(x), 2) for x in pos_weight_np])
    train_dl, val_dl = data.build_dataloaders(train_rows, val_rows, frame_dims)

    # --- 2. Model (optionally DataParallel) ---
    model = ATSTSED().to(device)
    if USE_DP:
        model = nn.DataParallel(model)
        print(f"DataParallel across {N_GPU} GPUs")
    print("params(M):",
          round(sum(p.numel() for p in core_model(model).parameters()) / 1e6, 2))

    # --- 3. Train ---
    ckpt = train(model, train_dl, val_dl, pos_weight)

    # --- 4. Inference wiring (bind trained model + frame constants) ---
    inference.bind(model, frame_dims)

    # --- 5. Local evaluation + optional decoding sweep ---
    if VAL_AUDIO_DIR.exists() and VAL_REF_JSONL.exists():
        evaluate.evaluate_local_folder(
            VAL_AUDIO_DIR, VAL_REF_JSONL,
            predict_fn=inference.predict_events_for_waveform, thr=0.5, med=7)
        if DO_DECODING_SWEEP:
            best, _ = evaluate.sweep_decoding_params(VAL_AUDIO_DIR, VAL_REF_JSONL)
            print("Best decode config:", best)
    else:
        print("Validation folder not found; skipping local eval.")

    print("done. checkpoint:", ckpt)


if __name__ == "__main__":
    main()
