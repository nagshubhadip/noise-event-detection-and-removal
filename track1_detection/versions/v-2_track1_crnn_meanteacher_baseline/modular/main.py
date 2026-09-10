"""End-to-end entrypoint for the modular v-2 CRNN + Mean-Teacher baseline.

    config -> data -> model(student/teacher) -> train(MT) -> inference -> evaluate

Usage:
    export HF_TOKEN=hf_xxx
    # optional real ATST-Frame:
    #   export USE_ATST_FRAME=1 ATST_FRAME_REPO=/path ATST_FRAME_CKPT=/path.ckpt
    python main.py
"""

from copy import deepcopy
from pathlib import Path

import torch

import config
import data
import inference
import evaluate
from config import device, describe_runtime, free_memory
from model import CRNN_SED, core_model
from train import train


# External validation audio + reference.jsonl (adjust to your mounts).
VAL_AUDIO_DIR = Path("/kaggle/input/datasets/shubhadipnag/input-audio/audio")
VAL_REF_JSONL = Path("/kaggle/input/datasets/shubhadipnag/input-data/reference.jsonl")


def main():
    config.seed_everything()
    print(describe_runtime())

    # --- 1. Data ---
    rows = data.load_train_subset(max_rows=4000)
    frame_dims = data.finalize_clip_length(rows)
    framing = data.make_framing(frame_dims)
    labeled, unlabeled = data.split_labeled_unlabeled(rows)
    train_dl, val_dl, unlab_dl, train_rows, val_rows = data.build_dataloaders(
        labeled, unlabeled, framing)

    pos_weight_np = data.estimate_pos_weight(train_rows, framing)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print("pos_weight:", [round(float(x), 2) for x in pos_weight_np])

    # --- 2. Student + EMA teacher (teacher starts as a frozen copy) ---
    free_memory()
    student = CRNN_SED(frames_out=frame_dims[2]).to(device)
    teacher = deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    print("student params(M):",
          round(sum(p.numel() for p in student.parameters()) / 1e6, 2))

    # --- 3. Train (returns the EMA teacher, our inference model) ---
    teacher = train(student, teacher, train_dl, val_dl, unlab_dl, pos_weight)

    # --- 4. Inference wiring + evaluation ---
    inference.bind(teacher, frame_dims)
    evaluate.test_saved_model_on_rows(val_rows, thr=0.5, med=5, tag="val")
    if VAL_AUDIO_DIR.exists() and VAL_REF_JSONL.exists():
        evaluate.test_saved_model_on_folder(VAL_AUDIO_DIR, VAL_REF_JSONL, thr=0.5, med=5)
    else:
        print("External validation folder not found; skipped.")


if __name__ == "__main__":
    main()
