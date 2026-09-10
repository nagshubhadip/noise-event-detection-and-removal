"""Two-stage training loop for v-1.

Stage 1 freezes the backbone and trains only the BiGRU + heads (fast warmup).
Stage 2 unfreezes the backbone at a lower LR to fine-tune end-to-end. Both
stages share one early-stopping counter that stops when val loss rises for a few
consecutive epochs. Uses AMP on GPU.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import CFG, device, USE_AMP
from losses import combined_loss
from model import core_model

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Early-stop policy.
EARLY_STOP_CONSECUTIVE_INCREASE = 3
MIN_DELTA = 1e-6


def _move_batch(batch):
    """Move a batch to the device; return (wav, lab, mask, sample_weight)."""
    wav = batch["wav"].to(device, non_blocking=True)
    lab = batch["lab"].to(device, non_blocking=True)
    m = batch["mask"].to(device, non_blocking=True)
    sw = batch.get("sw")
    if sw is not None:
        sw = sw.to(device, non_blocking=True)
    return wav, lab, m, sw


@torch.no_grad()
def eval_loss(model, dl, pos_weight=None, desc="val"):
    """Mean combined loss over a loader (no grad, eval mode)."""
    model.eval()
    vals = []
    for batch in tqdm(dl, desc=desc, leave=False):
        wav, lab, m, sw = _move_batch(batch)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            frame_logits, _ = model(wav, m)
            loss = combined_loss(frame_logits.float(), lab.float(), m.float(),
                                 sample_weight=sw, pos_weight=pos_weight)
        vals.append(float(loss.detach().cpu()))
    return float(np.mean(vals)) if vals else 0.0


def _run_stage(model, opt, scaler, train_dl, val_dl, epochs, params,
               pos_weight, prev_vl, inc_count, tag):
    """Run one training stage; return (prev_vl, inc_count, stopped)."""
    for ep in range(epochs):
        model.train()
        tr = []
        for batch in tqdm(train_dl, desc=f"{tag} ep{ep:02d}", leave=False):
            wav, lab, m, sw = _move_batch(batch)
            with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                frame_logits, _ = model(wav, m)
                loss = combined_loss(frame_logits.float(), lab.float(), m.float(),
                                     sample_weight=sw, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            tr.append(float(loss.detach().cpu()))

        tr_mean = float(np.mean(tr)) if tr else 0.0
        vl = eval_loss(model, val_dl, pos_weight=pos_weight, desc=f"{tag} val")
        print(f"[{tag}][{ep+1}/{epochs}] train={tr_mean:.4f} val={vl:.4f}")

        # Early-stop bookkeeping (shared across stages).
        if prev_vl is not None and (vl - prev_vl) > MIN_DELTA:
            inc_count += 1
        else:
            inc_count = 0
        prev_vl = vl
        if inc_count >= EARLY_STOP_CONSECUTIVE_INCREASE:
            print(f"[Early Stop] val rose {inc_count} epochs in {tag}.")
            return prev_vl, inc_count, True
    return prev_vl, inc_count, False


def train(model, train_dl, val_dl, pos_weight, ckpt_path="track1_atst.pt"):
    """Full two-stage fine-tune; saves the trained weights to ``ckpt_path``."""
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    mcore = core_model(model)
    prev_vl, inc_count = None, 0

    # ---- Stage 1: frozen backbone, train head only ----
    for p in mcore.backbone.parameters():
        p.requires_grad = False
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=CFG["lr_head"],
                            weight_decay=CFG["weight_decay"])
    print(f"[Stage 1] trainable params: {sum(p.numel() for p in head_params):,}")
    prev_vl, inc_count, stopped = _run_stage(
        model, opt, scaler, train_dl, val_dl, CFG["stage1_epochs"],
        head_params, pos_weight, prev_vl, inc_count, "stage1")

    # ---- Stage 2: unfreeze backbone at a lower LR ----
    if not stopped:
        for p in mcore.backbone.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW([
            {"params": mcore.backbone.parameters(), "lr": CFG["lr_backbone"]},
            {"params": [p for n, p in mcore.named_parameters()
                        if not n.startswith("backbone.")], "lr": CFG["lr_head"] * 0.3},
        ], weight_decay=CFG["weight_decay"])
        print(f"[Stage 2] unfroze backbone | lr_bb={CFG['lr_backbone']}")
        _run_stage(model, opt, scaler, train_dl, val_dl, CFG["stage2_epochs"],
                   list(model.parameters()), pos_weight, prev_vl, inc_count, "stage2")

    torch.save({"model": mcore.state_dict(), "cfg": CFG,
                "backbone": mcore.backbone_name}, ckpt_path)
    print("saved ->", ckpt_path)
    return ckpt_path
