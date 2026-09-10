"""Two-stage Mean-Teacher training loop for v-2.

Stage 1: freeze the sequence backbone, train the CRNN head only (no MT).
Stage 2: unfreeze the backbone at a low LR and add the Mean-Teacher consistency
loss on both labelled and unlabelled batches, with a ramped-up weight and an EMA
teacher update after every optimiser step.

Gradient accumulation gives a larger effective batch than fits in memory.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import CFG, device, USE_AMP, free_memory
from losses import (focal_bce_frame, consistency_mse, sigmoid_rampup, ema_update)
from model import core_model

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(student, teacher, ep, opt, scaler, train_dl, unlab_dl,
              pos_weight, mt_weight, use_mt):
    """Run one training epoch; return the mean loss.

    When ``use_mt`` is on, adds consistency between student and EMA teacher on the
    labelled batch and on one unlabelled batch, then EMA-updates the teacher.
    """
    student.train()
    tr = []
    accum = max(1, CFG["accum_steps"])
    unlab_iter = iter(unlab_dl) if (use_mt and unlab_dl is not None) else None
    pbar = tqdm(train_dl, desc=f"ep{ep:02d}", leave=False)
    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        wav = batch["wav"].to(device); mel = batch["mel"].to(device)
        lab = batch["lab"].to(device); m = batch["mask"].to(device)
        sw = batch["sw"].to(device)

        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            s_logits, _ = student(wav, mel, m)
            sup = focal_bce_frame(s_logits.float(), lab.float(), m.float(),
                                  sample_weight=sw, pos_weight=pos_weight)
            loss = sup
            if use_mt:
                # Consistency on the labelled batch.
                with torch.no_grad():
                    t_logits, _ = teacher(wav, mel, m)
                cons = consistency_mse(s_logits.float(), t_logits.float(), m.float())
                # Plus one unlabelled (bronze) batch.
                if unlab_iter is not None:
                    try:
                        ub = next(unlab_iter)
                    except StopIteration:
                        unlab_iter = iter(unlab_dl); ub = next(unlab_iter)
                    uw = ub["wav"].to(device); um = ub["mel"].to(device)
                    umask = ub["mask"].to(device)
                    us_logits, _ = student(uw, um, umask)
                    with torch.no_grad():
                        ut_logits, _ = teacher(uw, um, umask)
                    cons = cons + consistency_mse(us_logits.float(),
                                                  ut_logits.float(), umask.float())
                loss = sup + mt_weight * cons

        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if use_mt:
                ema_update(student, teacher, CFG["ema_decay"])

        tr.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=f"{tr[-1]:.4f}")
        del wav, mel, lab, m, sw, loss
    free_memory()
    return float(np.mean(tr)) if tr else 0.0


@torch.no_grad()
def eval_loss(model, val_dl, pos_weight):
    """Mean supervised frame loss over the validation loader."""
    model.eval()
    vals = []
    for batch in val_dl:
        wav = batch["wav"].to(device); mel = batch["mel"].to(device)
        lab = batch["lab"].to(device); m = batch["mask"].to(device)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            logits, _ = model(wav, mel, m)
            vals.append(float(focal_bce_frame(
                logits.float(), lab.float(), m.float(), pos_weight=pos_weight).cpu()))
    return float(np.mean(vals)) if vals else 0.0


def train(student, teacher, train_dl, val_dl, unlab_dl, pos_weight,
          ckpt_path="track1_crnn_mt.pt"):
    """Full Stage 1 -> Stage 2 schedule. Returns the (updated) teacher model.

    The EMA teacher is the better inference model, so it is what we return/save.
    """
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    sc = core_model(student)

    # ---- Stage 1: frozen backbone, head only, no Mean-Teacher ----
    for p in sc.seq_backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=CFG["lr_head"], weight_decay=CFG["weight_decay"])
    print(f"[Stage 1] frozen backbone, {CFG['stage1_epochs']} epochs")
    for ep in range(CFG["stage1_epochs"]):
        trl = run_epoch(student, teacher, ep, opt, scaler, train_dl, unlab_dl,
                        pos_weight, 0.0, use_mt=False)
        print(f"[S1][{ep+1}/{CFG['stage1_epochs']}] "
              f"train={trl:.4f} val={eval_loss(student, val_dl, pos_weight):.4f}")

    # Re-sync the teacher to the warmed-up student before MT starts.
    teacher = deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Stage 2: unfreeze backbone + heavy Mean-Teacher ----
    free_memory()
    for p in sc.seq_backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": sc.seq_backbone.parameters(), "lr": CFG["lr_backbone"]},
        {"params": [p for n, p in sc.named_parameters()
                    if not n.startswith("seq_backbone.")], "lr": CFG["lr_head"] * 0.3},
    ], weight_decay=CFG["weight_decay"])
    print(f"[Stage 2] fine-tune + Mean-Teacher, {CFG['stage2_epochs']} epochs")
    for ep in range(CFG["stage2_epochs"]):
        w = CFG["mt_max_weight"] * sigmoid_rampup(ep, CFG["mt_rampup_epochs"])
        trl = run_epoch(student, teacher, ep, opt, scaler, train_dl, unlab_dl,
                        pos_weight, w, use_mt=True)
        print(f"[S2][{ep+1}/{CFG['stage2_epochs']}] "
              f"train={trl:.4f} val={eval_loss(student, val_dl, pos_weight):.4f} mt_w={w:.2f}")

    torch.save({"student": sc.state_dict(), "cfg": CFG,
                "use_atst_frame": sc.use_atst_frame}, ckpt_path)
    print("saved ->", ckpt_path)
    return teacher
