"""Two-stage mean-teacher training for the v-3 WavLM baseline.

Stage 1 freezes the encoder and trains only the GRU + heads (skipping this stage is
the classic failure mode: a large pretrained encoder fine-tuned end-to-end from step
0 overfits immediately). Stage 2 unfreezes with a much smaller LR on the encoder.

Every step combines three losses (see losses.py and the notebook appendix):
  loss = l_strong + l_weak + cw * l_cons
  * l_strong - frame BCE, gold+silver only (masked by `strong`, weighted by `w`).
  * l_weak   - clip-tag BCE on the attention-pooled prob, all tiers (Bronze's path).
  * l_cons   - mean-teacher consistency between augmented student and clean teacher.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config import CFG, CKPT, LR_ENC, LR_HEAD
from model import WavLMSED
from losses import bce_pos, rampup, wav_augment, mixup, ema_update
from evaluate import evaluate


def run_epochs(stu, tea, opt, sched, epochs, scaler, train_dl, val_dl, device,
               state, tag=""):
    """Run `epochs` of mean-teacher training, updating `state` (best score + gstep).

    `state` is a shared dict {"best": float, "gstep": int} so the two stages share a
    global step counter (which drives the EMA decay ramp) and a single best checkpoint.
    Saves the better of student/teacher whenever the val Combined improves.
    """
    for ep in range(epochs):
        stu.train(); tea.train()
        cw = CFG["max_cons_w"] * rampup(ep, CFG["rampup_epochs"])   # consistency weight
        acc = defaultdict(float); t0 = time.time()

        for batch in train_dl:
            inp = batch["wav"].to(device, non_blocking=True)
            lab = batch["lab"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)
            strong = batch["strong"].to(device)
            w = batch["w"].to(device)

            # Mixup with union labels (see losses.mixup).
            if torch.rand(1).item() < CFG["mixup_prob"]:
                inp, lab, tags = mixup(inp, lab, tags)

            # Student sees an augmented view; teacher sees the clean view (no grad).
            with autocast("cuda"):
                f_s, c_s = stu(wav_augment(inp), mask)
                with torch.no_grad():
                    f_t, c_t = tea(inp, mask)

            # Losses computed in fp32 outside autocast (BCE on fp16 sigmoids is unsafe).
            m3 = mask.unsqueeze(1)
            f_s = f_s.float().clamp(1e-6, 1 - 1e-6)
            c_s = c_s.float().clamp(1e-6, 1 - 1e-6)

            # l_strong: per-frame BCE averaged over valid frames, then a weighted mean
            # over only timestamped clips (`* strong` zeros Bronze, `* w` weights gold/silver).
            ls = (bce_pos(f_s, lab) * m3).sum(dim=(1, 2)) / (m3.sum(dim=(1, 2)) * lab.shape[1] + 1e-6)
            l_strong = (ls * strong * w).sum() / (strong * w).sum().clamp(min=1e-6)

            # l_weak: clip-tag BCE on the attention-pooled prob (all tiers -> Bronze in).
            l_weak = bce_pos(c_s, tags).mean()

            # l_cons: frame + clip MSE between student and teacher (tier-agnostic).
            f_t, c_t = f_t.float(), c_t.float()
            l_cons = (((f_s - f_t) ** 2) * m3).sum() / (m3.sum() * lab.shape[1] + 1e-6) \
                     + ((c_s - c_t) ** 2).mean()

            loss = l_strong + l_weak + cw * l_cons

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(stu.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

            state["gstep"] += 1
            ema_update(stu, tea, CFG["ema_decay"], step=state["gstep"])

            acc["s"] += float(l_strong.detach()); acc["w"] += float(l_weak.detach())
            acc["c"] += float(l_cons.detach()); acc["n"] += 1

        # Validate both student and teacher; save the better if it improves.
        n = max(1, acc["n"])
        ms = evaluate(stu, val_dl, device)
        mt = evaluate(tea, val_dl, device)
        print(f"{tag}ep{ep:02d} [{(time.time()-t0)/60:.1f}m] s {acc['s']/n:.3f} "
              f"w {acc['w']/n:.3f} c {acc['c']/n:.3f} | "
              f"stu {ms['score']:.4f} (F1 {ms['f1']:.3f} D {ms['dice']:.3f}) | "
              f"tea {mt['score']:.4f}")
        if max(ms["score"], mt["score"]) > state["best"]:
            state["best"] = max(ms["score"], mt["score"])
            which = "student" if ms["score"] >= mt["score"] else "teacher"
            torch.save({"model": (stu if which == "student" else tea).state_dict(),
                        "which": which, "score": state["best"], "cfg": CFG}, CKPT)
            print(f"   saved {which} {state['best']:.4f}")


def train(train_dl, val_dl, device, stage1_epochs, stage2_epochs):
    """Full two-stage training. Returns the shared `state` dict (best score + gstep)."""
    student = WavLMSED().to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    scaler = GradScaler("cuda")
    head = [p for n_, p in student.named_parameters() if not n_.startswith("enc.")]
    state = dict(best=-1.0, gstep=0)

    # --- Stage 1: encoder frozen, train GRU + heads only ---
    print(f"=== stage 1: encoder frozen, {stage1_epochs} epochs ===")
    for p in student.enc.parameters():
        p.requires_grad_(False)
    opt1 = torch.optim.AdamW(head, lr=LR_HEAD, weight_decay=CFG["weight_decay"])
    run_epochs(student, teacher, opt1, None, stage1_epochs, scaler,
               train_dl, val_dl, device, state, tag="s1 ")

    # --- Stage 2: unfreeze encoder with a small LR, cosine schedule ---
    print(f"=== stage 2: unfrozen, enc lr {LR_ENC}, {stage2_epochs} epochs ===")
    for p in student.enc.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW([{"params": student.enc.parameters(), "lr": LR_ENC},
                              {"params": head, "lr": LR_HEAD * 0.3}],
                             weight_decay=CFG["weight_decay"])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=stage2_epochs * max(1, len(train_dl)))
    run_epochs(student, teacher, opt2, sched2, stage2_epochs, scaler,
               train_dl, val_dl, device, state, tag="s2 ")

    print("\nbest Combined:", state["best"], "/ 2.0")
    del student, teacher
    torch.cuda.empty_cache()
    return state
