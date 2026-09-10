"""Loss terms and mean-teacher utilities for the v-3 WavLM baseline.

The per-step training loss combines three signals (see train.py for how they are
assembled):

  l_strong  - frame-level BCE, gold+silver only (the only timing supervision).
  l_weak    - clip-tag BCE on the attention-pooled probability, all tiers (MIL;
              Bronze's main path).
  l_cons    - mean-teacher consistency between the augmented student and the clean
              EMA teacher, tier-agnostic.

This module provides the primitive pieces: positive-weighted BCE, the consistency
ramp-up schedule, augmentation/mixup for the student, and the EMA weight update.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from config import CFG


def bce_pos(p, target, pw=None):
    """Positive-weighted binary cross-entropy on probabilities (not logits).

    Noise frames are a small minority, so plain BCE under-fires; `pw` (default
    CFG['pos_weight']) up-weights the positive term. Inputs are clamped to avoid
    log(0); this is computed in fp32 outside autocast because BCE on fp16 sigmoids
    can saturate to exactly 1.0.
    """
    pw = CFG["pos_weight"] if pw is None else pw
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(pw * target * torch.log(p) + (1 - target) * torch.log(1 - p))


def rampup(ep, n):
    """Sigmoid-shaped consistency ramp-up: 0 at epoch 0, ~1 by epoch `n`.

    The consistency term must not dominate before the teacher has stabilised, so its
    weight ramps up over the first `n` epochs.
    """
    return 1.0 if n == 0 else float(np.exp(-5 * (1 - np.clip(ep / n, 0, 1)) ** 2))


def wav_augment(x, n=2, tmax=None):
    """Time-masking augmentation for the student: zero out up to `n` short spans.

    Operates on a copy so the teacher still sees the clean waveform.
    """
    tmax = tmax or int(0.15 * CFG["sr"])
    x = x.clone()
    for b_ in range(x.shape[0]):
        for _ in range(n):
            t = random.randint(0, tmax)
            t0 = random.randint(0, max(0, x.shape[1] - t))
            x[b_, t0:t0 + t] = 0
    return x


def mixup(inp, lab, tags, alpha=CFG["mixup_alpha"]):
    """Mixup with UNION labels.

    A mixed clip genuinely contains both event sets, so labels/tags combine with
    `max` (not a soft blend). `lam` is pushed toward the larger side so the dominant
    clip keeps its identity.
    """
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    perm = torch.randperm(inp.size(0), device=inp.device)
    return (lam * inp + (1 - lam) * inp[perm],
            torch.maximum(lab, lab[perm]),
            torch.maximum(tags, tags[perm]))


@torch.no_grad()
def ema_update(stu, tea, decay, step=None):
    """Exponential-moving-average update of the teacher from the student.

    The `step`-based cap `min(1 - 1/(step+1), decay)` ramps the effective decay up
    from ~0, otherwise the teacher would sit near random init for the first ~1000
    steps - exactly while the consistency weight is ramping up. Non-float buffers
    (e.g. batch-norm counts) are copied verbatim.
    """
    if step is not None:
        decay = min(1 - 1 / (step + 1), decay)
    for ts, ss in zip(tea.state_dict().values(), stu.state_dict().values()):
        if ts.dtype.is_floating_point:
            ts.mul_(decay).add_(ss.detach(), alpha=1 - decay)
        else:
            ts.copy_(ss)
