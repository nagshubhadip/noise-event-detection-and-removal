"""Losses + Mean-Teacher helpers for v-2.

- ``focal_bce_frame``  — supervised frame loss (same as v-1).
- ``consistency_mse``  — Mean-Teacher: MSE between student and (detached) teacher
  frame probabilities. Applied to both labelled and unlabelled batches.
- ``sigmoid_rampup``   — ramps the consistency weight up smoothly so the teacher
  is trusted only once it is worth trusting.
- ``ema_update``       — exponential moving average update of the teacher weights.
"""

import numpy as np
import torch
import torch.nn.functional as F


def focal_bce_frame(frame_logits, target, mask, sample_weight=None,
                    pos_weight=None, gamma=1.5):
    """Masked, class-weighted focal BCE over frames.

    ``pos_weight`` (per-channel) MUST be supplied by the caller in the modular
    layout (the original read a global); pass the tensor from
    ``data.estimate_pos_weight``.
    """
    if pos_weight is None:
        pos_weight = torch.ones(target.shape[1], device=target.device,
                                dtype=target.dtype)
    pos_weight = pos_weight.to(target.device, dtype=target.dtype)

    bce = F.binary_cross_entropy_with_logits(
        frame_logits, target, reduction="none",
        pos_weight=pos_weight.view(1, -1, 1))
    prob = torch.sigmoid(frame_logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    loss = bce * (1.0 - pt).pow(gamma)
    if sample_weight is not None:
        loss = loss * sample_weight.to(target.device, dtype=target.dtype).view(-1, 1, 1)
    m = mask.unsqueeze(1).to(target.device, dtype=target.dtype)
    return (loss * m).sum() / (m.sum() * target.shape[1] + 1e-6)


def consistency_mse(student_logits, teacher_logits, mask):
    """Masked MSE between student and detached-teacher frame probabilities."""
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits.detach())
    m = mask.unsqueeze(1).to(ps.device, dtype=ps.dtype)
    return ((ps - pt) ** 2 * m).sum() / (m.sum() * ps.shape[1] + 1e-6)


def sigmoid_rampup(current, rampup_length):
    """Gaussian-shaped ramp in [0, 1] over ``rampup_length`` epochs."""
    if rampup_length == 0:
        return 1.0
    p = np.clip(current / rampup_length, 0.0, 1.0)
    return float(np.exp(-5.0 * (1.0 - p) ** 2))


@torch.no_grad()
def ema_update(student, teacher, decay):
    """teacher = decay*teacher + (1-decay)*student (params); copy buffers."""
    for ts, tt in zip(student.parameters(), teacher.parameters()):
        tt.data.mul_(decay).add_(ts.data, alpha=1 - decay)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)
