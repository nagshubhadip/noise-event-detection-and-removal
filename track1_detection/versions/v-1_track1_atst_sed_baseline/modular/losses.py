"""Loss functions for v-1.

The training objective is a focal BCE on the frame logits (handles heavy class
imbalance and hard positives) plus a boundary-aware term that penalises
disagreement on the temporal derivative — i.e. the on/off *edges*, which is
exactly what the Event-F1 onset/offset tolerance measures.
"""

import torch
import torch.nn.functional as F


def focal_bce_frame(frame_logits, target, mask, sample_weight=None,
                    pos_weight=None, gamma=1.5):
    """Masked, class-weighted focal BCE over frames.

    Args:
        frame_logits: (B, C, T) raw logits.
        target:       (B, C, T) 0/1 labels.
        mask:         (B, T) 1 for valid frames.
        sample_weight:(B,) per-clip weight (annotation tier).
        pos_weight:   (C,) per-channel positive weighting.
        gamma:        focal exponent (0 = plain BCE).
    """
    if pos_weight is None:
        pos_weight = torch.ones(target.shape[1], device=target.device,
                                dtype=target.dtype)
    pos_weight = pos_weight.to(target.device, dtype=target.dtype)

    bce = F.binary_cross_entropy_with_logits(
        frame_logits, target, reduction="none",
        pos_weight=pos_weight.view(1, -1, 1))

    # Focal modulation: down-weight easy (confident, correct) frames.
    prob = torch.sigmoid(frame_logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    loss = bce * (1.0 - pt).pow(gamma)

    if sample_weight is not None:
        loss = loss * sample_weight.to(target.device, dtype=target.dtype).view(-1, 1, 1)

    m = mask.unsqueeze(1).to(target.device, dtype=target.dtype)
    return (loss * m).sum() / (m.sum() * target.shape[1] + 1e-6)


def boundary_aware_loss(frame_logits, target, mask):
    """MSE on the first temporal difference of prob vs. target (the edges)."""
    prob = torch.sigmoid(frame_logits)
    d_pred = prob[..., 1:] - prob[..., :-1]
    d_tgt = target[..., 1:] - target[..., :-1]
    m = mask.unsqueeze(1).to(target.device, dtype=target.dtype)[..., 1:]
    return ((d_pred - d_tgt) ** 2 * m).sum() / (m.sum() * target.shape[1] + 1e-6)


EDGE_LOSS_WEIGHT = 0.5


def combined_loss(frame_logits, target, mask, sample_weight=None,
                  pos_weight=None, edge_w=EDGE_LOSS_WEIGHT):
    """focal BCE + ``edge_w`` * boundary term (the training objective)."""
    base = focal_bce_frame(frame_logits, target, mask,
                           sample_weight=sample_weight, pos_weight=pos_weight)
    edge = boundary_aware_loss(frame_logits, target, mask)
    return base + edge_w * edge
