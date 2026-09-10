"""Training loss for the v-1 Track 2 baseline.

The loss IS the metric: SI-SDR in the time domain. The mask is applied to the complex
STFT and inverted (see model.enhance_batch), so gradients flow from the reconstructed
waveform back through the mask.
"""

from __future__ import annotations

import torch


def si_sdr_loss(est, ref, eps=1e-8):
    """Negative mean SI-SDR (a loss to minimise) between estimate and reference.

    Both signals are mean-centred; the reference is optimally rescaled onto the
    estimate (scale-invariant), the residual is the noise term, and the ratio is
    turned into dB. Negated so that higher SI-SDR = lower loss.
    """
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    t = a * ref
    n = est - t
    return -(10 * torch.log10(((t ** 2).sum(-1) + eps) / ((n ** 2).sum(-1) + eps))).mean()
