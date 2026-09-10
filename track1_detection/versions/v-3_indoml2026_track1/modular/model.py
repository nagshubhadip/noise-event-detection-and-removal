"""Model definition for the v-3 WavLM SED baseline.

`WavLMSED` is a WavLM-base+ encoder (last two transformer layers dropped) followed
by a BiGRU and two heads:

  * frame head  - per-frame event probabilities (the detection output).
  * attention head - softmax over time producing an attention-pooled clip
    probability. This is Multiple-Instance Learning: it is the only way Bronze
    clips (clip-level tags, no timing) can contribute a gradient.

WavLM emits 50 Hz frames, exactly the 20 ms label grid, so no interpolation is
needed during training - only a small safety interpolation for conv stride rounding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ENCODER_NAME, DROP_LAST_LAYERS, N_OUT


class WavLMSED(nn.Module):
    """WavLM encoder + BiGRU + frame/attention heads.

    Args:
        name: HF model id for the WavLM encoder.
        n_out: number of output channels (8 = any-noise + 7 categories).
        rnn_dim: hidden size per GRU direction.
        drop_layers: number of final transformer layers to drop (regularisation).
    """

    input_key = "wav"   # which batch key feeds the encoder (used by generic loops)

    def __init__(self, name=ENCODER_NAME, n_out=N_OUT, rnn_dim=256,
                 drop_layers=DROP_LAST_LAYERS):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        if drop_layers > 0:
            # Drop the final transformer blocks - they overfit the pretrain task.
            self.enc.encoder.layers = self.enc.encoder.layers[:-drop_layers]
        d = self.enc.config.hidden_size
        self.rnn = nn.GRU(d, rnn_dim, 2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)   # frame head
        self.att = nn.Linear(2 * rnn_dim, n_out)      # attention head

    def forward(self, x, mask=None):
        """Return (frame, clip):
          frame - (B, n_out, T) per-frame probabilities
          clip  - (B, n_out)    attention-pooled clip probabilities
        """
        h = self.enc(x).last_hidden_state             # (B, T_enc, d)
        T = mask.shape[1] if mask is not None else h.shape[1]
        if h.shape[1] != T:
            # Conv stride rounding can leave the sequence off by a frame or two;
            # align it to the label length with linear interpolation.
            h = F.interpolate(h.transpose(1, 2), size=T, mode="linear",
                              align_corners=False).transpose(1, 2)
        h, _ = self.rnn(h)
        frame = torch.sigmoid(self.strong(h))         # (B, T, n_out)
        a = self.att(h)                               # (B, T, n_out) attention logits
        if mask is not None:
            # Never let padding frames win the softmax.
            a = a.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        a = torch.softmax(a, dim=1)
        # Attention-pooled clip probability: weight frame probs by attention over time.
        clip = (frame * a).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return frame.transpose(1, 2), clip           # (B, n_out, T), (B, n_out)


def core_model(**kwargs):
    """Factory returning a fresh WavLMSED (keeps main.py free of import details)."""
    return WavLMSED(**kwargs)
