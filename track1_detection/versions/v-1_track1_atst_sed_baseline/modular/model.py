"""The ATST-style SED model for v-1.

``ATSTSED`` wraps a pretrained transformer backbone (AST by default, or a real
ATST checkpoint) and adds a BiGRU + two output heads:

- ``strong`` — per-frame logits (the detection target)
- ``att``    — attention weights for clip-level (weak) pooling

For AST specifically, the patch tokens form a (freq x time) grid preceded by two
special tokens; we drop those and max-pool over frequency to get a clean
time-ordered sequence suitable for frame-level SED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoFeatureExtractor

from config import CFG, N_OUT, ATST_BACKBONE


class ATSTSED(nn.Module):
    def __init__(self, backbone_name=ATST_BACKBONE, n_out=N_OUT, rnn_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone_name = backbone_name
        self.model_type = getattr(self.backbone.config, "model_type", "")
        self.is_ast = self.model_type == "audio-spectrogram-transformer"
        # AST needs its own feature extractor (mel front-end); waveform models
        # take the raw signal directly.
        self.fe = AutoFeatureExtractor.from_pretrained(backbone_name) if self.is_ast else None

        d = self.backbone.config.hidden_size
        self.rnn = nn.GRU(d, rnn_dim, 2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)
        self.att = nn.Linear(2 * rnn_dim, n_out)

        # Precompute AST grid geometry so forward() can drop special tokens and
        # reshape the flattened patch sequence back into (freq, time).
        self.ast_n_special = 0
        self.ast_f_dim = self.ast_t_dim = None
        if self.is_ast:
            cfg = self.backbone.config
            patch = int(getattr(cfg, "patch_size", 16))
            fstride = int(getattr(cfg, "frequency_stride", 10))
            tstride = int(getattr(cfg, "time_stride", 10))
            n_mels = int(getattr(cfg, "num_mel_bins", 128))
            max_len = int(getattr(cfg, "max_length", 1024))
            self.ast_f_dim = (n_mels - patch) // fstride + 1
            self.ast_t_dim = (max_len - patch) // tstride + 1
            self.ast_n_special = 2  # [CLS] + [distill]
            print(f"Backbone: AST | freq={self.ast_f_dim} time={self.ast_t_dim} "
                  f"special={self.ast_n_special}")
        else:
            print(f"Backbone: {self.model_type or 'waveform'} (direct waveform)")

    def _encode(self, wav):
        """Run the backbone, returning ``last_hidden_state``.

        AST path: numpy -> feature extractor -> model. Waveform path: model(wav).
        """
        if self.is_ast:
            wav_np = wav.detach().float().cpu().numpy()
            feats = self.fe([w for w in wav_np], sampling_rate=CFG["sr"],
                            return_tensors="pt")
            out = self.backbone(feats["input_values"].to(wav.device))
        else:
            out = self.backbone(wav)
        return out.last_hidden_state

    def forward(self, wav, mask=None):
        """Return ``(frame_logits [B, C, T], clip_probs [B, C])``."""
        h = self._encode(wav)   # AST: (B, 2 + f*t, D) ; else (B, T, D)

        # AST: strip special tokens, fold back to (f, t, D), max over frequency.
        # max (not mean) keeps sharp transient onsets instead of smearing them.
        if self.is_ast and self.ast_f_dim is not None:
            h = h[:, self.ast_n_special:, :]
            f, t = self.ast_f_dim, self.ast_t_dim
            if h.shape[1] == f * t:
                h = h.reshape(h.shape[0], f, t, h.shape[-1]).amax(dim=1)  # (B, t, D)

        # Interpolate the sequence to the label frame count when they differ.
        T = mask.shape[1] if mask is not None else h.shape[1]
        if h.shape[1] != T:
            h = F.interpolate(h.transpose(1, 2), size=T, mode="linear",
                              align_corners=False).transpose(1, 2)

        h, _ = self.rnn(h)
        frame_logits = self.strong(h)
        frame = torch.sigmoid(frame_logits)

        # Attention pooling for the clip-level probability; masked frames get a
        # huge negative logit so softmax ignores padding.
        att = self.att(h)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        att = torch.softmax(att, dim=1)
        clip = (frame * att).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return frame_logits.transpose(1, 2), clip


def core_model(m):
    """Unwrap a possible ``nn.DataParallel`` to reach the real module."""
    return m.module if isinstance(m, nn.DataParallel) else m
