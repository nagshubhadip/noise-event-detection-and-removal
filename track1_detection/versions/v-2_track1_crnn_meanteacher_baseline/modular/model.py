"""Backbones + the CRNN SED model for v-2.

Streams:
- ``ASTBackbone``        — patch-wise AST features (coarse ~100 ms), max-pooled
  over frequency. Gradient checkpointing keeps Stage-2 fine-tuning on 16 GB.
- ``ATSTFrameBackbone``  — optional real ATST-Frame (already ~40 ms, frame
  aligned); needs an external checkpoint + package.
- ``CNNBranch``          — log-mel CNN at a native 40 ms grid; supplies the local
  temporal detail that sharpens onset/offset boundaries.

``CRNN_SED`` fuses the sequence stream + CNN branch, runs a BiGRU, and emits
per-frame logits plus an attention-pooled clip probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoFeatureExtractor

from config import (CFG, N_OUT, USE_ATST_FRAME, AST_BACKBONE,
                    ATST_FRAME_REPO, ATST_FRAME_CKPT)


class ASTBackbone(nn.Module):
    """AST feature stream -> (B, T_ast, D). Drops CLS/distill, max-pools freq."""

    def __init__(self, name=AST_BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        try:
            # Trades compute for a big activation-memory saving.
            self.backbone.gradient_checkpointing_enable()
        except Exception as e:
            print("grad checkpointing not enabled:", e)
        cfg = self.backbone.config
        patch = int(getattr(cfg, "patch_size", 16))
        fstride = int(getattr(cfg, "frequency_stride", 10))
        tstride = int(getattr(cfg, "time_stride", 10))
        n_mels = int(getattr(cfg, "num_mel_bins", 128))
        max_len = int(getattr(cfg, "max_length", 1024))
        self.f_dim = (n_mels - patch) // fstride + 1
        self.t_dim = (max_len - patch) // tstride + 1
        self.n_special = 2                 # [CLS] + [distill]
        self.out_dim = cfg.hidden_size

    def forward(self, wav):
        wav_np = wav.detach().float().cpu().numpy()
        feats = self.fe([w for w in wav_np], sampling_rate=CFG["sr"], return_tensors="pt")
        h = self.backbone(feats["input_values"].to(wav.device)).last_hidden_state
        h = h[:, self.n_special:, :]                       # drop special tokens
        if h.shape[1] == self.f_dim * self.t_dim:          # (f, t) grid -> max over f
            h = h.reshape(h.shape[0], self.f_dim, self.t_dim, h.shape[-1]).amax(dim=1)
        return h                                           # (B, t_dim, D)


class ATSTFrameBackbone(nn.Module):
    """Optional real ATST-Frame stream (needs the Audio-WestlakeU code + ckpt).

    Steps to enable: clone audiossl/ATST-SED, download ``atst_as2M.ckpt``, then
    set ``USE_ATST_FRAME=1``, ``ATST_FRAME_REPO`` and ``ATST_FRAME_CKPT``.
    """

    def __init__(self, repo=ATST_FRAME_REPO, ckpt=ATST_FRAME_CKPT):
        super().__init__()
        if not repo or not ckpt:
            raise RuntimeError("ATST-Frame needs ATST_FRAME_REPO and ATST_FRAME_CKPT set.")
        import sys
        sys.path.insert(0, repo)
        from audiossl.methods.atstframe.embedding import load_model  # type: ignore
        self.atst = load_model(ckpt)
        self.out_dim = getattr(self.atst, "embed_dim", 768)

    def forward(self, wav):
        # Frame-level (B, T, D) embeddings.
        return self.atst.get_timestamp_embedding(wav)[0]


class CNNBranch(nn.Module):
    """Log-mel CNN -> (B, T_out, C). Pools freq to 1 and time by ``time_pool``."""

    def __init__(self, n_mels=CFG["n_mels"], out_ch=128, time_pool=CFG["time_pool"]):
        super().__init__()
        tp1 = 2 if time_pool >= 2 else 1        # split time pooling across 2 stages
        tp2 = time_pool // tp1

        def blk(ci, co, fp, tp):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.AvgPool2d((fp, tp)))

        self.net = nn.Sequential(
            blk(1, 32, 4, tp1),      # mel/4,  T/tp1
            blk(32, 64, 4, tp2),     # mel/16, T/time_pool
            blk(64, out_ch, 2, 1),   # mel/32
        )
        self.out_ch = out_ch

    def forward(self, mel):
        x = mel.unsqueeze(1)         # (B, 1, n_mels, T)
        x = self.net(x)              # (B, C, mel', T')
        x = x.mean(dim=2)            # collapse remaining freq -> (B, C, T')
        return x.transpose(1, 2)     # (B, T', C)


class CRNN_SED(nn.Module):
    """Sequence stream + CNN branch -> merge -> BiGRU -> strong + attention heads."""

    def __init__(self, n_out=N_OUT, rnn_dim=256, frames_out=None):
        super().__init__()
        # ``frames_out`` is the default output length when no mask is given.
        self.frames_out = frames_out
        self.use_atst_frame = USE_ATST_FRAME
        self.seq_backbone = ATSTFrameBackbone() if USE_ATST_FRAME else ASTBackbone()
        seq_dim = self.seq_backbone.out_dim

        self.cnn = CNNBranch()
        self.merge = nn.Linear(seq_dim + self.cnn.out_ch, 256)
        self.rnn = nn.GRU(256, rnn_dim, 2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)
        self.att = nn.Linear(2 * rnn_dim, n_out)

    @property
    def backbone(self):
        """The fine-tunable backbone module (for freeze/unfreeze in training)."""
        return (self.seq_backbone.backbone
                if hasattr(self.seq_backbone, "backbone") else self.seq_backbone)

    def forward(self, wav, mel, mask=None):
        T = mask.shape[1] if mask is not None else self.frames_out
        # Sequence stream -> resample to the output frame count.
        seq = self.seq_backbone(wav)
        seq = F.interpolate(seq.transpose(1, 2), size=T, mode="linear",
                            align_corners=False).transpose(1, 2)
        # CNN branch -> resample if needed.
        cnn = self.cnn(mel)
        if cnn.shape[1] != T:
            cnn = F.interpolate(cnn.transpose(1, 2), size=T, mode="linear",
                                align_corners=False).transpose(1, 2)
        # Fuse, recurse, predict.
        h = F.relu(self.merge(torch.cat([seq, cnn], dim=-1)))
        h, _ = self.rnn(h)
        frame_logits = self.strong(h)                # (B, T, C)
        att = self.att(h)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        att = torch.softmax(att, dim=1)
        clip = (torch.sigmoid(frame_logits) * att).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return frame_logits.transpose(1, 2), clip    # (B, C, T), (B, C)


def core_model(m):
    """Unwrap a possible DataParallel wrapper."""
    return m.module if isinstance(m, nn.DataParallel) else m
