"""Encoders and the mask network for the v-1 Track 2 baseline.

The mask network is a BiGRU predicting a magnitude mask on the mixture spectrogram;
the signal is reconstructed with the mixture phase. Optional encoder features condition
the network via **FiLM at two depths** (measured to beat input-only concatenation).

Encoders:
  * `SraVaaniEnc` - loads the mandated SraVaani-1.0 TorchScript graph directly,
    reproducing its exact 128-mel front end. An ASR encoder is the right restraint
    here because dWER is half the metric and over-suppression deletes phonemes.
  * `WavLMEnc`    - fallback self-supervised encoder.
  * `"none"`      - spectrogram only (a useful control).
"""

from __future__ import annotations

import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, N_FFT, ASR_REPO, WAVLM_NAME, ENC_PROJ, T2_ENCODER
from data import device, stft, istft


class SraVaaniEnc(nn.Module):
    """Frozen SraVaani-1.0 FastConformer encoder loaded from its exported TorchScript.

    Bypasses the HF wrapper, custom modeling code and SentencePiece. `config.json`
    specifies feat_in=128 (not the NeMo default 80); `preproc.pt` carries the exact
    window and mel filterbank so the front end is reproduced rather than guessed.
    FastConformer subsamples 8x (80 ms frames), later interpolated up to the STFT rate.
    """

    def __init__(self, repo=ASR_REPO, token=None):
        super().__init__()
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, token=token)
        ts = sorted(glob.glob(os.path.join(path, "*.ts")))
        if not ts:
            raise RuntimeError("no TorchScript graph")
        self.enc = torch.jit.load(ts[0], map_location="cpu").eval()
        # Detect the graph's parameter dtype (fp16 export) and freeze it.
        self.pdtype = torch.float32
        for p in self.enc.parameters():
            self.pdtype = p.dtype
            break
        for p in self.enc.parameters():
            p.requires_grad_(False)
        # Reproduce the exact preprocessing front end.
        pp = torch.load(os.path.join(path, "preproc.pt"), map_location="cpu",
                        weights_only=False)
        self.register_buffer("win", pp["window"].float(), persistent=False)
        self.register_buffer("fb", pp["fb"].float(), persistent=False)
        prm = dict(pp.get("params", {}) or {})
        self.n_fft = int(prm.get("n_fft", 512))
        self.hop = int(prm.get("hop_length", prm.get("hop", 160)))
        self.preemph = float(prm.get("preemph", 0.97) or 0.0)
        self.guard = float(prm.get("log_zero_guard_value", 2 ** -24))
        self.norm_c = float(prm.get("normalize_constant", 1e-5))
        self._call = None                       # cache which call signature works
        self.eval()

    @staticmethod
    def _norm(x, seq_len, c):
        """Per-feature mean/var normalisation over valid frames (NeMo-style)."""
        B, _, T = x.shape
        steps = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = steps < seq_len.unsqueeze(1)
        denom = valid.sum(dim=1)
        mean = torch.where(valid.unsqueeze(1), x, torch.zeros_like(x)).sum(2) / denom.unsqueeze(1)
        var = torch.sum(torch.where(valid.unsqueeze(1), x - mean.unsqueeze(2),
                                    torch.zeros_like(x)) ** 2, dim=2) / (denom.unsqueeze(1) - 1.0)
        std = torch.sqrt(var)
        std = std.masked_fill(std.isnan(), 0.0) + c
        return (x - mean.unsqueeze(2)) / std.unsqueeze(2)

    @torch.no_grad()
    def forward(self, wav):
        """Waveform -> (B, T, dim) encoder features at the FastConformer frame rate."""
        x = wav
        if self.preemph:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]], dim=1)
        X = torch.stft(x, self.n_fft, self.hop, win_length=self.win.numel(),
                       window=self.win.to(x.device), center=True, return_complex=True)
        mel = torch.log(torch.matmul(self.fb.to(x.device), X.abs() ** 2) + self.guard)
        ln = torch.full((mel.shape[0],), mel.shape[-1], device=mel.device, dtype=torch.float32)
        feats = self._norm(mel, ln, self.norm_c).to(self.pdtype)
        li = torch.full((feats.shape[0],), feats.shape[-1], dtype=torch.long, device=x.device)
        # The TorchScript graph may expect keyword or positional args; try both once.
        trials = [("kw", lambda: self.enc(audio_signal=feats, length=li)),
                  ("pos", lambda: self.enc(feats, li))]
        if self._call:
            trials = [t for t in trials if t[0] == self._call]
        err = []
        for name, fn in trials:
            try:
                out = fn()
            except Exception as e:
                err.append(f"{name}: {e}")
                continue
            self._call = name
            out = out[0] if isinstance(out, (tuple, list)) else out
            return out.transpose(1, 2).float()
        raise RuntimeError("encoder call failed: " + " | ".join(err))


class WavLMEnc(nn.Module):
    """Frozen WavLM-base+ fallback encoder."""

    def __init__(self, name=WAVLM_NAME):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, wav):
        x = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-5)
        return self.m(x).last_hidden_state.float()


def build_encoder(hf_token=None):
    """Instantiate the configured encoder with graceful fallback.

    Returns (encoder_or_None, enc_dim). If all encoders fail, falls back to
    spectrogram-only so an encoder problem can never block the run.
    """
    if T2_ENCODER == "none":
        print("spectrogram only")
        return None, 0
    order = ([("sravaani", SraVaaniEnc), ("wavlm", WavLMEnc)]
             if T2_ENCODER == "sravaani" else [("wavlm", WavLMEnc)])
    for name, cls in order:
        try:
            print(f"loading {name}...")
            e = (cls(token=hf_token) if name == "sravaani" else cls()).to(device)
            with torch.no_grad():
                pr = e(torch.zeros(1, CFG["sr"], device=device))
            print(f"   OK -> {pr.shape[1]} frames, dim {pr.shape[-1]}")
            return e, pr.shape[-1]
        except Exception as ex:
            print(f"   {name} failed ({type(ex).__name__}: {str(ex)[:100]})")
    print("all encoders failed - spectrogram only")
    return None, 0


class MaskNet(nn.Module):
    """BiGRU magnitude-mask predictor with optional two-depth FiLM conditioning."""

    def __init__(self, enc, enc_dim, n_freq=N_FFT // 2 + 1, hid=256, proj=ENC_PROJ):
        super().__init__()
        self.enc = enc
        self.use_enc = enc is not None and enc_dim > 0
        if self.use_enc:
            self.enc_proj = nn.Linear(enc_dim, proj)
            self.film1 = nn.Linear(proj, 2 * hid)   # FiLM before the RNN
            self.film2 = nn.Linear(proj, 4 * hid)   # FiLM after the RNN (2 dirs)
        self.inp = nn.Linear(n_freq, hid)
        self.rnn = nn.GRU(hid, hid, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.mid = nn.Linear(2 * hid, 2 * hid)
        self.out = nn.Linear(2 * hid, n_freq)

    def forward(self, wav, mag_log, T):
        """Return a sigmoid magnitude mask (B, T, n_freq)."""
        h = self.inp(mag_log)
        if self.use_enc:
            e = self.enc_proj(self.enc(wav))
            # Align encoder frames to the STFT frame count before FiLM.
            e = F.interpolate(e.transpose(1, 2), size=T, mode="linear",
                              align_corners=False).transpose(1, 2)
            g, b_ = self.film1(e).chunk(2, -1)
            h = h * (1 + g) + b_
            h, _ = self.rnn(h)
            g, b_ = self.film2(e).chunk(2, -1)
            h = h * (1 + g) + b_
        else:
            h, _ = self.rnn(h)
        return torch.sigmoid(self.out(F.relu(self.mid(h))))


def enhance_batch(model, mix):
    """Apply the predicted mask to the mixture STFT and invert with mixture phase.

    Returns (estimated_waveform, mask). Gradients flow from the waveform back through
    the mask, so the SI-SDR loss trains the mask end to end.
    """
    X = stft(mix)
    m = model(mix, torch.log1p(X.abs()).transpose(1, 2), X.shape[-1]).transpose(1, 2)
    return istft(X * m, mix.shape[-1]), m


def core_model(hf_token=None):
    """Factory: build the encoder then the mask network. Returns (model, encoder, enc_dim)."""
    encoder, enc_dim = build_encoder(hf_token)
    model = MaskNet(encoder, enc_dim).to(device)
    return model, encoder, enc_dim
