"""Output stage, quality checks and ASR transcription for the v-1 Track 2 baseline.

Output stage (`enhance`) applies three guards, each verified by `check_output`:
  1. NaN/Inf are zeroed so one bad frame cannot poison a file.
  2. Blend with the mixture: y = a*est + (1-a)*mix caps how much the model can damage
     speech, for a fraction of a dB.
  3. RMS restoration - the important one. A sigmoid mask only attenuates, and nothing
     in an SI-SDR loss penalises an output 30 dB too quiet (SI-SDR is scale-invariant,
     ASR is not). The submitted transcripts come from this audio.

Long clips are processed in overlapping windows with cross-fades so an 11 h test set
does not OOM.

ASR (`Transcriber`) wraps the mandated SraVaani-1.0 model - the audit checks that the
submitted transcripts match this model run on the submitted WAVs.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import soundfile as sf

from config import CFG, BLEND, MATCH_RMS, WAV_SUBTYPE, ASR_REPO
from data import device
from model import enhance_batch


MAX_ONESHOT = 30.0 * CFG["sr"]                    # longest clip run in a single pass


@torch.no_grad()
def _enh_core(model, wav, blend, match_rms):
    """Enhance a single (short) clip: mask -> blend -> RMS restore -> peak guard."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(device)
    est, _ = enhance_batch(model, x)
    y = est[0].float().cpu().numpy()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < len(wav):
        y = np.pad(y, (0, len(wav) - len(y)))
    y = y[:len(wav)]
    y = blend * y + (1.0 - blend) * wav           # blend guard
    if match_rms:                                  # restore level for ASR
        r_in = float(np.sqrt((wav ** 2).mean()))
        r_out = float(np.sqrt((y ** 2).mean()))
        if r_out > 1e-8 and r_in > 1e-8:
            y = y * (r_in / r_out)
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y / pk * 0.99
    return y.astype(np.float32)


def enhance(model, wav, blend=BLEND, match_rms=MATCH_RMS):
    """Enhance a clip of any length. Short clips go through in one pass; long clips use
    overlapping 30 s windows with cross-fades and overlap-add reconstruction."""
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) <= MAX_ONESHOT:
        return _enh_core(model, wav, blend, match_rms)
    win, ov = int(MAX_ONESHOT), int(1.0 * CFG["sr"])
    out = np.zeros(len(wav), np.float32)
    acc = np.zeros(len(wav), np.float32)
    for s0 in range(0, len(wav), win - ov):
        seg = wav[s0:s0 + win]
        if len(seg) < CFG["sr"] * 0.1:
            break
        y = _enh_core(model, seg, blend, match_rms)
        fade = np.ones(len(y), np.float32)
        k = min(ov, len(y) // 2)
        if k > 0 and s0 > 0:
            fade[:k] = np.linspace(0, 1, k)
        if k > 0 and s0 + win < len(wav):
            fade[-k:] = np.linspace(1, 0, k)
        out[s0:s0 + len(y)] += y * fade
        acc[s0:s0 + len(y)] += fade
    return (out / np.maximum(acc, 1e-6)).astype(np.float32)


def check_output(wav_in, wav_out):
    """Validate one enhanced clip: finiteness, length, dtype, mono, level, no-clip.

    `rms_ratio` should sit near 1.0; `level_ok`/`not_silent`/`no_clip` gate the
    submission (a silent or clipped output is a lost clip).
    """
    r_in = float(np.sqrt((wav_in ** 2).mean()))
    r_out = float(np.sqrt((wav_out ** 2).mean()))
    p = dict(finite=bool(np.isfinite(wav_out).all()),
             len_match=len(wav_out) == len(wav_in),
             dtype_f32=wav_out.dtype == np.float32, mono=wav_out.ndim == 1,
             rms_in=r_in, rms_out=r_out, rms_ratio=r_out / (r_in + 1e-12),
             peak=float(np.abs(wav_out).max()))
    p["level_ok"] = 0.5 <= p["rms_ratio"] <= 2.0
    p["not_silent"] = r_out > 1e-4
    p["no_clip"] = p["peak"] <= 0.999
    p["ok"] = all(p[k] for k in ["finite", "len_match", "dtype_f32", "mono",
                                 "level_ok", "not_silent", "no_clip"])
    return p


# ---------------------------------------------------------------------------
# Mandated ASR
# ---------------------------------------------------------------------------
class Transcriber:
    """Wraps the mandated SraVaani-1.0 ASR. The audit re-runs this model on the
    submitted WAVs, so the submitted transcripts must come from exactly this model."""

    def __init__(self, hf_token=None):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
        path = snapshot_download(ASR_REPO, token=hf_token)
        self.asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
        print("mandated ASR loaded:", path)

    def _tmp(self, wavs):
        """Write waveforms to temporary PCM-16 WAVs (the ASR reads file paths)."""
        ps = []
        for w in wavs:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(f.name, np.asarray(w, dtype=np.float32), CFG["sr"], subtype=WAV_SUBTYPE)
            ps.append(f.name)
        return ps

    @torch.no_grad()
    def transcribe_paths(self, paths):
        """Transcribe a list of WAV paths; falls back to one-by-one on batch failure."""
        try:
            hyps = self.asr.transcribe(paths, return_hypotheses=True)
            return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
        except Exception:
            out = []
            for p in paths:
                try:
                    h = self.asr.transcribe([p], return_hypotheses=True)[0]
                    out.append((h.text if hasattr(h, "text") else str(h)).strip())
                except Exception:
                    out.append("")
            return out

    def transcribe_many(self, wavs):
        ps = self._tmp(wavs)
        try:
            return self.transcribe_paths(ps)
        finally:
            for p in ps:
                try:
                    os.remove(p)
                except Exception:
                    pass

    def transcribe(self, w):
        return self.transcribe_many([w])[0]
