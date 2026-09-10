"""Official Track 2 scorer, reproduced verbatim.

Copied from the Evaluation page so the local number equals the leaderboard number.
Two details are easy to get wrong:

  * dWER is **pooled** - total edits over total reference words via `jiwer.wer` on
    lists - not a mean of per-clip ratios (averaging ratios lets one ASR repetition
    loop dominate a whole subset).
  * `normalize()` strips `<...>` and `[...]`, which matters because Vaani transcripts
    carry inline noise tags like `<horn> ... </horn>`.

Combined = SI-SDR(synthetic) + 100 * dWER(fraction).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np


def si_sdr(reference, enhanced):
    """Scale-Invariant SDR between a reference and an enhanced signal (dB, clipped).

    The reference is optimally rescaled onto the estimate, so SI-SDR is invariant to
    a global gain - which is exactly why the ASR-facing RMS restore is "free" here.
    """
    ref = np.asarray(reference, dtype=np.float64)
    enh = np.asarray(enhanced, dtype=np.float64)
    n = min(len(ref), len(enh))
    ref, enh = ref[:n], enh[:n]
    scale = np.dot(enh, ref) / np.dot(ref, ref)
    s_target = scale * ref
    e_noise = enh - s_target
    value = 10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))
    return float(np.clip(value, -100.0, 100.0))


# Strips inline tags like <horn>...</horn> and [noise] before scoring.
TAG_RE = re.compile(r"</?[^<>]*>|\[[^\[\]]*\]")


def normalize(text):
    """Lower-case, drop tags, replace punctuation with spaces, collapse whitespace."""
    s = TAG_RE.sub(" ", text or "")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.lower().split())


def delta_wer(gt, noisy_asr, submitted, clip_ids):
    """Pooled WER(noisy) - WER(enhanced) over `clip_ids`.

    Returns a fraction (x100 for percent). Empty hypotheses map to "@" so a blank
    line still counts as a full error rather than crashing jiwer.
    """
    from jiwer import wer as jiwer_wer
    refs, noisy, enh = [], [], []
    for cid in clip_ids:
        g = normalize(gt[cid])
        if not g:
            continue
        refs.append(g)
        noisy.append(normalize(noisy_asr.get(cid, "")) or "@")
        enh.append(normalize(submitted.get(cid, "")) or "@")
    return jiwer_wer(refs, noisy) - jiwer_wer(refs, enh)


def combined(si_sdr_synth, dwer_fraction):
    """The leaderboard's Combined metric."""
    return si_sdr_synth + 100.0 * dwer_fraction
