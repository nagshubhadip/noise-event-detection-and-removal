"""Training and fine-tuning for the v-1 Track 2 baseline.

Two phases:
  * `train_on_sim` - train the mask net on simulated mixtures with a time-domain
    SI-SDR loss. Must beat the identity function by >= 0.5 dB or a submission is
    pointless.
  * `finetune_on_real` - close the domain gap by fine-tuning on the organisers' real
    (noisy, clean) validation pairs at a lower LR, holding out 15% for honest model
    selection.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import CFG, CKPT, N_SAMP, WORK
from data import device, read_audio
from model import enhance_batch
from losses import si_sdr_loss
from evaluate import si_sdr


@torch.no_grad()
def eval_sisdr(model, dl):
    """Return (mean input SI-SDR, mean output SI-SDR) over a loader."""
    model.eval()
    ins, outs = [], []
    for batch in dl:
        mix, clean = batch["mix"].to(device), batch["clean"].to(device)
        est, _ = enhance_batch(model, mix)
        for e, c, m_ in zip(est.float().cpu().numpy(), clean.cpu().numpy(), mix.cpu().numpy()):
            ins.append(si_sdr(c, m_))
            outs.append(si_sdr(c, e))
    return float(np.mean(ins)), float(np.mean(outs))


def train_on_sim(model, encoder, train_dl, val_dl, epochs):
    """Train the mask net on simulated mixtures; checkpoint the best SI-SDRi to CKPT."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(train_dl)))
    best = -1e9
    for ep in range(epochs):
        model.train()
        if encoder is not None:
            encoder.eval()                          # keep the frozen encoder in eval mode
        tot = n = 0
        t0 = time.time()
        for batch in train_dl:
            mix, clean = batch["mix"].to(device), batch["clean"].to(device)
            est, _ = enhance_batch(model, mix)
            loss = si_sdr_loss(est, clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
        si, so = eval_sisdr(model, val_dl)
        print(f"ep{ep:02d} [{(time.time()-t0)/60:.1f}m] loss {tot/max(1,n):.3f} | "
              f"SI-SDR {si:.2f} -> {so:.2f} ({so-si:+.2f} dB)")
        if so - si > best:
            best = so - si
            torch.save({"model": model.state_dict(), "si_sdri": best}, CKPT)
    print(f"\nbest SI-SDRi {best:+.2f} dB")
    assert best > 0.5, "model does not beat the identity function - raise N_MIX or EPOCHS"
    return best


class RealPairs(Dataset):
    """Fixed-length (mix, clean) pairs from the real validation clips."""

    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v = self.items[i]
        mix, clean = read_audio(v["path"]), read_audio(v["ref"])
        n = min(len(mix), len(clean))
        mix, clean = mix[:n], clean[:n]
        if n > N_SAMP:
            s = random.randint(0, n - N_SAMP) if self.train else 0
            mix, clean = mix[s:s + N_SAMP], clean[s:s + N_SAMP]
        elif n < N_SAMP:
            mix = np.pad(mix, (0, N_SAMP - n))
            clean = np.pad(clean, (0, N_SAMP - n))
        return dict(mix=torch.from_numpy(mix.astype(np.float32)),
                    clean=torch.from_numpy(clean.astype(np.float32)))


def finetune_on_real(model, encoder, val_items, bs,
                     ft_epochs=30, ft_lr=3e-4, ft_hold=0.15):
    """Fine-tune on real (noisy, clean) pairs; return (hold_items or None).

    Holds out `ft_hold` of the pairs so the later blend sweep and model selection stay
    honest. Saves the best held-out ABSOLUTE SI-SDR (the leaderboard metric).
    """
    pairs = [v for v in val_items if v["ref"] is not None]
    print(f"{len(pairs)} real (noisy, clean) pairs available")
    if len(pairs) < 20:
        print("too few pairs - skipping fine-tune")
        return None

    random.Random(0).shuffle(pairs)
    n_hold = max(20, int(ft_hold * len(pairs)))
    hold_items, fit_items = pairs[:n_hold], pairs[n_hold:]
    print(f"  {len(fit_items)} fine-tune / {len(hold_items)} held out")

    ft_dl = DataLoader(RealPairs(fit_items), batch_size=bs, shuffle=True,
                       num_workers=2, drop_last=True)
    hold_dl = DataLoader(RealPairs(hold_items, train=False), batch_size=bs,
                         shuffle=False, num_workers=2)

    prm = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(prm, lr=ft_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs * max(1, len(ft_dl)))
    si0, so0 = eval_sisdr(model, hold_dl)
    print(f"before fine-tune: held-out SI-SDR {si0:+.2f} -> {so0:+.2f} ({so0-si0:+.2f} dB)")

    best_abs = so0
    for ep in range(ft_epochs):
        model.train()
        if encoder is not None:
            encoder.eval()
        tot = n = 0
        for batch in ft_dl:
            mix, clean = batch["mix"].to(device), batch["clean"].to(device)
            est, _ = enhance_batch(model, mix)
            loss = si_sdr_loss(est, clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(prm, 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
        si, so = eval_sisdr(model, hold_dl)
        print(f"ft ep{ep:02d} loss {tot/max(1,n):.3f} | held-out SI-SDR "
              f"{si:+.2f} -> {so:+.2f} ({so-si:+.2f} dB)")
        if so > best_abs:
            best_abs = so
            torch.save({"model": model.state_dict(), "si_sdr": so}, WORK / "t2_model_real.pt")
    if (WORK / "t2_model_real.pt").exists():
        model.load_state_dict(torch.load(WORK / "t2_model_real.pt", weights_only=False)["model"])
    model.eval()
    print(f"\nbest held-out ABSOLUTE SI-SDR {best_abs:+.2f} dB")
    return hold_items
