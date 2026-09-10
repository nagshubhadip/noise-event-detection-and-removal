# Datathon@IndoML 2026 — Track 1 Baseline v2
# CRNN (AST + parallel CNN branch) + Mean-Teacher semi-supervised training.
#
# WHY THIS FILE EXISTS
# --------------------
# The v1 file (track1_atst_sed_baseline.py) uses ONLY the patch-wise AST backbone.
# AST produces ~100 ms time tokens that are then interpolated to the SED grid, so
# onset/offset boundaries physically cannot land inside the 50-80 ms Event-F1
# tolerance. Error analysis on v1 confirmed this: ~54% of GT events were
# "boundary-miss" (overlap correct, but edges outside tolerance) and Event-F1 was
# capped at ~0.25 while Dice was ~0.66.
#
# Following "Fine-tune the pretrained ATST model for SED" (Shao et al., ICASSP 2024),
# the fix is (1) a frame-wise / fine-resolution feature stream and (2) heavy use of
# unlabelled (in-domain) data via a Mean-Teacher consistency loss. This file adds:
#   (b)  A parallel CNN branch on a log-mel spectrogram at a 40 ms output grid
#        (matching ATST-Frame's resolution), fused with the AST features. The CNN
#        supplies the local temporal detail that sharpens boundaries.
#   +    Mean-Teacher (EMA teacher) semi-supervised loss so the bronze / no-timestamp
#        clips (previously DISCARDED via USE_STRONG_ONLY) are used as unlabelled data.
#
#   (a)  Optional real ATST-Frame backbone hook (USE_ATST_FRAME). It is disabled by
#        default because it needs the Audio-WestlakeU checkpoint + package (see the
#        ATSTFrameBackbone docstring). When enabled it REPLACES the AST stream.
#
# The data-reading, label rasterization, decoding and evaluation logic mirror v1.

# ===== Cell 1 (code): imports + config =====
import os, io, json, math, random, zipfile, gc
from pathlib import Path
from collections import Counter
from copy import deepcopy

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, Audio
from transformers import AutoModel, AutoFeatureExtractor

# Reduce CUDA fragmentation (helps the exact OOM message you hit)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_AMP = torch.cuda.is_available()

CATS = [
    "animal", "vehicle_traffic", "baby_child", "singing_music",
    "phone_signal_alarm", "appliance_machine", "human_non_speech"
]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + len(CATS)      # channel 0 = any-noise, 1..7 = categories

CFG = dict(
    sr=16000,
    n_mels=128,
    n_fft=1024,
    hop=160,               # 10 ms mel frames
    time_pool=4,           # CNN pools time by 4 -> 40 ms output frames (ATST-Frame-like)
    clip_sec=10.0,         # finalized from data below (AST window cap = 10 s)
    val_frac=0.1,
    batch_size=4 if N_GPU < 2 else 12,   # lower: Stage-2 runs 4 fwd passes on unfrozen AST
    accum_steps=2,                        # effective batch = batch_size * accum_steps
    unlab_batch_frac=1.0,                 # unlabeled batch = ceil(batch_size * this)
    num_workers=2,
    stage1_epochs=6,       # frozen backbone, train CRNN head
    stage2_epochs=20,      # unfreeze + heavy Mean-Teacher
    lr_head=1e-3,
    lr_backbone=5e-5,
    weight_decay=1e-2,
    # Mean-Teacher
    ema_decay=0.999,
    mt_max_weight=30.0,    # ramp-up target for consistency weight (paper uses very high)
    mt_rampup_epochs=8,
)

# Backbone selection ---------------------------------------------------------
# (a) Real ATST-Frame: set USE_ATST_FRAME=True and provide the paths. Disabled by
#     default (needs external checkpoint + package). When True, the AST stream is
#     replaced by ATST-Frame features (already ~40 ms, frame-aligned).
USE_ATST_FRAME = os.environ.get("USE_ATST_FRAME", "0") == "1"
ATST_FRAME_CKPT = os.environ.get("ATST_FRAME_CKPT", "")      # e.g. .../atst_as2M.ckpt
ATST_FRAME_REPO = os.environ.get("ATST_FRAME_REPO", "")      # path to cloned audiossl/ATST-SED
AST_BACKBONE = os.environ.get("AST_BACKBONE", "MIT/ast-finetuned-audioset-10-10-0.4593")

print(f"device={device} | gpus={N_GPU} | amp={USE_AMP} | use_atst_frame={USE_ATST_FRAME}")

# ===== Cell 2 (code): data reading (mirrors v1) =====
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN or HF_TOKEN == "PASTE_YOUR_HF_TOKEN_HERE":
    raise RuntimeError("Set HF_TOKEN before running.")
try:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("HF auth: OK")
except Exception as e:
    print("HF login warning:", e)

import tempfile

def decode_audio(a, target_sr=CFG['sr']):
    w = sr = None
    if isinstance(a, dict):
        if a.get('array') is not None:
            w, sr = np.asarray(a['array'], dtype=np.float32), a['sampling_rate']
        elif a.get('bytes') is not None:
            b = a['bytes']
            try:
                w, sr = sf.read(io.BytesIO(b), dtype='float32', always_2d=False)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix='.audio', delete=True) as tmp:
                    tmp.write(b); tmp.flush()
                    w, sr = librosa.load(tmp.name, sr=None, mono=True)
        elif a.get('path'):
            try:
                w, sr = sf.read(a['path'], dtype='float32', always_2d=False)
            except Exception:
                w, sr = librosa.load(a['path'], sr=None, mono=True)
    elif isinstance(a, (str, Path)) and str(a):
        try:
            w, sr = sf.read(str(a), dtype='float32', always_2d=False)
        except Exception:
            w, sr = librosa.load(str(a), sr=None, mono=True)
    if w is None:
        raise ValueError(f'cannot decode audio payload type={type(a)}')
    if w.ndim > 1:
        w = w.mean(axis=0) if w.shape[0] < w.shape[1] else w.mean(axis=1)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(w, dtype=np.float32)

def spans_from(ex):
    spans = []
    for s in (ex.get('NoiseSubCategoryTimeStamp') or []):
        try:
            st, en = float(s.get('start')), float(s.get('end'))
        except Exception:
            continue
        if en > st:
            spans.append((st, en, s.get('category')))
    return sorted(spans)

def build_labels(spans, n_frames_100hz, has_strong=True):
    y = np.zeros((N_OUT, n_frames_100hz), dtype=np.uint8)
    if has_strong:
        for st, en, cat in spans:
            a = max(0, int(round(st * 100)))
            b = min(n_frames_100hz, int(round(en * 100)))
            if b <= a:
                continue
            y[0, a:b] = 1
            ci = CAT2IDX.get(cat)
            if ci is not None:
                y[1 + ci, a:b] = 1
    return y

def load_train_subset(max_rows=5000):
    ds = load_dataset(REPO, split='train', token=HF_TOKEN)
    ds = ds.cast_column('audio', Audio(decode=False))
    print('audio decode mode: datasets.Audio decode=False (manual decode)')
    rows, skipped = [], 0
    for i, ex in enumerate(ds):
        if i >= max_rows:
            break
        q = ex.get('annotationQuality')
        has_strong = q in ('verified_timestamps', 'unverified_timestamps')
        try:
            wav = decode_audio(ex['audio'])
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f'skip {i}: {e}')
            continue
        spans = spans_from(ex)
        nf = 1 + len(wav) // CFG['hop']
        lab = build_labels(spans, nf, has_strong=has_strong)
        rows.append(dict(wav=wav, lab=lab, quality=q, has_strong=has_strong))
    if not rows:
        raise RuntimeError('No training rows decoded. Check HF token/access.')
    print('loaded', len(rows), 'rows | skipped', skipped, '| mix=', Counter([r['quality'] for r in rows]))
    return rows

# ===== Cell 3 (code): data pipeline + labeled/unlabeled split =====
PAD_EXTRA_SEC = 2.0
AST_MAX_CLIP_SEC = 10.0
QUALITY_WEIGHTS = {'verified_timestamps': 1.00, 'unverified_timestamps': 0.75, 'no_timestamps': 0.25}

def prep_wav(w):
    return (w - w.mean()) / (w.std() + 1e-5)

rows = load_train_subset(max_rows=4000)
dur_sec = np.asarray([len(r['wav']) / CFG['sr'] for r in rows], dtype=np.float32)
mx = float(dur_sec.max()) if len(dur_sec) else CFG['clip_sec']
p95 = float(np.percentile(dur_sec, 95)) if len(dur_sec) else CFG['clip_sec']
print(f'audio duration stats (sec) | p95={p95:.2f} | max={mx:.2f}')
CFG['clip_sec'] = float(min(math.ceil(min(p95, mx) + PAD_EXTRA_SEC), AST_MAX_CLIP_SEC))

N_SAMP = int(CFG['clip_sec'] * CFG['sr'])
FRAMES_100HZ = int(CFG['clip_sec'] * CFG['sr'] / CFG['hop'])
FRAMES_OUT = FRAMES_100HZ // CFG['time_pool']
FRAME_SEC = CFG['hop'] * CFG['time_pool'] / CFG['sr']
MEL_FRAMES = FRAMES_100HZ
print(f"clip_sec={CFG['clip_sec']} | N_SAMP={N_SAMP} | FRAMES_OUT={FRAMES_OUT} | FRAME_SEC={FRAME_SEC:.4f}s")

# labeled = strong (verified/unverified timestamps); unlabeled = bronze (no timestamps)
labeled_rows = [r for r in rows if r.get('has_strong', False)]
unlabeled_rows = [r for r in rows if not r.get('has_strong', False)]
print(f'labeled(strong)={len(labeled_rows)} | unlabeled(bronze)={len(unlabeled_rows)}')

def logmel(wav):
    # (n_mels, T) log-mel on the 10 ms grid; T == MEL_FRAMES after pad/crop
    m = librosa.feature.melspectrogram(
        y=wav, sr=CFG['sr'], n_fft=CFG['n_fft'], hop_length=CFG['hop'],
        n_mels=CFG['n_mels'], power=2.0, center=True)
    m = librosa.power_to_db(m, ref=np.max).astype(np.float32)
    m = (m - m.mean()) / (m.std() + 1e-5)
    return m

def _fit_wav(wav, train):
    if len(wav) > N_SAMP:
        s = random.randint(0, len(wav) - N_SAMP) if train else 0
        return wav[s:s + N_SAMP], s, N_SAMP
    return np.pad(wav, (0, N_SAMP - len(wav))), 0, len(wav)

def _labels_20ms(lab, s):
    lab = lab[:, s // CFG['hop']: s // CFG['hop'] + FRAMES_100HZ]
    if lab.shape[1] < FRAMES_100HZ:
        lab = np.pad(lab, ((0, 0), (0, FRAMES_100HZ - lab.shape[1])))
    lab = lab[:, :FRAMES_100HZ]
    return lab.reshape(lab.shape[0], FRAMES_OUT, CFG['time_pool']).max(axis=2)

class LabeledDataset(Dataset):
    def __init__(self, rows, train=True):
        self.rows, self.train = rows, train
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        wav, s, valid = _fit_wav(r['wav'], self.train)
        lab20 = _labels_20ms(r['lab'].astype(np.float32), s)
        m = np.zeros(FRAMES_OUT, dtype=np.float32)
        n_valid = min(FRAMES_OUT, max(1, int(math.ceil(valid / CFG['sr'] / FRAME_SEC))))
        m[:n_valid] = 1.0
        wav = prep_wav(wav.astype(np.float32))
        return dict(
            wav=torch.from_numpy(wav),
            mel=torch.from_numpy(logmel(wav)),
            lab=torch.from_numpy(lab20.astype(np.float32)),
            mask=torch.from_numpy(m),
            sw=torch.tensor(float(QUALITY_WEIGHTS.get(r.get('quality'), 1.0)), dtype=torch.float32),
        )

class UnlabeledDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        wav, s, valid = _fit_wav(self.rows[i]['wav'], True)
        m = np.zeros(FRAMES_OUT, dtype=np.float32)
        n_valid = min(FRAMES_OUT, max(1, int(math.ceil(valid / CFG['sr'] / FRAME_SEC))))
        m[:n_valid] = 1.0
        wav = prep_wav(wav.astype(np.float32))
        return dict(
            wav=torch.from_numpy(wav),
            mel=torch.from_numpy(logmel(wav)),
            mask=torch.from_numpy(m),
        )

random.Random(SEED).shuffle(labeled_rows)
n_val = max(1, int(len(labeled_rows) * CFG['val_frac']))
val_rows, train_rows = labeled_rows[:n_val], labeled_rows[n_val:]

def estimate_pos_weight(train_rows):
    pos = np.zeros(N_OUT); total = np.zeros(N_OUT)
    for r in train_rows:
        lab20 = _labels_20ms(r['lab'].astype(np.float32), 0)
        n_valid = min(FRAMES_OUT, max(1, int(math.ceil(min(len(r['wav']), N_SAMP) / CFG['sr'] / FRAME_SEC))))
        y = lab20[:, :n_valid]
        pos += y.sum(axis=1); total += n_valid
    pw = np.clip(np.maximum(total - pos, 1.0) / np.maximum(pos, 1.0), 1.0, 50.0)
    return pw.astype(np.float32)

LOSS_POS_WEIGHT = torch.tensor(estimate_pos_weight(train_rows), dtype=torch.float32, device=device)
print('pos_weight:', [round(float(x), 2) for x in LOSS_POS_WEIGHT.cpu().tolist()])

_UNLAB_BS = max(1, int(math.ceil(CFG['batch_size'] * CFG['unlab_batch_frac'])))
train_dl = DataLoader(LabeledDataset(train_rows, True), batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'], drop_last=True)
val_dl = DataLoader(LabeledDataset(val_rows, False), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
unlab_dl = DataLoader(UnlabeledDataset(unlabeled_rows), batch_size=_UNLAB_BS, shuffle=True, num_workers=CFG['num_workers'], drop_last=True) if unlabeled_rows else None
print(f'train={len(train_rows)} | val={len(val_rows)} | unlab={len(unlabeled_rows)} | batch={CFG["batch_size"]} accum={CFG["accum_steps"]}')

# ===== Cell 4 (code): backbones =====
class ASTBackbone(nn.Module):
    """AST feature stream -> (B, T_ast, D). Drops CLS/distill, max-pools frequency."""
    def __init__(self, name=AST_BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        # Gradient checkpointing trades compute for a large activation-memory saving,
        # which is what makes Stage-2 fine-tuning of the ~87M AST fit on a 16GB GPU.
        try:
            self.backbone.gradient_checkpointing_enable()
        except Exception as e:
            print('grad checkpointing not enabled:', e)
        cfg = self.backbone.config
        patch = int(getattr(cfg, 'patch_size', 16))
        fstride = int(getattr(cfg, 'frequency_stride', 10))
        tstride = int(getattr(cfg, 'time_stride', 10))
        n_mels = int(getattr(cfg, 'num_mel_bins', 128))
        max_len = int(getattr(cfg, 'max_length', 1024))
        self.f_dim = (n_mels - patch) // fstride + 1
        self.t_dim = (max_len - patch) // tstride + 1
        self.n_special = 2
        self.out_dim = cfg.hidden_size

    def forward(self, wav):
        wav_np = wav.detach().float().cpu().numpy()
        feats = self.fe([w for w in wav_np], sampling_rate=CFG['sr'], return_tensors='pt')
        h = self.backbone(feats['input_values'].to(wav.device)).last_hidden_state
        h = h[:, self.n_special:, :]
        if h.shape[1] == self.f_dim * self.t_dim:
            h = h.reshape(h.shape[0], self.f_dim, self.t_dim, h.shape[-1]).amax(dim=1)
        return h  # (B, t_dim, D)

class ATSTFrameBackbone(nn.Module):
    """Optional REAL ATST-Frame stream (option a).

    Requires the Audio-WestlakeU code + checkpoint, which are NOT pip-installable:
      1. git clone https://github.com/Audio-WestlakeU/audiossl (and/or ATST-SED)
      2. download atst_as2M.ckpt (Google Drive, see ATST-SED README)
      3. set env: USE_ATST_FRAME=1  ATST_FRAME_REPO=/path/to/repo  ATST_FRAME_CKPT=/path/to/atst_as2M.ckpt

    ATST-Frame already outputs ~40 ms frame-aligned features, so no CNN branch is
    needed for resolution -- but we still fuse one for local detail if present.
    """
    def __init__(self, repo=ATST_FRAME_REPO, ckpt=ATST_FRAME_CKPT):
        super().__init__()
        if not repo or not ckpt:
            raise RuntimeError("ATST-Frame needs ATST_FRAME_REPO and ATST_FRAME_CKPT set.")
        import sys
        sys.path.insert(0, repo)
        # NOTE: exact import path depends on the repo layout; commonly:
        #   from audiossl.methods.atstframe.embedding import load_model, get_scene_embedding
        # We import lazily and store a callable that maps waveform -> (B, T, D).
        from audiossl.methods.atstframe.embedding import load_model  # type: ignore
        self.atst = load_model(ckpt)
        self.out_dim = getattr(self.atst, 'embed_dim', 768)

    def forward(self, wav):
        # ATST-Frame expects (B, 1, n_samples) or log-mel per their preprocessing;
        # get_timestamp_embedding returns frame-level (B, T, D).
        return self.atst.get_timestamp_embedding(wav)[0]

class CNNBranch(nn.Module):
    """Local CRNN CNN on log-mel -> (B, T_out, C). Pools frequency to 1 and time by
    time_pool (=4) giving a native 40 ms grid -> supplies sharp boundaries."""
    def __init__(self, n_mels=CFG['n_mels'], out_ch=128, time_pool=CFG['time_pool']):
        super().__init__()
        # time pooling split across two stages to reach total factor time_pool
        tp1 = 2 if time_pool >= 2 else 1
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
        x = mel.unsqueeze(1)                 # (B,1,n_mels,T)
        x = self.net(x)                      # (B,C,mel',T')
        x = x.mean(dim=2)                    # collapse remaining freq -> (B,C,T')
        return x.transpose(1, 2)             # (B,T',C)

class CRNN_SED(nn.Module):
    def __init__(self, n_out=N_OUT, rnn_dim=256):
        super().__init__()
        self.use_atst_frame = USE_ATST_FRAME
        if self.use_atst_frame:
            self.seq_backbone = ATSTFrameBackbone()
            seq_dim = self.seq_backbone.out_dim
        else:
            self.seq_backbone = ASTBackbone()
            seq_dim = self.seq_backbone.out_dim
        self.cnn = CNNBranch()
        merge_dim = seq_dim + self.cnn.out_ch
        self.merge = nn.Linear(merge_dim, 256)
        self.rnn = nn.GRU(256, rnn_dim, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)
        self.att = nn.Linear(2 * rnn_dim, n_out)

    @property
    def backbone(self):
        return self.seq_backbone.backbone if hasattr(self.seq_backbone, 'backbone') else self.seq_backbone

    def forward(self, wav, mel, mask=None):
        T = mask.shape[1] if mask is not None else FRAMES_OUT
        seq = self.seq_backbone(wav)                              # (B, t_seq, D)
        seq = F.interpolate(seq.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
        cnn = self.cnn(mel)                                       # (B, t_cnn, C)
        if cnn.shape[1] != T:
            cnn = F.interpolate(cnn.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
        h = torch.cat([seq, cnn], dim=-1)
        h = F.relu(self.merge(h))
        h, _ = self.rnn(h)
        frame_logits = self.strong(h)                            # (B,T,C)
        att = self.att(h)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        att = torch.softmax(att, dim=1)
        clip = (torch.sigmoid(frame_logits) * att).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return frame_logits.transpose(1, 2), clip                # (B,C,T), (B,C)

def core_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

# ===== Cell 5 (code): losses =====
def focal_bce_frame(frame_logits, target, mask, sample_weight=None, pos_weight=None, gamma=1.5):
    if pos_weight is None:
        pos_weight = LOSS_POS_WEIGHT
    pos_weight = pos_weight.to(target.device, dtype=target.dtype)
    bce = F.binary_cross_entropy_with_logits(frame_logits, target, reduction='none',
                                             pos_weight=pos_weight.view(1, -1, 1))
    prob = torch.sigmoid(frame_logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    loss = bce * (1.0 - pt).pow(gamma)
    if sample_weight is not None:
        loss = loss * sample_weight.to(target.device, dtype=target.dtype).view(-1, 1, 1)
    m = mask.unsqueeze(1).to(target.device, dtype=target.dtype)
    return (loss * m).sum() / (m.sum() * target.shape[1] + 1e-6)

def consistency_mse(student_logits, teacher_logits, mask):
    # Mean-Teacher: MSE between student & teacher FRAME PROBABILITIES (paper uses MSE)
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits.detach())
    m = mask.unsqueeze(1).to(ps.device, dtype=ps.dtype)
    return ((ps - pt) ** 2 * m).sum() / (m.sum() * ps.shape[1] + 1e-6)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    p = np.clip(current / rampup_length, 0.0, 1.0)
    return float(np.exp(-5.0 * (1.0 - p) ** 2))

@torch.no_grad()
def ema_update(student, teacher, decay):
    for ts, tt in zip(student.parameters(), teacher.parameters()):
        tt.data.mul_(decay).add_(ts.data, alpha=1 - decay)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)

# ===== Cell 6 (code): build student + teacher =====
free_memory()
student = CRNN_SED().to(device)
teacher = deepcopy(student).to(device)
for p in teacher.parameters():
    p.requires_grad = False
print('student params(M):', round(sum(p.numel() for p in student.parameters()) / 1e6, 2))

# ===== Cell 7 (code): training (Stage 1 frozen -> Stage 2 fine-tune + Mean-Teacher) =====
from tqdm.auto import tqdm
AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

def run_epoch(student, teacher, ep, opt, mt_weight, use_mt):
    student.train()
    tr = []
    accum = max(1, CFG['accum_steps'])
    unlab_iter = iter(unlab_dl) if (use_mt and unlab_dl is not None) else None
    pbar = tqdm(train_dl, desc=f"ep{ep:02d}", leave=False)
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(pbar):
        wav = batch['wav'].to(device); mel = batch['mel'].to(device)
        lab = batch['lab'].to(device); m = batch['mask'].to(device)
        sw = batch['sw'].to(device)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            s_logits, _ = student(wav, mel, m)
            sup = focal_bce_frame(s_logits.float(), lab.float(), m.float(), sample_weight=sw)
            loss = sup
            if use_mt:
                with torch.no_grad():
                    t_logits, _ = teacher(wav, mel, m)
                cons = consistency_mse(s_logits.float(), t_logits.float(), m.float())
                if unlab_iter is not None:
                    try:
                        ub = next(unlab_iter)
                    except StopIteration:
                        unlab_iter = iter(unlab_dl); ub = next(unlab_iter)
                    uw = ub['wav'].to(device); um = ub['mel'].to(device); umask = ub['mask'].to(device)
                    us_logits, _ = student(uw, um, umask)
                    with torch.no_grad():
                        ut_logits, _ = teacher(uw, um, umask)
                    cons = cons + consistency_mse(us_logits.float(), ut_logits.float(), umask.float())
                loss = sup + mt_weight * cons
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if use_mt:
                ema_update(student, teacher, CFG['ema_decay'])
        tr.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=f"{tr[-1]:.4f}")
        del wav, mel, lab, m, sw, loss
    free_memory()
    return float(np.mean(tr)) if tr else 0.0

@torch.no_grad()
def eval_loss(model):
    model.eval(); vals = []
    for batch in val_dl:
        wav = batch['wav'].to(device); mel = batch['mel'].to(device)
        lab = batch['lab'].to(device); m = batch['mask'].to(device)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            logits, _ = model(wav, mel, m)
            vals.append(float(focal_bce_frame(logits.float(), lab.float(), m.float()).cpu()))
    return float(np.mean(vals)) if vals else 0.0

sc = core_model(student)
# Stage 1: freeze the pretrained sequence backbone, train CRNN head only
for p in sc.seq_backbone.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=CFG['lr_head'], weight_decay=CFG['weight_decay'])
print(f"[Stage 1] frozen backbone, {CFG['stage1_epochs']} epochs")
for ep in range(CFG['stage1_epochs']):
    trl = run_epoch(student, teacher, ep, opt, 0.0, use_mt=False)
    print(f"[S1][{ep+1}/{CFG['stage1_epochs']}] train={trl:.4f} val={eval_loss(student):.4f}")

# sync teacher to the stage-1 student (paper: model init matters for pseudo-labels)
teacher = deepcopy(student).to(device)
for p in teacher.parameters():
    p.requires_grad = False

# Stage 2: unfreeze backbone + heavy Mean-Teacher on unlabeled data
free_memory()
for p in sc.seq_backbone.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([
    {'params': sc.seq_backbone.parameters(), 'lr': CFG['lr_backbone']},
    {'params': [p for n, p in sc.named_parameters() if not n.startswith('seq_backbone.')], 'lr': CFG['lr_head'] * 0.3},
], weight_decay=CFG['weight_decay'])
print(f"[Stage 2] fine-tune + Mean-Teacher, {CFG['stage2_epochs']} epochs")
for ep in range(CFG['stage2_epochs']):
    w = CFG['mt_max_weight'] * sigmoid_rampup(ep, CFG['mt_rampup_epochs'])
    trl = run_epoch(student, teacher, ep, opt, w, use_mt=True)
    print(f"[S2][{ep+1}/{CFG['stage2_epochs']}] train={trl:.4f} val={eval_loss(student):.4f} mt_w={w:.2f}")

torch.save({'student': sc.state_dict(), 'cfg': CFG, 'use_atst_frame': USE_ATST_FRAME}, 'track1_crnn_mt.pt')
print('saved -> track1_crnn_mt.pt')

# ===== Cell 8 (code): inference + decoding (mirrors v1; teacher used at test) =====
from scipy.ndimage import median_filter
infer_model = teacher  # EMA teacher is the better inference model in Mean-Teacher

def prob_to_events(p, thr=0.5, med=5, frame_sec=FRAME_SEC, min_dur=0.05, merge_gap=0.0):
    b = (np.asarray(p) >= thr).astype(np.uint8)
    if med > 1:
        b = median_filter(b, size=med, mode='nearest')
    events, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            events.append([start * frame_sec, i * frame_sec]); start = None
    if start is not None:
        events.append([start * frame_sec, len(b) * frame_sec])
    out = []
    for e in events:
        if out and e[0] - out[-1][1] <= merge_gap:
            out[-1][1] = e[1]
        else:
            out.append(e)
    return [e for e in out if e[1] - e[0] >= min_dur]

@torch.no_grad()
def _chunk_prob(wav_chunk):
    wav_chunk = prep_wav(wav_chunk.astype(np.float32))
    x = torch.from_numpy(wav_chunk).unsqueeze(0).to(device)
    mel = torch.from_numpy(logmel(wav_chunk)).unsqueeze(0).to(device)
    m = torch.ones(1, FRAMES_OUT, device=device)
    infer_model.eval()
    logits, _ = infer_model(x, mel, m)
    return torch.sigmoid(logits[0, 0]).float().cpu().numpy()

def _trim(events, true_sec):
    out = []
    for a, b in events:
        a = max(0.0, float(a)); b = min(float(true_sec), float(b))
        if b - a >= 0.05:
            out.append((a, b))
    return out

@torch.no_grad()
def predict_events_for_waveform(wav, thr=0.5, med=5, overlap=0.5):
    wav = wav.astype(np.float32); true_sec = len(wav) / CFG['sr']
    if len(wav) <= N_SAMP:
        w = np.pad(wav, (0, N_SAMP - len(wav))) if len(wav) < N_SAMP else wav
        ev = prob_to_events(_chunk_prob(w), thr=thr, med=med)
        ev = _trim(ev, true_sec)
        return [{'onset': round(a, 3), 'offset': round(b, 3)} for a, b in ev]
    frame_samp = CFG['hop'] * CFG['time_pool']
    total = int(math.ceil(len(wav) / frame_samp))
    acc = np.zeros(total, np.float32); cnt = np.zeros(total, np.float32)
    step = max(frame_samp, int(N_SAMP * (1.0 - overlap)))
    starts = list(range(0, max(1, len(wav) - N_SAMP + 1), step))
    if starts[-1] != len(wav) - N_SAMP:
        starts.append(max(0, len(wav) - N_SAMP))
    for s in starts:
        ch = wav[s:s + N_SAMP]
        if len(ch) < N_SAMP:
            ch = np.pad(ch, (0, N_SAMP - len(ch)))
        p = _chunk_prob(ch); g0 = int(round(s / frame_samp)); g1 = min(total, g0 + len(p))
        L = max(0, g1 - g0)
        if L > 0:
            acc[g0:g1] += p[:L]; cnt[g0:g1] += 1.0
    ev = prob_to_events(acc / np.maximum(cnt, 1e-6), thr=thr, med=med)
    ev = _trim(ev, true_sec)
    return [{'onset': round(a, 3), 'offset': round(b, 3)} for a, b in ev]

# ===== Cell 8b (code): test the SAVED model on test/held-out data =====
# Loads the checkpoint we just wrote (track1_crnn_mt.pt) into a FRESH model so we
# genuinely test the saved artifact, then scores it with the SAME event-level metric
# as the competition (Event-F1 micro + temporal Dice macro).

# -------- competition metric helpers (mirror v1 / scoring_program) --------
def _merge_intervals(intervals):
    if not intervals:
        return []
    itv = sorted(intervals)
    out = [list(itv[0])]
    for a, b in itv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]

def _total_dur(intervals):
    return float(sum(max(0.0, b - a) for a, b in intervals))

def _inter_dur(a_list, b_list):
    a_list, b_list = _merge_intervals(a_list), _merge_intervals(b_list)
    i = j = 0; inter = 0.0
    while i < len(a_list) and j < len(b_list):
        a1, a2 = a_list[i]; b1, b2 = b_list[j]
        lo, hi = max(a1, b1), min(a2, b2)
        if hi > lo:
            inter += (hi - lo)
        if a2 < b2:
            i += 1
        else:
            j += 1
    return inter

def _event_match(g, p):
    dur = max(1e-6, g[1] - g[0]); tol = max(0.2 * dur, 0.05)
    return (abs(g[0] - p[0]) <= tol) and (abs(g[1] - p[1]) <= tol)

def _event_counts(gt_events, pr_events):
    matched, tp = set(), 0
    for g in gt_events:
        best_j, best_err = -1, 1e18
        for j, p in enumerate(pr_events):
            if j in matched or not _event_match(g, p):
                continue
            err = abs(g[0] - p[0]) + abs(g[1] - p[1])
            if err < best_err:
                best_err, best_j = err, j
        if best_j >= 0:
            matched.add(best_j); tp += 1
    return tp, len(pr_events) - tp, len(gt_events) - tp

def _events_from_label(lab_row, frames_per_sec=100.0):
    """Reconstruct GT (onset, offset) spans from the rasterized any-noise channel."""
    b = (np.asarray(lab_row) >= 0.5).astype(np.uint8)
    events, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            events.append((start / frames_per_sec, i / frames_per_sec)); start = None
    if start is not None:
        events.append((start / frames_per_sec, len(b) / frames_per_sec))
    return events

def load_saved_model(ckpt_path='track1_crnn_mt.pt'):
    ck = torch.load(ckpt_path, map_location=device)
    m = CRNN_SED().to(device)
    m.load_state_dict(ck['student'])
    m.eval()
    print(f'loaded {ckpt_path} | use_atst_frame={ck.get("use_atst_frame")}')
    return m

@torch.no_grad()
def test_saved_model_on_rows(test_rows, thr=0.5, med=5, progress_every=100, tag='val'):
    """Event-F1 (micro) + Dice (macro) on rows that carry strong labels."""
    sum_tp = sum_fp = sum_fn = 0
    dice_scores, per_clip = [], []
    rows = [r for r in test_rows if r.get('has_strong', False)]
    print(f'[{tag}] scoring {len(rows)} labelled clips (thr={thr}, med={med})')
    for k, r in enumerate(tqdm(rows, desc=f'{tag}-test', leave=False), 1):
        gt_events = _events_from_label(r['lab'][0])
        pred = predict_events_for_waveform(r['wav'], thr=thr, med=med)
        pr_events = [(float(x['onset']), float(x['offset'])) for x in pred]

        tp, fp, fn = _event_counts(gt_events, pr_events)
        sum_tp += tp; sum_fp += fp; sum_fn += fn

        inter = _inter_dur(gt_events, pr_events)
        gt_d, pr_d = _total_dur(_merge_intervals(gt_events)), _total_dur(_merge_intervals(pr_events))
        dice_i = (2.0 * inter) / (gt_d + pr_d + 1e-9)
        dice_scores.append(dice_i)
        per_clip.append(dict(n_gt=len(gt_events), n_pred=len(pr_events), dice=dice_i))

        if progress_every and (k % progress_every == 0):
            p = sum_tp / (sum_tp + sum_fp + 1e-9); rc = sum_tp / (sum_tp + sum_fn + 1e-9)
            f1p = 2 * p * rc / (p + rc + 1e-9)
            print(f'  [prefix {k}] EventF1={f1p:.4f} Dice={np.mean(dice_scores):.4f}')

    p = sum_tp / (sum_tp + sum_fp + 1e-9); rc = sum_tp / (sum_tp + sum_fn + 1e-9)
    event_f1 = 2 * p * rc / (p + rc + 1e-9)
    dice_macro = float(np.mean(dice_scores)) if dice_scores else 0.0
    print(f'\n[{tag}] TEST RESULTS')
    print(f'  clips        : {len(per_clip)}')
    print(f'  TP/FP/FN     : {sum_tp}/{sum_fp}/{sum_fn}')
    print(f'  Precision    : {p:.4f}  Recall: {rc:.4f}')
    print(f'  Event F1     : {event_f1:.4f}')
    print(f'  Dice (macro) : {dice_macro:.4f}')
    print(f'  Combined     : {event_f1 + dice_macro:.4f}')
    return dict(event_f1=event_f1, dice=dice_macro, combined=event_f1 + dice_macro,
                tp=sum_tp, fp=sum_fp, fn=sum_fn)

# -------- folder-based scorer: external audio + reference.jsonl (mirrors v1) --------
def _read_audio_file(p):
    p = str(p)
    try:
        w, sr = sf.read(p, dtype='float32', always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
    except Exception:
        w, sr = librosa.load(p, sr=None, mono=True)
    if sr != CFG['sr']:
        w = librosa.resample(w, orig_sr=sr, target_sr=CFG['sr'])
    return np.asarray(w, dtype=np.float32)

def _load_reference_jsonl(p):
    gt = {}
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get('clip_id', ''))
            clean = []
            for e in (rec.get('events', []) or []):
                try:
                    a, b = float(e['onset']), float(e['offset'])
                except Exception:
                    continue
                if b > a:
                    clean.append((a, b))
            gt[cid] = sorted(clean)
    return gt

@torch.no_grad()
def test_saved_model_on_folder(audio_dir, ref_jsonl, thr=0.5, med=5,
                               exts=('.wav', '.flac', '.mp3', '.ogg'),
                               progress_every=100, tag='local-test'):
    gt_map = _load_reference_jsonl(ref_jsonl)
    files = sorted(p for p in Path(audio_dir).rglob('*') if p.suffix.lower() in exts)
    print(f'[{tag}] {len(files)} audio files | GT entries: {len(gt_map)} (thr={thr}, med={med})')

    sum_tp = sum_fp = sum_fn = 0
    dice_scores, per_clip, missing = [], [], 0
    for k, p in enumerate(tqdm(files, desc=f'{tag}', leave=False), 1):
        cid = p.stem
        if cid not in gt_map:
            missing += 1
            continue
        gt_events = gt_map[cid]
        pred = predict_events_for_waveform(_read_audio_file(p), thr=thr, med=med)
        pr_events = [(float(x['onset']), float(x['offset'])) for x in pred]

        tp, fp, fn = _event_counts(gt_events, pr_events)
        sum_tp += tp; sum_fp += fp; sum_fn += fn

        inter = _inter_dur(gt_events, pr_events)
        gt_d, pr_d = _total_dur(_merge_intervals(gt_events)), _total_dur(_merge_intervals(pr_events))
        dice_i = (2.0 * inter) / (gt_d + pr_d + 1e-9)
        dice_scores.append(dice_i)
        per_clip.append(dict(clip_id=cid, n_gt=len(gt_events), n_pred=len(pr_events), dice=dice_i,
                             f1=(2 * tp) / (2 * tp + fp + fn + 1e-9)))

        if progress_every and (len(per_clip) % progress_every == 0):
            pp = sum_tp / (sum_tp + sum_fp + 1e-9); rr = sum_tp / (sum_tp + sum_fn + 1e-9)
            f1p = 2 * pp * rr / (pp + rr + 1e-9)
            print(f'  [prefix {len(per_clip)}] EventF1={f1p:.4f} Dice={np.mean(dice_scores):.4f}')

    p_ = sum_tp / (sum_tp + sum_fp + 1e-9); r_ = sum_tp / (sum_tp + sum_fn + 1e-9)
    event_f1 = 2 * p_ * r_ / (p_ + r_ + 1e-9)
    dice_macro = float(np.mean(dice_scores)) if dice_scores else 0.0
    print(f'\n[{tag}] TEST RESULTS')
    print(f'  files scored : {len(per_clip)}  | missing GT: {missing}')
    print(f'  TP/FP/FN     : {sum_tp}/{sum_fp}/{sum_fn}')
    print(f'  Precision    : {p_:.4f}  Recall: {r_:.4f}')
    print(f'  Event F1     : {event_f1:.4f}')
    print(f'  Dice (macro) : {dice_macro:.4f}')
    print(f'  Combined     : {event_f1 + dice_macro:.4f}')
    worst = sorted(per_clip, key=lambda x: x['f1'] + x['dice'])[:5]
    print('  lowest-5 (F1+Dice):', [f"{w['clip_id']}(f1={w['f1']:.2f},d={w['dice']:.2f})" for w in worst])
    return dict(event_f1=event_f1, dice=dice_macro, combined=event_f1 + dice_macro,
                files=len(per_clip), missing_gt=missing, tp=sum_tp, fp=sum_fp, fn=sum_fn)

# Run: reload the saved artifact and test it on the external audio + reference.jsonl.
VAL_AUDIO_DIR = Path('/kaggle/input/datasets/shubhadipnag/input-audio/audio')
VAL_REF_JSONL = Path('/kaggle/input/datasets/shubhadipnag/input-data/reference.jsonl')

free_memory()
saved_model = load_saved_model('track1_crnn_mt.pt')
infer_model = saved_model  # decoding helpers above use `infer_model`
test_metrics = test_saved_model_on_folder(VAL_AUDIO_DIR, VAL_REF_JSONL, thr=0.5, med=5)
print(test_metrics)

# -------- OPTIONAL: run on an EXTERNAL test folder (unlabelled) -> submission --------
# If you have a folder of test audio files, uncomment and point TEST_AUDIO_DIR at it.
# TEST_AUDIO_DIR = '/kaggle/input/track1-test-audio'
# if 'TEST_AUDIO_DIR' in globals() and Path(TEST_AUDIO_DIR).exists():
#     AUD = ('.wav', '.flac', '.mp3', '.ogg')
#     def _read_audio(p):
#         try:
#             w, sr = sf.read(str(p), dtype='float32', always_2d=False)
#             if w.ndim > 1: w = w.mean(axis=1)
#         except Exception:
#             w, sr = librosa.load(str(p), sr=None, mono=True)
#         if sr != CFG['sr']:
#             w = librosa.resample(w, orig_sr=sr, target_sr=CFG['sr'])
#         return w.astype(np.float32)
#     files = sorted(p for p in Path(TEST_AUDIO_DIR).rglob('*') if p.suffix.lower() in AUD)
#     print('test clips:', len(files))
#     with open('predictions.jsonl', 'w', encoding='utf-8') as f:
#         for p in tqdm(files, desc='test-infer'):
#             ev = predict_events_for_waveform(_read_audio(p), thr=0.5, med=5)
#             f.write(json.dumps({'clip_id': p.stem, 'events': ev}, ensure_ascii=False) + '\n')
#     with zipfile.ZipFile('submission_track1.zip', 'w', zipfile.ZIP_DEFLATED) as z:
#         z.write('predictions.jsonl', 'predictions.jsonl')
#     print('wrote predictions.jsonl + submission_track1.zip')

# # ===== Cell 9 (code): submission packaging =====
# AUD = ('.wav', '.flac', '.mp3', '.ogg')

# def read_audio_file(p):
#     try:
#         w, sr = sf.read(str(p), dtype='float32', always_2d=False)
#         if w.ndim > 1:
#             w = w.mean(axis=1)
#     except Exception:
#         w, sr = librosa.load(str(p), sr=None, mono=True)
#     if sr != CFG['sr']:
#         w = librosa.resample(w, orig_sr=sr, target_sr=CFG['sr'])
#     return w.astype(np.float32)

# def build_submission(test_dir, out_jsonl='predictions.jsonl', out_zip='submission_track1.zip', thr=0.5, med=5):
#     files = sorted([p for p in Path(test_dir).rglob('*') if p.suffix.lower() in AUD])
#     print('clips:', len(files))
#     with open(out_jsonl, 'w', encoding='utf-8') as f:
#         for p in files:
#             ev = predict_events_for_waveform(read_audio_file(p), thr=thr, med=med)
#             f.write(json.dumps({'clip_id': p.stem, 'events': ev}, ensure_ascii=False) + '\n')
#     with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
#         z.write(out_jsonl, 'predictions.jsonl')
#     print('written:', out_jsonl, out_zip)

# # Example:
# # build_submission('/kaggle/input/track1-input-data')
