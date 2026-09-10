# Datathon@IndoML 2026 — Track 1 Baseline (ATST-style fine-tuning)
# Converted from notebook to a Python script.

# ===== Notebook Cell 1 (markdown) =====
# Title + notebook overview.
# ===== Notebook Cell 2 (markdown) =====
# Competition constraints summary.

# ===== Notebook Cell 3 (code) =====
# If running on Kaggle/Colab, uncomment:
# !pip -q install datasets huggingface_hub transformers soundfile librosa scipy tqdm

# ===== Notebook Cell 4 (code): imports + config =====
import os, io, json, math, random, zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, Audio
from transformers import AutoModel, AutoFeatureExtractor

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_AMP = torch.cuda.is_available()
USE_DP = N_GPU > 1

CFG = dict(
    sr=16000,
    hop=160,                # 10 ms frame grid for label rasterization
    clip_sec=5.0,
    time_pool=2,            # 20 ms output frame step
    val_frac=0.1,
    batch_size=16 if N_GPU >= 2 else 8,
    stage1_epochs=5,
    stage2_epochs=15,
    lr_head=1e-3,
    lr_backbone=5e-5,
    weight_decay=1e-2
)

# Set this to your official ATST checkpoint/repo if available.
# Fallback is an AST checkpoint to keep code runnable.
ATST_BACKBONE = os.environ.get("ATST_BACKBONE", "MIT/ast-finetuned-audioset-10-10-0.4593")

CATS = [
    "animal", "vehicle_traffic", "baby_child", "singing_music",
    "phone_signal_alarm", "appliance_machine", "human_non_speech"
]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + len(CATS)      # channel 0=any-noise, 1..7 categories

print(f"device={device} | gpus={N_GPU} | amp={USE_AMP} | data_parallel={USE_DP} | batch_size={CFG['batch_size']}")

# ===== Notebook Cell 5 (code): data reading =====
# -------- Data reading (adapted from your previous notebook) --------
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"

# Hardcoded HF token (as requested)
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set via env var; e.g. hf_xxxxxxxxxxxxxxxxx
if not HF_TOKEN or HF_TOKEN == "PASTE_YOUR_HF_TOKEN_HERE":
    raise RuntimeError("Please paste your Hugging Face token into HF_TOKEN before running.")

try:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("HF auth: OK")
except Exception as e:
    print("HF login warning:", e)

import tempfile

def decode_audio(a, target_sr=CFG['sr']):
    w = sr = None

    # Common cases from datasets Audio(decode=False)
    if isinstance(a, dict):
        if a.get('array') is not None:
            w, sr = np.asarray(a['array'], dtype=np.float32), a['sampling_rate']
        elif a.get('bytes') is not None:
            b = a['bytes']
            try:
                w, sr = sf.read(io.BytesIO(b), dtype='float32', always_2d=False)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix='.audio', delete=True) as tmp:
                    tmp.write(b)
                    tmp.flush()
                    w, sr = librosa.load(tmp.name, sr=None, mono=True)
        elif a.get('path'):
            p = a['path']
            try:
                w, sr = sf.read(p, dtype='float32', always_2d=False)
            except Exception:
                w, sr = librosa.load(p, sr=None, mono=True)
    elif isinstance(a, (str, Path)) and str(a):
        p = str(a)
        try:
            w, sr = sf.read(p, dtype='float32', always_2d=False)
        except Exception:
            w, sr = librosa.load(p, sr=None, mono=True)

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
    # Multi-event support: all events are rasterized; overlaps are naturally allowed
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
    # IMPORTANT: force decode=False to avoid torchcodec decode crashes on problematic files
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
                print(f'skipping sample {i} due to audio decode error: {e}')
            continue
        spans = spans_from(ex)
        nf = 1 + len(wav) // CFG['hop']
        lab = build_labels(spans, nf, has_strong=has_strong)
        rows.append(dict(wav=wav, lab=lab, quality=q, has_strong=has_strong))

    if len(rows) == 0:
        raise RuntimeError('No training rows could be decoded. Check HF token/access and audio backend packages.')

    print('loaded', len(rows), 'rows | skipped', skipped, '| mix=', Counter([r['quality'] for r in rows]))
    return rows













# ===== Notebook Cell 6 (code): data pipeline =====
# -------- Data pipeline + auto clip length + training weights --------
AUTO_SET_CLIP_SEC_FROM_DATA = True
PAD_EXTRA_SEC = 2.0  # train with max observed audio length + 2 sec

# IMPORTANT: bronze (no_timestamps) has weak labels only. Using it as strong negatives hurts event timing.
USE_STRONG_ONLY = True

# Validation policy: internal val split MUST contain event-bearing clips, otherwise the
# val loss (and early stopping) is minimized by predicting all-zeros (detect nothing).
VAL_CLEAN_ONLY = False

# Quality weighting in training loss (applied per sample)
QUALITY_WEIGHTS = {
    'verified_timestamps': 1.00,   # gold
    'unverified_timestamps': 0.75, # silver
    'no_timestamps': 0.25          # bronze (used only if USE_STRONG_ONLY=False)
}

def prep_wav(w):
    return (w - w.mean()) / (w.std() + 1e-5)

def is_clean_clip_row(r):
    # channel-0 is any-noise; sum==0 means no timestamped noise in this clip
    return float(np.asarray(r['lab'])[0].sum()) == 0.0

# 1) load rows first, then decide clip_sec from actual data lengths
rows = load_train_subset(max_rows=4000)
dur_sec = np.asarray([len(r['wav']) / CFG['sr'] for r in rows], dtype=np.float32)
mx = float(dur_sec.max()) if len(dur_sec) else CFG['clip_sec']
p95 = float(np.percentile(dur_sec, 95)) if len(dur_sec) else CFG['clip_sec']
med = float(np.median(dur_sec)) if len(dur_sec) else CFG['clip_sec']
print(f'audio duration stats (sec) | median={med:.2f} | p95={p95:.2f} | max={mx:.2f}')

# AST feature extractor pads/truncates every input to a FIXED window (max_length=1024
# mel frames = ~10.24s) that yields ~101 time tokens. A LARGER clip_sec spreads those
# fixed tokens over more time => coarser boundaries => Event-F1 cannot hit tolerance.
# So prefer p95 (robust to a few long outliers) and keep the window short.
AST_MAX_CLIP_SEC = 10.0
if AUTO_SET_CLIP_SEC_FROM_DATA:
    base = float(min(p95, mx))
    proposed = float(math.ceil(base + PAD_EXTRA_SEC))
    CFG['clip_sec'] = float(min(proposed, AST_MAX_CLIP_SEC))
    print(f"clip_sec set to {CFG['clip_sec']}s (p95-based={base:.2f}s + {PAD_EXTRA_SEC:.1f}s, capped at {AST_MAX_CLIP_SEC}s)")

# 2) recompute frame constants after clip_sec is finalized
N_SAMP = int(CFG['clip_sec'] * CFG['sr'])
FRAMES_100HZ = int(CFG['clip_sec'] * CFG['sr'] / CFG['hop'])
FRAMES_OUT = FRAMES_100HZ // CFG['time_pool']
FRAME_SEC = CFG['hop'] * CFG['time_pool'] / CFG['sr']
print(f'N_SAMP={N_SAMP} | FRAMES_OUT={FRAMES_OUT} | FRAME_SEC={FRAME_SEC:.4f}s')

# 3) inspect quality mix and optionally keep strong labels only
n_all = len(rows)
n_strong = sum(int(r.get('has_strong', False)) for r in rows)
n_weak = n_all - n_strong
print(f'rows loaded: total={n_all} | strong={n_strong} | weak(bronze)={n_weak}')

if USE_STRONG_ONLY:
    rows = [r for r in rows if r.get('has_strong', False)]
    print(f'using strong-only rows for frame training: {len(rows)}')
else:
    print('using all rows (including bronze weak labels)')

class Track1Dataset(Dataset):
    def __init__(self, rows, train=True):
        self.rows, self.train = rows, train
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        wav, lab = r['wav'], r['lab'].astype(np.float32)

        if len(wav) > N_SAMP:
            s = random.randint(0, len(wav)-N_SAMP) if self.train else 0
            wav = wav[s:s+N_SAMP]
            lab = lab[:, s//CFG['hop'] : s//CFG['hop'] + FRAMES_100HZ]
            valid = N_SAMP
        else:
            valid = len(wav)
            wav = np.pad(wav, (0, N_SAMP-len(wav)))

        if lab.shape[1] < FRAMES_100HZ:
            lab = np.pad(lab, ((0,0), (0, FRAMES_100HZ-lab.shape[1])))
        lab = lab[:, :FRAMES_100HZ]

        # 100 Hz -> 50 Hz (20 ms)
        lab_20ms = lab.reshape(lab.shape[0], FRAMES_OUT, CFG['time_pool']).max(axis=2)

        # valid mask on 20 ms grid
        m = np.zeros(FRAMES_OUT, dtype=np.float32)
        n_valid = max(1, int(math.ceil(valid / CFG['sr'] / FRAME_SEC)))
        m[:n_valid] = 1.0

        sw = float(QUALITY_WEIGHTS.get(r.get('quality'), 1.0))

        return dict(
            wav=torch.from_numpy(prep_wav(wav).astype(np.float32)),
            lab=torch.from_numpy(lab_20ms.astype(np.float32)),
            mask=torch.from_numpy(m),
            sw=torch.tensor(sw, dtype=torch.float32)
        )

# split
random.Random(SEED).shuffle(rows)
n_val = max(1, int(len(rows) * CFG['val_frac']))
val_rows, train_rows = rows[:n_val], rows[n_val:]

if VAL_CLEAN_ONLY:
    val_rows_clean = [r for r in val_rows if is_clean_clip_row(r)]
    if len(val_rows_clean) == 0:
        print('warning: clean-only val split is empty; reverting to mixed val split')
    else:
        val_rows = val_rows_clean
        print(f'val split forced to clean-only clips: {len(val_rows)}')

# 4) estimate class-wise positive weights from training rows
def estimate_pos_weight(train_rows):
    pos = np.zeros(N_OUT, dtype=np.float64)
    total = np.zeros(N_OUT, dtype=np.float64)
    for r in train_rows:
        lab = r['lab'].astype(np.float32)
        wav_len = len(r['wav'])

        if lab.shape[1] < FRAMES_100HZ:
            lab = np.pad(lab, ((0,0), (0, FRAMES_100HZ-lab.shape[1])))
        lab = lab[:, :FRAMES_100HZ]
        lab_20ms = lab.reshape(lab.shape[0], FRAMES_OUT, CFG['time_pool']).max(axis=2)

        n_valid = max(1, int(math.ceil(min(wav_len, N_SAMP) / CFG['sr'] / FRAME_SEC)))
        n_valid = min(FRAMES_OUT, n_valid)
        y = lab_20ms[:, :n_valid]

        pos += y.sum(axis=1)
        total += n_valid

    neg = np.maximum(total - pos, 1.0)
    pw = neg / np.maximum(pos, 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    return pw.astype(np.float32)

POS_W = estimate_pos_weight(train_rows)
LOSS_POS_WEIGHT = torch.tensor(POS_W, dtype=torch.float32, device=device)
print('channel pos_weight:', {i: round(float(w), 3) for i, w in enumerate(POS_W)})

# loaders
train_dl = DataLoader(Track1Dataset(train_rows, train=True), batch_size=CFG['batch_size'], shuffle=True, num_workers=2)
val_dl = DataLoader(Track1Dataset(val_rows, train=False), batch_size=CFG['batch_size'], shuffle=False, num_workers=2)

b = next(iter(train_dl))
print({k: tuple(v.shape) for k, v in b.items() if hasattr(v, 'shape')})
print(f"sample weight range in batch: min={float(b['sw'].min()):.3f} max={float(b['sw'].max()):.3f}")
print(f'train clips={len(train_rows)} | val clips={len(val_rows)}')

# ===== Notebook Cell 7 (code): model =====
# -------- ATST-style fine-tuning model --------
class ATSTSED(nn.Module):
    def __init__(self, backbone_name=ATST_BACKBONE, n_out=N_OUT, rnn_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone_name = backbone_name
        self.model_type = getattr(self.backbone.config, 'model_type', '')
        self.is_ast = self.model_type == 'audio-spectrogram-transformer'
        self.fe = AutoFeatureExtractor.from_pretrained(backbone_name) if self.is_ast else None

        d = self.backbone.config.hidden_size
        self.rnn = nn.GRU(d, rnn_dim, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)
        self.att = nn.Linear(2 * rnn_dim, n_out)

        # AST flattens patches as a (freq x time) grid preceded by [CLS] and [distill] tokens.
        # Precompute the grid dims so we can drop the 2 special tokens and average over the
        # frequency axis, yielding a clean time-ordered sequence for frame-level SED.
        self.ast_n_special = 0
        self.ast_f_dim = self.ast_t_dim = None
        if self.is_ast:
            cfg = self.backbone.config
            patch = int(getattr(cfg, 'patch_size', 16))
            fstride = int(getattr(cfg, 'frequency_stride', 10))
            tstride = int(getattr(cfg, 'time_stride', 10))
            n_mels = int(getattr(cfg, 'num_mel_bins', 128))
            max_len = int(getattr(cfg, 'max_length', 1024))
            self.ast_f_dim = (n_mels - patch) // fstride + 1
            self.ast_t_dim = (max_len - patch) // tstride + 1
            self.ast_n_special = 2  # cls_token + distillation_token
            print(f'Backbone mode: AST | freq_patches={self.ast_f_dim} time_patches={self.ast_t_dim} special_tokens={self.ast_n_special}')
        else:
            print(f'Backbone mode: {self.model_type or "waveform"} (direct waveform input)')

    def _encode(self, wav):
        if self.is_ast:
            wav_np = wav.detach().float().cpu().numpy()
            feats = self.fe(
                [w for w in wav_np],
                sampling_rate=CFG['sr'],
                return_tensors='pt'
            )
            inp = feats['input_values'].to(wav.device)
            out = self.backbone(inp)
        else:
            out = self.backbone(wav)
        return out.last_hidden_state

    def forward(self, wav, mask=None):
        h = self._encode(wav)  # AST: (B, 2 + f*t, D) ; else (B, T, D)

        # For AST: drop [CLS]+[distill] tokens and average over the frequency axis so the
        # remaining sequence is purely time-ordered (correct for frame-level SED).
        if self.is_ast and self.ast_f_dim is not None:
            h = h[:, self.ast_n_special:, :]              # (B, f*t, D)
            f, t = self.ast_f_dim, self.ast_t_dim
            if h.shape[1] == f * t:
                # max over frequency keeps sharp transient onsets (mean smears boundaries)
                h = h.reshape(h.shape[0], f, t, h.shape[-1]).amax(dim=1)  # (B, t, D)

        T = mask.shape[1] if mask is not None else h.shape[1]
        if h.shape[1] != T:
            h = F.interpolate(h.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
        h, _ = self.rnn(h)
        frame_logits = self.strong(h)
        frame = torch.sigmoid(frame_logits)
        att = self.att(h)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        att = torch.softmax(att, dim=1)
        clip = (frame * att).sum(dim=1).clamp(1e-6, 1-1e-6)
        return frame_logits.transpose(1, 2), clip

def focal_bce_frame(frame_logits, target, mask, sample_weight=None, pos_weight=None, gamma=1.5):
    if pos_weight is None and 'LOSS_POS_WEIGHT' in globals():
        pos_weight = LOSS_POS_WEIGHT
    if pos_weight is None:
        pos_weight = torch.ones(target.shape[1], device=target.device, dtype=target.dtype)
    pos_weight = pos_weight.to(target.device, dtype=target.dtype)

    # BCE with logits + focal modulation for class imbalance and harder positives
    bce = F.binary_cross_entropy_with_logits(
        frame_logits,
        target,
        reduction='none',
        pos_weight=pos_weight.view(1, -1, 1)
    )
    prob = torch.sigmoid(frame_logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    focal = (1.0 - pt).pow(gamma)
    loss = bce * focal

    if sample_weight is not None:
        sw = sample_weight.to(target.device, dtype=target.dtype).view(-1, 1, 1)
        loss = loss * sw
    m = mask.unsqueeze(1).to(target.device, dtype=target.dtype)
    return (loss * m).sum() / (m.sum() * target.shape[1] + 1e-6)

def boundary_aware_loss(frame_logits, target, mask):
    # Penalise disagreement on the TEMPORAL DERIVATIVE (the on/off edges), which is exactly
    # what the Event-F1 onset/offset tolerance measures. frame_logits,target: (B, C, T).
    prob = torch.sigmoid(frame_logits)
    d_pred = prob[..., 1:] - prob[..., :-1]
    d_tgt = target[..., 1:] - target[..., :-1]
    m = mask.unsqueeze(1).to(target.device, dtype=target.dtype)[..., 1:]
    return ((d_pred - d_tgt) ** 2 * m).sum() / (m.sum() * target.shape[1] + 1e-6)

EDGE_LOSS_WEIGHT = 0.5

def combined_loss(frame_logits, target, mask, sample_weight=None, edge_w=EDGE_LOSS_WEIGHT):
    base = focal_bce_frame(frame_logits, target, mask, sample_weight=sample_weight)
    edge = boundary_aware_loss(frame_logits, target, mask)
    return base + edge_w * edge

def core_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

model = ATSTSED().to(device)
if USE_DP:
    model = nn.DataParallel(model)
    print(f'Using DataParallel across {N_GPU} GPUs')
else:
    print('Using single GPU/CPU')

print('params(M):', round(sum(p.numel() for p in core_model(model).parameters()) / 1e6, 2))







# ===== Notebook Cell 8 (code): training =====
# -------- Two-stage training (AMP + optional multi-GPU DataParallel) --------
from tqdm.auto import tqdm

if 'train_dl' not in globals() or 'val_dl' not in globals():
    raise RuntimeError("train_dl/val_dl not found. Please run dataset + dataloaders first.")

AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# Early stop when val loss increases for N consecutive epochs
EARLY_STOP_CONSECUTIVE_INCREASE = 3
MIN_DELTA = 1e-6

def eval_loss(model, dl, desc='val'):
    model.eval()
    vals = []
    pbar = tqdm(dl, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            wav = batch['wav'].to(device, non_blocking=True)
            lab = batch['lab'].to(device, non_blocking=True)
            m = batch['mask'].to(device, non_blocking=True)
            sw = batch.get('sw', None)
            if sw is not None:
                sw = sw.to(device, non_blocking=True)
            with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                frame_logits, _ = model(wav, m)
                loss = combined_loss(frame_logits.float(), lab.float(), m.float(), sample_weight=sw)
            vals.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{vals[-1]:.4f}")
    return float(np.mean(vals)) if vals else 0.0

mcore = core_model(model)
print(f"Starting training | stage1_epochs={CFG['stage1_epochs']} | stage2_epochs={CFG['stage2_epochs']} | amp={USE_AMP} | dp={USE_DP}")
print(f"Early-stop rule: stop if val loss increases for {EARLY_STOP_CONSECUTIVE_INCREASE} consecutive epochs")
if 'LOSS_POS_WEIGHT' in globals():
    print('Using class pos_weight:', [round(float(x), 3) for x in LOSS_POS_WEIGHT.detach().cpu().tolist()])

# Stage 1: freeze backbone, train head
for p in mcore.backbone.parameters():
    p.requires_grad = False
head_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(head_params, lr=CFG['lr_head'], weight_decay=CFG['weight_decay'])
print(f"[Stage 1] Trainable params: {sum(p.numel() for p in head_params):,}")

prev_vl = None
inc_count = 0
stop_all_training = False

for ep in range(CFG['stage1_epochs']):
    model.train()
    tr = []
    pbar = tqdm(train_dl, desc=f"stage1 ep{ep:02d}", leave=False)
    for batch in pbar:
        wav = batch['wav'].to(device, non_blocking=True)
        lab = batch['lab'].to(device, non_blocking=True)
        m = batch['mask'].to(device, non_blocking=True)
        sw = batch.get('sw', None)
        if sw is not None:
            sw = sw.to(device, non_blocking=True)

        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            frame_logits, _ = model(wav, m)
            loss = combined_loss(frame_logits.float(), lab.float(), m.float(), sample_weight=sw)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(head_params, 5.0)
        scaler.step(opt)
        scaler.update()

        tr.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=f"{tr[-1]:.4f}")

    tr_mean = float(np.mean(tr)) if tr else 0.0
    vl = eval_loss(model, val_dl, desc=f"stage1 ep{ep:02d} val")
    print(f"[Stage 1][Epoch {ep+1}/{CFG['stage1_epochs']}] train={tr_mean:.4f} val={vl:.4f}")

    if prev_vl is not None and (vl - prev_vl) > MIN_DELTA:
        inc_count += 1
    else:
        inc_count = 0
    prev_vl = vl

    if inc_count >= EARLY_STOP_CONSECUTIVE_INCREASE:
        print(f"[Early Stop] Validation loss increased for {inc_count} consecutive epochs in Stage 1. Stopping training.")
        stop_all_training = True
        break

# Stage 2: unfreeze backbone, lower LR on backbone
if not stop_all_training:
    for p in mcore.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {'params': mcore.backbone.parameters(), 'lr': CFG['lr_backbone']},
        {'params': [p for n, p in mcore.named_parameters() if not n.startswith('backbone.')], 'lr': CFG['lr_head'] * 0.3}
    ], weight_decay=CFG['weight_decay'])
    print(f"[Stage 2] Unfroze backbone | lr_backbone={CFG['lr_backbone']} | lr_head={CFG['lr_head'] * 0.3}")

    # Continue early-stop tracking across stage boundary
    for ep in range(CFG['stage2_epochs']):
        model.train()
        tr = []
        pbar = tqdm(train_dl, desc=f"stage2 ep{ep:02d}", leave=False)
        for batch in pbar:
            wav = batch['wav'].to(device, non_blocking=True)
            lab = batch['lab'].to(device, non_blocking=True)
            m = batch['mask'].to(device, non_blocking=True)
            sw = batch.get('sw', None)
            if sw is not None:
                sw = sw.to(device, non_blocking=True)

            with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                frame_logits, _ = model(wav, m)
                loss = combined_loss(frame_logits.float(), lab.float(), m.float(), sample_weight=sw)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            tr.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{tr[-1]:.4f}")

        tr_mean = float(np.mean(tr)) if tr else 0.0
        vl = eval_loss(model, val_dl, desc=f"stage2 ep{ep:02d} val")
        print(f"[Stage 2][Epoch {ep+1}/{CFG['stage2_epochs']}] train={tr_mean:.4f} val={vl:.4f}")

        if prev_vl is not None and (vl - prev_vl) > MIN_DELTA:
            inc_count += 1
        else:
            inc_count = 0
        prev_vl = vl

        if inc_count >= EARLY_STOP_CONSECUTIVE_INCREASE:
            print(f"[Early Stop] Validation loss increased for {inc_count} consecutive epochs. Stopping Stage 2.")
            break

torch.save({'model': mcore.state_dict(), 'cfg': CFG, 'backbone': ATST_BACKBONE,
            'n_gpu': N_GPU, 'use_amp': USE_AMP}, 'track1_atst.pt')
print('saved -> track1_atst.pt')










# ===== Notebook Cell 9 (code): val-split testing =====
# -------- Explicit validation-set testing (after training) --------
@torch.no_grad()
def test_on_val_split(model, val_dl, thr=0.5):
    model.eval()
    losses = []
    tp = fp = fn = 0.0   # computed on channel-0 (any-noise)

    pbar = tqdm(val_dl, desc='val-test', leave=False)
    for batch in pbar:
        wav = batch['wav'].to(device, non_blocking=True)
        lab = batch['lab'].to(device, non_blocking=True)
        m = batch['mask'].to(device, non_blocking=True)

        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            frame_logits, _ = model(wav, m)
            loss = combined_loss(frame_logits.float(), lab.float(), m.float())
        losses.append(float(loss.detach().cpu()))

        # binary frame prediction on "any-noise" channel
        pred = (torch.sigmoid(frame_logits[:, 0]) >= thr).float()
        tgt = lab[:, 0].float()
        valid = m.float()

        tp += float(((pred == 1) & (tgt == 1)).float().mul(valid).sum().item())
        fp += float(((pred == 1) & (tgt == 0)).float().mul(valid).sum().item())
        fn += float(((pred == 0) & (tgt == 1)).float().mul(valid).sum().item())

        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    val_loss = float(np.mean(losses)) if losses else 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)

    print('Validation (frame-level proxy on any-noise channel):')
    print(f'  BCE loss : {val_loss:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall   : {recall:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Dice     : {dice:.4f}')
    return dict(val_loss=val_loss, precision=precision, recall=recall, f1=f1, dice=dice)

val_metrics = test_on_val_split(model, val_dl, thr=0.5)
print(val_metrics)









# ===== Notebook Cell 10 (code): inference helpers =====
# -------- Inference: posterior -> multiple events (full-length audio support) --------
from scipy.ndimage import median_filter

def prob_to_events(p, thr=0.5, med=7, frame_sec=FRAME_SEC, min_dur=0.05, merge_gap=0.10):
    b = (np.asarray(p) >= thr).astype(np.uint8)
    if med > 1:
        b = median_filter(b, size=med, mode='nearest')

    events, start = [], None
    for i, v in enumerate(b):
        if v and start is None:
            start = i
        elif (not v) and (start is not None):
            events.append([start * frame_sec, i * frame_sec])
            start = None
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
def _infer_chunk_any_prob(wav_chunk):
    x = torch.from_numpy(prep_wav(wav_chunk)).unsqueeze(0).to(device)
    m = torch.ones(1, FRAMES_OUT, device=device)
    frame_logits, _ = model(x, m)
    return torch.sigmoid(frame_logits[0, 0]).float().cpu().numpy()

def _trim_events_to_duration(events, true_sec):
    # Padding at inference produces undefined outputs over silence; clamp/drop events that
    # fall outside the clip's real duration to avoid fabricated false positives.
    out = []
    for a, b in events:
        a = max(0.0, float(a))
        b = min(float(true_sec), float(b))
        if b - a >= 0.05:
            out.append((a, b))
    return out

@torch.no_grad()
def compute_posterior_for_waveform(wav, overlap=0.5):
    """Return (p_any, true_sec): the any-noise frame posterior on the 20ms grid plus the
    clip's true duration. Separated from decoding so a sweep can re-decode cheaply."""
    wav = wav.astype(np.float32)
    true_sec = len(wav) / CFG['sr']

    if len(wav) <= N_SAMP:
        if len(wav) < N_SAMP:
            wav = np.pad(wav, (0, N_SAMP-len(wav)))
        return _infer_chunk_any_prob(wav), true_sec

    frame_samp = CFG['hop'] * CFG['time_pool']   # 320 samples = 20 ms
    total_frames = int(math.ceil(len(wav) / frame_samp))
    acc = np.zeros(total_frames, dtype=np.float32)
    cnt = np.zeros(total_frames, dtype=np.float32)

    step = int(N_SAMP * (1.0 - overlap))
    step = max(frame_samp, step)

    starts = list(range(0, max(1, len(wav) - N_SAMP + 1), step))
    if starts[-1] != len(wav) - N_SAMP:
        starts.append(max(0, len(wav) - N_SAMP))

    for s in starts:
        ch = wav[s:s+N_SAMP]
        if len(ch) < N_SAMP:
            ch = np.pad(ch, (0, N_SAMP-len(ch)))
        p = _infer_chunk_any_prob(ch)
        g0 = int(round(s / frame_samp))
        g1 = min(total_frames, g0 + len(p))
        L = max(0, g1 - g0)
        if L > 0:
            acc[g0:g1] += p[:L]
            cnt[g0:g1] += 1.0

    p_any = acc / np.maximum(cnt, 1e-6)
    return p_any, true_sec

def split_on_valleys(p_any, events, frame_sec=FRAME_SEC, valley_drop=0.3, min_dur=0.10):
    """Split one detected region into multiple events if the posterior dips (a valley)
    between two peaks -- this undoes over-merge, the 2nd biggest Event-F1 error source."""
    p_any = np.asarray(p_any)
    out = []
    for a, b in events:
        i0 = int(round(a / frame_sec)); i1 = int(round(b / frame_sec))
        seg = p_any[i0:i1]
        if len(seg) < 3:
            out.append((a, b)); continue
        peak = float(seg.max())
        cut = [k for k in range(1, len(seg) - 1)
               if seg[k] < peak - valley_drop and seg[k] <= seg[k-1] and seg[k] <= seg[k+1]]
        if not cut:
            out.append((a, b)); continue
        prev = 0
        for c in cut + [len(seg)]:
            s = (i0 + prev) * frame_sec
            e = (i0 + c) * frame_sec
            if e - s >= min_dur:
                out.append((s, e))
            prev = c
    return out

def decode_posterior_to_events(p_any, true_sec, thr=0.5, med=7, min_dur=0.05, merge_gap=0.10, split_valleys=True):
    events = prob_to_events(p_any, thr=thr, med=med, min_dur=min_dur, merge_gap=merge_gap)
    if split_valleys:
        events = split_on_valleys(p_any, events, min_dur=min_dur)
    events = _trim_events_to_duration(events, true_sec)
    return [{'onset': round(float(a), 3), 'offset': round(float(b), 3)} for a, b in events]

@torch.no_grad()
def predict_events_for_waveform(wav, thr=0.5, med=7, overlap=0.5, min_dur=0.05, merge_gap=0.10):
    p_any, true_sec = compute_posterior_for_waveform(wav, overlap=overlap)
    return decode_posterior_to_events(p_any, true_sec, thr=thr, med=med, min_dur=min_dur, merge_gap=merge_gap)













# ===== Notebook Cell 11 (code): local validation =====
# -------- Local validation on external audio + reference.jsonl --------
from tqdm.auto import tqdm

if 'predict_events_for_waveform' not in globals():
    raise RuntimeError("Please run inference helpers first (predict_events_for_waveform).")

VAL_AUDIO_DIR = Path('/kaggle/input/datasets/shubhadipnag/input-audio/audio')
VAL_REF_JSONL = Path('/kaggle/input/datasets/shubhadipnag/input-data/reference.jsonl')
CLEAN_ONLY_LOCAL_VAL = False
PROGRESS_EVERY = 100

def read_audio_file_local(p):
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

def load_reference_jsonl(p):
    gt = {}
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get('clip_id', ''))
            events = rec.get('events', []) or []
            clean = []
            for e in events:
                try:
                    a = float(e['onset']); b = float(e['offset'])
                except Exception:
                    continue
                if b > a:
                    clean.append((a, b))
            gt[cid] = sorted(clean)
    return gt

def merge_intervals(intervals):
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

def total_duration(intervals):
    return float(sum(max(0.0, b - a) for a, b in intervals))

def intersection_duration(a_list, b_list):
    i = j = 0
    inter = 0.0
    a_list = merge_intervals(a_list)
    b_list = merge_intervals(b_list)
    while i < len(a_list) and j < len(b_list):
        a1, a2 = a_list[i]
        b1, b2 = b_list[j]
        lo, hi = max(a1, b1), min(a2, b2)
        if hi > lo:
            inter += (hi - lo)
        if a2 < b2:
            i += 1
        else:
            j += 1
    return inter

def event_match(g, p):
    dur = max(1e-6, g[1] - g[0])
    tol = max(0.2 * dur, 0.05)
    return (abs(g[0] - p[0]) <= tol) and (abs(g[1] - p[1]) <= tol)

def event_f1(gt_events, pr_events):
    matched_pred = set()
    tp = 0
    for g in gt_events:
        best_j = -1
        best_err = 1e18
        for j, p in enumerate(pr_events):
            if j in matched_pred:
                continue
            if event_match(g, p):
                err = abs(g[0]-p[0]) + abs(g[1]-p[1])
                if err < best_err:
                    best_err = err
                    best_j = j
        if best_j >= 0:
            matched_pred.add(best_j)
            tp += 1
    fp = len(pr_events) - tp
    fn = len(gt_events) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return f1, tp, fp, fn

def _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores):
    p = sum_tp / (sum_tp + sum_fp + 1e-9)
    r = sum_tp / (sum_tp + sum_fn + 1e-9)
    event_f1_micro = 2 * p * r / (p + r + 1e-9)
    dice_macro = float(np.mean(dice_scores)) if dice_scores else 0.0
    combined = event_f1_micro + dice_macro
    return event_f1_micro, dice_macro, combined

def evaluate_local_folder(audio_dir, ref_jsonl, thr=0.5, med=7, exts=('.wav', '.flac', '.mp3', '.ogg'), clean_only=False, progress_every=100):
    gt_map = load_reference_jsonl(ref_jsonl)
    files = sorted([p for p in Path(audio_dir).rglob('*') if p.suffix.lower() in exts])
    print(f'Found {len(files)} audio files | GT entries: {len(gt_map)} | clean_only={clean_only}')

    sum_tp = sum_fp = sum_fn = 0
    dice_scores = []
    missing_gt = 0
    skipped_non_clean = 0
    per_clip = []

    for p in tqdm(files, desc='local-val'):
        cid = p.stem
        if cid not in gt_map:
            missing_gt += 1
            continue

        gt_events = gt_map[cid]
        if clean_only and len(gt_events) > 0:
            skipped_non_clean += 1
            continue

        wav = read_audio_file_local(p)
        pred = predict_events_for_waveform(wav, thr=thr, med=med)
        pr_events = [(float(x['onset']), float(x['offset'])) for x in pred]

        f1_i, tp, fp, fn = event_f1(gt_events, pr_events)
        sum_tp += tp; sum_fp += fp; sum_fn += fn

        inter = intersection_duration(gt_events, pr_events)
        gt_d = total_duration(merge_intervals(gt_events))
        pr_d = total_duration(merge_intervals(pr_events))
        dice_i = (2.0 * inter) / (gt_d + pr_d + 1e-9)
        dice_scores.append(dice_i)

        per_clip.append({
            'clip_id': cid,
            'n_gt': len(gt_events),
            'n_pred': len(pr_events),
            'f1': f1_i,
            'dice': dice_i
        })

        # Prefix score print: first 100, first 200, first 300, ...
        n_eval = len(per_clip)
        if progress_every and (n_eval % progress_every == 0):
            f1_p, dice_p, comb_p = _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores)
            print(f"[prefix {n_eval}] EventF1={f1_p:.4f} Dice={dice_p:.4f} Combined={comb_p:.4f}")

    event_f1_micro, dice_macro, combined = _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores)

    print('\nLocal validation summary')
    print(f'  files evaluated   : {len(per_clip)}')
    print(f'  missing GT IDs    : {missing_gt}')
    print(f'  skipped non-clean : {skipped_non_clean}')
    if len(per_clip) == 0 and clean_only:
        print("  note             : clean_only=True filtered out all clips; set CLEAN_ONLY_LOCAL_VAL=False")
    print(f'  Event F1 (micro)  : {event_f1_micro:.4f}')
    print(f'  Dice (macro)      : {dice_macro:.4f}')
    print(f'  Combined          : {combined:.4f}')

    per_clip = sorted(per_clip, key=lambda x: x['f1'] + x['dice'])
    print('\nLowest-5 clips by (F1 + Dice):')
    for row in per_clip[:5]:
        print(f"  {row['clip_id']}: gt={row['n_gt']} pred={row['n_pred']} f1={row['f1']:.3f} dice={row['dice']:.3f}")

    return {
        'event_f1_micro': event_f1_micro,
        'dice_macro': dice_macro,
        'combined': combined,
        'files_evaluated': len(per_clip),
        'missing_gt': missing_gt,
        'skipped_non_clean': skipped_non_clean
    }

# Run local validation
local_val_metrics = evaluate_local_folder(
    VAL_AUDIO_DIR,
    VAL_REF_JSONL,
    thr=0.5,
    med=7,
    clean_only=CLEAN_ONLY_LOCAL_VAL,
    progress_every=PROGRESS_EVERY
)
print(local_val_metrics)

# -------- Decoding-parameter sweep (raise Event F1 without retraining) --------
# High Dice + low Event F1 means the acoustic detection is good but the event
# segmentation/boundaries are off. These post-processing knobs move F1 a lot while
# barely touching Dice. We cache each clip's posterior ONCE, then re-decode cheaply
# across the grid so the model is not re-run per configuration.
DO_DECODING_SWEEP = True
SWEEP_MAX_CLIPS = 800            # cap for speed; set None to use all evaluable clips
SWEEP_OPTIMIZE = 'event_f1'      # 'event_f1' or 'combined'
SWEEP_THR = [0.35, 0.45, 0.5, 0.55, 0.65]
SWEEP_MED = [1, 3, 5, 7]
SWEEP_MERGE_GAP = [0.0, 0.03, 0.05, 0.10]
SWEEP_MIN_DUR = [0.05, 0.10]

def sweep_decoding_params(audio_dir, ref_jsonl, exts=('.wav', '.flac', '.mp3', '.ogg'),
                          max_clips=None, optimize='event_f1'):
    gt_map = load_reference_jsonl(ref_jsonl)
    files = sorted([p for p in Path(audio_dir).rglob('*') if p.suffix.lower() in exts])
    files = [p for p in files if p.stem in gt_map]
    if max_clips is not None:
        files = files[:max_clips]
    print(f'Sweep: caching posteriors for {len(files)} clips ...')

    cache = []  # (gt_events, p_any, true_sec)
    for p in tqdm(files, desc='sweep-cache'):
        wav = read_audio_file_local(p)
        p_any, true_sec = compute_posterior_for_waveform(wav)
        cache.append((gt_map[p.stem], p_any, true_sec))

    best = None
    results = []
    grid = [(t, m, g, d) for t in SWEEP_THR for m in SWEEP_MED
            for g in SWEEP_MERGE_GAP for d in SWEEP_MIN_DUR]
    print(f'Sweep: evaluating {len(grid)} decoding configs ...')

    for (thr, med, merge_gap, min_dur) in tqdm(grid, desc='sweep-grid'):
        sum_tp = sum_fp = sum_fn = 0
        dice_scores = []
        for gt_events, p_any, true_sec in cache:
            pred = decode_posterior_to_events(p_any, true_sec, thr=thr, med=med,
                                              min_dur=min_dur, merge_gap=merge_gap)
            pr_events = [(float(x['onset']), float(x['offset'])) for x in pred]
            _, tp, fp, fn = event_f1(gt_events, pr_events)
            sum_tp += tp; sum_fp += fp; sum_fn += fn
            inter = intersection_duration(gt_events, pr_events)
            gt_d = total_duration(merge_intervals(gt_events))
            pr_d = total_duration(merge_intervals(pr_events))
            dice_scores.append((2.0 * inter) / (gt_d + pr_d + 1e-9))
        f1_m, dice_m, comb = _metric_from_counts(sum_tp, sum_fp, sum_fn, dice_scores)
        score = comb if optimize == 'combined' else f1_m
        row = dict(thr=thr, med=med, merge_gap=merge_gap, min_dur=min_dur,
                   event_f1=f1_m, dice=dice_m, combined=comb, score=score)
        results.append(row)
        if best is None or score > best['score']:
            best = row

    results = sorted(results, key=lambda r: r['score'], reverse=True)
    print('\nTop-5 decoding configs (by ' + optimize + '):')
    for r in results[:5]:
        print(f"  thr={r['thr']} med={r['med']} merge_gap={r['merge_gap']} min_dur={r['min_dur']}"
              f" | EventF1={r['event_f1']:.4f} Dice={r['dice']:.4f} Combined={r['combined']:.4f}")
    print(f"\nBest: thr={best['thr']} med={best['med']} merge_gap={best['merge_gap']}"
          f" min_dur={best['min_dur']} -> EventF1={best['event_f1']:.4f}"
          f" Dice={best['dice']:.4f} Combined={best['combined']:.4f}")
    return best, results

if DO_DECODING_SWEEP:
    BEST_DECODE, SWEEP_RESULTS = sweep_decoding_params(
        VAL_AUDIO_DIR, VAL_REF_JSONL, max_clips=SWEEP_MAX_CLIPS, optimize=SWEEP_OPTIMIZE
    )
    print('\nUse these in build_submission / predict_events_for_waveform:', BEST_DECODE)






# # ===== Notebook Cell 12 (code): submission packaging =====
# # -------- Submission packaging (adapted from your previous notebook) --------
# AUD = ('.wav', '.flac', '.mp3', '.ogg')

# def read_audio_file(p):
#     w, sr = sf.read(str(p), dtype='float32', always_2d=False)
#     if w.ndim > 1:
#         w = w.mean(axis=1)
#     if sr != CFG['sr']:
#         w = librosa.resample(w, orig_sr=sr, target_sr=CFG['sr'])
#     return w.astype(np.float32)

# def build_submission(test_dir, out_jsonl='predictions.jsonl', out_zip='submission_track1.zip', thr=0.5, med=7):
#     test_dir = Path(test_dir)
#     files = sorted([p for p in test_dir.rglob('*') if p.suffix.lower() in AUD])
#     print('clips found:', len(files))

#     n_short_pad = 0
#     n_long = 0

#     with open(out_jsonl, 'w', encoding='utf-8') as f:
#         for p in files:
#             wav = read_audio_file(p)
#             if len(wav) < N_SAMP:
#                 n_short_pad += 1
#             elif len(wav) > N_SAMP:
#                 n_long += 1
#             ev = predict_events_for_waveform(wav, thr=thr, med=med)
#             rec = {'clip_id': p.stem, 'events': ev}
#             f.write(json.dumps(rec, ensure_ascii=False) + '\n')

#     with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
#         z.write(out_jsonl, 'predictions.jsonl')   # must be at ZIP root

#     print('written:', out_jsonl, 'and', out_zip)
#     print(f'inference padding summary | short_clips_padded={n_short_pad} | long_clips_windowed={n_long}')

# # Example:
# # build_submission('/kaggle/input/track1-input-data')

# # ===== Notebook Cell 13 (markdown) =====
# # Paper-summary notes are omitted in executable script form.
