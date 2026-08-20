# TDV vs MAE — Full Pipeline & Comparison

---

## Overview

Both pipelines answer the same question from different angles:

> **"What is the best self-supervised way to learn surgical visual
> representations from unlabelled video?"**

MAE asks: *what does a frame look like spatially?*
TDV asks: *how does a frame change over time?*

Both produce a 768-d CLS vector per frame that feeds into the same
128-d ProjectionModel, evaluated on the same labels. The comparison
is therefore controlled — only the self-supervised training signal
differs.

---

## Side-by-Side Pipeline

```
                    MAE PIPELINE                    TDV PIPELINE
                    ────────────                    ────────────

INPUT           Single frame                    Consecutive frame pair
                (B, 3, 224, 224)                (B, 3, 224, 224) × 2

ENCODER         MAE ViT-Base                    DINOv2 ViT-B/14
                ImageNet MAE pretrained          ImageNet DINO pretrained
                embed_dim = 768                 embed_dim = 768
                FROZEN                          FROZEN

ENCODER         Patch + positional embed        Patch + positional embed
INTERNALS       12 transformer blocks           12 transformer blocks
                Random masking (75%)            No masking
                CLS token aggregation           CLS token aggregation

ADAPTER         MLPAdapter [TRAINABLE]          MLPAdapter [TRAINABLE]
                768 → 128 → 768, residual       768 → 128 → 768, residual
                On CLS token only               On CLS token only
                Patch tokens unchanged          Patch tokens unchanged

CONDITIONING    adapted_cls + patch_tokens      adapted_cls + patch_tokens
                recombined → adapted_latent     recombined → condition_t
                (B, 257, 768)                   (B, 257, 768)
                passed to decoder               passed to motion encoder

TRAINABLE       MAE decoder [TRAINABLE]         TDV xattn motion encoder
COMPONENT       8 transformer blocks            [TRAINABLE]
                512-d                           dinoViT_xattn_base14
                Reconstructs masked pixels      depth=4, 768-d output
                                                Cross-attends to condition_t

SELF-SUPERVISED Masked reconstruction           Temporal difference
TASK            Given 25% of patches,           Given F_t and F_{t+1} − F_t,
                reconstruct the 75%             predict z_{t+1} from z_t
                hidden patches

LOSS            MSE on masked patches           MSE on L2-normalised
                (pixel space)                   representations (latent space)
                || pred_pixels − true_pixels ||²  || norm(z_t + Δz) −
                                                   norm(z_t1).detach() ||²

GRADIENT        loss → decoder → adapted_cls   loss → motion_encoder →
PATH            → MLP adapter                  Δz → z_t → MLP adapter

WHAT THE        Spatial structure:              Temporal dynamics:
ADAPTER         tissue texture, instrument      instrument motion, tissue
LEARNS          shape, scene layout             deformation, fluid flow

OUTPUT          adapted_cls (B, 768)            adapted_cls (B, 768)
(TRAINING)      reconstruction loss             temporal difference loss

OUTPUT          extract_cls(frame)              extract_cls(frame)
(INFERENCE)     → (B, 768)                     → (B, 768)
                identical interface             identical interface

DOWNSTREAM      ProjectionModel 768→128         ProjectionModel 768→128
                (same weights, same labels)     (same weights, same labels)
                → (B, 128) L2-normalised        → (B, 128) L2-normalised
```

---

## Step-by-Step TDV Pipeline

### Step 0 — Preprocessing (shared with MAE)
```
Raw .mp4 prostatectomy video
        │
Segment by surgical step annotations
        │
clip_0000.pt ... clip_0003.pt
    video: (T, 3, 224, 224)  ImageNet-normalised
    fps, timestamps, actions
```
**Why ImageNet normalisation?**
DINOv2 was trained on ImageNet with the same normalisation as MAE
(`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). No change
to preprocessing needed.

---

### Step 1 — SurgicalPairDataset
```
File: datasets/RARPclip_dataset.py → SurgicalPairDataset

Input:  clip_XXXX.pt files
Output: consecutive frame pairs (frame_t, frame_t1)

Index structure:
  A clip with T frames → T-1 pairs
  4 clips (452+761+64+27 frames) → 1300 pairs

Per __getitem__:
{
    "frame_t":   (3, 224, 224)    F_t
    "frame_t1":  (3, 224, 224)    F_{t+1}
    "frames":    (2, 3, 224, 224) stacked
    "pt_idx":    int              which .pt file
    "frame_t_idx": int            position of F_t
}

Key constraint: pairs never cross .pt file boundaries.
Temporal continuity guaranteed within each recording segment.
```

**MAE comparison:** MAE uses `SurgicalFrameDataset` — individual frames
shuffled randomly. TDV must preserve temporal order within clips.

---

### Step 2 — TDVFineTuner (models/tdv_wrapper.py)

```
Frame pair (B, 3, 224, 224) × 2
        │
        ├─── frame_t ──────────────────────────────────────────┐
        │    [FROZEN] DINOv2 forward_features()                │
        │         → cls_token    (B, 768)                      │
        │         → patch_tokens (B, 256, 768)                 │
        │                                                       │
        │    [TRAIN] MLPAdapter (residual)                     │
        │         adapted_cls = cls_token + adapter(cls_token) │
        │         adapted_cls: (B, 768)  ← z_t                │
        │                                                       │
        │    Recombine:                                         │
        │    condition_t = [adapted_cls | patch_tokens]        │
        │    condition_t: (B, 257, 768)                        │
        │                                                       │
        └─── frame_t1 ─────────────────────────────────────────┘
             [FROZEN] DINOv2 forward_features()
                  → z_t1 (B, 768)   ← stop-gradient target

pixel_diff = frame_t1 - frame_t    (B, 3, H, W)
        │
        ▼
[TRAIN] TDV dinoViT_xattn_base14
        Input:     pixel_diff    (B, 3, 224, 224)
        condition= condition_t   (B, 257, 768)    ← keyword arg
        Output:    dict
            x_norm_clstoken    (B, 768)  ← Δz
            x_norm_patchtokens (B, 256, 768)
        │
        Δz = motion_proj(motion_out["x_norm_clstoken"])
        │
        predicted = L2_norm(z_t + Δz)          (B, 768)
        sg_z_t1   = L2_norm(z_t1).detach()     (B, 768)
        │
        loss = MSE(predicted, sg_z_t1)
```

---

### Step 3 — Training Loop (training/train_tdv.py)

```
Mirrors train_mae.py exactly:

Same:
  - AdamW optimiser (betas 0.9, 0.95)
  - Cosine LR schedule with linear warm-up (5 epochs)
  - NativeScaler (mixed precision)
  - Per-iteration LR adjustment via lr_sched
  - Checkpoint every 20 epochs + final
  - log.txt: one JSON line per epoch
  - TensorBoard logging

Different:
  - Dataset: SurgicalClipDataset (clip_len=2) not SurgicalFrameDataset
  - Batch unpacking: clip[:,0] and clip[:,1] not batch["frame"]
  - Loss: temporal difference not reconstruction
  - Batch size: 8 not 16 (motion encoder is 38M params vs 8-block decoder)
```

**Terminal:**
```powershell
python training/train_tdv.py `
    --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
    --checkpoint "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --output_dir "checkpoints/tdv_finetuned" `
    --epochs     50 `
    --batch_size 8 `
    --num_workers 0 `
    --seed       42
```

**Checkpoint contents (same format as MAE):**
```python
{
    "model":      state_dict,   # frozen DINOv2 + adapter + motion encoder
    "optimizer":  ...,
    "scaler":     ...,
    "epoch":      49,
    "train_loss": float,
    "args":       {...},        # mlp_hidden_dim etc. for reproducibility
}
```

---

### Step 4 — CLS Extraction (models/extract_tdv_cls.py)

```
Input:  checkpoint-0049.pth + clip_XXXX.pt files
Output: data/embeddings/step_5_tdv_cls.pt
    {
        "embeddings":    (1304, 768)   one 768-d vector per frame
        "clip_indices":  (1304,)
        "frame_indices": (1304,)
        "model":         "tdv"         distinguishes from MAE embeddings
    }

Key difference from MAE extraction:
  MAE: mask_ratio=0.0 at inference (was 0.75 during training)
  TDV: no masking ever — DINOv2 always sees full frame
       motion encoder NOT used at inference
       extract_cls() = encoder + adapter only, single frame input
```

**Terminal:**
```powershell
python models/extract_tdv_cls.py `
    --checkpoint  "checkpoints/tdv_finetuned/checkpoint-0049.pth" `
    --dinov2_ckpt "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --clip_dir    "data/processed/prostatectomy/GSTT_010/step_5" `
    --output      "data/embeddings/step_5_tdv_cls.pt" `
    --step_label  5
```

---

### Step 5 — Label Preparation (shared with MAE)

```
datasets/prepare_labels.py
Input:  clip_XXXX.pt (reads actions field) + embedding file (for alignment)
Output: data/labels/step_5/labels.pt    LongTensor (1304,)
        data/labels/step_5/label_map.json

Already completed for MAE — same labels.pt used for TDV.
No re-run needed.
```

---

### Step 6 — Projection Model (models/projection_model.py)

```
Input:  data/embeddings/step_5_tdv_cls.pt  (768-d TDV embeddings)
        data/labels/step_5/labels.pt        (same labels as MAE)
Output: checkpoints/projection_tdv/
            checkpoint-best.pth
            log.txt

Architecture (identical to MAE projection):
    Linear(768→512) → GELU → LayerNorm → Dropout(0.1) → Linear(512→128)
    L2 normalise
    ClassifierHead: Linear(128→7)

Training:
    80/20 train/val split (same seed as MAE run for fair comparison)
    CrossEntropyLoss with label_smoothing=0.1
    Cosine annealing LR
```

**Terminal:**
```powershell
python models/projection_model.py `
    --embeddings "data/embeddings/step_5_tdv_cls.pt" `
    --labels     "data/labels/step_5/labels.pt" `
    --output_dir "checkpoints/projection_tdv" `
    --num_classes 7
```

---

## Comparison Framework

### What the numbers mean

| Metric | MAE | TDV | Interpretation |
|---|---|---|---|
| Self-supervised loss | Reconstruction MSE | Temporal diff MSE | Different scales — not directly comparable |
| Projection val acc | 42.7% | TBD | **Primary comparison metric** |
| Projection val loss | 1.670 | TBD | Secondary |
| Loss reduction | 80% (1.32→0.271) | TBD% | Relative learning signal |

### Why val accuracy is the right comparison metric

The self-supervised losses cannot be compared directly — they measure
fundamentally different things (pixel reconstruction vs latent
correspondence). What can be compared is: **given the 768-d embeddings
each method produces, how well can a small supervised model classify
surgical actions?** This is the projection model's validation accuracy.

A higher val accuracy means the self-supervised training produced
embeddings that better separate surgical action classes — which is
exactly the property needed for downstream robotics and planning.

### What a TDV win would mean

If TDV val accuracy > MAE val accuracy:
Temporal motion information (instrument speed, tissue dynamics, flow
patterns) is more discriminative for surgical action recognition than
spatial appearance alone. A robot should prioritise motion-based
representations.

### What a MAE win would mean

If MAE val accuracy > TDV val accuracy:
Spatial appearance (instrument type, tissue colour, scene geometry)
is more discriminative than motion for your step-5 surgical clips.
This could reflect the 1fps downsampling — at 1 fps, consecutive
frames may be too dissimilar for reliable temporal correspondence
learning. Higher fps preprocessing could change this outcome.

### Current MAE baseline

```
Reconstruction loss : 1.32 → 0.271  (↓80% over 50 epochs)
Projection val acc  : 42.7%
Projection val loss : 1.670
Projection train acc: 36.5%
```

### Quick comparison after TDV training

```powershell
python -c "
import json

mae_proj = [json.loads(l) for l in
            open('checkpoints/projection/log.txt')]
tdv_proj = [json.loads(l) for l in
            open('checkpoints/projection_tdv/log.txt')]

mae_best = max(e['val_acc'] for e in mae_proj)
tdv_best = max(e['val_acc'] for e in tdv_proj)

print(f'MAE best val acc : {mae_best:.2f}%')
print(f'TDV best val acc : {tdv_best:.2f}%')
print(f'Winner           : {\"TDV\" if tdv_best > mae_best else \"MAE\"} '
      f'(+{abs(tdv_best-mae_best):.2f}%)')
"
```

---

## File Map

```
RARP/
├── datasets/
│   ├── frame_dataset.py          MAE  — single frame index
│   ├── RARPclip_dataset.py       TDV  — consecutive pairs + eval clips
│   └── prepare_labels.py         Shared — frame → action label mapping
│
├── models/
│   ├── mae_wrapper.py            MAE encoder + adapter + MAE decoder
│   ├── tdv_wrapper.py            DINOv2 encoder + adapter + motion encoder
│   ├── extract_cls.py            MAE  — 768-d embedding extraction
│   ├── extract_tdv_cls.py        TDV  — 768-d embedding extraction
│   └── projection_model.py       Shared — 768→128 supervised compression
│
├── training/
│   ├── train_mae.py              MAE self-supervised training
│   └── train_tdv.py              TDV self-supervised training
│
├── checkpoints/
│   ├── mae_pretrained/           Meta MAE ViT-Base weights
│   ├── dinov2_pretrained/        Meta DINOv2 ViT-B weights
│   ├── mae_finetuned_128/        MAE training output
│   ├── tdv_finetuned/            TDV training output
│   ├── projection/               MAE downstream model
│   └── projection_tdv/           TDV downstream model
│
└── data/
    ├── processed/.../step_5/     clip_XXXX.pt tensors (shared)
    ├── embeddings/
    │   ├── step_5_cls.pt         MAE 768-d embeddings (1304, 768)
    │   └── step_5_tdv_cls.pt     TDV 768-d embeddings (1304, 768)
    └── labels/step_5/
        ├── labels.pt             Shared — LongTensor (1304,)
        └── label_map.json        Shared — class name → index
```

---

## Next Steps After Comparison

**If you have more surgical steps processed:**
Repeat Steps 1–6 for steps 2, 3, 4, 6, 7. Combine embedding files
before training the projection model — more data will give more
reliable accuracy estimates and reduce the effect of step-5-specific
quirks.

**Visualise the embedding spaces:**
```powershell
# t-SNE on MAE embeddings
python -c "
import torch, numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mae = torch.load('data/embeddings/step_5_cls.pt')
tdv = torch.load('data/embeddings/step_5_tdv_cls.pt')
labels = torch.load('data/labels/step_5/labels.pt')

mask = labels >= 0
for name, emb in [('MAE', mae), ('TDV', tdv)]:
    z = TSNE(n_components=2, random_state=42).fit_transform(
        emb['embeddings'][mask].numpy())
    plt.figure(figsize=(8,6))
    plt.scatter(z[:,0], z[:,1], c=labels[mask].numpy(), cmap='tab10', s=5)
    plt.title(f'{name} — t-SNE of 768-d surgical embeddings')
    plt.colorbar()
    plt.savefig(f'data/embeddings/{name.lower()}_tsne.png', dpi=150)
    plt.close()
    print(f'{name} t-SNE saved')
"
```

Well-separated clusters in the t-SNE plot indicate that the
self-supervised training learned action-discriminative features.
Comparing the MAE and TDV t-SNE plots visually shows which method
produces cleaner action boundaries in embedding space.
