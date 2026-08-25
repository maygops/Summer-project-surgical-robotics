# MASTERY RARP — TDV Fine-Tuning Project Timeline

## Overarching Goal

Having established a working MAE baseline (spatial reconstruction as the
self-supervised signal), this pipeline asks a second question of the same
surgical data: **is temporal motion more discriminative than spatial
appearance for surgical action recognition?**

TDV (Temporal Difference in Vision) answers this by training a lightweight
motion encoder to predict how a frame's representation changes over time,
rather than reconstructing pixels. Both pipelines produce a 768-d CLS
vector per frame that feeds into the same `ProjectionModel` and the same
`labels.pt` — the comparison is controlled, and the only thing that
differs is the self-supervised training signal.

The implementation adapts
[`ninaddaithankar/tdv`](https://github.com/ninaddaithankar/tdv) (PyTorch
code for *"You Don't Need Strong Assumptions: Visual Representation
Learning via Temporal Differences,"* Daithankar, Gladstone, LeCun & Ji,
2026) to surgical video, in the same way the MAE pipeline adapts the
official Meta MAE repository — reusing the paper's motion-encoder
architecture as-is, and writing a thin wrapper, dataset, and training loop
around it for the surgical domain.

---

## What came from the TDV repo vs. what's custom

| Component | Source |
|---|---|
| DINOv2 ViT-B/14 backbone (`model/cv/dinov2/vision_transformer.py`) | TDV repo, unmodified |
| Cross-attention motion encoder (`dinoViT_xattn_base14`, `model/cv/dinov2_with_cross_attention/`) | TDV repo, unmodified |
| `create_motion_encoder()` factory (`model/model_utils.py`) | TDV repo, unmodified |
| `MLPAdapter` (768→128→768 residual bottleneck) | Custom — not part of TDV repo; mirrors the MAE pipeline's adapter design so both pipelines share the same adaptation mechanism |
| `ProjectionHead` (768→128 linear) | Custom — identical Xavier init to `MAEFineTuner.ProjectionHead`, so both pipelines start downstream training from the same state |
| `TDVFineTuner` wrapper (`models/tdv_wrapper.py`) | Custom — combines the above into the surgical-specific forward pass, loss, and inference interface |
| `SurgicalPairDataset` / `SurgicalWindowDataset` (`datasets/RARPclip_dataset.py`) | Custom — TDV repo assumes generic video loaders; these adapt to the `clip_XXXX.pt` format shared with the MAE pipeline |
| `train_tdv.py` | Custom — mirrors `train_mae.py`'s structure (same optimiser, LR schedule, checkpoint format) rather than the TDV repo's own training script, so MAE and TDV runs are directly comparable |
| InfoNCE discriminative loss | Custom addition, not in the original TDV recipe — added to compensate for not using the paper's EMA-teacher + DINO-loss collapse prevention (see below) |
| Δz variance regularisation | Custom addition, VICReg-style |

**Important architectural divergence from the paper:** the TDV paper's
collapse-prevention relies on an EMA teacher copy of a *trainable* frame
encoder plus a DINO-style self-distillation loss, because in their setup
the same encoder produces both the prediction and the target, and a
trainable shared encoder can trivially collapse to a constant. This
pipeline instead keeps DINOv2 **fully frozen** for both frames — only the
adapter and motion encoder are trainable — so the paper's specific
collapse mode (whole-encoder collapse) doesn't apply. What this pipeline
is instead vulnerable to, and had to specifically fix, is described in
Step 2.

---

## Step 0 — Raw Video Preprocessing *(shared with MAE)*

Identical to the MAE pipeline — the same `clip_XXXX.pt` files are consumed
by both. See the MAE timeline doc for full detail. Briefly:

**Input:** raw `.mp4` + surgical step annotation `.txt`
**Output:** `data/processed/prostatectomy/GSTT_010/step_5/clip_0000.pt … clip_0003.pt`, each containing ImageNet-normalised `(T, C, H, W)` video, fps, timestamps, and action windows.

**Why ImageNet normalisation still applies:** DINOv2 was trained on
ImageNet with the same `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
as MAE, so no separate preprocessing branch is needed for TDV.

---

## Step 1 — Frame-Pair / Frame-Window Datasets

**File:** `datasets/RARPclip_dataset.py`

Two dataset classes, used at different points in the project:

### `SurgicalPairDataset`

**What it is:** builds a flat index of consecutive frame pairs
`(frame_t, frame_{t+1})` within each clip. Pairs never cross clip
boundaries, since a clip boundary means a real recording discontinuity,
not a real one-frame temporal gap.

**Input:** `clip_XXXX.pt` files (same folder as MAE)

**Output (per `__getitem__`):**
```python
{
    "frame_t":     Tensor (3, 224, 224),
    "frame_t1":    Tensor (3, 224, 224),
    "frames":      Tensor (2, 3, 224, 224),  # stacked
    "pt_idx":      int,                       # which clip
    "frame_t_idx": int,                       # position of frame_t
}
```

**Index structure:** a clip with `T` frames contributes `T-1` pairs. Four
clips (452+761+64+27 frames) → 1300 pairs total.

### `SurgicalWindowDataset`

**What it is:** generalises `SurgicalPairDataset` by sampling a *random*
offset `k ∈ [min_offset, max_offset]` per `__getitem__` call, instead of
always `k=1`. Motivation: at low fps, consecutive frames can already be
similar enough in DINOv2's representation space that a motion encoder can
satisfy an MSE loss by predicting near-zero motion. Varying the offset
means "predict no change" is penalised harder on average, since larger
gaps carry more genuine visual change by construction.

**Input:** same `clip_XXXX.pt` files

**Output (per `__getitem__`):**
```python
{
    "frame_t":     Tensor (3, 224, 224),
    "frame_t1":    Tensor (3, 224, 224),   # frame at t + k
    "frames":      Tensor (2, 3, 224, 224),
    "pt_idx":      int,
    "frame_t_idx": int,
    "delta_t":     int,   # the sampled offset k, for logging/analysis
}
```

**Constraint:** offsets are capped per-frame so `frame_t_idx + k` never
exceeds the clip length; frames near a clip's tail have a smaller `k_max`.

**Link to TDV repo:** neither dataset exists in the TDV repo, which
assumes larger, generic video datasets (SomethingSomethingV2) with its own
loader. Both are written from scratch to match the `.pt`-per-clip format
shared with the MAE pipeline, so the same preprocessing (Step 0) serves
both pipelines without duplication.

**Terminal (smoke-test):**
```powershell
python datasets/RARPclip_dataset.py "data/processed/prostatectomy/GSTT_010/step_5"
```

---

## Step 2 — TDVFineTuner (`models/tdv_wrapper.py`)

**What it is:** frozen DINOv2 ViT-B/14 + trainable MLP adapter + TDV
cross-attention motion encoder. The core architectural component,
analogous to `MAEFineTuner` in the MAE pipeline.

### Architecture (training)

```
                    DINOv2 ViT-B/14  [FROZEN, from TDV repo — unmodified]
                           │
              ┌────────────┴────────────┐
              │  frame_t                │  frame_t1
              ▼                         ▼
         CLS token (768)          CLS token (768)
         patch tokens (256×768)   (patch tokens discarded on this branch)
              │                         │
        MLP Adapter [TRAIN]             │   ← target branch is RAW,
          768→128→768, residual         │     no adapter applied
          adapted_cls = z_t             │     (see "Preventing collapse")
              │                         │
              │                         ▼
              │                    z_t1_raw  (frozen, stop-gradient target)
              │
              ▼
   condition_t = [z_t | patch_tokens]   (B, 257, 768)
              │
              ▼
   TDV dinoViT_xattn_base14  [TRAIN, from TDV repo — unmodified]
   Input:      pixel_diff = frame_t1 - frame_t     (B, 3, 224, 224)
   Condition:  condition_t (keyword arg)
   Output:     motion_out["x_norm_clstoken"]  (B, 768)
              │
              ▼  motion_proj (identity if dims already match)
             Δz  (B, 768)
              │
              ▼
   predicted = L2_norm(z_t + Δz)             (B, 768)
   target    = L2_norm(z_t1_raw).detach()    (B, 768)
              │
   ┌──────────┼──────────────────┐
   ▼          ▼                  ▼
 MSE loss   Variance reg.      InfoNCE loss
 (pred_loss) on Δz (var_loss)  (nce_loss, in-batch)
   └──────────┴──────────────────┘
              │
   loss = pred_loss + var_weight·var_loss + nce_weight·nce_loss
```

### Preventing collapse — how this differs from the TDV paper

The paper's collapse-prevention (EMA teacher + DINO-style self-distillation
loss) exists because their frame encoder is *trainable* and produces both
sides of the comparison — removing the motion encoder or the MSE term
collapses their recipe outright (KNN accuracy drops from ~17% to under 2%
in their ablations), and the DINO loss with centering is what stops the
shared trainable encoder from mapping every frame to a constant.

This pipeline keeps DINOv2 fully frozen, so that specific collapse mode
isn't available — but an earlier version of this wrapper hit a **different**
collapse mode: routing *both* `frame_t` and `frame_t1` through the same
trainable `mlp_adapter` let the adapter minimise the loss by making its own
output nearly frame-invariant (measured: `cos(z_t, z_t1)` rose from
DINOv2's natural 0.86 to 0.99 after adapting), which meant the motion
encoder had almost no real temporal gap left to bridge. The fix — routing
the target branch through raw, un-adapted DINOv2 only
(`_encode_frame_raw`, no gradient) — restores the natural temporal
diversity of the frozen backbone as the prediction target, so the adapter
can no longer cheat by degrading both sides of the comparison at once.

The InfoNCE term is a second, independent safeguard: even with a
non-collapsing target, nothing prevents `predicted` from being merely
"close enough on average" to every target in the batch rather than
distinctively close to its *own* target. InfoNCE penalises exactly that
failure mode. It is **not** part of the original TDV recipe (which uses
DINO-style cross-entropy with teacher centering for this purpose) — it's
a simpler substitute chosen because this pipeline doesn't have an EMA
teacher to center against.

**Practical note on loss-term balancing:** `pred_loss` operates on a very
different numeric scale (~1e-3 to 1e-4, since it's MSE between
L2-normalised vectors) than `nce_loss` (cross-entropy, natural range
0–~2 for small batches) or `var_loss`. Weighting these with `nce_weight`
and `var_weight` set too high (e.g. `1.0`, matching `pred_loss`'s
coefficient) will make the discriminative/variance terms dominate the
gradient, actively pushing the adapter toward outputs that satisfy those
terms at the expense of real temporal correspondence. Both weights need
tuning relative to the observed scale of `pred_loss` on your data, not
left at unit weight by default.

### Why the MLP is applied only to the CLS token, not patch tokens

Same rationale as MAE: patch tokens carry the spatial context the motion
encoder needs via cross-attention (`condition_t`), and adapting them too
would let the adapter alter the very context the motion encoder relies on
to interpret `pixel_diff`, rather than adapting only the frame-level
summary the loss actually supervises.

### Why the residual connection

Identical logic to the MAE adapter: the pretrained DINOv2 CLS token is
already a strong feature; the adapter should learn the *delta* needed for
the surgical domain and for temporal correspondence, not relearn features
from scratch. This is also, as noted above, precisely the mechanism that
made the shared-adapter collapse easy — the residual makes it cheap for
the adapter to drift toward near-identity behaviour if nothing prevents it.

### Frozen vs. trainable parameters

```
Frozen (85.6M):    DINOv2 ViT-B/14 encoder (patch_embed, pos_embed,
                   cls_token, 12 transformer blocks, norm) — TDV repo
Frozen (98K):      ProjectionHead (frozen during TDV training)
Trainable (198K):  MLPAdapter — custom
Trainable (38.5M): TDV cross-attention motion encoder (dinoViT_xattn_base14,
                   depth=4) — TDV repo factory, custom depth/config
```

Note the trainable parameter count is dominated by the motion encoder
(38.5M) rather than the adapter (198K) — very different from MAE, where
the trainable share splits between the adapter and an 8-block decoder.
This is expected: the motion encoder is a full cross-attention transformer,
not a lightweight bottleneck.

### `condition` keyword-argument contract

`dinoViT_xattn_base14`'s forward signature requires `condition` to be
passed as a **keyword** argument, not positional — `motion_encoder(pixel_diff, condition=condition_t)`.
It returns a dict with `x_norm_clstoken` (B, 768) and `x_norm_patchtokens`
(B, 256, 768) when called in eval mode; **verify this contract when
switching `model.train()`/`model.eval()`,** since some motion-encoder
implementations branch on `self.training` and can return a bare tensor
instead of the dict during training — check with an `isinstance` guard
after the call rather than assuming the dict form unconditionally.

### Model registry

```python
MOTION_ENCODER_TYPE  = "dinoViT_xattn_base14"
MOTION_ENCODER_DEPTH = 4
DINOV2_EMBED_DIM     = 768
```

Analogous to MAE's `MODEL_CONFIGS` dict, though TDV currently only targets
the ViT-B/14 + `dinoViT_xattn_base14` combination rather than a
size-parameterised family.

**Terminal (smoke-test):**
```powershell
python models/tdv_wrapper.py `
    --checkpoint "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --tdv_repo   "tdv"
```

---

## Step 3 — Self-Supervised TDV Fine-Tuning

**File:** `training/train_tdv.py`

**What it is:** the main training loop, structurally mirroring
`train_mae.py` so the two pipelines' loss curves and checkpoints are
directly comparable — same optimiser, same LR schedule, same checkpoint
format — while swapping in the TDV-specific dataset and loss.

**Input:**
- Clip tensors: `data/processed/prostatectomy/GSTT_010/step_5/`
- DINOv2 pretrained checkpoint: `checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth`

**Output:**
```
checkpoints/tdv_finetuned/
    checkpoint-0000.pth
    checkpoint-0020.pth
    checkpoint-0040.pth
    checkpoint-0049.pth
    log.txt
```

**Training loop (per iteration):**
```
batch["frame_t"], batch["frame_t1"] → GPU
        │
        ▼  torch.amp.autocast("cuda")
loss_dict = model(frame_t, frame_t1, nce_weight=..., nce_temp=...)
loss = loss_dict["loss"]
        │
        ▼  loss /= accum_iter
loss_scaler(loss, optimizer, parameters=model.trainable_parameters(), ...)
        │
        ▼  optimizer.step()   (only MLP adapter + motion encoder updated)
        │
        ▼  lr_sched.adjust_learning_rate(...)
```

**Important: `forward()` returns a dict, not a scalar.** Unlike MAE's
`train_one_epoch`, which calls `.item()` and `.backward()` directly on the
model's output, TDV's `forward()` returns
`{"loss", "pred_loss", "var_loss", "nce_loss", "delta_z_norm"}` so each
component can be logged and diagnosed independently — a single combined
scalar previously hid a case where the loss dropped 98.6% while validation
accuracy stayed completely flat, which was only diagnosable once the
components were logged separately.

**Learning rate schedule:** identical cosine-with-warmup to MAE — 5 epoch
warm-up, cosine decay to `min_lr` over the remaining epochs, per-iteration
adjustment via the same `lr_sched.adjust_learning_rate`.

**Batch size:** 8, not MAE's 16 — the motion encoder (38.5M trainable
params) is substantially larger than MAE's 8-block decoder, and each TDV
sample is two frames rather than one.

**Checkpoint contents:**
```python
{
    "model":      model.state_dict(),   # frozen DINOv2 + trained adapter + motion encoder
    "optimizer":  optimizer.state_dict(),
    "scaler":     loss_scaler.state_dict(),
    "epoch":      49,
    "train_loss": float,   # the *combined* loss (pred+var+nce), not pred_loss alone
    "args":       {...},   # includes mlp_hidden_dim, var_weight, nce_weight, nce_temp
}
```

Note `train_loss` in the checkpoint is the **combined** loss — when
comparing against MAE's reconstruction loss, or against the zero-motion
baseline computed in diagnostics, use the checkpoint's `args` to recover
`nce_weight`/`var_weight` and reconstruct `pred_loss` alone if a clean
comparison is needed, rather than comparing combined-loss numbers directly
against MAE's single-term loss.

**Link to TDV repo:**
- DINOv2 loading and motion-encoder construction reuse `create_motion_encoder`
  and the DINOv2 `vision_transformer` module directly from the TDV repo.
- The training loop itself, optimiser, and LR schedule are **not** taken
  from the TDV repo's own training script — they're written to match
  `train_mae.py` instead, so both pipelines are comparable apples-to-apples
  rather than each following its source repo's own conventions.

**Terminal:**
```powershell
python training/train_tdv.py `
    --clip_dir    "data/processed/prostatectomy/GSTT_010/step_5" `
    --checkpoint  "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --output_dir  "checkpoints/tdv_finetuned" `
    --epochs      50 `
    --batch_size  8 `
    --num_workers 0 `
    --seed        42 `
    --nce_weight  0.02 `
    --var_weight  0.02

# Resume
python training/train_tdv.py ... `
    --resume "checkpoints/tdv_finetuned/checkpoint-0020.pth"

# TensorBoard (second terminal)
tensorboard --logdir "checkpoints/tdv_finetuned"
```

---

## Step 4 — CLS Token Extraction

**File:** `models/extract_tdv_cls.py`

**What it is:** runs the trained encoder + adapter over every frame in
inference mode (no motion encoder, no pixel differences — single-frame
input only) and saves the resulting 768-d vectors.

**Input:**
- Fine-tuned checkpoint: `checkpoints/tdv_finetuned/checkpoint-0049.pth`
- Clip tensors: `data/processed/prostatectomy/GSTT_010/step_5/`

**Output:**
```
data/embeddings/step_5_tdv_cls.pt
{
    "embeddings":    Tensor (1304, 768),
    "clip_indices":  Tensor (1304,),
    "frame_indices": Tensor (1304,),
    "model":         "tdv",     # distinguishes from MAE embeddings
}
```

**Key difference from MAE extraction:** MAE's inference setting
(`mask_ratio=0.0`) exists to give the reconstruction pathway full image
context instead of the 25% used during training. TDV has no equivalent
knob — DINOv2 always sees the full frame, with or without training — and
critically, **the motion encoder is not used at inference at all.**
`extract_cls()` = frozen DINOv2 + trained adapter, applied to a single
frame, exactly the `_encode_frame()` path used for the `z_t` branch during
training (not `_encode_frame_raw`, which is target-only and never has the
adapter applied).

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

## Step 5 — Label Preparation *(shared with MAE)*

**File:** `datasets/prepare_labels.py`

No re-run needed if MAE's labels already exist — `labels.pt` and
`label_map.json` are frame-index-aligned and independent of which
embedding model produced the vectors. See the MAE timeline doc for full
detail.

---

## Step 6 — Projection Model Training

**File:** `models/projection_model.py` *(shared architecture with MAE)*

Identical architecture, training procedure, and hyperparameters as the
MAE pipeline's projection model — same 768→512→128 MLP with L2
normalisation, same `CrossEntropyLoss` with label smoothing, same 80/20
split with the same seed, so validation accuracy is the controlled
comparison metric between the two self-supervised signals.

**Input:**
- `data/embeddings/step_5_tdv_cls.pt`
- `data/labels/step_5/labels.pt` (same file used for MAE)

**Output:**
```
checkpoints/projection_tdv/
    checkpoint-best.pth
    checkpoint-0000.pth
    checkpoint-0029.pth
    log.txt
```

**Terminal:**
```powershell
python models/projection_model.py `
    --embeddings "data/embeddings/step_5_tdv_cls.pt" `
    --labels     "data/labels/step_5/labels.pt" `
    --output_dir "checkpoints/projection_tdv" `
    --num_classes 7
```

**What this step achieves:** the actual answer to "did TDV's temporal
signal produce more discriminative embeddings than MAE's spatial
signal?" — validation accuracy here, not the self-supervised loss value,
is the metric that's comparable across the two pipelines, since the raw
losses measure fundamentally different things (pixel reconstruction MSE
vs. latent temporal-correspondence MSE) on different scales.

---

## Step 6.5 — Diagnostics (TDV-specific, no MAE equivalent)

**File:** `eval/tdv_diagnostics.py`

**What it is:** a post-hoc analysis suite that distinguishes "the motion
encoder learned nothing" from "the adapter is collapsing the target
space" from "the loss terms are unbalanced" — three distinct failure
modes that all present initially as "TDV's validation accuracy is flat."
Not part of the MAE pipeline because MAE's single reconstruction loss
doesn't have this many interacting terms to diagnose.

**Input:** TDV checkpoint, MAE checkpoint (for side-by-side comparison),
DINOv2 pretrained checkpoint, clip directory.

**Tests:**

1. **Zero-motion baseline** — `Lzero = MSE(norm(z_t), norm(z_t1))` computed
   in the adapter's own output space, compared against the checkpoint's
   trained loss. If trained loss is *lower* than this baseline while
   `Δz ≈ 0`, the motion encoder found a shortcut rather than learning real
   motion.

2. **Δz statistics** — norm, per-dimension variance across the batch, and
   directional cosine alignment against the true latent difference
   (`z_t1_raw - z_t`, using the un-adapted target, not the adapted one —
   the adapted comparison measures something training no longer optimises
   against). Low norm/variance indicates collapse to a trivial output;
   low cosine alignment with reasonable norm/variance indicates the motion
   encoder is producing large but *directionally wrong* outputs, typically
   from over-weighted auxiliary loss terms.

3. **Latent difference loss** — reports what the loss would be under
   alternative formulations (`MSE(Δz, z_t1-z_t)`, its normalised form, and
   a pure cosine loss) without retraining, to separate "Δz has the right
   direction but wrong scale" from "Δz is genuinely misaligned."

4. **Temporal similarity comparison** — `cos(z_t, z_t1)` for raw DINOv2,
   the TDV adapter, and the MAE adapter side by side. This is the single
   most informative number for catching shared-adapter collapse: if the
   adapter's temporal similarity is markedly *higher* than raw DINOv2's,
   the adapter is actively destroying the temporal variation the motion
   encoder needs; if it drops too far *below* raw DINOv2, an over-weighted
   discriminative loss term is likely pushing embeddings apart in ways
   unrelated to genuine content or motion.

**Terminal:**
```powershell
python eval/tdv_diagnostics.py `
    --tdv_checkpoint  "checkpoints/tdv_finetuned/checkpoint-0049.pth" `
    --mae_checkpoint  "checkpoints/mae_finetuned_128/checkpoint-0049.pth" `
    --dinov2_ckpt     "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --mae_pretrained  "checkpoints/pretrained/mae_pretrain_vit_base.pth" `
    --clip_dir        "data/processed/prostatectomy/GSTT_010/step_5" `
    --output_dir      "eval/diagnostics" `
    --batch_size      16
```

---

## Complete File Structure

```
RARP/
├── data/
│   ├── processed/prostatectomy/GSTT_010/step_5/
│   │   ├── clip_0000.pt ... clip_0003.pt        ← Step 0 output (shared)
│   ├── embeddings/
│   │   ├── step_5_cls.pt                         ← MAE Step 4 output
│   │   └── step_5_tdv_cls.pt                     ← TDV Step 4 output
│   └── labels/step_5/
│       ├── labels.pt                              ← Step 5 output (shared)
│       └── label_map.json                         ← Step 5 output (shared)
│
├── tdv/                                           ← cloned TDV repo
│   └── model/
│       ├── cv/dinov2/                             ← frozen backbone source
│       ├── cv/dinov2_with_cross_attention/        ← motion encoder source
│       └── model_utils.py                         ← create_motion_encoder()
│
├── checkpoints/
│   ├── dinov2_pretrained/
│   │   └── dinov2_vitb14_pretrain.pth             ← Downloaded from Meta
│   ├── tdv_finetuned/
│   │   ├── checkpoint-0000.pth ... 0049.pth       ← Step 3 output
│   │   └── log.txt                                ← Step 3 output
│   └── projection_tdv/
│       ├── checkpoint-best.pth                    ← Step 6 output
│       └── log.txt                                ← Step 6 output
│
├── datasets/
│   ├── RARPclip_dataset.py                        ← Step 1 (SurgicalPairDataset,
│   │                                                  SurgicalWindowDataset)
│   └── prepare_labels.py                          ← Step 5 (shared)
│
├── models/
│   ├── tdv_wrapper.py                             ← Step 2
│   ├── extract_tdv_cls.py                         ← Step 4
│   └── projection_model.py                        ← Step 6 (shared)
│
├── training/
│   └── train_tdv.py                               ← Step 3
│
└── eval/
    └── tdv_diagnostics.py                         ← Step 6.5
```

---

## Next Steps

**Immediate (loss balancing):**
Rerun Step 3 with `nce_weight` and `var_weight` scaled down to the same
order of magnitude as `pred_loss` (starting point ~0.02, not the previous
default of 1.0), then rerun Step 6.5 diagnostics before regenerating
embeddings — confirm `cos(z_t, z_t1)` for the TDV adapter lands between
the DINOv2-raw baseline (~0.86) and a clearly-collapsed value (~0.99),
rather than overshooting below it.

**Short term (same as MAE):**
- t-SNE/UMAP of the 128-d TDV projection vectors vs. MAE's, side by side.
- Nearest-neighbour retrieval comparison between the two embedding spaces.

**Medium term (more surgical data):**
Repeat Steps 0–4 for steps 2, 3, 4, 6, 7, combining embedding files before
projection-model training — same rationale as the MAE pipeline, plus the
added benefit of more temporal variety for the motion encoder to learn
from, which directly addresses the "1fps frames may be too similar"
concern from `SurgicalWindowDataset`'s design.

**Longer term (architecture):**
If InfoNCE + variance regularisation prove insufficient once properly
balanced, the next escalation is closer to the paper's actual recipe: an
EMA-updated copy of the adapter (not the full DINOv2 backbone, which stays
frozen) producing the target, paired with a proper DINO-style
self-distillation loss with running-mean centering — the paper's own
ablations show centering matters more than temperature sharpening for
avoiding the subtler, dimensional form of collapse.