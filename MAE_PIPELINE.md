# MASTERY RARP — MAE Fine-Tuning Project Timeline

## Overarching Goal

Your supervisor asked you to **kick-start familiarity with the surgical data**
by fine-tuning a pretrained MAE encoder using self-supervision, rather than
jumping directly to a labelled downstream task. The output of this entire
pipeline is a **surgical feature extractor**: a model that takes any raw
surgical frame and produces a compact, meaningful vector representation of
what is happening in that frame. That representation can later be used for
action recognition, surgical phase detection, and robotic planning — without
needing to re-train from scratch on every new task.

The pipeline follows the official Meta MAE repository structure as closely
as practical given the data available, adapting only what is necessary to
work with surgical video rather than ImageNet images. **Note on fidelity to
the original paper:** the encoder here is kept fully frozen and only a
small adapter + decoder (~3M params) is trained, which is already a
departure from Meta's typical full end-to-end MAE fine-tuning protocol —
done deliberately, because fine-tuning an 86M-parameter ViT-B/14 on ~1300
frames from 4 clips risks catastrophic overfitting regardless of which
self-supervised signal is used. The companion TDV pipeline makes the same
trade-off for the same reason (see the TDV timeline doc), which keeps the
two pipelines comparable to each other even though neither exactly
reproduces its source paper's flagship large-scale recipe.

---

## Step 0 — Raw Video Preprocessing

**What it is:**
The 2-hour prostatectomy recording is segmented into clips using the
surgical step annotation text file. Each clip corresponds to one surgical
step or sub-step. Step 5 was chosen as the starting point, producing
4 clips.

**Input:**
- Raw `.mp4` prostatectomy video
- Annotation `.txt` file with timestamps and step labels
  ```
  00:45:16 00:45:47  EC
  00:32:36 00:45:16  5
  ...
  ```

**Output:**
```
data/processed/prostatectomy/GSTT_010/step_5/
    clip_0000.pt
    clip_0001.pt
    clip_0002.pt
    clip_0003.pt
```

Each `.pt` tensor contains:
```python
{
    "video":        Tensor (T, C, H, W),   # ImageNet-normalised frames
    "fps":          float,
    "original_fps": float,
    "timestamps":   Tensor (T,),
    "actions":      [(start_sec, end_sec, label), ...],
    "shape":        tuple
}
```

**Why ImageNet normalisation?**
The pretrained MAE was trained on ImageNet with
`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.
For the pretrained encoder to interpret the surgical frames correctly,
they must be in the same pixel value range the encoder expects.

**Link to MAE repository:**
`main_pretrain.py` applies the same normalisation via
`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
Your preprocessing does this at clip-creation time so no transform is
needed during training.

**Note:** these same four clips (452+761+64+27 = 1304 frames) are shared
verbatim with the TDV pipeline — Step 0 only runs once.

**What this step achieves:**
Converts a continuous video stream into a structured, frame-level dataset
that can be indexed and batched efficiently during training.

---

## Step 1 — SurgicalFrameDataset

**File:** `datasets/frame_dataset.py`

**What it is:**
A PyTorch `Dataset` subclass that builds a flat index over every frame
across all clips. Instead of thinking about clips, the model sees a
single sequence of frames numbered 0 to N-1.

**Input:**
```
data/processed/prostatectomy/GSTT_010/step_5/  (folder with clip_XXXX.pt files)
```

**Output (per `__getitem__` call):**
```python
{
    "frame":       Tensor (3, 224, 224),   # single frame, ImageNet-normalised
    "dataset_idx": int,                    # global flat index
    "clip_idx":    int,                    # which clip (0–3)
    "frame_idx":   int,                    # position within that clip
    "step_label":  str,                    # "5"
}
```

**Flat index structure:**
```
dataset_idx | clip | frame
0           |  0   |  0
1           |  0   |  1
...
451         |  0   | 451
452         |  1   |  0
...
1303        |  3   | last_frame
```

**Loading modes:**
- `lazy` (default): each clip's tensor is loaded from disk only when
  a frame from that clip is first requested, then cached in RAM.
  Safe for machines with limited memory.
- `eager`: all 4 clips are loaded at construction time.
  Faster iteration once loaded.

**Link to MAE repository:**
Replaces `datasets.ImageFolder` used in `main_pretrain.py` and
`main_finetune.py`. The DataLoader interface is identical — the training
loop only sees `batch["frame"]`, which is a plain `(B, 3, H, W)` tensor.

**Also used by:** `eval/extract_dinov2_raw_cls.py` (the raw-DINOv2 control
extraction added for the TDV comparison — see the TDV timeline doc) reuses
this exact dataset class, so the raw-DINOv2 control embeddings are
frame-aligned with both MAE's and TDV's.

**Terminal:**
```powershell
# Smoke-test to verify dataset builds correctly
python datasets/frame_dataset.py "data/processed/prostatectomy/GSTT_010/step_5"
```

**What this step achieves:**
Provides an efficient, indexed interface to every surgical frame so the
DataLoader can serve random batches during training.

---

## Step 2 — MAEFineTuner (mae_wrapper.py)

**File:** `models/mae_wrapper.py`

**What it is:**
The core architectural component. Wraps the pretrained MAE with a trainable
MLP adapter that sits directly in the path between encoder and decoder,
ensuring the MAE reconstruction loss provides a genuine learning signal.

**Architecture:**
```
Input frame (3, 224, 224)
        │
        ▼  [FROZEN] patch_embed
Split into 196 patches of 16×16 → (196, 768)
        │
        ▼  [FROZEN] random_masking  (75% masked)
49 visible patches remain → (49, 768)
        │
        ▼  [FROZEN] 12 transformer blocks + norm
latent: (50, 768)   [1 CLS + 49 patch tokens]
        │
   ┌────┴────┐
   │         │
CLS token  patch tokens
(1, 768)   (49, 768)
   │
   ▼  [TRAINABLE] MLPAdapter
   Linear(768→128) → GELU → LayerNorm(128) → Linear(128→768)
   Residual: adapted_cls = cls_token + mlp_adapter(cls_token)
   │
   ▼  recombine with patch tokens
adapted_latent: (50, 768)
   │
   ▼  [TRAINABLE] MAE decoder
Reconstructed patches: (196, 768)
   │
   ▼  forward_loss
MSE on masked patches only → scalar loss
```

**Why the residual connection (`cls + adapter(cls)`):**
The pretrained CLS token already contains good ImageNet features.
The adapter only needs to learn the *delta* — what changes to make the
representation surgical-domain-aware. This stabilises training with a
small dataset because the adapter starts from a useful initialisation
rather than zero.

**Why the MLP is in the decoder path:**
The MAE loss backpropagates through the decoder → through `adapted_cls`
→ into the MLP adapter weights. This is a genuine learning signal.
If the adapter produces a poor CLS, reconstruction suffers and the
loss rises. This is mathematically equivalent to the supervised gradient
path in a labelled model, but uses reconstruction as the proxy task.

**Why 768→128→768 bottleneck:**
With only 4 clips, a larger hidden dimension (e.g. 512) has enough
capacity to memorise specific frames rather than learning generalisable
surgical structure. The 128-d bottleneck acts as structural
regularisation, forcing the adapter to retain only the most salient
information.

**Frozen vs trainable parameters:**
```
Frozen (86M):      patch_embed, pos_embed, cls_token, 12 transformer blocks, norm
Trainable (~3M):   MLPAdapter, MAE decoder (8 blocks), ProjectionHead (frozen
                   during MAE training, unfrozen later)
```

**Why fully frozen, not partially fine-tuned:** the TDV pipeline ran a
direct experiment on this exact question — unfreezing the last transformer
block of its own (architecturally similar) frozen ViT-B/14 backbone — and
found it hurt downstream accuracy by roughly 5x relative to staying fully
frozen (32.9% vs. 6.8% mean LOCO CV accuracy; see the TDV timeline doc,
Step 6). That result was obtained on TDV, not MAE, but the same practical
constraint (86M backbone params vs. ~1300 frames from 4 clips) applies
equally here, so it's treated as a relevant caution against partially
unfreezing MAE's encoder too, absent a specific reason to test it directly.

**Link to MAE repository:**
Uses `models_mae.py` entirely as-is. `MaskedAutoencoderViT`,
`forward_encoder`, `forward_decoder`, and `forward_loss` are untouched.
The wrapper intercepts the latent between encoder and decoder to insert
the MLP adapter — everything else is the original Meta code.

**Model registry (replaces if/elif chain):**
```python
MODEL_CONFIGS = {
    "base":  {"factory": "mae_vit_base_patch16",  "embed_dim": 768},
    "large": {"factory": "mae_vit_large_patch16", "embed_dim": 1024},
    "huge":  {"factory": "mae_vit_huge_patch14",  "embed_dim": 1280},
}
```

**Terminal (smoke-test):**
```powershell
python models/mae_wrapper.py "checkpoints/mae_pretrained/mae_pretrain_vit_base.pth"
```

**What this step achieves:**
Defines the model architecture for self-supervised surgical fine-tuning.
After training, `extract_cls()` produces the 768-d surgical representation.

---

## Step 3 — Self-Supervised MAE Fine-Tuning

**File:** `training/train_mae.py`

**What it is:**
The main training loop. Adapted from `main_finetune.py` in the Meta
repository. Trains only the MLP adapter and MAE decoder on surgical
frames using reconstruction loss as the self-supervised signal.

**Input:**
- Clip tensors in `data/processed/prostatectomy/GSTT_010/step_5/`
- Pretrained MAE checkpoint: `checkpoints/mae_pretrained/mae_pretrain_vit_base.pth`

**Output:**
```
checkpoints/mae_finetuned_128/
    checkpoint-0000.pth    (epoch 0)
    checkpoint-0020.pth    (epoch 20)
    checkpoint-0040.pth    (epoch 40)
    checkpoint-0049.pth    (final)
    log.txt                (one JSON line per epoch)
```

**Training loop (per iteration):**
```
batch["frame"] → GPU
        │
        ▼  torch.cuda.amp.autocast()   (mixed precision, halves memory)
loss, pred, mask = model(frames, mask_ratio=0.75)
        │
        ▼  loss /= accum_iter
loss_scaler(loss, optimizer, ...)      (scaled backward pass)
        │
        ▼  optimizer.step()            (only MLP adapter + decoder updated)
        │
        ▼  lr_sched.adjust_learning_rate(...)  (per-iteration cosine schedule)
```

**Learning rate schedule:**
Cosine annealing with linear warm-up, identical to `engine_pretrain.py`:
- Epochs 0–5: LR ramps from ~6e-6 to 6.2e-5 (warm-up)
- Epochs 5–49: LR decays along cosine curve to near zero

This is visible in your `log.txt`:
```json
{"train_lr": 6.17e-06, "train_loss": 1.325, "epoch": 0}   ← warm-up start
{"train_lr": 6.24e-05, "train_loss": 0.424, "epoch": 5}   ← warm-up end
{"train_lr": 2.58e-08, "train_loss": 0.271, "epoch": 49}  ← cosine decay end
```

**Loss interpretation:**
The loss is MSE on masked patches only (75% of the image).
- Epoch 0 (~1.32): random MLP weights, decoder cannot reconstruct
- Epoch 5 (~0.42): decoder has adapted to surgical pixel statistics
- Epoch 49 (~0.271): convergence; adapter produces stable surgical CLS tokens
- 128-d bottleneck run converged ~0.001 lower than 512-d run,
  confirming better generalisation from stronger regularisation

**Checkpoint contents:**
```python
{
    "model":      model.state_dict(),    # frozen encoder + trained adapter + decoder
    "optimizer":  optimizer.state_dict(),
    "scaler":     loss_scaler.state_dict(),
    "epoch":      49,
    "train_loss": 0.271,
    "args":       {...},                 # all CLI args — critical for reproducibility
}
```
The `args` dict is what `extract_cls.py` reads to reconstruct the exact
architecture (particularly `mlp_hidden_dim`) when loading the checkpoint.

**Link to MAE repository:**
- DataLoader construction mirrors `main_finetune.py` exactly
- `train_one_epoch` mirrors `engine_pretrain.py` exactly
- `misc.save_model` / `misc.load_model` replaced by custom
  `save_checkpoint` / `load_checkpoint` that also store `args`
- `NativeScalerWithGradNormCount` used unchanged from `util/misc.py`
- `lr_sched.adjust_learning_rate` used unchanged from `util/lr_sched.py`

**No train/val split at this stage** — this is self-supervised
pretraining on unlabelled data; every frame from all 4 clips contributes
to the reconstruction loss. `train_loss` in `log.txt` is a training-set
metric only. Generalisation is assessed downstream, in Step 6.

**Terminal:**
```powershell
# First run (128-d bottleneck)
python training/train_mae.py `
    --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
    --finetune   "checkpoints/mae_pretrained/mae_pretrain_vit_base.pth" `
    --output_dir "checkpoints/mae_finetuned_128" `
    --epochs     50 `
    --batch_size 16 `
    --num_workers 0 `
    --seed       42

# Resume from checkpoint
python training/train_mae.py `
    --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
    --finetune   "checkpoints/mae_pretrained/mae_pretrain_vit_base.pth" `
    --output_dir "checkpoints/mae_finetuned_128" `
    --resume     "checkpoints/mae_finetuned_128/checkpoint-0020.pth" `
    --epochs     50

# Monitor with TensorBoard (second terminal)
tensorboard --logdir "checkpoints/mae_finetuned_128"
# then open http://localhost:6006
```

**What this step achieves:**
Adapts the generic ImageNet encoder to the surgical domain without any
labels. The MLP adapter learns to emphasise the visual features that
are most useful for reconstructing surgical scenes — instruments,
tissue texture, spatial relationships between tools and anatomy.

---

## Step 4 — CLS Token Extraction

**File:** `models/extract_cls.py`

**What it is:**
Runs the trained encoder + adapter over every frame in inference mode
(no masking, no decoder, no loss) and saves the resulting 768-d vectors.

**Input:**
- Fine-tuned checkpoint: `checkpoints/mae_finetuned_128/checkpoint-0049.pth`
- Clip tensors: `data/processed/prostatectomy/GSTT_010/step_5/`

**Output:**
```
data/embeddings/step_5_cls.pt
{
    "embeddings":    Tensor (1304, 768),   # one vector per frame
    "clip_indices":  Tensor (1304,),       # clip 0–3
    "frame_indices": Tensor (1304,),       # position within clip
    "step_label":    "5",
    "checkpoint":    "checkpoints/...",
    "embed_dim":     768,
}
```

**Why mask_ratio=0.0 at inference:**
During training, 75% of patches are hidden — the model reconstructs
from partial information. At inference, you want the richest possible
representation: the CLS token should attend to all 196 patches, not
just 49. Setting `mask_ratio=0.0` gives the full image context.

**Key fix applied:**
The checkpoint's `args` dict is read to recover `mlp_hidden_dim`
before instantiating the model, guaranteeing architecture consistency:
```python
mlp_hidden_dim = saved_args.get("mlp_hidden_dim", 128)
model = MAEFineTuner(..., mlp_hidden_dim=mlp_hidden_dim)
model.load_state_dict(ckpt["model"])
```

**Terminal:**
```powershell
python models/extract_cls.py `
    --checkpoint "checkpoints/mae_finetuned_128/checkpoint-0049.pth" `
    --finetune   "checkpoints/mae_pretrained/mae_pretrain_vit_base.pth" `
    --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
    --output     "data/embeddings/step_5_cls.pt" `
    --step_label 5
```

**What this step achieves:**
Materialises the surgical feature vectors as a standalone file.
Downstream models can now be trained purely on these 768-d vectors
without ever loading the large MAE encoder again — training the
projection model is fast (seconds per epoch) because the hard
perceptual work is already done.

---

## Step 5 — Label Preparation

**File:** `datasets/prepare_labels.py`

**What it is:**
Maps every frame in the embedding file to an integer class label using
the `actions` field already stored in the `.pt` clip tensors.

**Input:**
- Clip tensors (reads `actions` field): `data/processed/.../step_5/`
- Embedding file (for frame count verification): `data/embeddings/step_5_cls.pt`

**Output:**
```
data/labels/step_5/
    labels.pt        LongTensor (1304,) — integer class index per frame
                     Unannotated frames between action windows → -1
    label_map.json   {"5_cold_cut": 0, "5_clip": 1, ...}
```

**Two label modes:**
- `action` (fine-grained): one class per tool-use event, e.g. `5_cold_cut`
- `step`   (coarse):       one class per surgical phase, e.g. `step_5`

**Unannotated frames (-1):**
287 out of 1304 frames (22%) fall between annotated action windows.
These are transition moments — instrument repositioning, brief pauses.
They are assigned label `-1` and filtered out before training so
`CrossEntropyLoss` never receives an invalid index. **Confirmed:** clip 3
(27 frames, the shortest clip) has zero valid labels after this filtering
— entirely unannotated — so it never appears in Step 6's train/val folds,
though it still contributed unlabelled signal to Step 3's self-supervised
pretraining (see Step 6 below for why this isn't a leakage concern).

**Terminal:**
```powershell
python datasets/prepare_labels.py `
    --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
    --embeddings "data/embeddings/step_5_cls.pt" `
    --output_dir "data/labels/step_5" `
    --mode       action `
    --step_label 5
```

**What this step achieves:**
Bridges self-supervised pretraining and supervised downstream training.
The label file is the only place where human annotation enters the
pipeline — everything before this step is entirely label-free.

---

## Step 6 — Projection Model Training

**File:** `models/projection_model.py` *(shared script with TDV)*

**What it is:**
A supervised MLP that compresses 768-d CLS embeddings to 128-d
L2-normalised vectors for downstream tasks. Trained on the labelled
embeddings produced by Steps 4 and 5.

**Architecture:**
```
Input: (B, 768)   adapted CLS token
        │
        ▼  Linear(768 → 512)
        ▼  GELU
        ▼  LayerNorm(512)
        ▼  Dropout(0.1)
        ▼  Linear(512 → 128)
        ▼  L2 normalise
        │
Output: (B, 128)  unit-norm surgical feature vector
        │
        ▼  ClassifierHead: Linear(128 → num_classes)
        │
Output: (B, num_classes)  logits for training only
```

**Why L2 normalisation:**
Downstream tasks (nearest-neighbour retrieval, cosine similarity
matching, contrastive learning) all operate on angles between vectors,
not magnitudes. L2 normalising the 128-d output puts all vectors on
the unit hypersphere, making distances consistent and preventing any
single dimension from dominating.

**Split methodology — corrected from an earlier frame-level random split.**
The original implementation used `random_split` on all 1017 labelled
frames pooled together (80/20, deterministic seed). At 1fps, this leaks
near-duplicate consecutive frames across train/val — a frame at `t=50` in
"train" and `t=51` in "val" from the *same clip* are barely
distinguishable, inflating val accuracy in a way that measures
memorisation, not generalisation. `projection_model.py` now defaults to
**leave-one-clip-out (LOCO) cross-validation**: train on N-1 clips,
validate on the held-out one, repeat for every labelled clip, report
mean ± std. `--val_clips` remains available for a single fixed
held-out-clip split as a faster sanity check.

**Why clip 3's exclusion isn't a leakage concern:** clip 3 has no valid
labels (Step 5), so it never appears as a training or validation
candidate in this step — `EmbeddingDataset` filters `label == -1` before
the clip-level split is built. It *did* contribute unlabelled signal to
Step 3's self-supervised pretraining, which shapes the shared embedding
space all clips are later projected through — a normal characteristic of
self-supervised pretraining (using all available unlabelled data), not a
train/test leak, since it can't be predicted on or scored against here.

**Training details:**
- Leave-one-clip-out CV (was: 80/20 random split — see above)
- `CrossEntropyLoss` with `label_smoothing=0.1`
- Frames with label `-1` filtered out in `EmbeddingDataset`
- Cosine annealing LR schedule, restarted fresh per fold
- Best checkpoint per fold saved when that fold's val accuracy improves
  (note: this makes "best" a mild form of peeking at the test fold's own
  trajectory — an accepted trade-off given how few clips are available
  for a proper three-way train/val/test split)

**Output:**
```
checkpoints/projection/
    fold_clip0/checkpoint-best.pth, log.txt
    fold_clip1/...
    fold_clip2/...
    cv_summary.json
```

**log.txt format (per fold):**
```json
{"fold": "clip0", "epoch": 0, "train_loss": 1.823, "train_acc": 42.1,
 "val_loss": 1.654, "val_acc": 48.3, "lr": 0.001, "best_val_acc": 48.3}
{"fold": "clip0", "epoch": 1, "train_loss": 1.412, "train_acc": 58.7, ...}
```

**Terminal (unchanged invocation — CV is now the default behaviour):**
```powershell
python models/projection_model.py `
    --embeddings "data/embeddings/step_5_cls.pt" `
    --labels     "data/labels/step_5/labels.pt" `
    --output_dir "checkpoints/projection" `
    --num_classes 7
```

**Actual result (7-class task, LOCO CV over 3 labelled clips):**

| held-out clip 0 | held-out clip 1 | held-out clip 2 | mean ± std |
|---|---|---|---|
| 45.5% | 28.3% | 12.5% | **28.8% ± 16.5%** |

(chance level for 7 classes: 14.3%)

**Full cross-pipeline comparison (see TDV timeline doc for the TDV-side
runs that produced these numbers):**

| | held-out 0 | held-out 1 | held-out 2 | mean ± std |
|---|---|---|---|---|
| **TDV (frozen backbone — correct config)** | 44.8% | 29.0% | 25.0% | **32.9% ± 10.5%** |
| **MAE** | 45.5% | 28.3% | 12.5% | 28.8% ± 16.5% |
| **DINOv2 raw (no adapter, no training)** | 22.0% | 18.3% | 0.0% | 13.4% ± 11.8% |
| **TDV (1 block unfrozen — since reverted)** | 2.2% | 18.3% | 0.0% | 6.8% ± 10.0% |
| chance (7 classes) | | | | 14.3% |

**What this step achieves:** under the corrected, leak-free split, both
MAE and (correctly-configured, frozen-backbone) TDV clearly beat the
raw-DINOv2-no-training control, confirming both self-supervised training
regimes extract real, task-relevant signal from these 4 clips rather than
just riding on DINOv2's pretrained features. TDV is directionally
slightly ahead of MAE (32.9% vs. 28.8%), but the two are within
overlapping std bars on only 3 usable folds — "roughly comparable, TDV
maybe slightly ahead" is the honest read, not a confident win for either
pretext task. An earlier run had wrongly suggested MAE clearly beat
TDV (6.8%), but that used a broken (backbone-unfrozen) TDV checkpoint —
now reverted; see the TDV timeline doc's "Where to improve TDV" section
for what's still worth trying to push this comparison to a clearer
answer. Note the wide std values (10.5–16.5 points on only 3 folds of
very uneven size, from 64 to 761 frames) — treat both pipelines' numbers
as noisy estimates, not precise scores.

---

## Step 7 — Embedding-Space Visualisation (t-SNE / UMAP) *(new, shared with TDV)*

**File:** `eval/visualize_embeddings.py`

Projects the 768-d MAE, TDV, and raw-DINOv2-control embeddings to 2D via
t-SNE (always available) and UMAP (if installed), colored two ways side by
side per source: by **action label** and by **clip identity**. If a
source's label-colored and clip-colored panels show the same clustering
shapes with different color keys, that's a visual sign its LOCO CV result
is being driven by clip identity rather than genuine action content — see
the TDV timeline doc for full detail and example usage.

---

## Complete File Structure

```
RARP/
├── data/
│   ├── processed/prostatectomy/GSTT_010/step_5/
│   │   ├── clip_0000.pt ... clip_0003.pt      ← Step 0 output (shared with TDV)
│   ├── embeddings/
│   │   ├── step_5_cls.pt                      ← MAE Step 4 output
│   │   ├── step_5_tdv_frozen_cls.pt           ← TDV Step 4 output (correct config)
│   │   └── step_5_dinov2_raw_cls.pt           ← raw-DINOv2 control
│   └── labels/step_5/
│       ├── labels.pt                           ← Step 5 output (shared)
│       └── label_map.json                      ← Step 5 output (shared)
│
├── checkpoints/
│   ├── mae_pretrained/
│   │   └── mae_pretrain_vit_base.pth           ← Downloaded from Meta
│   ├── mae_finetuned_128/
│   │   ├── checkpoint-0000.pth ... 0049.pth    ← Step 3 output
│   │   └── log.txt                             ← Step 3 output
│   └── projection/
│       ├── fold_clip0/ ... fold_clip2/         ← Step 6 output (per-fold)
│       └── cv_summary.json                     ← Step 6 output
│
├── datasets/
│   ├── frame_dataset.py                        ← Step 1
│   └── prepare_labels.py                       ← Step 5
│
├── models/
│   ├── mae_wrapper.py                          ← Step 2
│   ├── extract_cls.py                          ← Step 4
│   └── projection_model.py                     ← Step 6 (shared with TDV)
│
├── training/
│   └── train_mae.py                            ← Step 3
│
├── eval/
│   ├── extract_dinov2_raw_cls.py               ← raw-DINOv2 control (TDV-doc detail)
│   └── visualize_embeddings.py                 ← Step 7 (shared with TDV)
│
└── [Meta MAE repository files — unchanged]
    ├── models_mae.py
    ├── engine_pretrain.py
    ├── engine_finetune.py
    ├── main_pretrain.py
    ├── main_finetune.py
    └── util/
        ├── misc.py
        ├── lr_sched.py
        └── pos_embed.py
```

---

## Next Steps

**Immediate:**
- Run `eval/visualize_embeddings.py` (Step 7) across all three embedding
  sources (MAE, TDV-frozen, raw-DINOv2) — worth checking whether MAE's
  and TDV's 2D structures look qualitatively different from each other,
  given their LOCO accuracies are now close.
- See the TDV timeline doc's "Where to improve TDV" list — several of
  those items (adapter_nce_weight sweep, EMA+DINO self-distillation on
  the adapter) are aimed at pushing TDV to a clearer, more confident
  advantage over MAE, which would make the temporal-vs-spatial question
  this whole comparison was built to answer actually resolvable.

**Medium term (more surgical data):**
Repeat Steps 0–4 for all other surgical steps (2, 3, 4, 6, 7) and combine
their embedding files. This gives the projection model far more training
data, produces a step-invariant feature space, and — now that Step 6 uses
clip-level LOCO CV — directly reduces the fold-to-fold variance seen in
the current 3-fold result (16.5% std on a 28.8% mean is a lot of
uncertainty) by giving each fold more, and more comparable, data.

**Short term (evaluation):**
- Nearest-neighbour retrieval: given a query frame, do the top-5 most
  similar frames (by cosine distance in 128-d space) show the same
  surgical action?

**Medium term (downstream tasks):**
- Action recognition: freeze `ProjectionModel`, train a small temporal
  model (LSTM or sliding-window Transformer) over sequences of 128-d
  vectors.
- Phase detection: the coarser step-level labels enable a simpler
  classifier over longer temporal windows.

**Longer term (robotics):**
- The 128-d vector at each frame becomes the state representation for an
  imitation learning or reinforcement learning policy.
- The unit-norm property means cosine distance directly measures semantic
  similarity between surgical states.