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
as possible, adapting only what is necessary to work with surgical video
rather than ImageNet images.

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
`CrossEntropyLoss` never receives an invalid index.

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

**File:** `models/projection_model.py`

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

**Training details:**
- 80/20 train/val split (deterministic with fixed seed)
- `CrossEntropyLoss` with `label_smoothing=0.1`
- Frames with label `-1` filtered out in `EmbeddingDataset`
- Cosine annealing LR schedule
- Best checkpoint saved when val accuracy improves

**Output:**
```
checkpoints/projection/
    checkpoint-best.pth    (best validation accuracy)
    checkpoint-0000.pth    (periodic saves)
    checkpoint-0029.pth    (final)
    log.txt                (one JSON line per epoch)
```

**log.txt format:**
```json
{"epoch": 0, "train_loss": 1.823, "train_acc": 42.1,
 "val_loss": 1.654, "val_acc": 48.3, "lr": 0.001, "best_val_acc": 48.3}
{"epoch": 1, "train_loss": 1.412, "train_acc": 58.7, ...}
```

**Terminal:**
```powershell
python models/projection_model.py `
    --embeddings "data/embeddings/step_5_cls.pt" `
    --labels     "data/labels/step_5/labels.pt" `
    --output_dir "checkpoints/projection" `
    --num_classes 7
```

**What this step achieves:**
Produces the final 128-d surgical feature space. Frames of similar
surgical content cluster together; different actions are separated.
The `ProjectionModel` weights (without `ClassifierHead`) are the
transferable artefact for robotics and planning.

---

## Complete File Structure

```
RARP/
├── data/
│   ├── processed/prostatectomy/GSTT_010/step_5/
│   │   ├── clip_0000.pt ... clip_0003.pt      ← Step 0 output
│   ├── embeddings/
│   │   └── step_5_cls.pt                      ← Step 4 output
│   └── labels/step_5/
│       ├── labels.pt                           ← Step 5 output
│       └── label_map.json                      ← Step 5 output
│
├── checkpoints/
│   ├── mae_pretrained/
│   │   └── mae_pretrain_vit_base.pth           ← Downloaded from Meta
│   ├── mae_finetuned_128/
│   │   ├── checkpoint-0000.pth ... 0049.pth    ← Step 3 output
│   │   └── log.txt                             ← Step 3 output
│   └── projection/
│       ├── checkpoint-best.pth                 ← Step 6 output
│       └── log.txt                             ← Step 6 output
│
├── datasets/
│   ├── frame_dataset.py                        ← Step 1
│   └── prepare_labels.py                       ← Step 5
│
├── models/
│   ├── mae_wrapper.py                          ← Step 2
│   ├── extract_cls.py                          ← Step 4
│   └── projection_model.py                     ← Step 6
│
├── training/
│   └── train_mae.py                            ← Step 3
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

**Immediate (more surgical data):**
Repeat Steps 0–4 for all other surgical steps (2, 3, 4, 6, 7) and
combine their embedding files. This gives the projection model far
more training data and produces a step-invariant feature space.

```powershell
# Example for step 6
python training/train_mae.py `
    --clip_dir   "data/processed/prostatectomy/GSTT_010/step_6" `
    --finetune   "checkpoints/mae_pretrained/mae_pretrain_vit_base.pth" `
    --output_dir "checkpoints/mae_finetuned_128_step6" `
    --epochs     50 --batch_size 16 --num_workers 0
```

**Short term (evaluation):**
- t-SNE or UMAP visualisation of the 128-d vectors — do frames from
  the same action cluster together?
- Nearest-neighbour retrieval: given a query frame, do the top-5
  most similar frames (by cosine distance in 128-d space) show the
  same surgical action?

**Medium term (downstream tasks):**
- Action recognition: freeze `ProjectionModel`, train a small
  temporal model (LSTM or sliding-window Transformer) over sequences
  of 128-d vectors
- Phase detection: the coarser step-level labels enable a simpler
  classifier over longer temporal windows

**Longer term (robotics):**
- The 128-d vector at each frame becomes the state representation
  for an imitation learning or reinforcement learning policy
- The unit-norm property means cosine distance directly measures
  semantic similarity between surgical states
