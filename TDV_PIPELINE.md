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
| `MLPAdapter` (768→128→768 residual bottleneck) | Custom — mirrors the MAE pipeline's adapter design |
| `ProjectionHead` (768→128 linear) | Custom — identical Xavier init to `MAEFineTuner.ProjectionHead` |
| `TDVFineTuner` wrapper (`models/tdv_wrapper.py`) | Custom |
| `SurgicalPairDataset` / `SurgicalWindowDataset` | Custom |
| `train_tdv.py` | Custom — mirrors `train_mae.py`'s structure |
| InfoNCE discriminative loss | Custom addition, not in the original TDV recipe |
| Δz variance regularisation | Custom addition, VICReg-style |
| Optional last-N-block backbone unfreezing + differential-LR optimiser groups | Custom addition — see "Optional backbone unfreezing" below |

**Important architectural divergence from the paper:** the TDV paper's
collapse-prevention relies on an EMA teacher copy of a *trainable* frame
encoder plus a DINO-style self-distillation loss. This pipeline instead
keeps DINOv2 **frozen by default** for both frames, so that specific
collapse mode doesn't apply. What this pipeline is instead vulnerable to,
and had to specifically fix, is described in Step 2.

---

## Step 0 — Raw Video Preprocessing *(shared with MAE)*

Identical to the MAE pipeline. See the MAE timeline doc for full detail.

---

## Step 1 — Frame-Pair / Frame-Window Datasets

**File:** `datasets/RARPclip_dataset.py`

### `SurgicalPairDataset`

Builds a flat index of consecutive frame pairs `(frame_t, frame_{t+1})`
within each clip, fixed `k=1`. **Used by:** `eval/tdv_diagnostics.py`.

### `SurgicalWindowDataset`

Generalises `SurgicalPairDataset` with a *random* offset `k` per
`__getitem__` call. **Used by:** `training/train_tdv.py`.

**Note — train/eval gap-distribution mismatch:** training samples a
*variable* offset `k`, but `eval/tdv_diagnostics.py` uses a *fixed* `k=1`
for every test. This is intentional (diagnostics need a fixed, reproducible
offset to compare configurations against each other), but it means the
diagnostic numbers characterise the model's behaviour at the *smallest*
offset in its training distribution, not its average behaviour across the
offsets it actually trained on. Worth rerunning the diagnostic tests at a
few different fixed `k` values (2, 4, 8) to check how much this matters.

---

## Step 2 — TDVFineTuner (`models/tdv_wrapper.py`)

**What it is:** frozen-by-default DINOv2 ViT-B/14 + trainable MLP adapter +
TDV cross-attention motion encoder, with an optional path to unfreeze the
last few DINOv2 blocks.

### Architecture (training)

```
                    DINOv2 ViT-B/14  [frozen by default — TDV repo, unmodified;
                                       optionally last N blocks trainable]
                           |
              +------------+------------+
              |  frame_t                |  frame_t1
              v                         v
         CLS token (768)          CLS token (768)
         patch tokens (256x768)   (patch tokens discarded on this branch)
              |                         |
        MLP Adapter [TRAIN]             |   <- target branch is RAW,
          768->128->768, residual       |     no adapter applied
          adapted_cls = z_t             |
              |                         v
              |                    z_t1_raw  (always stop-gradient target,
              |                              regardless of backbone unfreeze)
              v
   condition_t = [z_t | patch_tokens]   (B, 257, 768)
              |
              v
   TDV dinoViT_xattn_base14  [TRAIN, from TDV repo - unmodified]
   Input:      pixel_diff = frame_t1 - frame_t     (B, 3, 224, 224)
   Condition:  condition_t (keyword arg)
   Output:     motion_out["x_norm_clstoken"]  (B, 768)
              |
              v  motion_proj (identity if dims already match)
             dz  (B, 768)
              |
              v
   predicted = L2_norm(z_t + dz)             (B, 768)
   target    = L2_norm(z_t1_raw).detach()    (B, 768)
              |
   +----------+------------------+
   v          v                  v
 MSE loss   Variance reg.      InfoNCE loss
 (pred_loss) on dz (var_loss)  (nce_loss, in-batch)
   +----------+------------------+
              |
   loss = pred_loss + var_weight*var_loss + nce_weight*nce_loss
        + adapter_nce_weight*adapter_nce_loss
```

### Preventing collapse — how this differs from the TDV paper

The paper's collapse-prevention (EMA teacher + DINO-style self-distillation
loss) exists because their frame encoder is *trainable* and produces both
sides of the comparison. This pipeline keeps DINOv2 frozen by default, so
that specific collapse mode isn't available — but an earlier version of
this wrapper hit a **different** collapse mode: routing *both* `frame_t`
and `frame_t1` through the same trainable `mlp_adapter` let the adapter
minimise the loss by making its own output nearly frame-invariant
(measured: `cos(z_t, z_t1)` rose from DINOv2's natural 0.86 to 0.99 after
adapting). The fix — routing the target branch through raw, un-adapted
DINOv2 only (`_encode_frame_raw`, no gradient) — restores the natural
temporal diversity of the frozen backbone as the prediction target.

The InfoNCE term is a second, independent safeguard against `predicted`
being merely "close enough on average" to every target in the batch rather
than distinctively close to its own target. Not part of the original TDV
recipe (which uses DINO-style cross-entropy with teacher centering) — a
simpler substitute chosen because this pipeline has no EMA teacher.

**Practical note on loss-term balancing:** `pred_loss` operates on a very
different numeric scale (~1e-3 to 1e-4) than `nce_loss` (cross-entropy,
0-~2) or `var_loss`. Weighting `nce_weight`/`var_weight` too high (e.g.
`1.0`, matching `pred_loss`'s implicit coefficient of 1) makes the
discriminative/variance terms dominate the gradient at the expense of real
temporal correspondence. Both weights need tuning relative to the observed
scale of `pred_loss` on your data.

### Optional backbone unfreezing

`unfreeze_last_n_blocks` (default `0`) unfreezes the last N DINOv2
transformer blocks plus the final `norm` layer, on the reasoning that
clip-identity is baked into frozen DINOv2's features before any training
touches them (71.8% nearest-centroid clip accuracy on raw DINOv2 CLS vs.
25% chance) — a downstream adapter sitting on top of a frozen backbone
structurally cannot remove information the backbone already committed to
encoding; only the backbone itself has that leverage.

**Known hard-won pitfall — `torch.no_grad()` silently defeats
`requires_grad`.** `_encode_frame`'s DINOv2 forward pass originally ran
inside an unconditional `torch.no_grad()` block (correct when the encoder
is fully frozen). Setting `requires_grad=True` on the last block's
parameters alone is **not sufficient** to train them if the forward pass
that uses them still runs under `no_grad` — `no_grad` blocks autograd from
building a graph at all, independent of `requires_grad`. The first attempt
at this feature had exactly this bug: training completed without error,
losses looked plausible, but the run was **bit-identical** to a
fully-frozen baseline down to 6 significant figures, because the "unfrozen"
block never received a single gradient update. The fix makes the `no_grad`
conditional:
```python
encoder_grad_enabled = self.training and any(
    p.requires_grad for p in self.encoder.parameters()
)
with torch.set_grad_enabled(encoder_grad_enabled):
    out = self.encoder.forward_features(frame)
    ...
```
`_encode_frame_raw` (the target branch) correctly stays under
unconditional `@torch.no_grad()` regardless of this setting — the target
must remain a stop-gradient no matter what the backbone-unfreeze
configuration is, or the adapter-collapse failure mode above reopens.
**Lesson for any future frozen-backbone-with-optional-unfreeze design:**
"is `requires_grad` set correctly" and "does gradient actually reach the
parameter" are two different questions, and only the second one matters —
verify with an actual `.backward()` plus a check of `.grad is not None` on
the specific parameters in question, not just a static inspection of
`requires_grad` flags.

`get_param_groups(lr, backbone_lr_mult)` gives any unfrozen backbone
parameters a separate, smaller learning rate (default `0.1x` the
adapter/motion-encoder LR) via the `lr_scale` convention read by
`util/lr_sched.py`'s `adjust_learning_rate`.

**Empirical result so far (1 block unfrozen, `backbone_lr_mult=0.1`, 50
epochs, same 4-clip data):** mixed-to-negative on the proxy metrics that
motivated trying it — `cos(dz, true_diff)` moved from 0.2235 -> 0.2047
(worse), and clip-identity on the (now partially fine-tuned) backbone
moved from 0.718 -> 0.757 (worse). See Step 6 for the more decisive
downstream comparison, which points the same direction. With only 50
epochs / 4 clips / a ~3e-6 effective backbone LR, this is a lot to ask of
one transformer block — inconclusive rather than a clean negative result,
but not yet a demonstrated win either.

### Why the MLP is applied only to the CLS token, not patch tokens

Patch tokens carry the spatial context the motion encoder needs via
cross-attention (`condition_t`); adapting them too would let the adapter
alter the context the motion encoder relies on to interpret `pixel_diff`.

### Why the residual connection

The pretrained DINOv2 CLS token is already a strong feature; the adapter
should learn the *delta*, not relearn features from scratch. This is also
precisely the mechanism that made the shared-adapter collapse easy.

**Open question worth investigating (see "Where to improve" below):** the
same residual design that stabilises training also means the adapter has
no explicit incentive to *preserve* whatever appearance information isn't
useful for the temporal-prediction objective. A purely motion-driven loss
optimises for "what changed," which is not obviously the same thing as
"what's discriminative for a static-frame action label" — the two could
even be in tension.

### Frozen vs. trainable parameters

```
Frozen (85.6M by default; 78.4M with 1 block unfrozen):
                   DINOv2 ViT-B/14 encoder — TDV repo
Frozen (98K):      ProjectionHead (frozen during TDV training)
Trainable (198K):  MLPAdapter — custom
Trainable (19.5M at motion_depth=2, 38.5M at depth=4): TDV cross-attention
                   motion encoder — TDV repo factory, custom depth/config
Optional trainable (7.09M per block): last N DINOv2 blocks + final norm,
                   via unfreeze_last_n_blocks
```

With only ~1300 frame pairs from 4 clips, the motion encoder alone is a
lot of trainable capacity relative to the data — see "Where to improve"
for regularisation ideas targeting this specifically.

### `condition` keyword-argument contract

`dinoViT_xattn_base14`'s forward signature requires `condition` to be
passed as a **keyword** argument. It returns a dict with
`x_norm_clstoken`/`x_norm_patchtokens` in eval mode; verify this contract
with an `isinstance` guard rather than assuming the dict form
unconditionally, since some motion-encoder implementations branch on
`self.training` and can return a bare tensor during training.

**Terminal (smoke-test):**
```powershell
python models/tdv_wrapper.py `
    --checkpoint "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --tdv_repo   "tdv"
```

---

## Step 3 — Self-Supervised TDV Fine-Tuning

**File:** `training/train_tdv.py`

Mirrors `train_mae.py`'s structure. Uses `SurgicalWindowDataset` (variable
offset `k`), not `SurgicalPairDataset` — see the train/eval gap-distribution
note in Step 1.

**Checkpoint `args` now also include:** `unfreeze_blocks`, `backbone_lr_mult`.

**Terminal:**
```powershell
python training/train_tdv.py `
    --clip_dir    "data/processed/prostatectomy/GSTT_010/step_5" `
    --checkpoint  "checkpoints/dinov2_pretrained/dinov2_vitb14_pretrain.pth" `
    --output_dir  "checkpoints/tdv_finetuned" `
    --epochs      50 --batch_size 8 --num_workers 0 --seed 42 `
    --nce_weight  0.02 --var_weight 0.02 `
    --unfreeze_blocks 0 --backbone_lr_mult 0.1
```

---

## Step 4 — CLS Token Extraction

**File:** `models/extract_tdv_cls.py`

Same as before. `extract_cls()` = DINOv2 (frozen, or partially fine-tuned
if `unfreeze_blocks>0`) + trained adapter, applied to a single frame — the
motion encoder is **never used at inference**. Real limitation worth
keeping in mind: the only directly reusable trained artifact for
downstream tasks is the 198K-parameter adapter — the 19.5-38.5M-parameter
motion encoder that absorbed most of training's gradient signal is
entirely discarded at this step.

---

## Step 5 — Label Preparation *(shared with MAE)*

No change. See MAE timeline doc.

---

## Step 6 — Projection Model Training

**File:** `models/projection_model.py` *(shared script with MAE)*

**Split methodology — corrected from an earlier frame-level random split.**
The original implementation used `random_split` on all frames pooled
together (80/20, same seed for both pipelines). At 1fps this leaks
near-duplicate consecutive frames across train/val, inflating val accuracy
in a way that measures memorisation rather than generalisation, and
re-introduces the clip-identity shortcut this pipeline was built to
investigate. `projection_model.py` now defaults to **leave-one-clip-out
(LOCO) cross-validation**: train on N-1 clips, validate on the held-out
one, repeat for every clip, report mean +/- std. `--val_clips` is available
for a faster single-fixed-split sanity check.

**Actual result (1 block unfrozen, 7-class task, LOCO CV over 3 clips —
clip 3 had no valid labels and is excluded, worth double-checking that's
expected):**

| | held-out clip 0 | held-out clip 1 | held-out clip 2 | mean +/- std |
|---|---|---|---|---|
| **TDV** | 2.2% | 18.3% | 0.0% | **6.8% +/- 10.0%** |
| **MAE** | 45.5% | 28.3% | 12.5% | **28.8% +/- 16.5%** |
| chance (7 classes) | | | | 14.3% |

**What this step achieved, concretely:** under a leak-free split, **MAE
clearly outperforms TDV**, and TDV performs *below chance* on average
(0.0% on one held-out clip). This is the opposite of what the earlier
(leaky) 80/20-split run suggested (TDV: 79.9%). Both numbers are noisy
given only 3 usable folds — treat "MAE beats TDV, TDV underperforms
chance" as the working conclusion, not a settled one. See "Where to
improve TDV" below.

---

## Step 6.5 — Diagnostics (TDV-specific, no MAE equivalent)

**File:** `eval/tdv_diagnostics.py`

**Known gap (not yet fixed):** `main()` rebuilds `TDVFineTuner` from the
checkpoint's saved `args` but doesn't pass `unfreeze_last_n_blocks`
through — so its printed parameter summary always claims 0 trainable
encoder params regardless of what the checkpoint actually used.
`load_state_dict` still loads the correct trained *values* either way, so
the numeric test results are unaffected — only the printed summary is
misleading. Fix: read `saved_args.get("unfreeze_blocks", 0)` and pass it
into the `TDVFineTuner(...)` call in `main()`.

**Note on Test 4 (temporal similarity):** once `unfreeze_blocks>0` is
used, "raw DINOv2" in this test is computed via `model_tdv.encoder` — it
reflects whatever fine-tuning happened to the backbone, not a pristine
untouched checkpoint. A genuine pristine-backbone control needs a second,
separately-loaded, never-fine-tuned DINOv2 encoder.

Tests unchanged otherwise: zero-motion baseline, delta-z statistics,
latent difference loss, temporal similarity comparison, clip-identity
nearest-centroid probe.

---

## Step 7 — Embedding-Space Visualisation (t-SNE / UMAP) *(new)*

**File:** `eval/visualize_embeddings.py`

Projects the 768-d TDV and MAE embeddings to 2D via t-SNE (always
available via scikit-learn) and UMAP (if installed), colored two ways side
by side: by **action label** and by **clip identity**. Seeing both
colorings on the same layout makes the clip-identity-shortcut question
visually immediate — if the label-colored and clip-colored plots look like
the *same* clustering with different color keys, that's a strong visual
indication the classifier is really picking up clip identity, not action
content.

**Terminal:**
```powershell
pip install umap-learn --break-system-packages   # optional; degrades gracefully without it

python eval/visualize_embeddings.py `
    --tdv_embeddings "data/embeddings/step_5_tdv_cls.pt" `
    --mae_embeddings "data/embeddings/step_5_cls.pt" `
    --labels         "data/labels/step_5/labels.pt" `
    --label_map      "data/labels/step_5/label_map.json" `
    --output         "eval/diagnostics/embedding_viz.png"
```

---

## Where to improve TDV specifically (excluding "collect more data")

1. **Run Step 6 on the original fully-frozen-backbone checkpoint too.**
   Every downstream number so far comes from the `unfreeze_blocks=1`
   checkpoint. Not yet established whether unfreezing helped, hurt, or was
   neutral for the actual task — only that its proxy metrics were
   flat-to-slightly-worse. Highest-value, lowest-effort next step: rerun a
   script you already have on a checkpoint you already have.

2. **Add a raw-DINOv2 (no adapter, no training) control to Step 6.**
   Extract embeddings directly from frozen, untouched DINOv2 and run the
   same LOCO CV. Answers whether the adapter is helping, hurting, or
   irrelevant relative to doing nothing.

3. **Consider whether the training objective and eval task are actually
   aligned.** TDV's adapter is optimised purely to make `z_t` useful for
   predicting temporal *change* — nothing rewards it for being a good
   static-frame action descriptor. Try adding a small-weight anchor term
   (e.g. `MSE(adapted_cls, raw_cls)`) so the adapter's residual delta is
   discouraged from drifting further than the temporal objective actually
   requires. Genuinely open question, not a known fix.

4. **Rebalance capacity given data size.** ~1300 frame pairs from 4 clips
   is very little data for a 19.5M-parameter motion encoder. Try
   `motion_depth=1` if supported, and/or a separate (higher) weight decay
   for the motion-encoder param group.

5. **Sweep `adapter_nce_weight` upward relative to `nce_weight`.** It's
   the only loss term that shapes `z_t` without the motion encoder's help,
   and Step 6 only ever uses `z_t` (motion encoder discarded at
   inference) — currently a minor term (0.01 vs. nce_weight=0.02).

6. **Check the `k=1`-only diagnostics blind spot.** Rerun the diagnostic
   tests at a few different fixed `k` values (2, 4, 8) to see whether the
   proxy metrics genuinely represent average training-time behaviour.

7. **If InfoNCE + variance regularisation prove structurally
   insufficient**, revisit the paper's actual recipe: an EMA-updated copy
   of the adapter (not the frozen backbone) producing the target, paired
   with a proper DINO-style self-distillation loss with running-mean
   centering.

---

## Next Steps

**Immediate:**
- Items 1 and 2 above — needed to interpret the current LOCO result
  correctly before investing more training runs.
- Run `eval/visualize_embeddings.py` (Step 7) on the existing embeddings.
- Fix the `eval/tdv_diagnostics.py` gap noted in Step 6.5.

**Medium term (more data — lower priority until the current
proxy-metric/downstream-accuracy disconnect is understood):**
Repeat Steps 0-4 for steps 2, 3, 4, 6, 7 and combine embedding files —
also means LOCO CV folds become larger and less noisy individually.

**Longer term (architecture):** see item 7 above.