
# MASTERY RARP — MAE Fine-Tuning Project Timeline

## Overarching Goal

Your supervisor asked you to **kick-start familiarity with the surgical data**
by fine-tuning a pretrained MAE encoder using self-supervision, rather than
jumping directly to a labelled downstream task. The output of this entire
pipeline is a **surgical feature extractor**: a model that takes any raw
surgical frame and produces a compact, meaningful vector representation of
what is happening in that frame.

The pipeline follows the official Meta MAE repository structure as closely
as possible, adapting only what is necessary to work with surgical video
rather than ImageNet images.

(Steps 0-5 unchanged from the original doc — see prior version for full
detail on preprocessing, `SurgicalFrameDataset`, `MAEFineTuner`,
`train_mae.py`, `extract_cls.py`, and label preparation. Only Step 6 is
updated below.)

---

## Step 6 — Projection Model Training

**File:** `models/projection_model.py` *(shared script with TDV)*

**Split methodology — corrected from the original 80/20 random split.**
The original description here said "80/20 train/val split (deterministic
with fixed seed)" using `random_split` over all frames pooled together.
That leaks near-duplicate consecutive frames across train/val at 1fps —
inflating val accuracy in a way that reflects memorisation of specific
frames rather than genuine generalisation, and (for TDV specifically)
re-opening the clip-identity shortcut the whole TDV investigation was
built to catch. Both this pipeline's and TDV's earlier reported numbers
used this leaky split.

`projection_model.py` now defaults to **leave-one-clip-out (LOCO)
cross-validation**: train on N-1 clips, validate on the held-out one,
repeat for every clip, report mean +/- std across folds. `--val_clips`
remains available for a single fixed held-out split if you want a faster
sanity check before committing to full CV.

**Output (new structure):**
```
checkpoints/projection/
    fold_clip0/checkpoint-best.pth, log.txt
    fold_clip1/...
    fold_clip2/...
    cv_summary.json
```

**Terminal (unchanged invocation — CV is now the default behaviour):**
```powershell
python models/projection_model.py `
    --embeddings "data/embeddings/step_5_cls.pt" `
    --labels     "data/labels/step_5/labels.pt" `
    --output_dir "checkpoints/projection" `
    --num_classes 7
```

**Actual result (7-class task, LOCO CV over 3 clips — clip 3 had no valid
labels and is excluded from all folds):**

| held-out clip 0 | held-out clip 1 | held-out clip 2 | mean +/- std |
|---|---|---|---|
| 45.5% | 28.3% | 12.5% | **28.8% +/- 16.5%** |

(chance level for 7 classes: 14.3%)

**What this step achieves:** under the corrected, leak-free split, MAE's
embeddings clearly beat chance. See the TDV timeline doc for the full
comparison — an initial run there wrongly suggested MAE clearly beat TDV,
but that used a broken (backbone-unfrozen) TDV checkpoint; against the
correct frozen-backbone TDV checkpoint (32.9% +/- 10.5%), the two are
roughly comparable, with TDV directionally slightly ahead but not by a
statistically confident margin on only 3 folds. The earlier 80/20-split
numbers for both pipelines should still be considered unreliable and
superseded by this LOCO result. Note the high variance across folds
(16.5% std against a 28.8% mean) — with only 3 usable folds from very
unevenly sized clips, this is a noisy estimate; more surgical clips (see
Next Steps) would tighten it considerably and could settle which pipeline
actually comes out ahead.

---

## Complete File Structure

(unchanged from original doc, except `checkpoints/projection/` now
contains `fold_clip*/` subdirectories and a `cv_summary.json` instead of a
single flat set of checkpoints — see Step 6 above.)

---

## Next Steps

**Immediate (more surgical data):**
Repeat Steps 0-4 for all other surgical steps (2, 3, 4, 6, 7) and combine
their embedding files. This gives the projection model far more training
data, produces a step-invariant feature space, and — now that Step 6 uses
clip-level LOCO CV — directly reduces the fold-to-fold variance seen in
the current 3-fold result by giving each fold more (and more comparable)
data to validate on.

**Short term (evaluation):**
- t-SNE / UMAP visualisation of the 128-d vectors — see the TDV timeline
  doc's new Step 7 (`eval/visualize_embeddings.py`), which handles both
  pipelines' embeddings side by side.
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