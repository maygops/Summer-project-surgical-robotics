"""
datasets/prepare_labels.py

Prepares a frame-level label tensor that aligns with the embedding file
produced by extract_cls.py.

The Problem
-----------
extract_cls.py produces an embedding file with shape (N, 768) where each
row corresponds to one frame in the SurgicalFrameDataset flat index.
To train the projection model (768 → 128) with supervision, we need a
parallel label tensor of shape (N,) with one integer class per frame.

We already have this information — the surgical step annotations in your
original .pt tensors contain (start_sec, end_sec, label) tuples in the
"actions" field. This script reads those, maps every frame to its label,
and saves a LongTensor that can be passed directly to projection_model.py.

Two label granularities are supported:

    --mode step
        One class per surgical step (coarse).
        e.g. all frames in step_5 get class 0, step_6 → 1, step_7 → 2
        Useful for: phase recognition, high-level planning

    --mode action
        One class per fine-grained action string (fine).
        e.g. "5_cold_cut" → 0, "5_clip" → 1, "6_dissect" → 2 ...
        Useful for: action recognition, robot skill detection

Output
------
Saves two files to --output_dir:

    labels.pt           LongTensor of shape (N,) — integer class per frame
                        Frames with no annotation get label -1 (excluded
                        from loss by CrossEntropyLoss automatically when
                        ignore_index=-1 is set).

    label_map.json      Dict mapping class_name → integer index
                        Keep this — you need it to interpret predictions.

Usage
-----
    python datasets/prepare_labels.py \
        --clip_dir   data/processed/prostatectomy/GSTT_010/step_5 \
        --embeddings data/embeddings/step_5_cls.pt \
        --output_dir data/labels/step_5 \
        --mode       action

PowerShell:
    python datasets/prepare_labels.py `
        --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
        --embeddings "data/embeddings/step_5_cls.pt" `
        --output_dir "data/labels/step_5" `
        --mode       action
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.RARPframe_dataset import SurgicalFrameDataset


# =============================================================================
# Core label builder
# =============================================================================

def build_labels(
    clip_dir:    str,
    embed_path:  str,
    output_dir:  str,
    mode:        str = "action",
    step_label:  str | None = None,
) -> None:
    """
    Build and save frame-level labels aligned to the embedding file.

    Parameters
    ----------
    clip_dir   : folder with clip_XXXX.pt files
    embed_path : .pt file from extract_cls.py (used to verify frame count)
    output_dir : where to save labels.pt and label_map.json
    mode       : "step" (coarse) or "action" (fine-grained)
    step_label : surgical step string e.g. "5" — used as fallback class name
                 when mode="step" and no action annotation is available
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset to get frame count and metadata
    # ------------------------------------------------------------------
    dataset = SurgicalFrameDataset(
        clip_dir   = clip_dir,
        load_mode  = "lazy",
        step_label = step_label,
    )
    n_frames = len(dataset)
    print(f"[Labels] Dataset: {n_frames} frames across "
          f"{len(dataset.clip_paths)} clips")

    # ------------------------------------------------------------------
    # 2. Verify alignment with embedding file
    # ------------------------------------------------------------------
    embed_payload = torch.load(embed_path, weights_only=False)
    n_embeddings  = embed_payload["embeddings"].shape[0]

    if n_embeddings != n_frames:
        raise ValueError(
            f"Frame count mismatch: dataset has {n_frames} frames but "
            f"embedding file has {n_embeddings} rows. "
            f"Make sure both were built from the same clip_dir."
        )
    print(f"[Labels] Embedding alignment verified ({n_embeddings} rows)")

    # ------------------------------------------------------------------
    # 3. Build raw label string per frame
    # ------------------------------------------------------------------
    # Each clip's _clip_meta stores fps and the actions list.
    # actions = [(start_sec, end_sec, label_str), ...]
    # We look up the frame's timestamp and find which interval it falls in.

    raw_labels = []     # list of str or None, length = n_frames

    for dataset_idx in range(n_frames):
        clip_idx, frame_idx = dataset._index[dataset_idx]
        meta = dataset._clip_meta[clip_idx]

        fps     = meta.get("fps")
        actions = None

        # Actions are stored in the original .pt tensor — reload the file
        # to access them (they were stripped from _clip_meta to save RAM).
        clip_data = torch.load(meta["path"], weights_only=False)
        actions   = clip_data.get("actions", [])

        label_str = None

        if fps and actions:
            frame_time = frame_idx / fps
            for start, end, act_label in actions:
                if start <= frame_time <= end:
                    if mode == "action":
                        label_str = act_label           # e.g. "5_cold_cut"
                    else:
                        # step mode: use the step portion of the label
                        # "5_cold_cut" → "step_5"
                        # or fall back to step_label argument
                        step_part = act_label.split("_")[0]
                        label_str = f"step_{step_part}"
                    break

        if label_str is None:
            # Frame falls between annotated actions or has no annotation
            if mode == "step" and step_label:
                label_str = f"step_{step_label}"   # assign to the step anyway
            # else: remains None → will become -1 (ignored in loss)

        raw_labels.append(label_str)

    # ------------------------------------------------------------------
    # 4. Build integer label map
    # ------------------------------------------------------------------
    # Collect all unique non-None label strings, sort for determinism
    unique_labels = sorted(set(l for l in raw_labels if l is not None))
    label_map     = {name: idx for idx, name in enumerate(unique_labels)}

    print(f"\n[Labels] Found {len(label_map)} classes ({mode} mode):")
    for name, idx in label_map.items():
        count = sum(1 for l in raw_labels if l == name)
        print(f"  [{idx:3d}] {name:30s}  {count:5d} frames  "
              f"({100*count/n_frames:.1f}%)")

    unannotated = sum(1 for l in raw_labels if l is None)
    if unannotated:
        print(f"  [ -1] (unannotated)                    "
              f"{unannotated:5d} frames  ({100*unannotated/n_frames:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Convert to integer tensor
    # ------------------------------------------------------------------
    label_tensor = torch.tensor(
        [label_map[l] if l is not None else -1 for l in raw_labels],
        dtype=torch.long,
    )   # shape: (N,)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    labels_path   = output_dir / "labels.pt"
    labelmap_path = output_dir / "label_map.json"

    torch.save(label_tensor, labels_path)
    with open(labelmap_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n[Labels] Saved:")
    print(f"  {labels_path}    — LongTensor {tuple(label_tensor.shape)}")
    print(f"  {labelmap_path}  — class name → integer index")
    print(
        f"\nNext step:\n"
        f"  python models/projection_model.py \\\n"
        f"      --embeddings {embed_path} \\\n"
        f"      --labels     {labels_path} \\\n"
        f"      --output_dir checkpoints/projection \\\n"
        f"      --num_classes {len(label_map)}"
    )

    return label_tensor, label_map


# =============================================================================
# Argument parser
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("Surgical frame label preparation")

    parser.add_argument(
        "--clip_dir", required=True, type=str,
        help="Folder containing clip_XXXX.pt files "
             "(same folder used for training and extraction)"
    )
    parser.add_argument(
        "--embeddings", required=True, type=str,
        help="Path to .pt embedding file from extract_cls.py "
             "(used to verify frame count alignment)"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Directory to save labels.pt and label_map.json"
    )
    parser.add_argument(
        "--mode", default="action", type=str,
        choices=["step", "action"],
        help="Label granularity. "
             "'step'   → one class per surgical step (coarse, e.g. 'step_5'). "
             "'action' → one class per fine-grained action string (fine, "
             "e.g. '5_cold_cut'). Default: action"
    )
    parser.add_argument(
        "--step_label", default=None, type=str,
        help="Surgical step string for this clip dir (e.g. '5'). "
             "Used as fallback class name in step mode when a frame has no "
             "action annotation."
    )

    return parser


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    build_labels(
        clip_dir   = args.clip_dir,
        embed_path = args.embeddings,
        output_dir = args.output_dir,
        mode       = args.mode,
        step_label = args.step_label,
    )