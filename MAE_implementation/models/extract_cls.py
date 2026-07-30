"""
models/extract_cls.py

Loads a trained MAEFineTuner checkpoint and extracts 768-d adapted CLS tokens
from every frame in a SurgicalFrameDataset.

What this script produces
--------------------------
A single .pt file containing:
    {
        "embeddings":   Tensor (N, 768),   # one vector per frame
        "clip_indices": Tensor (N,),       # which clip each frame came from
        "frame_indices":Tensor (N,),       # position within that clip
        "step_label":   str,               # surgical step, e.g. "5"
        "checkpoint":   str,               # path to the checkpoint used
        "embed_dim":    int,               # 768
    }

This file is the input to the downstream projection model (768 → 128).

Usage
-----
    python models/extract_cls.py \
        --checkpoint checkpoints/mae_finetuned/checkpoint-0049.pth \
        --clip_dir   data/processed/prostatectomy/GSTT_010/step_5 \
        --output     data/embeddings/step_5_cls.pt \
        --step_label 5 \
        --batch_size 32

PowerShell (from project root):
    python models/extract_cls.py `
        --checkpoint "checkpoints/mae_finetuned/checkpoint-0049.pth" `
        --clip_dir   "data/processed/prostatectomy/GSTT_010/step_5" `
        --output     "data/embeddings/step_5_cls.pt" `
        --step_label 5 `
        --batch_size 32
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.RARPframe_dataset import SurgicalFrameDataset
from models.mae_wrapper import MAEFineTuner


# =============================================================================
# Argument parser
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("CLS token extractor", add_help=True)

    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to trained MAEFineTuner checkpoint "
             "(e.g. checkpoints/mae_finetuned/checkpoint-0049.pth)"
    )
    parser.add_argument(
        "--finetune", default=None, type=str,
        help="Path to the original pretrained MAE checkpoint. Only needed if "
             "the fine-tuned checkpoint does not include the full model state. "
             "In normal use this can be omitted — the fine-tuned checkpoint "
             "contains all weights."
    )
    parser.add_argument(
        "--clip_dir", required=True, type=str,
        help="Path to folder containing clip_XXXX.pt files"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Path to save the output .pt embedding file "
             "(e.g. data/embeddings/step_5_cls.pt)"
    )
    parser.add_argument(
        "--step_label", default=None, type=str,
        help="Surgical step label for metadata (e.g. '5')"
    )
    parser.add_argument(
        "--model_name", default="base", type=str,
        choices=["base", "large", "huge"],
        help="MAE ViT variant matching the checkpoint (default: base)"
    )
    parser.add_argument(
        "--mlp_hidden_dim", default=128, type=int,
        help="MLP adapter bottleneck dim — used only if not found in checkpoint "
             "saved args. Must match the value used during training (default: 128)."
    )
    parser.add_argument(
        "--batch_size", default=32, type=int,
        help="Frames per batch during extraction (no gradient — use larger "
             "values than during training, limited only by GPU memory)"
    )
    parser.add_argument(
        "--num_workers", default=0, type=int,
        help="DataLoader workers (0 is safe on Windows)"
    )
    parser.add_argument(
        "--device", default="cuda", type=str,
        help="Device for extraction ('cuda' or 'cpu')"
    )

    return parser


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"[Extractor] Device : {device}")

    # ------------------------------------------------------------------
    # 1. Build dataset  (sequential — we want frame order preserved)
    # ------------------------------------------------------------------
    print(f"\n[Extractor] Loading dataset from {args.clip_dir}")
    dataset = SurgicalFrameDataset(
        clip_dir   = args.clip_dir,
        transform  = None,
        load_mode  = "lazy",
        step_label = args.step_label,
    )
    print(dataset.summary_df().to_string(index=False))

    # SequentialSampler: preserve frame order so embedding index matches
    # dataset index exactly (important for downstream alignment with labels).
    loader = DataLoader(
        dataset,
        sampler     = SequentialSampler(dataset),
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,    # we want every frame, including the last partial batch
    )

    # ------------------------------------------------------------------
    # 2. Load model
    #    Read ALL architecture hyperparameters from the checkpoint's saved
    #    args so the model we build here exactly matches what was trained.
    # ------------------------------------------------------------------
    print(f"\n[Extractor] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    saved_args    = ckpt.get("args", {})
    finetune_path = args.finetune or saved_args.get("finetune")

    if finetune_path is None:
        raise ValueError(
            "Could not determine pretrained checkpoint path. "
            "Pass --finetune checkpoints/mae_pretrained/mae_pretrain_vit_base.pth"
        )

    # mlp_hidden_dim is the critical field — must match training exactly.
    # Falls back to CLI arg, then to 128 (current default).
    mlp_hidden_dim = saved_args.get("mlp_hidden_dim", getattr(args, "mlp_hidden_dim", 128))
    model_name     = saved_args.get("model_name",     args.model_name)
    norm_pix_loss  = saved_args.get("norm_pix_loss",  True)

    print(f"[Extractor] Rebuilding architecture from checkpoint args:")
    print(f"  model_name     : {model_name}")
    print(f"  mlp_hidden_dim : {mlp_hidden_dim}")
    print(f"  norm_pix_loss  : {norm_pix_loss}")

    model = MAEFineTuner(
        checkpoint_path = finetune_path,
        model_name      = model_name,
        mlp_hidden_dim  = mlp_hidden_dim,
        norm_pix_loss   = norm_pix_loss,
    )

    # Overwrite encoder + adapter + decoder with fine-tuned weights.
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"[Extractor] Checkpoint epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"[Extractor] Checkpoint loss  : {ckpt.get('train_loss', 0.0):.4f}")

    # ------------------------------------------------------------------
    # 3. Extract CLS tokens
    # ------------------------------------------------------------------
    all_embeddings    = []
    all_clip_indices  = []
    all_frame_indices = []

    print(f"\n[Extractor] Extracting {len(dataset)} frame embeddings...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames = batch["frame"].to(device, non_blocking=True)

            # extract_cls() runs forward_encoder with mask_ratio=0.0
            # (full image, no masking) then passes CLS through the adapter.
            # Shape: (B, 768)
            cls_vecs = model.extract_cls(frames)

            all_embeddings.append(cls_vecs.cpu())
            all_clip_indices.append(batch["clip_idx"])
            all_frame_indices.append(batch["frame_idx"])

            if batch_idx % 10 == 0:
                n_done = min((batch_idx + 1) * args.batch_size, len(dataset))
                print(f"  {n_done:>5} / {len(dataset)} frames")

    # Concatenate all batches into single tensors
    embeddings    = torch.cat(all_embeddings,    dim=0)   # (N, 768)
    clip_indices  = torch.cat(all_clip_indices,  dim=0)   # (N,)
    frame_indices = torch.cat(all_frame_indices, dim=0)   # (N,)

    print(f"\n[Extractor] Done. Embedding tensor shape: {embeddings.shape}")
    print(f"  min  : {embeddings.min():.4f}")
    print(f"  max  : {embeddings.max():.4f}")
    print(f"  mean : {embeddings.mean():.4f}")
    print(f"  std  : {embeddings.std():.4f}")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "embeddings":    embeddings,           # (N, 768)  float32
        "clip_indices":  clip_indices,         # (N,)      int64
        "frame_indices": frame_indices,        # (N,)      int64
        "step_label":    args.step_label,      # str or None
        "checkpoint":    args.checkpoint,      # provenance
        "embed_dim":     embeddings.shape[1],  # 768
    }

    torch.save(payload, output_path)
    print(f"\n[Extractor] Saved → {output_path}")
    print(
        f"\nNext step: run models/projection_model.py to compress "
        f"these 768-d vectors to 128-d for downstream tasks."
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)