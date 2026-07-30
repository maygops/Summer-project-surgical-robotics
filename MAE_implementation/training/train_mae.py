"""
training/train_mae.py

Self-supervised MAE fine-tuning on surgical clips.
Adapted from the official Meta MAE repository (main_finetune.py).

What this script does (in order)
---------------------------------
1.  Parse arguments (paths, hyper-parameters, hardware settings).
2.  Set random seeds for reproducibility.
3.  Build SurgicalFrameDataset  →  DataLoader.
4.  Instantiate MAEFineTuner (loads pretrained encoder, freezes it,
    attaches MLP adapter, freezes projection head).
5.  Build AdamW optimiser over trainable parameters only
    (MLP adapter + MAE decoder).
6.  Run the training loop:
        for each epoch:
            for each batch:
                frames = batch["frame"]
                loss, pred, mask = model(frames, mask_ratio)
                loss.backward()
                optimizer.step()
7.  Save a checkpoint every `--save_every` epochs and at the final epoch.
8.  Log loss + lr to TensorBoard.

Checkpoint contents
-------------------
Each checkpoint saved to  checkpoints/mae_finetuned/checkpoint-<epoch>.pth
contains:
    {
        "model":        model.state_dict(),   # encoder (frozen) + adapter + decoder
        "optimizer":    optimizer.state_dict(),
        "epoch":        epoch,
        "loss":         train_loss,
        "args":         vars(args),
    }

After training, use extract_cls() from MAEFineTuner for 768-d embeddings.

Usage
-----
    python training/train_mae.py \
        --clip_dir   data/processed/prostatectomy/GSTT_010/step_5 \
        --finetune   checkpoints/mae_pretrained/mae_pretrain_vit_base.pth \
        --output_dir checkpoints/mae_finetuned \
        --epochs     50 \
        --batch_size 16 \
        --mask_ratio 0.75

Resume from checkpoint:
    python training/train_mae.py ... --resume checkpoints/mae_finetuned/checkpoint-20.pth
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# -- project imports ----------------------------------------------------------
# Assumes the repository root is on PYTHONPATH, matching the Meta repo layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.RARPframe_dataset import SurgicalFrameDataset
from models.mae_wrapper import MAEFineTuner
import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler


# =============================================================================
# Argument parser
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE surgical fine-tuning", add_help=True
    )

    # ---- Data ---------------------------------------------------------------
    parser.add_argument(
        "--clip_dir", required=True, type=str,
        help="Path to folder containing clip_XXXX.pt files "
             "(e.g. data/processed/prostatectomy/GSTT_010/step_5)"
    )
    parser.add_argument(
        "--step_label", default=None, type=str,
        help="Optional surgical step label stored in dataset metadata (e.g. '5')"
    )
    parser.add_argument(
        "--input_size", default=224, type=int,
        help="Spatial size frames will be resized to before encoding. "
             "Must match the pretrained model (224 for ViT-Base/Large)."
    )

    # ---- Model --------------------------------------------------------------
    parser.add_argument(
        "--finetune", required=True, type=str,
        help="Path to pretrained MAE checkpoint "
             "(e.g. checkpoints/mae_pretrained/mae_pretrain_vit_base.pth)"
    )
    parser.add_argument(
        "--model_name", default="base", type=str,
        choices=["base", "large", "huge"],
        help="MAE ViT variant to use (default: base → embed_dim=768)"
    )
    parser.add_argument(
        "--mlp_hidden_dim", default=128, type=int,
        help="Hidden dimension of the MLP adapter bottleneck (default: 128). "
             "768→128→768 creates an aggressive information bottleneck that "
             "acts as structural regularisation — important for small datasets."
    )
    parser.add_argument(
        "--mask_ratio", default=0.75, type=float,
        help="Fraction of patches masked during training (default: 0.75)"
    )
    parser.add_argument(
        "--norm_pix_loss", action="store_true", default=True,
        help="Normalise pixel targets per-patch before MSE (recommended for "
             "surgical video due to high contrast variability)"
    )

    # ---- Optimiser ----------------------------------------------------------
    parser.add_argument(
        "--epochs", default=50, type=int,
        help="Total number of training epochs"
    )
    parser.add_argument(
        "--warmup_epochs", default=5, type=int,
        help="Number of epochs for cosine LR warm-up"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int,
        help="Batch size (frames per step). With a small dataset keep this "
             "small (8–32) to get more gradient updates per epoch."
    )
    parser.add_argument(
        "--accum_iter", default=1, type=int,
        help="Gradient accumulation steps. Effective batch = batch_size × accum_iter."
    )
    parser.add_argument(
        "--lr", default=None, type=float,
        help="Absolute learning rate. If not set, computed from --blr."
    )
    parser.add_argument(
        "--blr", default=1e-3, type=float,
        help="Base learning rate. Actual lr = blr × effective_batch / 256."
    )
    parser.add_argument(
        "--min_lr", default=0.0, type=float,
        help="Minimum learning rate at the end of cosine schedule."
    )
    parser.add_argument(
        "--weight_decay", default=0.05, type=float,
        help="AdamW weight decay."
    )

    # ---- Checkpointing / logging --------------------------------------------
    parser.add_argument(
        "--output_dir", default="checkpoints/mae_finetuned", type=str,
        help="Directory to save checkpoints and log.txt"
    )
    parser.add_argument(
        "--log_dir", default=None, type=str,
        help="TensorBoard log directory. Defaults to --output_dir."
    )
    parser.add_argument(
        "--save_every", default=20, type=int,
        help="Save a checkpoint every N epochs (always saves the final epoch)."
    )
    parser.add_argument(
        "--resume", default="", type=str,
        help="Path to a checkpoint to resume training from."
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int,
        help="Epoch to start from (set automatically when resuming)."
    )

    # ---- Hardware -----------------------------------------------------------
    parser.add_argument(
        "--device", default="cuda", type=str,
        help="Device for training ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--num_workers", default=4, type=int,
        help="DataLoader worker processes. Set to 0 on Windows if you hit "
             "multiprocessing errors."
    )
    parser.add_argument(
        "--pin_mem", action="store_true", default=True,
        help="Pin CPU memory in DataLoader for faster GPU transfer."
    )
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Random seed for reproducibility."
    )

    return parser


# =============================================================================
# One training epoch
# =============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
    log_writer: SummaryWriter | None,
    args: argparse.Namespace,
) -> dict:
    """
    Run one full pass over the dataset.

    Closely follows engine_pretrain.py from the Meta repository, adapted for:
    - Our DataLoader returning dicts (we unpack 'frame' here)
    - Single-GPU training (no DistributedSampler needed for one machine)
    - Per-iteration cosine LR scheduling (identical to Meta repo)

    Returns
    -------
    dict with 'loss' and 'lr' averages for the epoch.
    """
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header      = f"Epoch [{epoch}]"
    print_freq  = 20          # print a log line every 20 iterations
    accum_iter  = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"TensorBoard log_dir: {log_writer.log_dir}")

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # -- Per-iteration cosine LR schedule (from Meta repo) ----------------
        # lr_sched.adjust_learning_rate updates the optimiser's lr in-place.
        # This gives a smoother schedule than per-epoch stepping.
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args,
            )

        # -- Unpack batch -----------------------------------------------------
        # SurgicalFrameDataset.__getitem__ returns a dict.
        # The MAE model expects a plain tensor of shape (B, 3, H, W).
        frames = batch["frame"].to(device, non_blocking=True)

        # -- Forward pass with automatic mixed precision ----------------------
        # torch.cuda.amp.autocast() runs the forward pass in float16 where
        # safe (most linear layers and activations), keeping float32 where
        # numerical stability matters (softmax, layer norm).
        # This halves memory usage and speeds up training on modern GPUs.
        with torch.amp.autocast(device_type= "cuda"):
            loss, _, _ = model(frames, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        # -- Sanity check -----------------------------------------------------
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            print(f"  Last batch clip_idx  : {batch['clip_idx']}")
            print(f"  Last batch frame_idx : {batch['frame_idx']}")
            sys.exit(1)

        # -- Backward pass ----------------------------------------------------
        # loss_scaler wraps the GradScaler for mixed precision.
        # It scales the loss up before backward() (to avoid float16 underflow),
        # then unscales before the optimiser step, and skips the step if
        # gradients contain inf/nan (can happen during warm-up).
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.trainable_parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # -- Logging ----------------------------------------------------------
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # epoch_1000x: x-axis calibration so curves align across batch sizes
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("train/loss", loss_value, epoch_1000x)
            log_writer.add_scalar("train/lr",   optimizer.param_groups[0]["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(
    output_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    train_loss: float,
    args: argparse.Namespace,
):
    """
    Save model + optimiser state to output_dir/checkpoint-<epoch>.pth.

    The checkpoint contains everything needed to:
      - Resume training  (model, optimizer, loss_scaler, epoch, args)
      - Run inference    (model weights only — encoder frozen but saved for
                          completeness; MLP adapter weights are what changed)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint-{epoch:04d}.pth")

    torch.save(
        {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scaler":       loss_scaler.state_dict(),
            "epoch":        epoch,
            "train_loss":   train_loss,
            "args":         vars(args),
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    args: argparse.Namespace,
) -> int:
    """
    Load model + optimiser state from a checkpoint.
    Returns the epoch to resume from (checkpoint epoch + 1).
    """
    print(f"[Checkpoint] Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    loss_scaler.load_state_dict(ckpt["scaler"])

    resume_epoch = ckpt["epoch"] + 1
    print(f"[Checkpoint] Resuming from epoch {resume_epoch}")
    return resume_epoch


# =============================================================================
# Main
# =============================================================================

def main(args):

    # -- Device & reproducibility ---------------------------------------------
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True          # fastest convolution for fixed input size

    print(f"Device : {device}")
    print(f"Args   :\n{json.dumps(vars(args), indent=2)}")

    # -- Dataset & DataLoader -------------------------------------------------
    print("\n[Data] Building SurgicalFrameDataset...")
    dataset_train = SurgicalFrameDataset(
        clip_dir   = args.clip_dir,
        transform  = None,          # frames are already ImageNet-normalised
        load_mode  = "lazy",        # load clip tensors on first access
        step_label = args.step_label,
    )
    print(dataset_train.summary_df().to_string(index=False))

    # RandomSampler: shuffle frame order every epoch so the model does not
    # memorise clip-level temporal ordering.
    sampler_train = RandomSampler(dataset_train)

    data_loader_train = DataLoader(
        dataset_train,
        sampler     = sampler_train,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = args.pin_mem,
        drop_last   = True,         # keeps batch sizes uniform; safe here
                                    # because we have thousands of frames
    )
    print(f"[Data] {len(dataset_train)} frames | "
          f"{len(data_loader_train)} batches per epoch "
          f"(batch_size={args.batch_size})\n")

    # -- Model ----------------------------------------------------------------
    print("[Model] Instantiating MAEFineTuner...")
    model = MAEFineTuner(
        checkpoint_path = args.finetune,
        model_name      = args.model_name,
        mlp_hidden_dim  = args.mlp_hidden_dim,
        norm_pix_loss   = args.norm_pix_loss,
    )
    model.to(device)

    # -- Effective batch size & learning rate ---------------------------------
    # Follows the Meta repo convention:
    #   actual_lr = base_lr × (batch_size × accum_iter) / 256
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"[Optimiser] Effective batch size : {eff_batch_size}")
    print(f"[Optimiser] Base lr              : {args.blr:.2e}")
    print(f"[Optimiser] Actual lr            : {args.lr:.2e}")

    # -- Optimiser ------------------------------------------------------------
    # Only trainable parameters (MLP adapter + MAE decoder) are passed.
    # The frozen encoder weights are excluded automatically by
    # model.trainable_parameters().
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr           = args.lr,
        betas        = (0.9, 0.95),   # from Meta MAE paper
        weight_decay = args.weight_decay,
    )
    loss_scaler = NativeScaler()

    # -- Resume ---------------------------------------------------------------
    start_epoch = args.start_epoch
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, loss_scaler, args
        )

    # -- TensorBoard ----------------------------------------------------------
    log_dir    = args.log_dir or args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)
    print(f"[Logging] TensorBoard → {log_dir}")

    # -- Training loop --------------------------------------------------------
    print(f"\n[Training] Starting from epoch {start_epoch}, "
          f"running to epoch {args.epochs - 1}\n")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model       = model,
            data_loader = data_loader_train,
            optimizer   = optimizer,
            device      = device,
            epoch       = epoch,
            loss_scaler = loss_scaler,
            log_writer  = log_writer,
            args        = args,
        )

        # -- Checkpoint -------------------------------------------------------
        is_save_epoch  = (epoch % args.save_every == 0)
        is_final_epoch = (epoch + 1 == args.epochs)

        if args.output_dir and (is_save_epoch or is_final_epoch):
            save_checkpoint(
                output_dir  = args.output_dir,
                epoch       = epoch,
                model       = model,
                optimizer   = optimizer,
                loss_scaler = loss_scaler,
                train_loss  = train_stats["loss"],
                args        = args,
            )

        # -- Log stats --------------------------------------------------------
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_stats['loss']:.4f} | "
            f"lr {train_stats['lr']:.2e}"
        )

    total_time = time.time() - start_time
    print(
        f"\n[Done] Training complete. "
        f"Total time: {datetime.timedelta(seconds=int(total_time))}"
    )
    print(f"Checkpoints saved to: {args.output_dir}")
    print(
        f"\nNext step: load the final checkpoint and call "
        f"model.extract_cls(frames) to obtain 768-d surgical embeddings."
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)