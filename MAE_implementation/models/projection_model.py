"""
models/projection_model.py

Supervised projection head that compresses 768-d adapted CLS embeddings to
128-d representations for downstream tasks (action recognition, phase
detection, surgical robotics).

This is trained AFTER self-supervised MAE fine-tuning, using labelled data.
The frozen encoder + frozen adapter produce stable 768-d embeddings; this
model only needs to learn the task-specific compression.

Architecture
------------
    Input: (B, 768)   adapted CLS token from MAEFineTuner.extract_cls()

    ProjectionModel:
        Linear(768 → 512)
        GELU
        LayerNorm(512)
        Dropout(0.1)
        Linear(512 → 128)
        L2 normalisation          ← unit-norm output, standard for metric
                                     learning and similarity-based tasks

    Output: (B, 128)   L2-normalised surgical feature vector

Why L2 normalisation?
---------------------
Downstream models (nearest-neighbour retrieval, cosine similarity matching,
contrastive losses) all work in terms of angles between vectors, not raw
magnitudes.  L2-normalising the output makes the geometry consistent and
prevents any single dimension dominating by having a large magnitude.

Two training modes
------------------
1.  Supervised classification (action recognition, phase detection):
        loss = CrossEntropyLoss(classifier_head(features), labels)
    A lightweight nn.Linear(128, num_classes) sits on top.

2.  Metric learning / contrastive (robotics, retrieval):
        loss = SupConLoss or TripletLoss on the 128-d L2-normalised vectors
    Frames of the same surgical step should cluster; different steps should
    separate.

Usage — training
----------------
    python models/projection_model.py \
        --embeddings  data/embeddings/step_5_cls.pt \
        --labels      data/labels/step_5_labels.pt \
        --output_dir  checkpoints/projection \
        --num_classes 10 \
        --epochs      30

PowerShell (from project root):
    python models/projection_model.py `
        --embeddings  "data/embeddings/step_5_cls.pt" `
        --labels      "data/labels/step_5_labels.pt" `
        --output_dir  "checkpoints/projection" `
        --num_classes 10 `
        --epochs      30

Usage — inference (in code)
----------------------------
    from models.projection_model import ProjectionModel, load_projection_model

    proj = load_projection_model("checkpoints/projection/checkpoint-best.pth")
    proj.eval()

    # embeddings: (B, 768) from MAEFineTuner.extract_cls()
    features_128 = proj(embeddings)   # (B, 128) L2-normalised
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =============================================================================
# Projection model
# =============================================================================

class ProjectionModel(nn.Module):
    """
    MLP that compresses 768-d CLS embeddings to 128-d L2-normalised vectors.

    Parameters
    ----------
    in_dim      : Input dimension. 768 for ViT-Base, 1024 for ViT-Large.
    hidden_dim  : Intermediate dimension. Default 512.
    out_dim     : Output dimension. Default 128.
    dropout     : Dropout probability for regularisation. Default 0.1.
    normalise   : If True, L2-normalise the output. Recommended True for
                  metric learning; set False if using raw logits downstream.
    """

    def __init__(
        self,
        in_dim:     int   = 768,
        hidden_dim: int   = 512,
        out_dim:    int   = 128,
        dropout:    float = 0.1,
        normalise:  bool  = True,
    ):
        super().__init__()
        self.normalise = normalise

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_dim)  — 768-d CLS embeddings
        returns (B, out_dim) — 128-d features, L2-normalised if self.normalise
        """
        out = self.net(x)
        if self.normalise:
            out = F.normalize(out, p=2, dim=-1)
        return out


# =============================================================================
# Classifier head  (sits on top of ProjectionModel for supervised training)
# =============================================================================

class ClassifierHead(nn.Module):
    """
    Lightweight linear classifier on top of 128-d features.
    Used during supervised training for action/phase recognition.

    At inference for robotics/retrieval, this head is discarded and only
    the ProjectionModel is used to produce the 128-d feature vectors.

    Parameters
    ----------
    in_dim      : 128 (output of ProjectionModel)
    num_classes : Number of action or phase classes
    """

    def __init__(self, in_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 128) → (B, num_classes) raw logits"""
        return self.fc(x)


# =============================================================================
# Dataset wrapper for pre-extracted embeddings
# =============================================================================

class EmbeddingDataset(Dataset):
    """
    Wraps a pre-extracted embedding .pt file and its corresponding labels.

    The embeddings file is produced by extract_cls.py.
    The labels file should be a .pt containing a 1-D LongTensor of length N
    with one integer class label per frame.

    Parameters
    ----------
    embeddings_path : str  Path to the .pt file from extract_cls.py
    labels_path     : str  Path to a .pt file: torch.LongTensor of shape (N,)
    """

    def __init__(self, embeddings_path: str, labels_path: str):
        payload = torch.load(embeddings_path, weights_only=False)
        self.embeddings   = payload["embeddings"]    # (N, 768)
        self.clip_indices = payload["clip_indices"]
        self.frame_indices= payload["frame_indices"]

        self.labels = torch.load(labels_path, weights_only=False)  # (N,)
        mask = self.labels >= 0

        self.embeddings    = self.embeddings[mask]
        self.labels        = self.labels[mask]
        self.clip_indices  = self.clip_indices[mask]
        self.frame_indices = self.frame_indices[mask]
        assert len(self.embeddings) == len(self.labels), (
            f"Embedding count ({len(self.embeddings)}) does not match "
            f"label count ({len(self.labels)})"
        )
        valid = self.labels[self.labels >= 0]
        print(
            f"[EmbeddingDataset] {len(self.embeddings)} frames | "
            f"{self.labels.unique().numel()} classes"
        )
        print("Unique labels:", torch.unique(self.labels))

        bad = self.labels[
        (self.labels < 0) | (self.labels >= 7)
                        ]

        print("Bad labels:", torch.unique(bad))
        print("Num bad:", len(bad))

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_projection_checkpoint(
    output_dir: str,
    tag: str,
    projection: ProjectionModel,
    classifier: ClassifierHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    args: argparse.Namespace,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"checkpoint-{tag}.pth"
    torch.save(
        {
            "projection":  projection.state_dict(),
            "classifier":  classifier.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "epoch":       epoch,
            "val_acc":     val_acc,
            "args":        vars(args),
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}  (val_acc={val_acc:.2f}%)")


def load_projection_model(
    checkpoint_path: str,
    in_dim:     int = 768,
    hidden_dim: int = 512,
    out_dim:    int = 128,
    normalise:  bool = True,
    device:     str  = "cpu",
) -> ProjectionModel:
    """
    Load a trained ProjectionModel from a checkpoint for inference.

    Returns the ProjectionModel only (no classifier head — that is
    task-specific and not needed for feature extraction).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    proj = ProjectionModel(
        in_dim     = saved_args.get("in_dim",     in_dim),
        hidden_dim = saved_args.get("hidden_dim", hidden_dim),
        out_dim    = saved_args.get("out_dim",    out_dim),
        normalise  = normalise,
    )
    proj.load_state_dict(ckpt["projection"])
    proj.to(device)
    proj.eval()
    print(f"[ProjectionModel] Loaded from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch','?')}, val_acc {ckpt.get('val_acc',0):.2f}%)")
    return proj


# =============================================================================
# Training loop
# =============================================================================

def train_projection(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Projection] Device : {device}")

    # -- Dataset --------------------------------------------------------------
    full_dataset = EmbeddingDataset(args.embeddings, args.labels)

    # 80/20 train/val split — deterministic with fixed seed
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    print(f"[Projection] Train: {n_train} | Val: {n_val}")

    # -- Model ----------------------------------------------------------------
    projection = ProjectionModel(
        in_dim     = args.in_dim,
        hidden_dim = args.hidden_dim,
        out_dim    = args.out_dim,
        dropout    = args.dropout,
        normalise  = True,
    ).to(device)

    classifier = ClassifierHead(
        in_dim      = args.out_dim,
        num_classes = args.num_classes,
    ).to(device)

    print(f"[Projection] Params: "
          f"{sum(p.numel() for p in projection.parameters()):,} (projection) + "
          f"{sum(p.numel() for p in classifier.parameters()):,} (classifier)")

    # -- Optimiser & loss -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(projection.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -- Training loop --------------------------------------------------------
    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(args.epochs):

        # Train
        projection.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels     = labels.to(device)

            features = projection(embeddings)     # (B, 128)
            logits   = classifier(features)       # (B, num_classes)
            loss     = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()

        scheduler.step()

        train_loss /= n_train
        train_acc   = 100.0 * train_correct / n_train

        # Validate
        projection.eval()
        classifier.eval()
        val_loss    = 0.0
        val_correct = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels     = labels.to(device)
                features   = projection(embeddings)
                logits     = classifier(features)
                loss       = criterion(logits, labels)
                val_loss      += loss.item() * len(labels)
                val_correct   += (logits.argmax(1) == labels).sum().item()

        val_loss /= n_val
        val_acc   = 100.0 * val_correct / n_val

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.1f}% | "
            f"val loss {val_loss:.4f}  acc {val_acc:.1f}%"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_projection_checkpoint(
                args.output_dir, "best",
                projection, classifier, optimizer, epoch, val_acc, args
            )

        # Save periodic checkpoint
        if epoch % args.save_every == 0 or epoch + 1 == args.epochs:
            save_projection_checkpoint(
                args.output_dir, f"{epoch:04d}",
                projection, classifier, optimizer, epoch, val_acc, args
            )

    elapsed = time.time() - start_time
    print(f"\n[Done] Best val accuracy: {best_val_acc:.2f}%  |  "
          f"Time: {elapsed/60:.1f} min")
    print(f"Best checkpoint → {args.output_dir}/checkpoint-best.pth")
    print(
        "\nInference:\n"
        "    from models.projection_model import load_projection_model\n"
        "    proj = load_projection_model('checkpoints/projection/checkpoint-best.pth')\n"
        "    features_128 = proj(embeddings_768)   # (B, 128) L2-normalised"
    )


# =============================================================================
# Argument parser
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("Projection model training", add_help=True)

    parser.add_argument("--embeddings",  required=True,  type=str,
        help="Path to .pt embedding file from extract_cls.py")
    parser.add_argument("--labels",      required=True,  type=str,
        help="Path to .pt label file: LongTensor of shape (N,)")
    parser.add_argument("--output_dir",  default="checkpoints/projection", type=str)
    parser.add_argument("--num_classes", default=10,  type=int,
        help="Number of action/phase classes")

    parser.add_argument("--in_dim",      default=768, type=int)
    parser.add_argument("--hidden_dim",  default=512, type=int)
    parser.add_argument("--out_dim",     default=128, type=int)
    parser.add_argument("--dropout",     default=0.1, type=float)

    parser.add_argument("--epochs",      default=30,   type=int)
    parser.add_argument("--batch_size",  default=256,  type=int,
        help="Can be large — embeddings are small vectors, not raw frames")
    parser.add_argument("--lr",          default=1e-3, type=float)
    parser.add_argument("--weight_decay",default=0.05, type=float)
    parser.add_argument("--save_every",  default=10,   type=int)
    parser.add_argument("--num_workers", default=0,    type=int)
    parser.add_argument("--seed",        default=42,   type=int)

    return parser


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_projection(args)