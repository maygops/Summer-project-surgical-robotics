"""
models/mae_wrapper.py

Wraps the pretrained MAE encoder with a trainable MLP adapter that sits
directly in the decoder's critical path, so the MAE reconstruction loss
provides a genuine learning signal to the adapter.

Architecture
------------

  [Frozen]   MAE ViT-Base encoder
                     │
               CLS token (768-d)
               patch tokens (196 × 768-d)
                     │
  [Trainable] MLP Adapter  (on CLS token only)
               Linear(768 → 512) → GELU → LayerNorm(512) → Linear(512 → 768)
                     │
               Adapted CLS token (768-d)   ← replaces original CLS in latent
                     │
               Recombine with patch tokens
                     │
  [Trainable] MAE Decoder
                     │
               Reconstruction loss  (MSE on masked patches)

Why this works
--------------
The adapted CLS token is passed to the decoder as part of the latent sequence.
The decoder uses it during cross-patch attention to reconstruct masked regions.
If the adapter produces a poor CLS representation, reconstruction suffers.
Backpropagation therefore flows:

    MAE loss → decoder → adapted CLS → MLP adapter weights

This is a genuine learning signal, not regularisation.

Inference / downstream tasks
-----------------------------
After training, two extraction modes are available:

  model.extract_cls(frames)        →  (B, 768)  full adapted CLS token
  model.extract_features(frames)   →  (B, 128)  compact projection for robotics

The 128-d projection head is a separate lightweight Linear(768 → 128) that can
be trained later with labelled data (action recognition, phase detection, etc.)
on top of the frozen encoder + frozen adapter.

Usage
-----
    from models.mae_wrapper import MAEFineTuner

    model = MAEFineTuner(
        checkpoint_path="checkpoints/mae_pretrained/mae_pretrain_vit_base.pth",
        model_name="base",        # "base" → ViT-Base (768-d), "large" → ViT-Large (1024-d)
        norm_pix_loss=True,
    )
    model.to(device)

    # Training
    loss, pred, mask = model(frames, mask_ratio=0.75)

    # Inference — adapted CLS token (primary surgical representation)
    cls_vec  = model.extract_cls(frames)       # (B, 768)

    # Inference — compact projection (for downstream robotics/action tasks)
    features = model.extract_features(frames)  # (B, 128)
"""

import torch
import torch.nn as nn

from models import models_mae


# ----------------------------------------------------------------------
# Model configuration registry
# ----------------------------------------------------------------------

MODEL_CONFIGS = {
    "base": {
        "factory":   "mae_vit_base_patch16",
        "embed_dim": 768,
    },
    "large": {
        "factory":   "mae_vit_large_patch16",
        "embed_dim": 1024,
    },
    "huge": {
        "factory":   "mae_vit_huge_patch14",
        "embed_dim": 1280,
    },
}


# ----------------------------------------------------------------------
# MLP Adapter
# ----------------------------------------------------------------------

class MLPAdapter(nn.Module):
    """
    Bottleneck MLP that transforms the encoder CLS token before it reaches
    the decoder.  By sitting in the decoder's critical path, it receives
    genuine gradients from the MAE reconstruction loss.

    Architecture:
        Linear(embed_dim → hidden_dim) → GELU → LayerNorm → Linear(hidden_dim → embed_dim)

    Input  : (B, embed_dim)   the CLS token from the frozen encoder
    Output : (B, embed_dim)   an adapted CLS token, same shape, inserted back
                               into the latent sequence before the decoder

    The bottleneck (embed_dim → hidden_dim → embed_dim) forces the adapter
    to learn a compressed but reconstruction-useful CLS representation.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
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

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """cls_token: (B, embed_dim) → (B, embed_dim)"""
        return self.net(cls_token)


# ----------------------------------------------------------------------
# Downstream projection head (used after training, not during)
# ----------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Lightweight linear projection from the adapted CLS token to a compact
    feature vector for downstream tasks (action recognition, robotics, etc.).

    This head is NOT used during self-supervised training.  It is trained
    separately on top of the frozen encoder + frozen adapter using labelled
    downstream data.

    Input  : (B, embed_dim)   e.g. (B, 768)
    Output : (B, out_dim)     e.g. (B, 128)
    """

    def __init__(self, embed_dim: int = 768, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ----------------------------------------------------------------------
# Main wrapper
# ----------------------------------------------------------------------

class MAEFineTuner(nn.Module):
    """
    Fine-tuning wrapper: frozen MAE encoder + trainable MLP adapter in the
    decoder critical path + optional downstream projection head.

    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained .pth file from Meta.
    model_name : str
        One of "base" (768-d), "large" (1024-d), "huge" (1280-d).
    mlp_hidden_dim : int
        Bottleneck dimension inside the MLP adapter.  Default 512 for base.
    proj_out_dim : int
        Output dimension of the downstream projection head.  Default 128.
    norm_pix_loss : bool
        Normalise patch targets before MSE.  Recommended True for surgical
        video due to high pixel value variance across scenes.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "base",
        mlp_hidden_dim: int = 512,
        proj_out_dim: int = 128,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"model_name must be one of {list(MODEL_CONFIGS.keys())}, got '{model_name}'"
            )

        cfg = MODEL_CONFIGS[model_name]
        self.embed_dim = cfg["embed_dim"]

        # ------------------------------------------------------------------
        # 1. Build full MAE (encoder + decoder)
        # ------------------------------------------------------------------
        self.mae = models_mae.__dict__[cfg["factory"]](norm_pix_loss=norm_pix_loss)

        # ------------------------------------------------------------------
        # 2. Load pretrained weights
        # ------------------------------------------------------------------
        self._load_checkpoint(checkpoint_path)

        # ------------------------------------------------------------------
        # 3. Freeze encoder
        #    Encoder components: patch_embed, blocks, norm, pos_embed, cls_token
        #    Decoder components: decoder_embed, mask_token, decoder_pos_embed,
        #                        decoder_blocks, decoder_norm, decoder_pred
        # ------------------------------------------------------------------
        for module in [self.mae.patch_embed, self.mae.blocks, self.mae.norm]:
            for param in module.parameters():
                param.requires_grad = False

        self.mae.pos_embed.requires_grad = False
        self.mae.cls_token.requires_grad = False

        # ------------------------------------------------------------------
        # 4. MLP adapter  (trainable — sits between encoder and decoder)
        # ------------------------------------------------------------------
        self.mlp_adapter = MLPAdapter(
            embed_dim=self.embed_dim,
            hidden_dim=mlp_hidden_dim,
        )

        # ------------------------------------------------------------------
        # 5. Downstream projection head  (separate; trained later with labels)
        # ------------------------------------------------------------------
        self.projection_head = ProjectionHead(
            embed_dim=self.embed_dim,
            out_dim=proj_out_dim,
        )

        # Frozen during self-supervised MAE training.
        # Unfreeze later for downstream tasks (action recognition, robotics):
        #     model.projection_head.requires_grad_(True)
        #     optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=1e-4)
        for p in self.projection_head.parameters():
            p.requires_grad = False

        self._print_param_summary()

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, checkpoint_path: str):
        print(f"[MAEFineTuner] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        msg = self.mae.load_state_dict(state_dict, strict=False)
        print("[MAEFineTuner] Checkpoint loaded.")
        if msg.missing_keys:
            print(f"  Missing keys   : {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys}")

    # ------------------------------------------------------------------
    # Parameter summary
    # ------------------------------------------------------------------

    def _print_param_summary(self):
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MAEFineTuner] Parameters:")
        print(f"  Frozen    (encoder)                  : {frozen:,}")
        print(f"  Trainable (MLP adapter + decoder + proj head) : {trainable:,}")

    # ------------------------------------------------------------------
    # Forward pass  (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass used during self-supervised training.

        Data flow
        ---------
        imgs
          → [frozen encoder]  → latent (B, 1+N_vis, D)
                                          │
                               CLS token (B, D)   patch tokens (B, N_vis, D)
                                          │
                               [MLP adapter]
                                          │
                               adapted CLS (B, D)
                                          │
                               recombine → adapted latent (B, 1+N_vis, D)
                                          │
                               [trainable decoder]
                                          │
                               pred (B, L, p²×3)
                                          │
                               [forward_loss]  ← imgs, mask
                                          │
                               MAE loss (scalar)

        Gradient path
        -------------
        MAE loss → decoder weights → adapted CLS → MLP adapter weights
        Encoder weights receive no gradients (frozen).

        Parameters
        ----------
        imgs       : (B, 3, H, W) ImageNet-normalised surgical frames
        mask_ratio : fraction of patches to mask  (default 0.75)

        Returns
        -------
        loss : scalar
        pred : (B, L, patch_size²×3)
        mask : (B, L)  1 = masked, 0 = visible
        """
        # --- Step 1: Frozen encoder ---
        # torch.no_grad() prevents PyTorch from building a computation graph
        # through the encoder, saving memory and compute.
        with torch.no_grad():
            latent, mask, ids_restore = self.mae.forward_encoder(imgs, mask_ratio)
            # latent shape: (B, 1 + N_visible_patches, embed_dim)
            # latent[:, 0, :]  → CLS token
            # latent[:, 1:, :] → visible patch tokens

        # --- Step 2: MLP adapter on CLS token ---
        # Split latent into CLS and patch tokens
        cls_token    = latent[:, 0, :]    # (B, embed_dim)
        patch_tokens = latent[:, 1:, :]   # (B, N_visible, embed_dim)

        # Transform CLS token through the adapter
        # This is the only step that requires grad tracking for the adapter
        adapted_cls = self.mlp_adapter(cls_token)   # (B, embed_dim)

        # --- Step 3: Recombine into adapted latent ---
        # The decoder receives the adapted CLS + original patch tokens.
        # Because adapted_cls has requires_grad=True, the full graph from
        # MAE loss → decoder → adapted_cls → mlp_adapter is maintained.
        adapted_latent = torch.cat(
            [adapted_cls.unsqueeze(1), patch_tokens], dim=1
        )   # (B, 1 + N_visible, embed_dim)

        # --- Step 4: Trainable decoder ---
        pred = self.mae.forward_decoder(adapted_latent, ids_restore)
        # pred shape: (B, L, patch_size²×3)

        # --- Step 5: Reconstruction loss on masked patches only ---
        loss = self.mae.forward_loss(imgs, pred, mask)

        return loss, pred, mask

    # ------------------------------------------------------------------
    # Inference: adapted CLS token (primary surgical representation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_cls(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract the 768-d adapted CLS token for a batch of frames.

        Uses mask_ratio=0.0 so the encoder sees the full image — no masking
        at inference gives a richer, more stable representation.

        Input  : (B, 3, H, W)
        Output : (B, embed_dim)  e.g. (B, 768)
        """
        latent, _, _ = self.mae.forward_encoder(imgs, mask_ratio=0.0)
        cls_token = latent[:, 0, :]
        return self.mlp_adapter(cls_token)

    # ------------------------------------------------------------------
    # Inference: compact 128-d projection (downstream tasks)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract compact 128-d feature vectors for downstream tasks
        (action recognition, phase detection, robotics planning).

        This uses the projection head on top of the adapted CLS token.
        The projection head should be trained separately with labelled data.

        Input  : (B, 3, H, W)
        Output : (B, proj_out_dim)  e.g. (B, 128)
        """
        cls_vec = self.extract_cls(imgs)          # (B, 768)
        return self.projection_head(cls_vec)      # (B, 128)

    # ------------------------------------------------------------------
    # Convenience: parameters for the optimiser
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """
        Returns only trainable parameters for the optimiser.
        During self-supervised training this excludes the frozen encoder.

        Usage:
            optimiser = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
        """
        return [p for p in self.parameters() if p.requires_grad]


# ----------------------------------------------------------------------
# Smoke-test  —  python models/mae_wrapper.py <ckpt_path>
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ckpt = (
        sys.argv[1] if len(sys.argv) > 1
        else "checkpoints/mae_pretrained/mae_pretrain_vit_base.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = MAEFineTuner(
        checkpoint_path=ckpt,
        model_name="base",
        mlp_hidden_dim=512,
        proj_out_dim=128,
        norm_pix_loss=True,
    ).to(device)

    dummy = torch.randn(4, 3, 224, 224, device=device)

    # Training pass
    model.train()
    loss, pred, mask = model(dummy, mask_ratio=0.75)
    print(f"\n--- Training forward pass ---")
    print(f"  loss : {loss.item():.4f}")
    print(f"  pred : {pred.shape}")
    print(f"  mask : {mask.shape}")

    # Verify gradients exist on adapter but not encoder
    loss.backward()
    print(f"\n--- Gradient check ---")
    for name, param in model.named_parameters():
        has_grad = param.grad is not None
        status   = "GRAD ✓" if has_grad else "frozen"
        print(f"  {status:8s}  {name}")

    # Inference
    model.eval()
    cls_vec  = model.extract_cls(dummy)
    features = model.extract_features(dummy)
    print(f"\n--- Inference ---")
    print(f"  CLS vector (768-d) : {cls_vec.shape}")
    print(f"  Features   (128-d) : {features.shape}")