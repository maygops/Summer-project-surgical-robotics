"""
Frame-level dataset for MAE fine-tuning on surgical clips.

Builds a flat index over all frames across all clips:
    Dataset index | Clip | Frame
    0             |  0   |  0
    1             |  0   |  1
    ...
    451           |  0   | 451
    452           |  1   |  0
    ...
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable
import pandas as pd


class SurgicalFrameDataset(Dataset):
    """
    Flat frame-level dataset built from a directory of .pt clip tensors.

    Each __getitem__ returns a single frame (C, H, W) and its metadata,
    enabling direct use with any frame-level MAE.

    Args:
        clip_dir   : Path to folder containing clip_XXXX.pt files.
        transform  : Optional torchvision transform applied to each frame.
                     If None, frames are returned as-is (ImageNet-normalised
                     floats from the preprocessing pipeline).
        load_mode  : "lazy"  – tensors are loaded on first access and cached.
                     "eager" – all tensors are loaded at construction time.
                               Faster iteration but higher RAM usage.
        step_label : Optional string label for the surgical step (e.g. "5").
                     Stored as metadata only; not used for indexing.
    """

    def __init__(
        self,
        clip_dir: str | Path,
        transform: Optional[Callable] = None,
        load_mode: str = "lazy",
        step_label: Optional[str] = None,
    ):
        assert load_mode in ("lazy", "eager"), "load_mode must be 'lazy' or 'eager'"

        self.clip_dir = Path(clip_dir)
        self.transform = transform
        self.load_mode = load_mode
        self.step_label = step_label

        # Discover and sort clip files
        self.clip_paths = sorted(self.clip_dir.glob("clip_*.pt"))
        if not self.clip_paths:
            raise FileNotFoundError(f"No clip_*.pt files found in {self.clip_dir}")

        # Build flat index: list of (clip_idx, frame_idx) tuples
        # and cache metadata without loading full video tensors
        self._index: list[tuple[int, int]] = []   # (clip_idx, frame_idx)
        self._clip_meta: list[dict] = []           # fps, timestamps, actions per clip
        self._clip_cache: dict[int, torch.Tensor] = {}  # video tensors (lazy or eager)

        self._build_index()

        if load_mode == "eager":
            self._load_all()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self):
        """Read each clip file just enough to know its frame count."""
        for clip_idx, path in enumerate(self.clip_paths):
            data = torch.load(path, weights_only=False)

            n_frames = data["video"].shape[0]   # T in (T, C, H, W)

            # Store lightweight metadata (no video tensor retained here)
            self._clip_meta.append({
                "path":         path,
                "n_frames":     n_frames,
                "fps":          data.get("fps"),
                "original_fps": data.get("original_fps"),
                "timestamps":   data.get("timestamps"),
                "actions":      data.get("actions"),
                "shape":        data.get("shape"),
            })

            # Extend flat index
            for frame_idx in range(n_frames):
                self._index.append((clip_idx, frame_idx))

            # Eager: keep tensor in cache immediately
            if self.load_mode == "eager":
                self._clip_cache[clip_idx] = data["video"]

        print(
            f"[SurgicalFrameDataset] {len(self.clip_paths)} clips | "
            f"{len(self._index)} frames total"
        )

    def _load_all(self):
        """Eagerly load every clip tensor into memory."""
        for clip_idx, meta in enumerate(self._clip_meta):
            if clip_idx not in self._clip_cache:
                data = torch.load(meta["path"], weights_only=False)
                self._clip_cache[clip_idx] = data["video"]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, dataset_idx: int) -> dict:
        clip_idx, frame_idx = self._index[dataset_idx]

        # Lazy-load: load clip tensor on first access, then cache
        if clip_idx not in self._clip_cache:
            data = torch.load(self._clip_meta[clip_idx]["path"], weights_only=False)
            self._clip_cache[clip_idx] = data["video"]

        frame: torch.Tensor = self._clip_cache[clip_idx][frame_idx]  # (C, H, W)

        if self.transform is not None:
            frame = self.transform(frame)

        # Resolve action label for this frame (if available)
        #action_label = self._get_action_label(clip_idx, frame_idx)

        return {
            "frame":       frame,            # (C, H, W) float32
            "dataset_idx": dataset_idx,      # flat dataset index
            "clip_idx":    clip_idx,         # which clip
            "frame_idx":   frame_idx,        # position within clip
            #"step_label":  self.step_label,  # surgical step, e.g. "5"
            #"action":      action_label,     # fine-grained action string or None
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    """
        Map a frame index to its action label via the stored timestamps.

        The actions list contains (start_sec, end_sec, label) tuples.
        We compute the frame's timestamp from fps and do an interval lookup.
    
    def _get_action_label(self, clip_idx: int, frame_idx: int) -> Optional[str]:
        
        meta = self._clip_meta[clip_idx]
        fps = meta.get("fps")
        actions = meta.get("actions")

        if fps is None or not actions:
            return None

        frame_time = frame_idx / fps
        for start, end, label in actions:
            if start <= frame_time <= end:
                return label
        return None
    """
    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame with one row per clip summarising frame counts."""
        rows = []
        offset = 0
        for clip_idx, meta in enumerate(self._clip_meta):
            n = meta["n_frames"]
            rows.append({
                "clip_idx":          clip_idx,
                "clip_file":         meta["path"].name,
                "n_frames":          n,
                "dataset_idx_start": offset,
                "dataset_idx_end":   offset + n - 1,
                "fps":               meta["fps"],
                "shape":             meta["shape"],
            })
            offset += n
        return pd.DataFrame(rows)

    def flat_index_df(self, max_rows: int = 20) -> pd.DataFrame:
        """
        Return a DataFrame mirroring the target format:
            dataset_idx | clip | frame
        """
        rows = [
            {"dataset_idx": i, "clip": c, "frame": f}
            for i, (c, f) in enumerate(self._index[:max_rows])
        ]
        if len(self._index) > max_rows:
            rows.append({"dataset_idx": "...", "clip": "...", "frame": "..."})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    clip_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\KCL\Research\MASTERY\RARP\data\processed\prostatectomy\GSTT_010\step_5"

    # --- Build dataset (lazy loading) ---
    dataset = SurgicalFrameDataset(
        clip_dir=clip_dir,
        transform=None,   # frames are already ImageNet-normalised
        load_mode="lazy",
        step_label="5",
    )

    # --- Print flat index (first and last few rows) ---
    print("\n--- Flat index (first 10 rows) ---")
    print(dataset.flat_index_df(max_rows=10).to_string(index=False))

    print("\n--- Clip summary ---")
    print(dataset.summary_df().to_string(index=False))

    # --- Inspect one sample ---
    sample = dataset[0]
    print("\n--- Sample[0] ---")
    for k, v in sample.items():
        print(f"  {k}: {v if not isinstance(v, torch.Tensor) else v.shape}")

    sample_mid = dataset[452]
    print("\n--- Sample[452] ---")
    for k, v in sample_mid.items():
        print(f"  {k}: {v if not isinstance(v, torch.Tensor) else v.shape}")

    # --- DataLoader smoke-test ---
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\n--- Batch shape ---")
    print(f"  frames : {batch['frame'].shape}")   # (8, C, H, W)
    print(f"  clip   : {batch['clip_idx']}")
    print(f"  frame  : {batch['frame_idx']}")