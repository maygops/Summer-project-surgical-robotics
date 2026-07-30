from pathlib import Path
import torch
import av
import torchvision.transforms.functional as F

CLIP_DIR = Path("data/clips/prostatectomy")
OUT_DIR = Path("data/processed/prostatectomy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 1
FRAME_SIZE = (224, 224)
CHUNK = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global mean/std (GPU once)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# Load per-clip action annotations
def load_actions(action_txt_path):
    actions = []
    if not action_txt_path.exists():
        return actions

    with open(action_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            actions.append((start, end, label))

    return actions

# Preprocess a single clip
def preprocess_clip(clip_path, out_path):
    container = av.open(str(clip_path))
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate)

    # Load actions
    action_path = clip_path.with_name(clip_path.stem + "_actions.txt")
    actions = load_actions(action_path)

    target_interval = 1.0 / TARGET_FPS
    last_t = -1e9

    processed_chunks = []
    chunk_frames = []
    kept_timestamps = []

    for frame in container.decode(video=0):
        t = frame.time
        if t is None:
            continue

        # FPS resampling
        if t - last_t < target_interval:
            continue
        last_t = t
        kept_timestamps.append(t)

        # Convert frame to tensor (CPU)
        img = frame.to_rgb().to_ndarray()
        tensor = torch.from_numpy(img)  # HWC
        chunk_frames.append(tensor)

        # Process chunk when full
        if len(chunk_frames) == CHUNK:
            chunk = torch.stack(chunk_frames).to(device)
            chunk = chunk.permute(0, 3, 1, 2).float() / 255.0
            chunk = F.resize(chunk, FRAME_SIZE)
            chunk = (chunk - MEAN) / STD

            processed_chunks.append(chunk.cpu())
            chunk_frames = []
            torch.cuda.empty_cache()

    # Process leftover frames
    if chunk_frames:
        chunk = torch.stack(chunk_frames).to(device)
        chunk = chunk.permute(0, 3, 1, 2).float() / 255.0
        chunk = F.resize(chunk, FRAME_SIZE)
        chunk = (chunk - MEAN) / STD
        processed_chunks.append(chunk.cpu())

    # Concatenate all processed chunks
    if len(processed_chunks) == 0:
        print(f"Warning: no frames extracted for {clip_path}")
        return

    video = torch.cat(processed_chunks, dim=0)
    timestamps = torch.tensor(kept_timestamps)

    # Save
    torch.save({
        "video": video,
        "fps": TARGET_FPS,
        "original_fps": original_fps,
        "timestamps": timestamps,
        "actions": actions,
        "shape": tuple(video.shape)
    }, out_path)

    print(f"Saved: {out_path}")

# Preprocess all clips (filtering for step 5)
def preprocess_all_clips():
    target_steps = {"5"}  # only preprocess step 5

    for video_dir in CLIP_DIR.iterdir():
        if not video_dir.is_dir():
            continue

        for step_dir in video_dir.iterdir():
            if not step_dir.is_dir():
                continue

            step_name = step_dir.name.replace("step_", "")
            if step_name not in target_steps:
                continue

            for clip_path in step_dir.glob("*.mp4"):
                rel_path = clip_path.relative_to(CLIP_DIR)
                out_path = OUT_DIR / rel_path.with_suffix(".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists():
                    print(f"Skipping (exists): {out_path}")

                print(f"Processing: {clip_path}")
                preprocess_clip(clip_path, out_path)
                
if __name__ == "__main__":
    preprocess_all_clips()