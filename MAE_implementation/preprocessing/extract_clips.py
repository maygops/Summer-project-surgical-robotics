import json
from tracemalloc import start
import av
from pathlib import Path

RAW_DIR = Path("data/raw/prostatectomy")
META_DIR = Path("data/metadata/prostatectomy")
CLIP_DIR = Path("data/clips/prostatectomy")

CLIP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Utility: convert HH:MM:SS → seconds
# ---------------------------------------------------------
def to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------------------------------------------------
# Parse TXT phase annotation file
#
# Format:
# HH:MM:SS HH:MM:SS LABEL
# ---------------------------------------------------------
def parse_phase_txt(txt_path):

    intervals = []

    with open(txt_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            start, end, label = parts[:3]

            # Skip EC regions if desired
            if label.upper() == "EC":
                continue

            intervals.append(
                (
                    to_seconds(start),
                    to_seconds(end),
                    label
                )
            )

    return intervals


# ---------------------------------------------------------
# Parse VIA JSON action annotations
#
# VIA format:
# "z": [start_sec, end_sec]
# "av": {"1": "5_cold_cut"}
# ---------------------------------------------------------
def parse_via_json(json_path):

    MIN_ACTION_DURATION = 0.5  # seconds
    MIN_START_TIME = 1.0

    data = json.loads(open(json_path, "r").read())

    actions = []

    for _, meta in data["metadata"].items():

        if "z" not in meta or "av" not in meta:
            continue

        start, end = meta["z"]

        if (end - start) < MIN_ACTION_DURATION:
           continue

        if start < MIN_START_TIME:
            continue

        if len(meta["av"]) == 0:
            continue

        label = list(meta["av"].values())[0]

        actions.append(
            (
                float(start),
                float(end),
                label
            )
        )

    return actions


# ---------------------------------------------------------
# Extract MP4 clip using PyAV
# ---------------------------------------------------------
def extract_clip(video_path, start_sec, end_sec, out_path):

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    # More accurate seek
    seek_ts = int(start_sec / stream.time_base)

    container.seek(
        seek_ts,
        stream=stream,
        any_frame=False,
        backward=True
    )

    output = av.open(str(out_path), mode="w")

    out_stream = output.add_stream(
        "h264",
        rate=stream.average_rate
    )

    out_stream.width = stream.width
    out_stream.height = stream.height
    out_stream.pix_fmt = "yuv420p"

    for frame in container.decode(video=0):

        t = frame.time

        if t is None:
            continue

        if t < start_sec:
            continue

        if t > end_sec:
            break

        packet = out_stream.encode(frame)

        if packet:
            output.mux(packet)

    # Flush encoder
    packet = out_stream.encode(None)

    if packet:
        output.mux(packet)

    output.close()
    container.close()


# ---------------------------------------------------------
# Save clip-local action annotations
#
# Converts:
# global timestamps
#
# → clip-relative timestamps
# ---------------------------------------------------------
def save_clip_actions(
    actions,
    clip_start,
    clip_end,
    out_txt
):

    with open(out_txt, "w") as f:

        for action_start, action_end, label in actions:

            # Keep actions overlapping clip
            if action_end < clip_start:
                continue

            if action_start > clip_end:
                continue

            # Convert to clip-relative timestamps
            rel_start = max(0.0, action_start - clip_start)
            rel_end = min(
                clip_end - clip_start,
                action_end - clip_start
            )
            if rel_end <= rel_start:
                continue
            f.write(
                f"{rel_start:.3f} "
                f"{rel_end:.3f} "
                f"{label}\n"
            )


# ---------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------
def extract_all_clips():

    for video_path in RAW_DIR.glob("*.mp4"):

        video_id = video_path.stem

        print(f"\nProcessing video: {video_id}")

        txt_path = META_DIR / f"{video_id}.txt"
        json_path = META_DIR / f"{video_id}.json"

        # -------------------------------------------------
        # Load phase annotations (TXT)
        # -------------------------------------------------

        if not txt_path.exists():

            print(f"Missing TXT annotations for {video_id}")
            continue

        phase_intervals = parse_phase_txt(txt_path)

        print(f"Loaded {len(phase_intervals)} phase intervals")

        # -------------------------------------------------
        # Load action annotations (JSON)
        # -------------------------------------------------

        action_annotations = []

        if json_path.exists():

            action_annotations = parse_via_json(json_path)

            print(
                f"Loaded "
                f"{len(action_annotations)} "
                f"action annotations"
            )

        else:
            print("No JSON action annotations found")

        # -------------------------------------------------
        # Sort chronologically
        # -------------------------------------------------

        phase_intervals.sort(key=lambda x: x[0])

        # -------------------------------------------------
        # Output directory
        # -------------------------------------------------

        vid_out_dir = CLIP_DIR / video_id

        vid_out_dir.mkdir(
            exist_ok=True,
            parents=True
        )

        # -------------------------------------------------
        # Extract clips
        # -------------------------------------------------

        counters = {}

        for start, end, phase_label in phase_intervals:

            step_dir = vid_out_dir / f"step_{phase_label}"

            step_dir.mkdir(
                exist_ok=True,
                parents=True
            )

            counters.setdefault(phase_label, 0)

            clip_idx = counters[phase_label]

            counters[phase_label] += 1

            # ---------------------------------------------
            # Output paths
            # ---------------------------------------------

            clip_name = f"clip_{clip_idx:04d}"

            clip_path = step_dir / f"{clip_name}.mp4"

            action_txt_path = (
                step_dir /
                f"{clip_name}_actions.txt"
            )

            # ---------------------------------------------
            # Extract video clip
            # ---------------------------------------------

            print(
                f"Extracting step {phase_label}: "
                f"{start:.2f}s → {end:.2f}s"
            )

            extract_clip(
                video_path,
                start,
                end,
                clip_path
            )

            # ---------------------------------------------
            # Save action annotations
            # ---------------------------------------------

            save_clip_actions(
                action_annotations,
                start,
                end,
                action_txt_path
            )

            print(
                f"Saved:\n"
                f"  {clip_path.name}\n"
                f"  {action_txt_path.name}"
            )

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":

    extract_all_clips()