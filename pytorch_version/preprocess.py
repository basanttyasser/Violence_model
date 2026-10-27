"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Preprocesses violence/non-violence video datasets:
  1. Extracts the N frames with the MOST MOTION per video (optical flow)
  2. Normalises every frame to a fixed square canvas so the model gets
     identical spatial content regardless of the original aspect ratio
     (letterbox / pillarbox padding → no stretching).
  3. Saves the selected frames as a single .npy tensor per video, plus a
     master CSV manifest used by the DataLoader.

Usage
-----
python preprocess.py \
    --data_root /path/to/raw_videos \
    --out_dir   /path/to/preprocessed \
    --num_frames 16 \
    --img_size 224 \
    --num_workers 4

Expected raw layout
-------------------
data_root/
    violence/
        vid001.mp4
        vid002.avi
        ...
    nonviolence/
        vid101.mp4
        ...
"""

import os
import csv
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# ─── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── constants ───────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
LABEL_MAP = {"violence": 1, "nonviolence": 0}


# ─── helpers ─────────────────────────────────────────────────────────────────

def letterbox(frame: np.ndarray, target: int = 224) -> np.ndarray:
    """
    Resize frame to target×target while preserving aspect ratio.
    Pads with black (zeros) so content is never stretched.
    This makes the model aspect-ratio invariant: the same scene filmed
    at 16:9 or 4:3 produces identical content region after letterboxing.
    """
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Dense Farneback optical flow → scalar motion magnitude for the frame."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


def select_top_motion_frames(
    video_path: str,
    num_frames: int = 16,
    img_size: int = 224,
) -> np.ndarray | None:
    """
    Read every frame of the video, compute inter-frame optical-flow magnitude,
    and return the `num_frames` frames with the highest motion scores as a
    uint8 array of shape (num_frames, img_size, img_size, 3).

    Falls back to uniform sampling if the video is too short.
    Returns None on read failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open %s", video_path)
        return None

    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    total = len(frames_bgr)
    if total == 0:
        log.warning("No frames decoded from %s", video_path)
        return None

    # ── compute motion scores ────────────────────────────────────────────────
    if total < 2:
        scores = [0.0]
    else:
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
        scores = [0.0]  # first frame has no predecessor → score 0
        for i in range(1, total):
            scores.append(optical_flow_magnitude(grays[i - 1], grays[i]))

    scores_arr = np.array(scores)

    # ── pick top-N indices ───────────────────────────────────────────────────
    if total <= num_frames:
        # video shorter than desired clip → use all, pad with last frame
        chosen_idx = list(range(total))
        while len(chosen_idx) < num_frames:
            chosen_idx.append(chosen_idx[-1])
    else:
        top_idx = np.argsort(scores_arr)[::-1][:num_frames]
        chosen_idx = sorted(top_idx.tolist())  # keep temporal order

    # ── letterbox & stack ────────────────────────────────────────────────────
    clip = np.stack(
        [letterbox(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB), img_size)
         for i in chosen_idx],
        axis=0,
    )  # (num_frames, H, W, 3) uint8
    return clip


def process_single_video(args):
    """Worker function (runs in subprocess)."""
    video_path, out_path, num_frames, img_size = args
    clip = select_top_motion_frames(video_path, num_frames, img_size)
    if clip is None:
        return video_path, False
    np.save(out_path, clip)
    return video_path, True


# ─── main ────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    data_root: str,
    out_dir: str,
    num_frames: int = 16,
    img_size: int = 224,
    num_workers: int = 4,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    data_root = Path(data_root)
    out_dir = Path(out_dir)

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # ── collect all videos ───────────────────────────────────────────────────
    all_videos: list[tuple[Path, int]] = []
    for label_name, label_id in LABEL_MAP.items():
        label_dir = data_root / label_name
        if not label_dir.exists():
            log.warning("Directory not found: %s — skipping", label_dir)
            continue
        for f in label_dir.iterdir():
            if f.suffix.lower() in SUPPORTED_EXT:
                all_videos.append((f, label_id))

    log.info("Found %d videos total", len(all_videos))

    # ── build task list ──────────────────────────────────────────────────────
    tasks = []
    metadata: list[dict] = []
    for vid_path, label_id in all_videos:
        stem = vid_path.stem
        rel_name = f"{vid_path.parent.name}_{stem}.npy"
        out_path = clips_dir / rel_name
        tasks.append((str(vid_path), str(out_path), num_frames, img_size))
        metadata.append({
            "clip_path": str(out_path),
            "label": label_id,
            "label_name": vid_path.parent.name,
            "original_video": str(vid_path),
        })

    # ── parallel processing ──────────────────────────────────────────────────
    failed: set[str] = set()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(process_single_video, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Preprocessing"):
            vid_path, ok = fut.result()
            if not ok:
                failed.add(vid_path)

    # ── filter out failures ───────────────────────────────────────────────────
    metadata = [m for m in metadata if m["original_video"] not in failed]
    log.info("%d clips successfully saved, %d failed", len(metadata), len(failed))

    # ── train/val/test split ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(metadata))
    n = len(indices)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    for split_name, idxs in splits.items():
        csv_path = out_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["clip_path", "label", "label_name", "original_video"]
            )
            writer.writeheader()
            for i in idxs:
                writer.writerow(metadata[i])
        log.info("Wrote %s: %d samples → %s", split_name, len(idxs), csv_path)

    log.info("✅ Preprocessing complete. Output: %s", out_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection Preprocessor")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    preprocess_dataset(
        data_root=args.data_root,
        out_dir=args.out_dir,
        num_frames=args.num_frames,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
