"""
inference.py
─────────────────────────────────────────────────────────────────────────────
Run violence detection on:
  • A single video file
  • Live webcam / RTSP stream (real-time)

Usage
-----
# Single video
python inference.py --mode video --source /path/to/video.mp4 --checkpoint best_model.pt

# Live webcam
python inference.py --mode live  --source 0             --checkpoint best_model.pt

# RTSP stream
python inference.py --mode live  --source rtsp://...     --checkpoint best_model.pt
"""

import argparse
import time

import cv2
import numpy as np
import torch

from dataset import ViolenceClassifier, RealTimeDetector


def run_on_video(source: str, detector: RealTimeDetector, show: bool = True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    print(f"\n▶ Processing: {source}  ({total} frames @ {fps:.1f} fps)")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = detector.push_frame(frame)
        if result:
            results.append({**result, "frame": frame_idx})
            label = f"{result['prediction']} ({result['confidence']:.0%})"
            color = (0, 0, 255) if result["alert"] else (0, 200, 0)
            print(f"  Frame {frame_idx:5d}: {label}  {'⚠ ALERT' if result['alert'] else ''}")

            if show:
                cv2.putText(frame, label, (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if result["alert"]:
                    cv2.rectangle(frame, (5, 5),
                                  (frame.shape[1]-5, frame.shape[0]-5), color, 4)

        if show:
            cv2.imshow("Violence Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    alerts = [r for r in results if r["alert"]]
    print(f"\nSummary: {len(results)} windows evaluated, {len(alerts)} ALERTS")
    return results


def run_live(source, detector: RealTimeDetector):
    """Same as run_on_video but prints latency per prediction."""
    if str(source).isdigit():
        source = int(source)
    run_on_video(source, detector, show=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["video", "live"], default="video")
    p.add_argument("--source", required=True, help="File path, 0 for webcam, or RTSP URL")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--window",     type=int, default=16, help="Frames per inference window")
    p.add_argument("--stride",     type=int, default=8,  help="Sliding window stride")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--threshold",  type=float, default=0.6)
    p.add_argument("--no_display", action="store_true")
    args = p.parse_args()

    # Load model
    model = ViolenceClassifier.from_checkpoint(args.checkpoint, device=args.device)
    detector = RealTimeDetector(
        model=model,
        device=args.device,
        window_size=args.window,
        stride=args.stride,
        img_size=args.img_size,
        confidence_threshold=args.threshold,
    )

    if args.mode == "video":
        run_on_video(args.source, detector, show=not args.no_display)
    else:
        run_live(args.source, detector)


if __name__ == "__main__":
    main()
