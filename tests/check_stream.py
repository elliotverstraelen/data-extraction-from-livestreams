#!/usr/bin/env python3
"""Quickly verify whether a livestream source is usable for this project.

Opens the source, grabs a handful of frames, runs the person detector, and
reports whether it's live, its resolution, and how many people are visible --
then saves an annotated preview image so you can eyeball it. Use it to vet any
candidate public webcam before wiring it into config.yaml.

    python tests/check_stream.py --source "https://www.youtube.com/watch?v=XXXX"
    python tests/check_stream.py --source "https://example.com/live.m3u8" --frames 40
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.stream import StreamReader  # noqa: E402
from src.visualize import draw_boxes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a livestream source.")
    ap.add_argument("--source", required=True,
                    help="a preset name (see config.yaml), or a raw URL / file / webcam index")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--backend", choices=["auto", "yolo", "hog"], default="yolo")
    ap.add_argument("--out", default="data/stream_check.jpg")
    args = ap.parse_args()

    source = load_config().resolve_source(args.source)   # preset name -> URL, else as-is
    print(f"[check] opening {source!r} ...")
    reader = StreamReader(source=source, process_width=960)
    try:
        reader.start()
    except Exception as exc:
        print(f"[check] FAILED to open source: {exc}")
        return 2

    # Wait for the first frame.
    first = reader.read_new(timeout=25.0)
    if first is None:
        print("[check] FAILED: no frames received within 25s (stream offline or geo-blocked?).")
        reader.stop()
        return 2
    h, w = first.shape[:2]
    print(f"[check] connected. processing resolution: {w}x{h}")

    detector = Detector(backend=args.backend)
    print(f"[check] detector backend: {detector.backend}")

    counts = []
    best_frame, best_dets, best_n = first, [], -1
    t0 = time.time()
    for i in range(args.frames):
        frame = reader.read_new(timeout=10.0)
        if frame is None:
            print(f"[check] stream stalled after {i} frames.")
            break
        dets = detector.detect_and_track(frame)
        counts.append(len(dets))
        if len(dets) > best_n:
            best_n, best_frame, best_dets = len(dets), frame, dets
        print(f"  frame {i + 1:>3}/{args.frames}: {len(dets)} people", end="\r")
    elapsed = time.time() - t0
    reader.stop()

    print("\n" + "-" * 50)
    if not counts:
        print("[check] FAILED: connected but could not read frames.")
        return 2
    fps = len(counts) / elapsed if elapsed else 0
    print(f"  frames analysed : {len(counts)}  (~{fps:.1f} fps processing)")
    print(f"  people  avg/max : {statistics.mean(counts):.1f} / {max(counts)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, draw_boxes(best_frame.copy(), best_dets))
    print(f"  preview saved   : {args.out}  (best frame: {best_n} people)")
    print("-" * 50)

    verdict = max(counts) >= 1
    print("VERDICT:", "USABLE ✅ (people detected)" if verdict
          else "connected, but NO people detected — pick a busier scene/time.")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
