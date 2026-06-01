#!/usr/bin/env python3
"""Generate a synthetic sample clip for OFFLINE pipeline testing.

This produces moving person-like figures over a changing background so you can
exercise the full stream -> features -> events -> storage path without a live
internet stream (handy for grading offline). NOTE: these are synthetic shapes,
so the YOLO/HOG *person* detector will mostly not fire on them -- the clip is
meant to validate the pipeline plumbing (motion, heatmap, aggregation, storage),
not detection accuracy. For real detections, point main.py at a real public
livestream (see the README).

    python tests/make_sample_video.py            # -> data/sample_walk.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make(path: Path, seconds: int = 20, fps: int = 15, w: int = 640, h: int = 360) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = seconds * fps
    rng = np.random.default_rng(42)
    # A few "people" with their own speed; crowd grows in the middle of the clip.
    agents = [{"x": rng.uniform(0, w), "y": rng.uniform(h * 0.5, h * 0.9),
               "vx": rng.uniform(-3, 3), "speed": rng.uniform(0.8, 1.6)} for _ in range(10)]
    for i in range(n):
        bright = int(60 + 150 * (i / n))            # gradual day-break (light change)
        frame = np.full((h, w, 3), bright, np.uint8)
        cv2.rectangle(frame, (0, int(h * 0.85)), (w, h), (40, 40, 40), -1)  # ground strip
        active = agents[: 3 + int(7 * np.sin(np.pi * i / n) ** 2)]          # crowd swells mid-clip
        for a in active:
            a["x"] = (a["x"] + a["vx"] * a["speed"]) % w
            cx, cy = int(a["x"]), int(a["y"])
            cv2.rectangle(frame, (cx - 8, cy - 45), (cx + 8, cy), (30, 30, 30), -1)  # body
            cv2.circle(frame, (cx, cy - 52), 7, (30, 30, 30), -1)                    # head
        writer.write(frame)
    writer.release()
    print(f"[make_sample_video] wrote {path} ({seconds}s @ {fps}fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_walk.mp4")
    ap.add_argument("--seconds", type=int, default=20)
    args = ap.parse_args()
    make(Path(args.out), seconds=args.seconds)
