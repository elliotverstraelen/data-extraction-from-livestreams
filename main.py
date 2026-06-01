#!/usr/bin/env python3
"""Public-Space Vibe Analytics -- live entrypoint.

Reads a public livestream, extracts crowd/activity/social features, generates
events, and stores everything to CSV + SQLite (+ optional MQTT).

Examples
--------
    # Webcam (works offline, good for a first run)
    python main.py

    # A YouTube live public square
    python main.py --source "https://www.youtube.com/watch?v=XXXXXXXX"

    # A direct HLS public webcam, headless, for 60 seconds
    python main.py --source "https://example.com/live/stream.m3u8" --no-window --duration 60

    # A local clip (offline demo / grading)
    python main.py --source samples/square.mp4
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Public-Space Vibe Analytics from a livestream.")
    p.add_argument("--config", default="config.yaml", help="path to config file")
    p.add_argument("--source", default=None,
                   help="stream source: a preset name (see --list-sources), or a raw "
                        "webcam index / YouTube / HLS / RTSP URL / file path")
    p.add_argument("--pick", action="store_true",
                   help="interactively choose a source from the presets before starting")
    p.add_argument("--list-sources", action="store_true",
                   help="print the available source presets and exit")
    p.add_argument("--backend", choices=["auto", "yolo", "hog"], default=None,
                   help="override detector backend")
    p.add_argument("--no-window", action="store_true", help="run headless (no live OpenCV window)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-stop after N seconds (0 = run until 'q'/Ctrl-C)")
    return p.parse_args()


def print_sources(cfg) -> None:
    print("Available source presets:")
    for i, s in enumerate(cfg.sources, 1):
        print(f"  {i:>2}. {s['name']:<12} {s.get('note', '')}")
        print(f"      {s['url']}")
    print("\nUse:  --source <name>   |   --pick   |   --source <raw URL/path/index>")


def pick_source(cfg) -> str:
    """Show a numbered menu and return the chosen source string."""
    print_sources(cfg)
    while True:
        raw = input("\nChoose a number, a preset name, or paste a URL/path: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(cfg.sources):
                return cfg.sources[idx]["url"]
            print("  number out of range, try again.")
            continue
        # A preset name resolves to its URL; anything else is used as-is.
        return cfg.resolve_source(raw)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.list_sources:
        print_sources(cfg)
        return

    if args.pick:
        source = pick_source(cfg)
    else:
        source = cfg.resolve_source(args.source)

    if args.backend:
        cfg.detector.backend = args.backend

    pipeline = Pipeline(
        cfg,
        source=source,
        show_window=False if args.no_window else None,
        duration=args.duration,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
