#!/usr/bin/env python3
"""End-to-end smoke test with *synthetic* detections (no YOLO / no network).

Drives the real feature -> event -> storage pipeline with hand-crafted frames
and detections, then asserts that CSV + SQLite were written and that the
expected events fired. Lets you verify the data pipeline on a minimal install
(numpy + opencv + pyyaml) before downloading the YOLO model.

    python tests/smoke_test.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MqttCfg, StorageCfg  # noqa: E402
from src.detector import Detection  # noqa: E402
from src.events import EventEngine  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.storage import Storage  # noqa: E402

W, H = 640, 360
PERSISTENT = [1, 2]  # two long-staying visitors


def make_dets(count: int, window_idx: int):
    """Build `count` detections; ids 1,2 persist across windows, rest are unique."""
    ids = PERSISTENT + list(range(100 + window_idx * 50, 100 + window_idx * 50 + max(0, count - 2)))
    ids = ids[:count]
    dets = []
    for i, tid in enumerate(ids):
        x = 20 + (i % 10) * (W - 60) / 10.0
        y = H * 0.5 + (i // 10) * 30
        dets.append(Detection(x, y - 60, x + 24, y, 0.9, tid))
    return dets


def run() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="vibe_smoke_"))
    storage = Storage(StorageCfg(out_dir=str(tmp)), MqttCfg(enabled=False))
    fx = FeatureExtractor(capacity=20, dwell_exit_timeout=2.0)
    ev = EventEngine(baseline_window=60, spike_zscore=2.0, spike_min_jump=3,
                     high_density=20, quiet_motion=0.001, quiet_windows=6,
                     loiter_seconds=30, cooldown_seconds=0)

    counts = [2, 3] * 10 + [22] + [3, 2, 3, 2, 3]  # quiet baseline, then a spike
    t = 1000.0
    for wi, c in enumerate(counts):
        brightness = 20 if wi < 10 else 200  # day/night transition at window 10
        frame = np.full((H, W, 3), brightness, dtype=np.uint8)
        for _ in range(5):  # 5 frames per aggregation window
            fx.process_frame(frame, make_dets(c, wi), t)
            t += 1.0
        finished = fx.finalize_absent_tracks(t)
        storage.write_tracks(finished)
        track_events = ev.evaluate_tracks(finished)
        row = fx.aggregate(t)
        metric_events = ev.evaluate(row) if row else []
        storage.write_metrics(row)
        storage.write_events(track_events + metric_events)
        t += 1.0

    storage.write_tracks(fx.dwell.finalize_absent(t + 1e9))  # flush remaining
    storage.close()

    # -- assertions --------------------------------------------------------
    conn = sqlite3.connect(str(tmp / "public_space.db"))
    n_metrics = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    types = {r[0] for r in conn.execute("SELECT DISTINCT type FROM events")}
    max_visitors = conn.execute("SELECT MAX(visitors_total) FROM metrics").fetchone()[0]
    conn.close()

    checks = {
        "metrics rows >= 20": n_metrics >= 20,
        "events recorded": n_events >= 3,
        "crowd_spike fired": "crowd_spike" in types,
        "high_density fired": "high_density" in types,
        "lighting_change fired": "lighting_change" in types,
        "tracks recorded": n_tracks >= 1,
        "visitors counted": (max_visitors or 0) >= 2,
        "metrics.csv written": (tmp / "metrics.csv").exists(),
        "events.csv written": (tmp / "events.csv").exists(),
    }

    print("\nSMOKE TEST RESULTS")
    print("-" * 48)
    print(f"metrics={n_metrics}  events={n_events}  tracks={n_tracks}")
    print(f"event types: {sorted(types)}")
    print("-" * 48)
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("-" * 48)
    print(f"data dir: {tmp}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
