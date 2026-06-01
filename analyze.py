#!/usr/bin/env python3
"""Offline analysis / reporting (Assignment Part 5, static version).

Reads the SQLite database produced by main.py and generates:
  - a printed summary of KPIs, and
  - PNG charts in data/report/ (time-series, dwell distribution, event counts).

This is the "no extra services" analysis deliverable; dashboard.py is the
interactive equivalent. Run it any time after (or during) a capture:

    python analyze.py                       # uses data/public_space.db
    python analyze.py --db data/public_space.db --out data/report
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load(db_path: Path):
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run `python main.py` first.")
    conn = sqlite3.connect(str(db_path))
    try:
        metrics = pd.read_sql_query("SELECT * FROM metrics ORDER BY epoch", conn)
        events = pd.read_sql_query("SELECT * FROM events ORDER BY epoch", conn)
        tracks = pd.read_sql_query("SELECT * FROM tracks ORDER BY enter_ts", conn)
    finally:
        conn.close()
    for df in (metrics, events):
        if not df.empty:
            df["time"] = pd.to_datetime(df["epoch"], unit="s")
    return metrics, events, tracks


def print_summary(metrics: pd.DataFrame, events: pd.DataFrame, tracks: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(" PUBLIC-SPACE VIBE -- ANALYSIS SUMMARY")
    print("=" * 60)
    if metrics.empty:
        print(" No metrics recorded yet.")
        return
    dur_min = (metrics["epoch"].max() - metrics["epoch"].min()) / 60.0
    print(f" Window rows        : {len(metrics)}  (~{dur_min:.1f} min captured)")
    print(f" People  avg / peak : {metrics['person_count'].mean():.1f} / "
          f"{int(metrics['person_count_max'].max())}")
    print(f" Vibe    avg / peak : {metrics['vibe_index'].mean():.0f} / "
          f"{metrics['vibe_index'].max():.0f}")
    print(f" Unique visitors    : {int(metrics['visitors_total'].max())}")
    if not tracks.empty:
        print(f" Dwell  avg / max   : {tracks['dwell_seconds'].mean():.0f}s / "
              f"{tracks['dwell_seconds'].max():.0f}s")
    busiest = metrics.loc[metrics["vibe_index"].idxmax()]
    print(f" Busiest moment     : {busiest['ts']} (vibe {busiest['vibe_index']:.0f})")
    print(f" Light states       : {metrics['light_state'].value_counts().to_dict()}")
    print(f" Events total       : {len(events)}")
    if not events.empty:
        for t, n in events["type"].value_counts().items():
            print(f"     - {t:<16}: {n}")
    print("=" * 60 + "\n")


def make_charts(metrics, events, tracks, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        print("[analyze] nothing to plot yet.")
        return

    # 1) Crowd + vibe over time
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(metrics["time"], metrics["person_count"], color="#1f77b4", label="people")
    ax1.set_ylabel("people", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(metrics["time"], metrics["vibe_index"], color="#d62728", alpha=0.7, label="vibe")
    ax2.set_ylabel("vibe index", color="#d62728")
    ax1.set_title("Crowd and vibe index over time")
    fig.tight_layout(); fig.savefig(out_dir / "timeseries_crowd_vibe.png", dpi=110); plt.close(fig)

    # 2) Motion with event markers
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(metrics["time"], metrics["motion_level"], color="#2ca02c", label="motion")
    if not events.empty:
        sev_y = metrics["motion_level"].max() or 1.0
        for sev, color in [("alert", "red"), ("warning", "orange"), ("info", "gray")]:
            sub = events[events["severity"] == sev]
            if not sub.empty:
                ax.scatter(sub["time"], [sev_y] * len(sub), c=color, s=30, label=sev, zorder=5)
    ax.set_title("Activity (motion) with events"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_dir / "timeseries_motion_events.png", dpi=110); plt.close(fig)

    # 3) Group vs solo over time
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.stackplot(metrics["time"], metrics["solo_count"], metrics["group_count"],
                 labels=["solo", "groups"], colors=["#9edae5", "#ff9896"])
    ax.set_title("Social structure: solo vs groups"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_dir / "timeseries_groups.png", dpi=110); plt.close(fig)

    # 4) Dwell-time distribution
    if not tracks.empty and len(tracks) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(tracks["dwell_seconds"], bins=min(30, len(tracks)), color="#9467bd")
        ax.set_xlabel("dwell (s)"); ax.set_ylabel("visitors")
        ax.set_title("Dwell-time distribution")
        fig.tight_layout(); fig.savefig(out_dir / "dwell_histogram.png", dpi=110); plt.close(fig)

    # 5) Event counts by type
    if not events.empty:
        counts = events["type"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index, counts.values, color="#ff7f0e")
        ax.set_title("Events by type"); ax.set_ylabel("count")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(out_dir / "events_by_type.png", dpi=110); plt.close(fig)

    print(f"[analyze] charts written to {out_dir.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse Public-Space Vibe data.")
    ap.add_argument("--db", default="data/public_space.db")
    ap.add_argument("--out", default="data/report")
    args = ap.parse_args()

    metrics, events, tracks = load(Path(args.db))
    print_summary(metrics, events, tracks)
    make_charts(metrics, events, tracks, Path(args.out))


if __name__ == "__main__":
    main()
