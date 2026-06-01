#!/usr/bin/env python3
"""Interactive dashboard (Assignment Part 5).

A Streamlit dashboard over the same SQLite database: KPIs, live time-series, the
live stream next to the hotspot heatmap, dwell distribution and the event log.

It can also start and stop the capture itself, so the whole demo runs from a
single command and a single terminal:

    streamlit run dashboard.py

On load it auto-starts a capture of the default source; use the sidebar to
stop/start it or switch source. Auto-refresh is on by default, so everything
updates live while the capture runs.
"""
from __future__ import annotations

import atexit
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config

st.set_page_config(page_title="Public-Space Vibe", layout="wide")

CFG = load_config()
ROOT = Path(__file__).resolve().parent
MAIN_PY = ROOT / "main.py"
CAPTURE_LOG = ROOT / "data" / "capture.log"


@st.cache_data(ttl=3)
def load(db_path: str):
    p = Path(db_path)
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        metrics = pd.read_sql_query("SELECT * FROM metrics ORDER BY epoch", conn)
        events = pd.read_sql_query("SELECT * FROM events ORDER BY epoch DESC", conn)
        tracks = pd.read_sql_query("SELECT * FROM tracks ORDER BY enter_ts", conn)
    finally:
        conn.close()
    if not metrics.empty:
        metrics["time"] = pd.to_datetime(metrics["epoch"], unit="s")
    return metrics, events, tracks


# -- capture control: run the pipeline straight from the dashboard ----------
# One shared handle across every session and rerun, so websocket reconnects (the
# blocking refresh below drops the socket each cycle) can't spawn duplicate
# captures. cache_resource returns the same dict for the server's lifetime.
@st.cache_resource
def _capture_state() -> dict:
    return {"proc": None, "source": None, "started": False}


CM = _capture_state()


def capture_running() -> bool:
    proc = CM["proc"]
    return proc is not None and proc.poll() is None


def start_capture(source_name: str) -> None:
    """Launch main.py (headless) as a subprocess writing into data/."""
    if capture_running():
        return
    url = CFG.resolve_source(source_name)
    CAPTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    CM["proc"] = subprocess.Popen(
        [sys.executable, str(MAIN_PY), "--no-window", "--source", url],
        cwd=str(ROOT),
        stdout=CAPTURE_LOG.open("ab"),
        stderr=subprocess.STDOUT,
    )
    CM["source"] = url


def stop_capture() -> None:
    proc = CM["proc"]
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)          # let it release the SQLite file
        except Exception:
            proc.kill()
    CM["proc"] = None


atexit.register(stop_capture)   # don't leave the capture orphaned on shutdown


def clear_data(source_name: str) -> None:
    """Delete the generated data (DB, CSVs, heatmap, frame, log) for a fresh
    session. Stops the capture first so it releases the files, then restarts it
    on the same source if it was running."""
    was_running = capture_running()
    stop_capture()
    out = Path(CFG.storage.out_dir)
    db = out / CFG.storage.sqlite_db
    targets = [
        db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm"),
        out / CFG.storage.metrics_csv, out / CFG.storage.events_csv,
        out / "heatmap_latest.png", out / "latest_frame.jpg", out / "capture.log",
    ]
    for p in targets:
        try:
            p.unlink()
        except (FileNotFoundError, PermissionError, OSError):
            pass
    load.clear()
    if was_running:
        start_capture(source_name)


# -- sidebar ---------------------------------------------------------------
st.sidebar.title("Public-Space Vibe")

st.sidebar.header("Capture")
preset_names = [s["name"] for s in CFG.sources]
default_idx = next((i for i, s in enumerate(CFG.sources) if s["url"] == CFG.stream.source), 0)
chosen = st.sidebar.selectbox("Source", preset_names, index=default_idx)
chosen_url = CFG.resolve_source(chosen)

# Auto-start a capture the first time the server runs (single `streamlit run`,
# no second terminal). After that, changing the Source dropdown while a capture
# is running restarts it on the newly selected source.
if not CM["started"]:
    CM["started"] = True
    try:
        start_capture(chosen)
    except Exception as exc:
        st.sidebar.error(f"Could not start capture: {exc}")
elif capture_running() and CM["source"] != chosen_url:
    stop_capture()
    start_capture(chosen)
    st.rerun()

c_start, c_stop = st.sidebar.columns(2)
if c_start.button("Start", use_container_width=True, disabled=capture_running()):
    start_capture(chosen)
    st.rerun()
if c_stop.button("Stop", use_container_width=True, disabled=not capture_running()):
    stop_capture()
    st.rerun()

if st.sidebar.button("Clear data", use_container_width=True,
                     help="Wipe the DB, CSVs, heatmap and frame, then start fresh."):
    clear_data(chosen)
    st.rerun()

if capture_running():
    st.sidebar.success(f"Capturing: {chosen}")
else:
    st.sidebar.info("Capture stopped. Press Start (or pick a source).")

st.sidebar.divider()
db_path = st.sidebar.text_input("SQLite database", "data/public_space.db")
heatmap_path = st.sidebar.text_input("Heatmap image", "data/heatmap_latest.png")
frame_path = st.sidebar.text_input("Live frame image", "data/latest_frame.jpg")
auto = st.sidebar.checkbox("Auto-refresh", value=True)
interval = st.sidebar.slider("Charts refresh (s)", 2, 30, 5, disabled=not auto,
                             help="Charts redraw on each refresh (a Streamlit limitation); "
                                  "a higher value means less flicker. The frame, heatmap and KPIs refresh faster.")


# -- panels, each refreshed by its own fragment ----------------------------
# Fragments refresh independently without a blocking rerun: the analysed frame
# updates fast (~0.8s) so it looks live, while the heavier DB-backed panels
# refresh on the slower `interval`. The embedded video is drawn once in the body
# below, so it is never reloaded by a refresh.
live_src = chosen_url


# Images use width="stretch" to fill their (equal) columns, so the analysed frame
# and the heatmap always render at the same size. Charts get an explicit height so
# they keep their footprint (no page shift) when they redraw. The fast panels
# (frame/heatmap/KPIs) and the slower charts panel are separate fragments, so the
# heavier chart redraw happens only on the (slower) charts interval.
_CHART_H = 240


@st.fragment(run_every=(0.7 if auto else None))
def live_view_panel():
    # Both images live in ONE fragment that builds its own equal columns. A
    # fragment that auto-reruns renders in isolation, so columns created *outside*
    # it lose their width on refresh (the images end up different sizes); creating
    # the columns inside the fragment keeps them equal on every refresh.
    col_frame, col_heat = st.columns(2, vertical_alignment="top")
    with col_frame:
        p = Path(frame_path)
        if p.exists():
            st.image(p.read_bytes(), width="stretch",
                     caption="Latest analysed frame (detected people boxed)")
        else:
            st.caption("No analysed frame yet -- the capture is warming up.")
    with col_heat:
        p = Path(heatmap_path)
        if p.exists():
            st.image(p.read_bytes(), width="stretch",
                     caption="Hotspot heatmap -- where people congregate (warmer = more)")
        else:
            st.caption("Heatmap not generated yet.")


@st.fragment(run_every=(2.0 if auto else None))
def kpi_panel():
    data = load(db_path)
    if data is None:
        st.info("Capture starting up -- waiting for the first rows..." if capture_running()
                else "No data yet. Pick a source and press Start in the sidebar.")
        return
    metrics, events, tracks = data
    if metrics.empty:
        st.info("Capture running -- waiting for the first aggregation window...")
        return
    last = metrics.iloc[-1]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current vibe", f"{last['vibe_index']:.0f}")
    c2.metric("People now", f"{last['person_count']:.0f}")
    c3.metric("Peak people", f"{int(metrics['person_count_max'].max())}")
    c4.metric("Unique visitors", f"{int(metrics['visitors_total'].max())}")
    c5.metric("Avg dwell", f"{(tracks['dwell_seconds'].mean() if not tracks.empty else 0):.0f}s")
    c6.metric("Events", f"{len(events)}")


@st.fragment(run_every=(float(interval) if auto else None))
def charts_panel():
    data = load(db_path)
    if data is None or data[0].empty:
        return
    metrics, events, tracks = data
    left, right = st.columns(2)
    with left:
        st.subheader("Crowd & vibe")
        st.line_chart(metrics.set_index("time")[["person_count", "vibe_index"]], height=_CHART_H)
        st.subheader("Social structure (solo vs groups)")
        st.area_chart(metrics.set_index("time")[["solo_count", "group_count"]], height=_CHART_H)
    with right:
        st.subheader("Activity (motion) & brightness")
        st.line_chart(metrics.set_index("time")[["motion_level", "brightness"]], height=_CHART_H)
        st.subheader("Dwell-time distribution")
        if not tracks.empty:
            hist = pd.cut(tracks["dwell_seconds"], bins=20).value_counts().sort_index()
            hist.index = [f"{int(i.left)}-{int(i.right)}s" for i in hist.index]
            st.bar_chart(hist, height=_CHART_H)
        else:
            st.caption("No finished visitor tracks yet.")

    st.subheader("Recent events")
    if not events.empty:
        st.dataframe(events[["ts", "type", "severity", "message"]].head(25),
                     use_container_width=True, hide_index=True, height=220)
    else:
        st.caption("No events yet.")


st.title("Public-Space Vibe Analytics")
st.subheader("Live view")
live_view_panel()
if isinstance(live_src, str) and live_src.startswith(("http://", "https://")):
    st.link_button("Open the live video feed", live_src)

kpi_panel()
charts_panel()

st.caption("Each camera node writes to CSV + SQLite (+ optional MQTT). "
           "At city scale these streams fan into a time-series DB / data lake for fleet-wide analytics.")
