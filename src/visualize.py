"""Live annotated view for the demo screencast.

Pure drawing on top of the processed frame: detection boxes + track IDs, an
info panel with the live KPIs, a colour-coded vibe bar, the hotspot heatmap
overlay, and a ticker of the most recent events. None of this changes the data;
it only makes the live demonstration readable.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .detector import Detection
from .events import Event
from .features import FrameFeatures, Heatmap

_GREEN = (80, 220, 80)
_YELLOW = (60, 220, 240)
_WHITE = (245, 245, 245)
_PANEL = (28, 28, 28)
_SEV_COLOR = {"info": (200, 200, 200), "warning": (40, 170, 250), "alert": (60, 60, 240)}


def _put(img, text, org, scale=0.5, color=_WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w]
    if sub.size == 0:
        return
    overlay = np.full_like(sub, _PANEL)
    img[y:y + h, x:x + w] = cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0)


def _vibe_color(v: float):
    """Green (calm) -> yellow -> red (buzzing)."""
    t = max(0.0, min(1.0, v / 100.0))
    if t < 0.5:
        return (80, 200, int(80 + t * 2 * 120))
    return (int(80 + (t - 0.5) * 2 * 60), int(200 - (t - 0.5) * 2 * 140), 240)


def draw_boxes(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for d in detections:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(frame, p1, p2, _GREEN, 2)
        fx, fy = d.foot
        cv2.circle(frame, (int(fx), int(fy)), 3, _YELLOW, -1)
        label = f"#{d.track_id}" if d.track_id is not None else f"{d.conf:.2f}"
        _put(frame, label, (p1[0], max(12, p1[1] - 5)), 0.45, _GREEN, 1)
    return frame


def draw_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    ff: FrameFeatures,
    *,
    fps: float = 0.0,
    visitors_total: int = 0,
    avg_dwell: float = 0.0,
    recent_events: Optional[List[Event]] = None,
    heatmap: Optional[Heatmap] = None,
    show_heatmap: bool = True,
) -> np.ndarray:
    out = frame.copy()
    if show_heatmap and heatmap is not None:
        out = heatmap.overlay(out)

    out = draw_boxes(out, detections)

    # -- info panel (top-left) --------------------------------------------
    _panel(out, 8, 8, 250, 170)
    x, y = 18, 30
    _put(out, "PUBLIC-SPACE VIBE", (x, y), 0.55, _WHITE, 2); y += 24
    _put(out, f"People now : {ff.person_count}", (x, y)); y += 19
    _put(out, f"Groups/Solo: {ff.group_count} / {ff.solo_count}", (x, y)); y += 19
    _put(out, f"Motion     : {ff.motion_level*100:4.1f}%", (x, y)); y += 19
    _put(out, f"Light      : {ff.light_state}", (x, y)); y += 19
    _put(out, f"Visitors   : {visitors_total}  dwell {avg_dwell:.0f}s", (x, y)); y += 22

    # vibe bar
    _put(out, f"Vibe {ff.vibe_index:5.1f}", (x, y), 0.5, _WHITE, 1)
    bx, by, bw, bh = x + 90, y - 10, 130, 12
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
    fillw = int(bw * max(0.0, min(1.0, ff.vibe_index / 100.0)))
    cv2.rectangle(out, (bx, by), (bx + fillw, by + bh), _vibe_color(ff.vibe_index), -1)

    if fps:
        _put(out, f"{fps:4.1f} fps", (out.shape[1] - 80, 22), 0.5, _WHITE, 1)

    # -- recent events ticker (bottom) ------------------------------------
    if recent_events:
        evs = recent_events[-4:]
        ph = 18 * len(evs) + 12
        _panel(out, 8, out.shape[0] - ph - 8, min(560, out.shape[1] - 16), ph)
        ey = out.shape[0] - ph + 6
        for ev in evs:
            color = _SEV_COLOR.get(ev.severity, _WHITE)
            _put(out, f"[{ev.severity.upper()}] {ev.message}", (18, ey), 0.45, color, 1)
            ey += 18
    return out
