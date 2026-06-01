"""Feature extraction -- the "useful information" (Assignment Part 2).

From the raw per-frame person detections we derive higher-level, *interpretable*
signals about a public space rather than just a head count:

  - crowd_count      : how busy is it right now
  - motion_level     : how much activity / movement (frame differencing)
  - brightness/light : a daylight proxy (night / dusk / overcast / sunny)
  - groups vs solo   : social structure (spatial clustering of people)
  - dwell time       : how long people linger (needs persistent track IDs)
  - hotspot heatmap  : *where* in the scene people congregate
  - vibe_index       : a single 0-100 "liveliness" KPI combining the above

The FeatureExtractor keeps state across frames, accumulates per-frame values,
and emits one aggregated MetricsRow every aggregation window.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection


def iso_utc(epoch: float) -> str:
    """Format an epoch timestamp as a second-resolution UTC ISO string."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Small value objects
# --------------------------------------------------------------------------- #
@dataclass
class FrameFeatures:
    """Instantaneous, single-frame features (used for the live overlay)."""
    person_count: int = 0
    group_count: int = 0
    solo_count: int = 0
    avg_group_size: float = 0.0
    brightness: float = 0.0
    light_state: str = "unknown"
    motion_level: float = 0.0
    vibe_index: float = 0.0


@dataclass
class FinishedTrack:
    track_id: int
    enter_ts: float
    exit_ts: float
    dwell_seconds: float


@dataclass
class MetricsRow:
    """One aggregated row written to CSV/SQLite every aggregation window."""
    ts: str = ""
    epoch: float = 0.0
    person_count: float = 0.0
    person_count_max: int = 0
    active_tracks: int = 0
    visitors_total: int = 0
    avg_dwell: float = 0.0
    group_count: float = 0.0
    solo_count: float = 0.0
    avg_group_size: float = 0.0
    brightness: float = 0.0
    light_state: str = "unknown"
    motion_level: float = 0.0
    vibe_index: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Pure helpers (easy to unit-test in isolation)
# --------------------------------------------------------------------------- #
def classify_light(brightness: float) -> str:
    """Map mean brightness (0..1) to a coarse daylight state. A proxy, not a
    calibrated lux meter -- but stable enough to flag day/night transitions."""
    if brightness < 0.18:
        return "night"
    if brightness < 0.35:
        return "dusk/dawn"
    if brightness < 0.55:
        return "overcast"
    return "bright/sunny"


def cluster_labels(points: List[Tuple[float, float]], dist_thresh: float) -> List[int]:
    """Union-find clustering of 2D points. Returns a cluster label per point
    (two points within dist_thresh share a label)."""
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    t2 = dist_thresh * dist_thresh
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= t2:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return [find(i) for i in range(n)]


def summarize_groups(labels: List[int]) -> Tuple[int, int, float, int]:
    """From cluster labels return (n_groups, solo_count, avg_group_size, in_group_people)
    where a 'group' is any cluster of 2+ people."""
    if not labels:
        return 0, 0, 0.0, 0
    sizes = Counter(labels)
    group_sizes = [s for s in sizes.values() if s >= 2]
    n_groups = len(group_sizes)
    solo = sum(1 for s in sizes.values() if s == 1)
    avg_group_size = float(statistics.mean(group_sizes)) if group_sizes else 0.0
    in_group_people = sum(group_sizes)
    return n_groups, solo, avg_group_size, in_group_people


# --------------------------------------------------------------------------- #
# Dwell tracking
# --------------------------------------------------------------------------- #
class DwellTracker:
    """Turns persistent track IDs into enter/leave times and dwell durations."""

    def __init__(self, exit_timeout: float = 3.0):
        self.exit_timeout = exit_timeout
        self.first_seen: Dict[int, float] = {}
        self.last_seen: Dict[int, float] = {}
        self.ever_seen: set[int] = set()
        self.finished_dwells: List[float] = []

    def update(self, track_ids: List[int], ts: float) -> None:
        for tid in track_ids:
            if tid is None:
                continue
            if tid not in self.first_seen:
                self.first_seen[tid] = ts
            self.last_seen[tid] = ts
            self.ever_seen.add(tid)

    def finalize_absent(self, ts: float) -> List[FinishedTrack]:
        finished: List[FinishedTrack] = []
        for tid in list(self.last_seen.keys()):
            if ts - self.last_seen[tid] > self.exit_timeout:
                dwell = self.last_seen[tid] - self.first_seen[tid]
                finished.append(FinishedTrack(tid, self.first_seen[tid], self.last_seen[tid], dwell))
                self.finished_dwells.append(dwell)
                self.first_seen.pop(tid, None)
                self.last_seen.pop(tid, None)
        return finished

    @property
    def avg_dwell(self) -> float:
        return float(statistics.mean(self.finished_dwells)) if self.finished_dwells else 0.0

    @property
    def visitors_total(self) -> int:
        return len(self.ever_seen)


# --------------------------------------------------------------------------- #
# Hotspot heatmap
# --------------------------------------------------------------------------- #
class Heatmap:
    """Accumulates where people stand to reveal congregation hotspots."""

    def __init__(self):
        self._acc: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None

    def add(self, foot_points: List[Tuple[float, float]], frame_shape: Tuple[int, int, int]) -> None:
        h, w = frame_shape[:2]
        if self._acc is None or self._shape != (h, w):
            self._acc = np.zeros((h, w), dtype=np.float32)
            self._shape = (h, w)
        for (x, y) in foot_points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w:
                self._acc[yi, xi] += 1.0

    def colorized(self, sigma: float = 12.0) -> Optional[np.ndarray]:
        if self._acc is None or self._acc.max() <= 0:
            return None
        blurred = cv2.GaussianBlur(self._acc, (0, 0), sigma)
        norm = np.clip(blurred / (blurred.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        heat = self.colorized()
        if heat is None:
            return frame
        if heat.shape[:2] != frame.shape[:2]:
            heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
        mask = (cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY) > 12).astype(np.float32)[..., None]
        return (frame * (1 - mask * alpha) + heat * (mask * alpha)).astype(np.uint8)

    def save(self, path: str) -> bool:
        heat = self.colorized()
        if heat is None:
            return False
        return bool(cv2.imwrite(path, heat))


# --------------------------------------------------------------------------- #
# Main extractor
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    def __init__(
        self,
        capacity: int = 25,
        group_distance_frac: float = 0.08,
        dwell_exit_timeout: float = 3.0,
        motion_ref: float = 0.05,
    ):
        self.capacity = max(1, int(capacity))
        self.group_distance_frac = group_distance_frac
        self.motion_ref = motion_ref
        self.dwell = DwellTracker(exit_timeout=dwell_exit_timeout)
        self.heatmap = Heatmap()

        self._prev_gray: Optional[np.ndarray] = None
        self._window: List[FrameFeatures] = []
        self._window_ids: set[int] = set()

    # -- per frame ---------------------------------------------------------
    def process_frame(self, frame: np.ndarray, detections: List[Detection], ts: float) -> FrameFeatures:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean() / 255.0)
        motion = self._motion(gray)
        self._prev_gray = gray

        feet = [d.foot for d in detections]
        self.heatmap.add(feet, frame.shape)

        ids = [d.track_id for d in detections if d.track_id is not None]
        self.dwell.update(ids, ts)
        self._window_ids.update(ids)

        thresh = self.group_distance_frac * frame.shape[1]
        labels = cluster_labels(feet, thresh)
        n_groups, solo, avg_group_size, in_group = summarize_groups(labels)

        ff = FrameFeatures(
            person_count=len(detections),
            group_count=n_groups,
            solo_count=solo,
            avg_group_size=round(avg_group_size, 2),
            brightness=brightness,
            light_state=classify_light(brightness),
            motion_level=motion,
            vibe_index=self._vibe(len(detections), motion, in_group),
        )
        self._window.append(ff)
        return ff

    def _motion(self, gray: np.ndarray) -> float:
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return 0.0
        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        return float(thresh.mean() / 255.0)

    def _vibe(self, count: int, motion: float, in_group: int) -> float:
        """0-100 liveliness KPI. Weighted blend of how full, how active, and how
        social the space is. Weights are documented in the README."""
        norm_count = min(1.0, count / self.capacity)
        norm_motion = min(1.0, motion / self.motion_ref)
        group_factor = (in_group / count) if count else 0.0
        return round(100.0 * (0.45 * norm_count + 0.35 * norm_motion + 0.20 * group_factor), 1)

    # -- aggregation -------------------------------------------------------
    def aggregate(self, ts: float) -> Optional[MetricsRow]:
        if not self._window:
            return None
        w = self._window
        counts = [f.person_count for f in w]
        row = MetricsRow(
            ts=iso_utc(ts),
            epoch=round(ts, 3),
            person_count=round(statistics.mean(counts), 2),
            person_count_max=max(counts),
            active_tracks=len(self._window_ids),
            visitors_total=self.dwell.visitors_total,
            avg_dwell=round(self.dwell.avg_dwell, 2),
            group_count=round(statistics.mean(f.group_count for f in w), 2),
            solo_count=round(statistics.mean(f.solo_count for f in w), 2),
            avg_group_size=round(statistics.mean(f.avg_group_size for f in w), 2),
            brightness=round(statistics.mean(f.brightness for f in w), 3),
            light_state=Counter(f.light_state for f in w).most_common(1)[0][0],
            motion_level=round(statistics.mean(f.motion_level for f in w), 4),
            vibe_index=round(statistics.mean(f.vibe_index for f in w), 1),
        )
        self._window.clear()
        self._window_ids.clear()
        return row

    def finalize_absent_tracks(self, ts: float) -> List[FinishedTrack]:
        return self.dwell.finalize_absent(ts)
