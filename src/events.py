"""Event generation (Assignment Part 3).

Raw metrics are continuous; an *event* is a discrete "something relevant just
happened" signal -- the thing a downstream system would alert on, count, or act
upon. We use a rolling statistical baseline (mean +/- z*std over a sliding
window) so the definition of "unusual" adapts to each location and time of day,
rather than relying on hand-tuned absolute numbers. Per-type cooldowns keep the
event stream meaningful instead of chattering every window.

Event types:
    crowd_spike     (alert)   crowd suddenly far above its recent baseline
    crowd_drop      (warning) crowd suddenly disperses well below baseline
    high_density    (alert)   absolute occupancy crosses a safety threshold
    quiet_period    (info)    sustained low activity (scene effectively idle)
    lighting_change (info)    day/night/overcast transition detected
    loiter          (warning) a single visitor lingered beyond loiter_seconds
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

from .features import FinishedTrack, MetricsRow, iso_utc


@dataclass
class Event:
    ts: str
    epoch: float
    type: str
    severity: str          # info | warning | alert
    value: float
    message: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class EventEngine:
    def __init__(
        self,
        baseline_window: int = 60,
        spike_zscore: float = 2.0,
        spike_min_jump: int = 3,
        high_density: int = 20,
        quiet_motion: float = 0.01,
        quiet_windows: int = 6,
        loiter_seconds: float = 120.0,
        cooldown_seconds: float = 20.0,
        light_min_windows: int = 3,
    ):
        self.baseline_window = baseline_window
        self.spike_zscore = spike_zscore
        self.spike_min_jump = spike_min_jump
        self.high_density = high_density
        self.quiet_motion = quiet_motion
        self.quiet_windows = quiet_windows
        self.loiter_seconds = loiter_seconds
        self.cooldown_seconds = cooldown_seconds
        self.light_min_windows = max(1, int(light_min_windows))

        self._counts: Deque[float] = deque(maxlen=baseline_window)
        self._last_emit: Dict[str, float] = {}
        self._prev_light: Optional[str] = None
        self._light_candidate: Optional[str] = None
        self._light_candidate_streak = 0
        self._quiet_streak = 0

    # -- helpers -----------------------------------------------------------
    def _cooled_down(self, etype: str, epoch: float) -> bool:
        last = self._last_emit.get(etype)
        return last is None or (epoch - last) >= self.cooldown_seconds

    def _emit(self, events: List[Event], epoch: float, etype: str, severity: str,
              value: float, message: str, details: Optional[dict] = None) -> None:
        events.append(Event(iso_utc(epoch), round(epoch, 3), etype, severity,
                            round(float(value), 3), message, details or {}))
        self._last_emit[etype] = epoch

    # -- main entry points -------------------------------------------------
    def evaluate(self, row: MetricsRow) -> List[Event]:
        """Evaluate one aggregated metrics row; returns any events it triggered."""
        events: List[Event] = []
        epoch = row.epoch
        count = row.person_count

        # Baseline statistics from history *before* including this row.
        if len(self._counts) >= 5:
            mean = statistics.mean(self._counts)
            std = statistics.pstdev(self._counts) or 0.0
            band = self.spike_zscore * std

            if std > 0 and count > mean + band and (count - mean) >= self.spike_min_jump:
                if self._cooled_down("crowd_spike", epoch):
                    self._emit(events, epoch, "crowd_spike", "alert", count,
                               f"Crowd spike: {count:.0f} people (baseline ~{mean:.0f}).",
                               {"baseline": round(mean, 1), "std": round(std, 2)})

            if std > 0 and count < mean - band and (mean - count) >= self.spike_min_jump:
                if self._cooled_down("crowd_drop", epoch):
                    self._emit(events, epoch, "crowd_drop", "warning", count,
                               f"Sudden dispersal: {count:.0f} people (baseline ~{mean:.0f}).",
                               {"baseline": round(mean, 1), "std": round(std, 2)})

        # Absolute density threshold (safety-style, not relative).
        if row.person_count_max >= self.high_density and self._cooled_down("high_density", epoch):
            self._emit(events, epoch, "high_density", "alert", row.person_count_max,
                       f"High density: {row.person_count_max} people in frame "
                       f"(threshold {self.high_density}).")

        # Sustained quiet period.
        if row.motion_level < self.quiet_motion:
            self._quiet_streak += 1
            if self._quiet_streak == self.quiet_windows and self._cooled_down("quiet_period", epoch):
                self._emit(events, epoch, "quiet_period", "info", row.motion_level,
                           f"Quiet period: low activity for {self.quiet_windows} windows.")
        else:
            self._quiet_streak = 0

        # Lighting transition, debounced. Brightness near a classify_light()
        # cut-off makes the label flicker between two states every window, so a
        # differing label is only a candidate: we commit it (and emit one event)
        # once it has held for light_min_windows consecutive windows.
        new_state = row.light_state
        if self._prev_light is None:
            self._prev_light = new_state          # first window: adopt silently
        elif new_state == self._prev_light:
            self._light_candidate = None          # back to the committed state
            self._light_candidate_streak = 0
        else:
            if new_state == self._light_candidate:
                self._light_candidate_streak += 1
            else:
                self._light_candidate = new_state
                self._light_candidate_streak = 1
            if self._light_candidate_streak >= self.light_min_windows:
                self._emit(events, epoch, "lighting_change", "info", row.brightness,
                           f"Lighting changed: {self._prev_light} -> {new_state}.",
                           {"from": self._prev_light, "to": new_state})
                self._prev_light = new_state
                self._light_candidate = None
                self._light_candidate_streak = 0

        self._counts.append(count)
        return events

    def evaluate_tracks(self, finished: List[FinishedTrack]) -> List[Event]:
        """Emit loiter events for visitors who lingered beyond the threshold."""
        events: List[Event] = []
        for tr in finished:
            if tr.dwell_seconds >= self.loiter_seconds:
                self._emit(events, tr.exit_ts, "loiter", "warning", tr.dwell_seconds,
                           f"Loiter: visitor #{tr.track_id} stayed {tr.dwell_seconds:.0f}s.",
                           {"track_id": tr.track_id})
        return events
