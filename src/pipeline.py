"""Pipeline orchestration -- ties Parts 1-4 together.

    stream -> detect+track -> features -> (every window) aggregate
           -> events -> storage (CSV/SQLite/MQTT) -> live overlay

The processing loop always pulls the freshest frame from the threaded reader,
so the analysis stays live even when the model is slower than the stream.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2

from .config import Config
from .detector import Detector
from .events import EventEngine
from .features import FeatureExtractor
from .storage import Storage
from .stream import StreamReader
from .visualize import draw_boxes, draw_overlay


class Pipeline:
    def __init__(self, cfg: Config, *, source: Optional[str] = None,
                 show_window: Optional[bool] = None, duration: float = 0.0):
        self.cfg = cfg
        self.duration = duration
        self.show_window = cfg.display.show_window if show_window is None else show_window

        self.reader = StreamReader(
            source=source or cfg.stream.source,
            process_width=cfg.stream.process_width,
            reconnect_delay=cfg.stream.reconnect_delay,
        )
        self.detector = Detector(
            backend=cfg.detector.backend, model=cfg.detector.model,
            conf=cfg.detector.conf, device=cfg.detector.device, tracker=cfg.detector.tracker,
        )
        self.features = FeatureExtractor(
            capacity=cfg.features.capacity,
            group_distance_frac=cfg.features.group_distance_frac,
            dwell_exit_timeout=cfg.features.dwell_exit_timeout,
        )
        self.events = EventEngine(
            baseline_window=cfg.events.baseline_window, spike_zscore=cfg.events.spike_zscore,
            spike_min_jump=cfg.events.spike_min_jump, high_density=cfg.events.high_density,
            quiet_motion=cfg.events.quiet_motion, quiet_windows=cfg.events.quiet_windows,
            loiter_seconds=cfg.events.loiter_seconds, cooldown_seconds=cfg.events.cooldown_seconds,
            light_min_windows=cfg.events.light_min_windows,
        )
        self.storage = Storage(cfg.storage, cfg.mqtt)

        self._recent_events = deque(maxlen=20)
        self._fps = 0.0
        self._window_open = False

    # ----------------------------------------------------------------------
    def run(self) -> None:
        print(f"[pipeline] backend={self.detector.backend} source={self.reader.source!r}")
        print(f"[pipeline] writing to {self.storage.db_path} (+ CSV). Press 'q' or Ctrl-C to stop.")
        self.reader.start()

        start = time.time()
        last_agg = start
        last_heat = start
        last_frame = start
        frame_every = 0.3          # how often to refresh the dashboard's live frame
        agg_every = self.cfg.features.aggregation_seconds
        heat_every = self.cfg.storage.heatmap_every_seconds
        min_dt = (1.0 / self.cfg.stream.target_fps) if self.cfg.stream.target_fps > 0 else 0.0

        try:
            while True:
                t0 = time.time()
                if self.duration and (t0 - start) >= self.duration:
                    break

                frame = self.reader.read_new(timeout=15.0)
                if frame is None:
                    if self.reader.stopped_at_eof:
                        print("[pipeline] source ended.")
                        break
                    print("[pipeline] no frame for 15s; still waiting/reconnecting...")
                    continue

                ts = time.time()
                detections = self.detector.detect_and_track(frame)
                ff = self.features.process_frame(frame, detections, ts)

                # -- aggregation window ----------------------------------
                if ts - last_agg >= agg_every:
                    finished = self.features.finalize_absent_tracks(ts)
                    self.storage.write_tracks(finished)
                    track_events = self.events.evaluate_tracks(finished)

                    row = self.features.aggregate(ts)
                    metric_events = self.events.evaluate(row) if row else []
                    if row:
                        self.storage.write_metrics(row)
                        print(f"[{row.ts}] people={row.person_count:.1f} "
                              f"vibe={row.vibe_index:.0f} motion={row.motion_level:.3f} "
                              f"light={row.light_state} visitors={row.visitors_total}")

                    all_events = track_events + metric_events
                    if all_events:
                        self.storage.write_events(all_events)
                        for ev in all_events:
                            self._recent_events.append(ev)
                            print(f"   >> EVENT [{ev.severity}] {ev.message}")
                    last_agg = ts

                # -- live annotated frame (for the dashboard) ------------
                # Saved several times a second, independent of the slower
                # aggregation, so the dashboard's live view looks responsive.
                # Written via a temp file + atomic rename so a reader never
                # picks up a half-written image.
                if ts - last_frame >= frame_every:
                    tmp = self.storage.frame_path.with_suffix(".tmp.jpg")
                    cv2.imwrite(str(tmp), draw_boxes(frame.copy(), detections))
                    tmp.replace(self.storage.frame_path)
                    last_frame = ts

                # -- heatmap snapshot (atomic, like the frame) -----------
                if ts - last_heat >= heat_every:
                    tmp = self.storage.heatmap_path.with_suffix(".tmp.png")
                    if self.features.heatmap.save(str(tmp)):
                        tmp.replace(self.storage.heatmap_path)
                    last_heat = ts

                # -- display ---------------------------------------------
                dt = time.time() - t0
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
                if self.show_window:
                    self._render(frame, detections, ff)

                if min_dt:
                    sleep = min_dt - (time.time() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
        except KeyboardInterrupt:
            print("\n[pipeline] interrupted by user.")
        finally:
            self._shutdown(time.time())

    # ----------------------------------------------------------------------
    def _render(self, frame, detections, ff) -> None:
        try:
            out = draw_overlay(
                frame, detections, ff,
                fps=self._fps,
                visitors_total=self.features.dwell.visitors_total,
                avg_dwell=self.features.dwell.avg_dwell,
                recent_events=list(self._recent_events),
                heatmap=self.features.heatmap,
                show_heatmap=self.cfg.display.show_heatmap_overlay,
            )
            cv2.imshow("Public-Space Vibe Analytics", out)
            self._window_open = True
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            raise
        except cv2.error:
            # No display available (headless server) -> disable window, keep running.
            print("[pipeline] no display available; continuing headless.")
            self.show_window = False

    def _shutdown(self, ts: float) -> None:
        print("[pipeline] shutting down, flushing final data...")
        # Force-finalise everyone still on screen so their dwell is recorded.
        finished = self.features.dwell.finalize_absent(ts + 1e9)
        self.storage.write_tracks(finished)
        final_events = self.events.evaluate_tracks(finished)
        row = self.features.aggregate(ts)
        if row:
            self.storage.write_metrics(row)
            final_events += self.events.evaluate(row)
        if final_events:
            self.storage.write_events(final_events)
        self.features.heatmap.save(str(self.storage.heatmap_path))
        self.reader.stop()
        self.storage.close()
        if self._window_open:
            cv2.destroyAllWindows()
        print(f"[pipeline] done. Data in: {self.storage.out_dir.resolve()}")
