"""Person detection + tracking (Assignment Part 2, foundation).

Primary backend: Ultralytics YOLO with a built-in tracker (ByteTrack). The
tracker assigns a *persistent ID* to each person across frames, which is the
prerequisite for measuring dwell time and counting unique visitors.

Fallback backend: OpenCV's built-in HOG pedestrian detector paired with a tiny
centroid tracker. Less accurate and CPU-only, but needs no PyTorch download, so
the project still runs on a minimal install.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    track_id: Optional[int] = None

    @property
    def foot(self) -> Tuple[float, float]:
        """Ground-contact point (bottom-centre) -- used for heatmap & grouping."""
        return (0.5 * (self.x1 + self.x2), self.y2)

    @property
    def centroid(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class _CentroidTracker:
    """Minimal greedy nearest-neighbour tracker for the HOG fallback path."""

    def __init__(self, max_distance: float = 80.0, max_missing: int = 15):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self._next_id = 1
        self._objects: dict[int, Tuple[float, float]] = {}
        self._missing: dict[int, int] = {}

    def update(self, centroids: List[Tuple[float, float]]) -> List[int]:
        if not self._objects:
            ids = []
            for c in centroids:
                self._objects[self._next_id] = c
                self._missing[self._next_id] = 0
                ids.append(self._next_id)
                self._next_id += 1
            return ids

        obj_ids = list(self._objects.keys())
        obj_pts = np.array([self._objects[i] for i in obj_ids], dtype=np.float32)
        assigned: dict[int, int] = {}  # centroid index -> object id

        if centroids:
            new_pts = np.array(centroids, dtype=np.float32)
            dists = np.linalg.norm(obj_pts[:, None, :] - new_pts[None, :, :], axis=2)
            used_rows, used_cols = set(), set()
            # Greedy: repeatedly take the globally smallest remaining distance.
            for _ in range(min(len(obj_ids), len(centroids))):
                r, c = np.unravel_index(np.argmin(dists), dists.shape)
                if dists[r, c] > self.max_distance:
                    break
                if r in used_rows or c in used_cols:
                    dists[r, c] = np.inf
                    continue
                oid = obj_ids[r]
                self._objects[oid] = centroids[c]
                self._missing[oid] = 0
                assigned[c] = oid
                used_rows.add(r)
                used_cols.add(c)
                dists[r, c] = np.inf

        result_ids: List[int] = []
        for idx in range(len(centroids)):
            if idx in assigned:
                result_ids.append(assigned[idx])
            else:
                self._objects[self._next_id] = centroids[idx]
                self._missing[self._next_id] = 0
                result_ids.append(self._next_id)
                self._next_id += 1

        # Age out objects that were not matched this frame.
        matched = set(assigned.values()) | set(result_ids)
        for oid in obj_ids:
            if oid not in matched:
                self._missing[oid] += 1
                if self._missing[oid] > self.max_missing:
                    self._objects.pop(oid, None)
                    self._missing.pop(oid, None)
        return result_ids


class Detector:
    PERSON_CLASS = 0  # 'person' in the COCO dataset

    def __init__(
        self,
        backend: str = "auto",
        model: str = "yolov8n.pt",
        conf: float = 0.35,
        device: str = "",
        tracker: str = "bytetrack.yaml",
    ):
        self.conf = float(conf)
        self.device = device or None
        self.tracker = tracker
        self.backend = backend
        self._yolo = None
        self._hog = None
        self._centroids = _CentroidTracker()

        if backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(model)
                self.backend = "yolo"
            except Exception as exc:
                if backend == "yolo":
                    raise RuntimeError(
                        f"YOLO backend requested but unavailable: {exc}. "
                        "Install with `pip install ultralytics`, or set "
                        "detector.backend: hog in config.yaml."
                    ) from exc
                print(f"[detector] YOLO unavailable ({exc}); falling back to HOG.")
                self.backend = "hog"

        if self.backend == "hog":
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @property
    def supports_tracking(self) -> bool:
        return True  # YOLO tracker or centroid fallback both provide IDs

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        if self.backend == "yolo":
            return self._yolo_track(frame)
        return self._hog_detect(frame)

    # -- backends ----------------------------------------------------------
    def _yolo_track(self, frame: np.ndarray) -> List[Detection]:
        results = self._yolo.track(
            frame,
            persist=True,
            classes=[self.PERSON_CLASS],
            conf=self.conf,
            tracker=self.tracker,
            verbose=False,
            device=self.device,
        )
        dets: List[Detection] = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
            dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c),
                                   int(tid) if tid is not None else None))
        return dets

    def _hog_detect(self, frame: np.ndarray) -> List[Detection]:
        rects, weights = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        dets: List[Detection] = []
        centroids: List[Tuple[float, float]] = []
        boxes: List[Tuple[float, float, float, float, float]] = []
        for (x, y, w, h), wt in zip(rects, weights):
            if wt < 0.3:
                continue
            boxes.append((x, y, x + w, y + h, float(wt)))
            centroids.append((x + w / 2.0, y + h / 2.0))
        ids = self._centroids.update(centroids)
        for (x1, y1, x2, y2, wt), tid in zip(boxes, ids):
            dets.append(Detection(x1, y1, x2, y2, min(1.0, wt / 2.0), tid))
        return dets
