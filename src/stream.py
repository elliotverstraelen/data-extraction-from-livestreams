"""Robust livestream reader (Assignment Part 1).

A livestream is unbounded and arrives faster than we can run YOLO on every
frame. If you read sequentially you fall further and further behind real time.
So a background thread continuously grabs frames and keeps only the *latest*
one; the processing loop always pulls the freshest frame and silently drops the
rest. This keeps the analysis "live" no matter how slow the model is.

Supported sources (auto-detected):
    "0", "1", ...                webcam index
    youtube.com / youtu.be URL   resolved to a stream URL via yt-dlp
    *.m3u8 / http(s) / rtsp://   passed straight to OpenCV's FFmpeg backend
    path/to/file.mp4             local video file (loops, for offline testing)
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False


def resolve_source(source: str) -> Tuple[object, bool]:
    """Turn a config 'source' string into something cv2.VideoCapture accepts.

    Returns (capture_target, is_live) where is_live tells callers whether the
    source is an unbounded stream (so we should reconnect on EOF) or a finite
    file/loop.
    """
    source = str(source).strip()

    if _is_int(source):
        return int(source), True  # webcam: treat as live (never EOF normally)

    low = source.lower()
    if "youtube.com" in low or "youtu.be" in low:
        return _resolve_youtube(source), True

    if low.startswith("rtsp://") or low.endswith(".m3u8") or low.startswith("http"):
        return source, True

    # Otherwise assume a local file path (finite).
    return source, False


def _resolve_youtube(url: str) -> str:
    """Use yt-dlp to extract a directly-playable stream URL from a YouTube page."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "yt-dlp is required to read YouTube streams. Install it with "
            "`pip install yt-dlp`, or use a direct .m3u8 / file source instead."
        ) from exc

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        # Prefer a mid-resolution mp4/HLS variant: high enough to detect people,
        # low enough to keep YOLO real-time.
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
        # Use the Android client: the default "web" client returns live segment
        # URLs that Google answers with HTTP 403, so OpenCV's FFmpeg stalls on
        # the first segment until its 30s timeout. Android's URLs play directly.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if "url" in info:
        return info["url"]
    # Some live formats nest the playable URL inside 'formats'.
    fmts = info.get("formats") or []
    if fmts:
        return fmts[-1]["url"]
    raise RuntimeError(f"Could not resolve a playable stream URL from: {url}")


class StreamReader:
    """Threaded, always-latest-frame video reader with auto-reconnect."""

    def __init__(self, source: str, process_width: int = 960, reconnect_delay: float = 5.0):
        self.source = source
        self.process_width = int(process_width)
        self.reconnect_delay = float(reconnect_delay)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_live = True
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_returned_id = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stopped_at_eof = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "StreamReader":
        self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _open(self) -> None:
        target, self._is_live = resolve_source(self.source)
        cap = cv2.VideoCapture(target)
        # Keep OpenCV's internal buffer tiny so we stay close to real time.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open stream source: {self.source!r}. "
                "Check the URL/path, your network, and that the stream is live."
            )
        self._cap = cap

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            if not ok:
                if self._is_live:
                    # Live stream hiccup or drop -> wait and reconnect.
                    self._reconnect()
                    continue
                # Finite file: loop it so offline demos keep running.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    self.stopped_at_eof = True
                    break
            frame = self._resize(frame)
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def _reconnect(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        time.sleep(self.reconnect_delay)
        if self._stop.is_set():
            return
        try:
            self._open()
        except RuntimeError:
            # Keep trying until told to stop.
            pass

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.process_width and frame.shape[1] != self.process_width:
            h, w = frame.shape[:2]
            new_h = int(round(h * self.process_width / w))
            frame = cv2.resize(frame, (self.process_width, new_h), interpolation=cv2.INTER_AREA)
        return frame

    # -- consumer API ------------------------------------------------------
    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame, or None if none is available yet."""
        with self._lock:
            if self._frame is None:
                return None
            self._last_returned_id = self._frame_id
            return self._frame.copy()

    def read_new(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Block until a frame *newer* than the last returned one arrives."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.stopped_at_eof:
                return None
            with self._lock:
                if self._frame is not None and self._frame_id != self._last_returned_id:
                    self._last_returned_id = self._frame_id
                    return self._frame.copy()
            time.sleep(0.005)
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
