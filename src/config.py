"""Typed configuration loader.

Loads config.yaml over a set of dataclass defaults so the app still runs even
if the YAML is missing keys (or missing entirely). Access is attribute-style:

    cfg = load_config()
    cfg.stream.source, cfg.events.high_density, ...
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StreamCfg:
    source: str = "0"
    process_width: int = 960
    target_fps: float = 0.0
    reconnect_delay: float = 5.0


@dataclass
class DetectorCfg:
    backend: str = "auto"
    model: str = "yolov8n.pt"
    conf: float = 0.35
    device: str = ""
    tracker: str = "bytetrack.yaml"


@dataclass
class FeaturesCfg:
    aggregation_seconds: float = 5.0
    dwell_exit_timeout: float = 3.0
    group_distance_frac: float = 0.08
    capacity: int = 25


@dataclass
class EventsCfg:
    baseline_window: int = 60
    spike_zscore: float = 2.0
    spike_min_jump: int = 3
    high_density: int = 20
    quiet_motion: float = 0.01
    quiet_windows: int = 6
    loiter_seconds: float = 120.0
    cooldown_seconds: float = 20.0
    light_min_windows: int = 3


@dataclass
class StorageCfg:
    out_dir: str = "data"
    sqlite_db: str = "public_space.db"
    metrics_csv: str = "metrics.csv"
    events_csv: str = "events.csv"
    heatmap_every_seconds: float = 30.0


@dataclass
class MqttCfg:
    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    topic_prefix: str = "publicspace"


@dataclass
class DisplayCfg:
    show_window: bool = True
    show_heatmap_overlay: bool = True


# The full source menu lives in config.yaml. These two offline-safe entries are
# only a fallback so the picker still works if config.yaml is missing.
_FALLBACK_SOURCES = [
    {"name": "webcam", "url": "0", "note": "Local webcam (index 0)."},
    {"name": "sample", "url": "data/sample_walk.mp4", "note": "Bundled clip, loops offline."},
]


@dataclass
class Config:
    stream: StreamCfg = field(default_factory=StreamCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    events: EventsCfg = field(default_factory=EventsCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    mqtt: MqttCfg = field(default_factory=MqttCfg)
    display: DisplayCfg = field(default_factory=DisplayCfg)
    sources: list = field(default_factory=lambda: [dict(s) for s in _FALLBACK_SOURCES])

    def resolve_source(self, value: str | None) -> str | None:
        """Turn a --source value into an actual stream source.

        A value matching a preset name returns that preset's URL; anything else
        is returned unchanged (raw URL, file path or webcam index). None passes
        through so the caller can fall back to stream.source.
        """
        if value is None:
            return None
        for s in self.sources:
            if s.get("name") == value:
                return s["url"]
        return value


def _merge(node: Any, data: dict | None) -> Any:
    """Recursively overlay a dict onto a dataclass instance, keeping defaults."""
    if not data:
        return node
    for f in fields(node):
        if f.name not in data:
            continue
        current = getattr(node, f.name)
        if is_dataclass(current) and isinstance(data[f.name], dict):
            _merge(current, data[f.name])
        else:
            setattr(node, f.name, data[f.name])
    return node


def load_config(path: str | Path = "config.yaml") -> Config:
    cfg = Config()
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _merge(cfg, raw)
    return cfg
