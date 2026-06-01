"""Structured storage (Assignment Part 4).

Every aggregated metrics row, event, and finished visitor track is persisted in
three complementary ways:

  - CSV    : flat files, trivially openable in Excel / pandas (the "raw data"
             deliverable). One file each for metrics and events.
  - SQLite : a real relational store with metrics / events / tracks tables --
             queryable, indexed, and a faithful local stand-in for the
             time-series database you'd use at scale.
  - MQTT   : optional. When enabled, each row/event is also published to a
             broker, which is how this single camera node would feed a city-wide
             IoT pipeline (Part 5).

Reads are done by dashboard.py / analyze.py straight from the SQLite file.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from pathlib import Path
from typing import List

from .config import MqttCfg, StorageCfg
from .events import Event
from .features import FinishedTrack, MetricsRow, iso_utc

_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT,
    epoch           REAL,
    person_count    REAL,
    person_count_max INTEGER,
    active_tracks   INTEGER,
    visitors_total  INTEGER,
    avg_dwell       REAL,
    group_count     REAL,
    solo_count      REAL,
    avg_group_size  REAL,
    brightness      REAL,
    light_state     TEXT,
    motion_level    REAL,
    vibe_index      REAL
);
"""

_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT,
    epoch     REAL,
    type      TEXT,
    severity  TEXT,
    value     REAL,
    message   TEXT,
    details   TEXT
);
"""

_TRACKS_DDL = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id      INTEGER,
    enter_ts      REAL,
    exit_ts       REAL,
    dwell_seconds REAL,
    enter_iso     TEXT,
    exit_iso      TEXT,
    PRIMARY KEY (track_id, enter_ts)
);
"""


class Storage:
    def __init__(self, cfg: StorageCfg, mqtt_cfg: MqttCfg | None = None):
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.out_dir / cfg.sqlite_db
        self.metrics_csv = self.out_dir / cfg.metrics_csv
        self.events_csv = self.out_dir / cfg.events_csv
        self.heatmap_path = self.out_dir / "heatmap_latest.png"
        self.frame_path = self.out_dir / "latest_frame.jpg"

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        for ddl in (_METRICS_DDL, _EVENTS_DDL, _TRACKS_DDL):
            self._conn.execute(ddl)
        self._conn.commit()

        self._mqtt = None
        if mqtt_cfg and mqtt_cfg.enabled:
            self._init_mqtt(mqtt_cfg)

    # -- MQTT --------------------------------------------------------------
    def _init_mqtt(self, mqtt_cfg: MqttCfg) -> None:
        try:
            import paho.mqtt.client as mqtt

            client = mqtt.Client()
            client.connect(mqtt_cfg.host, mqtt_cfg.port, keepalive=60)
            client.loop_start()
            self._mqtt = client
            self._mqtt_prefix = mqtt_cfg.topic_prefix
            print(f"[storage] MQTT connected to {mqtt_cfg.host}:{mqtt_cfg.port}")
        except Exception as exc:
            print(f"[storage] MQTT disabled (could not connect): {exc}")
            self._mqtt = None

    def _publish(self, topic: str, payload: dict) -> None:
        if self._mqtt is None:
            return
        try:
            self._mqtt.publish(f"{self._mqtt_prefix}/{topic}", json.dumps(payload), qos=0)
        except Exception:
            pass

    # -- writers -----------------------------------------------------------
    def write_metrics(self, row: MetricsRow) -> None:
        d = row.as_dict()
        with self._lock:
            cols = ",".join(d.keys())
            ph = ",".join("?" * len(d))
            self._conn.execute(f"INSERT INTO metrics ({cols}) VALUES ({ph})", list(d.values()))
            self._conn.commit()
            self._append_csv(self.metrics_csv, d)
        self._publish("metrics", d)

    def write_events(self, events: List[Event]) -> None:
        if not events:
            return
        with self._lock:
            for ev in events:
                d = ev.as_dict()
                self._conn.execute(
                    "INSERT INTO events (ts,epoch,type,severity,value,message,details) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (d["ts"], d["epoch"], d["type"], d["severity"], d["value"],
                     d["message"], json.dumps(d["details"])),
                )
                self._append_csv(
                    self.events_csv,
                    {"ts": d["ts"], "type": d["type"], "severity": d["severity"],
                     "value": d["value"], "message": d["message"]},
                )
            self._conn.commit()
        for ev in events:
            self._publish("events", ev.as_dict())

    def write_tracks(self, finished: List[FinishedTrack]) -> None:
        if not finished:
            return
        with self._lock:
            for tr in finished:
                self._conn.execute(
                    "INSERT OR REPLACE INTO tracks "
                    "(track_id,enter_ts,exit_ts,dwell_seconds,enter_iso,exit_iso) "
                    "VALUES (?,?,?,?,?,?)",
                    (tr.track_id, tr.enter_ts, tr.exit_ts, round(tr.dwell_seconds, 2),
                     iso_utc(tr.enter_ts), iso_utc(tr.exit_ts)),
                )
            self._conn.commit()

    def _append_csv(self, path: Path, row: dict) -> None:
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
        if self._mqtt is not None:
            try:
                self._mqtt.loop_stop()
                self._mqtt.disconnect()
            except Exception:
                pass
