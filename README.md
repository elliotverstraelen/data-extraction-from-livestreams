# Public-Space Vibe Analytics

**Big Data Extraction from Livestreams - Postgraduate IoT & Big Data**

A Python + OpenCV + YOLO application that watches a **public livestream of a
public space** (a square, promenade, beach or busy street) and turns the raw
video into a structured, queryable stream of *interpretable* signals about the
place - not just a head count, but how **busy**, **active**, **social**, and
**well-lit** it is over time, **where** people congregate, and **how long** they
stay. It generates **events** when something noteworthy happens and stores
everything to **CSV + SQLite (+ optional MQTT)** for downstream Big Data use.

---

## 1. What information is extracted, and why is it useful?

> *"Just listing people and their positions is not interesting."* - so we don't.
> We derive higher-level features and a single headline KPI from them.

| Signal | How it's computed | Why it's useful / who cares |
|---|---|---|
| **Crowd count** | YOLO person detections per frame, averaged per window | Baseline occupancy - retailers, transport, safety |
| **Vibe index (0-100)** | Weighted blend of fullness + activity + sociability | One glanceable "liveliness" KPI for a place over time |
| **Dwell time** | Persistent track IDs -> enter/leave timestamps per visitor | Do people *linger* or pass through? Retail & placemaking |
| **Groups vs solo** | Spatial clustering of people's foot positions | Social structure - events, nightlife, family-friendliness |
| **Activity / motion** | Frame differencing (independent of detection) | Robust "is anything happening" signal, even at distance |
| **Light state** | Mean brightness -> night / dusk / overcast / sunny | Daylight & weather proxy to correlate with behaviour |
| **Hotspot heatmap** | Accumulated foot positions, blurred & colour-mapped | *Where* people gather - bench/stage/entrance placement |
| **Unique visitors** | Distinct track IDs over the session | Footfall estimate without counting the same person twice |

**Who could be interested?** City/tourism boards (how lively is the
promenade by hour?), retailers and HoReCa near the square (when to staff up),
event & security teams (anomalous gatherings/dispersals), and urban planners
(where to put seating, lighting, or a stage - straight from the heatmap).

### The "vibe index" formula

```
norm_count  = min(1, people / capacity)          # how full        (weight 0.45)
norm_motion = min(1, motion / motion_ref)         # how active       (weight 0.35)
group_factor= people_in_groups / people           # how social       (weight 0.20)
vibe        = 100 * (0.45*norm_count + 0.35*norm_motion + 0.20*group_factor)
```

`capacity` (the crowd size considered "full") and the motion reference are
configurable per location in `config.yaml`.

---

## 2. Architecture / pipeline

```
                                                    ┌────────── Big Data context (Part 5) ──────────┐
 livestream     frame        feature        event   │  CSV  ┐                                        │
 (YouTube/  →  capture   →  extraction  →  generation→ SQLite ├→ dashboard.py (Streamlit, live KPIs)  │
  HLS/RTSP)    (threaded,   (YOLO+track,   (rolling   │  MQTT ┘   analyze.py  (offline charts/report) │
               latest      heatmap,dwell,  baseline,  │           …or fan-in to a city-wide TSDB/lake │
               frame)      groups,light)   cooldowns) └───────────────────────────────────────────────┘
```

| Stage | File | Assignment part |
|---|---|---|
| Open & read the livestream (threaded, auto-reconnect) | `src/stream.py` | **Part 1** |
| Detect & track people, extract features | `src/detector.py`, `src/features.py` | **Part 2** |
| Generate events on relevant changes | `src/events.py` | **Part 3** |
| Store to CSV / SQLite / MQTT | `src/storage.py` | **Part 4** |
| Analyse & visualise (dashboard + report) | `dashboard.py`, `analyze.py` | **Part 5** |
| Orchestration & live overlay | `src/pipeline.py`, `src/visualize.py`, `main.py` | glue |

---

## 3. Data model (Part 4)

Everything lands in `data/`. SQLite (`data/public_space.db`) has three tables;
the same metrics/events are mirrored to flat CSVs for the "raw data" deliverable.

**`metrics`** - one row per aggregation window (default 5 s):

`ts, epoch, person_count, person_count_max, active_tracks, visitors_total,
avg_dwell, group_count, solo_count, avg_group_size, brightness, light_state,
motion_level, vibe_index`

**`events`** - one row per generated event:

`ts, epoch, type, severity(info|warning|alert), value, message, details(JSON)`

**`tracks`** - one row per finished visitor (for dwell analysis):

`track_id, enter_ts, exit_ts, dwell_seconds, enter_iso, exit_iso`

---

## 4. Events generated (Part 3)

Events use a **rolling statistical baseline** (mean ± z·σ over a sliding window)
so "unusual" adapts to each location and time of day, with **per-type cooldowns**
to keep the stream meaningful.

| Event | Severity | Fires when |
|---|---|---|
| `crowd_spike` | alert | crowd jumps far above its recent baseline |
| `crowd_drop` | warning | crowd suddenly disperses below baseline |
| `high_density` | alert | absolute occupancy crosses a safety threshold |
| `quiet_period` | info | sustained low activity (scene effectively idle) |
| `lighting_change` | info | day/night/overcast transition detected |
| `loiter` | warning | a single visitor lingers beyond `loiter_seconds` |

### 4.1 Debouncing `lighting_change`

`lighting_change` comes from mean frame brightness, which `classify_light()`
buckets into `night` / `dusk/dawn` / `overcast` / `bright/sunny` at fixed cut-offs
(0.18, 0.35, 0.55). Brightness jitters by a few percent on any real stream (clouds,
auto-exposure, neon at night), so when it sits near a cut-off the label flips back
and forth every window and fires an event each time. Early test runs produced
dozens of these on a single night scene.

The fix is to debounce on persistence: a label different from the current one is
kept as a candidate, and only becomes the reported state (firing one event) after
it has held for `light_min_windows` windows in a row (default 3). A one- or
two-window blip never crosses that threshold. Because the rule is about how long
the new label persists rather than the exact brightness values, it needs no
per-camera tuning. The state machine is a few lines in `EventEngine.evaluate()`
(`src/events.py`); the threshold is `events.light_min_windows` in `config.yaml`.

---

## 5. Installation

> Requires **Python 3.9-3.12**. Works on Windows, macOS and Linux.
> A GPU is **not** required (runs on CPU; a GPU just makes YOLO faster).

### 5.1 Get the code
```bash
git clone <your-repo-url>
cd "Data Extraction from Livestreams"
```

### 5.2 Create a virtual environment

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3 Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This installs OpenCV, YOLO (`ultralytics` + PyTorch), `yt-dlp`, Streamlit and
the rest. The YOLO weights file (`yolov8n.pt`, ~6 MB) downloads automatically on
the first detection run.

**Minimal install (no PyTorch).** If you can't/don't want to install PyTorch,
the app falls back to OpenCV's built-in HOG pedestrian detector. Just install
the core and run with `--backend hog`:
```bash
pip install opencv-python numpy pandas pyyaml yt-dlp
python main.py --backend hog
```

### 5.4 Verify the install (no camera or internet needed)
```bash
python tests/smoke_test.py
```
Runs the full feature -> event -> storage pipeline on synthetic data and prints a
`PASS/FAIL` checklist. Expect **OVERALL: PASS**.

> *macOS note:* on the system Python you may see a one-line
> `NotOpenSSLWarning (LibreSSL)`. It is harmless and does not appear on Windows.

---

## 6. Choosing a public livestream

Per the assignment, only use public livestreams of public places. The sources
menu lives in `config.yaml` (`sources:`); `config.py` keeps a small offline
fallback so the app still runs if the YAML is missing. The default source is a
busy Shinjuku/Kabukicho (Tokyo) night street with continuous foot traffic.

Pick a source in any of these ways:
```bash
python main.py --list-sources          # print the menu and exit
python main.py --pick                  # interactive numbered menu, then run
python main.py --source shinjuku       # by preset name
python main.py --source "https://..."  # or any raw URL / file path / webcam index
```

**Built-in presets:**

| Name | Scene | Good for |
|---|---|---|
| `shinjuku` *(default)* | Tokyo night street, dense crowd | main crowd-analytics demo |
| `kabukicho` | Tokyo night street, alt angle | crowd demo / second scene |
| `shibuya` | Shibuya Scramble Crossing | busy peaks *(has an on-screen AI overlay)* |
| `kusatsu` | onsen town, usually sparse | a calm "quiet scene" contrast |
| `nyc_skyline` | rooftop skyline, **no people** | lighting/scene only - not crowds |
| `watertown` | town square (often empty) | backup |
| `webcam` | local camera index 0 | fully offline |
| `sample` | bundled looping clip | fully offline demo / grading |

> **Live URLs rotate**, and a busy plaza can be empty at 4 a.m. local time. Before
> relying on any source, confirm it with the bundled checker - it reports
> live?/resolution/people-count and saves an annotated preview to
> `data/stream_check.jpg`:
> ```bash
> python tests/check_stream.py --source "<URL or preset name>"
> ```

### 6.1 Why YouTube live needs the Android player client

Reading a YouTube live stream through OpenCV used to hang for about 30 seconds and
then report no frames. The cause is not OpenCV: yt-dlp resolves the page fine, but
the segment URLs from the default (`web`) player client are answered by Google with
HTTP 403 on a direct fetch. FFmpeg, inside OpenCV, keeps retrying the 403'ing `.ts`
segments until its 30-second timeout fires, which looks like a freeze. Requesting
the `android` player client instead returns segment URLs that play directly. It is
a one-line option in `src/stream.py`:
```python
"extractor_args": {"youtube": {"player_client": ["android"]}}
```

**Finding your own** - reliable source types:
- **YouTube live** - search *"live cam crossing"*, *"24/7 city live"*, *"shibuya
  live"*. Copy the watch URL; `yt-dlp` + the Android-client fix above resolve it.
- **Direct HLS** (`…/playlist.m3u8`) - many municipal/tourism webcams; these need
  no `yt-dlp` and open straight in OpenCV. Browse <https://www.skylinewebcams.com>.
- **Local file** - for fully offline testing/grading, use `sample` or the generator
  (`python tests/make_sample_video.py`).

---

## 7. Running it

### Easiest: everything from the dashboard (one command)

```bash
streamlit run dashboard.py
```
The dashboard starts a capture of the default source by itself, so you don't need
a second terminal. It opens at http://localhost:8501 and shows the live stream
next to the hotspot heatmap, the KPIs, time-series and event log, all updating
live. In the sidebar you can pick a different source and Start/Stop the capture;
Auto-refresh is on by default.

> The capture runs as a background `python main.py` process that the dashboard
> launches. Press **Stop** in the sidebar (or quit the dashboard) to end it.

### Or run the capture on its own

```bash
# Live, with the annotated window (press 'q' to quit). Uses the default source.
python main.py

# Pick a source from the menu, or pass one directly
python main.py --pick
python main.py --source shinjuku
python main.py --source "<URL>" --no-window --duration 300   # headless, stop after 5 min

# Static report (summary + PNG charts in data/report/), no server
python analyze.py
```

CLI flags: `--source <name|URL|path>`, `--pick`, `--list-sources`,
`--backend {auto,yolo,hog}`, `--no-window`, `--duration <seconds>`,
`--config <path>`.

The live window (and the dashboard's "Latest analysed frame") shows detection
boxes + track IDs; the dashboard also has KPIs, a colour-coded vibe trend, the
hotspot heatmap, dwell distribution and the event log.

---

## 8. Configuration

All behaviour is driven by `config.yaml` (every key is documented inline and has
a safe default in `src/config.py`). Highlights:

- `stream.source`, `stream.process_width`, `stream.target_fps`
- `detector.backend / model / conf`
- `features.aggregation_seconds`, `dwell_exit_timeout`, `group_distance_frac`, `capacity`
- `events.*` thresholds (z-score, density, quiet, loiter, cooldown,
  `light_min_windows` - the lighting-change debounce, see §4.1)
- `sources` - the source-picker menu (see §6)
- `mqtt.enabled` (+ host/port/topic) for the IoT story
- `display.show_window`, `display.show_heatmap_overlay`

---

## 9. Big Data / IoT context (Part 5)

This program is **one camera node**. The design is deliberately the per-node
slice of a fleet-scale pipeline:

```
[ N camera nodes ]  →  edge feature extraction (this app, only ~14 numbers/window,
                       NOT raw video → cheap to transmit, privacy-friendly)
        │  MQTT topics  publicspace/metrics , publicspace/events
        ▼
[ broker / ingestion ]  (Mosquitto / Kafka)
        ▼
[ time-series DB / data lake ]  (TimescaleDB / InfluxDB / Parquet on S3)
        ▼
[ batch + stream analytics ]   trends by hour/weekday, cross-camera correlation,
                               anomaly alerting, ML on dwell/vibe
        ▼
[ dashboards / APIs ]          city operations, tourism, retail siting
```

Why this scales:
- **Edge reduction.** We transmit ~14 numeric features per 5 s window, not video
  - orders of magnitude less data, and **privacy-preserving** (no faces/images
  leave the node; only counts and aggregates).
- **Schema-first.** The fixed `metrics`/`events`/`tracks` schema drops straight
  into a time-series DB; partition by camera_id + day.
- **Event-driven.** `events` are exactly the alerting/triggering layer a stream
  processor (Kafka Streams/Flink) would consume.
- **Pluggable sinks.** CSV -> SQLite -> MQTT here; swap MQTT's broker for Kafka and
  SQLite for TimescaleDB with no change to the extraction code.

Enable the MQTT leg locally:
```bash
# start any MQTT broker, e.g. Mosquitto, then set mqtt.enabled: true in config.yaml
mosquitto_sub -t 'publicspace/#' -v        # watch metrics + events stream out live
```

---

## 10. Repository layout

```
.
├── main.py                 # live entrypoint (Parts 1-4)
├── dashboard.py            # Streamlit dashboard (Part 5)
├── analyze.py              # offline summary + charts (Part 5)
├── config.yaml             # all settings, documented inline
├── requirements.txt
├── src/
│   ├── config.py           # typed config loader (defaults + YAML overlay)
│   ├── stream.py           # threaded livestream reader + yt-dlp resolution
│   ├── detector.py         # YOLO person detection + tracking (HOG fallback)
│   ├── features.py         # crowd/dwell/heatmap/groups/light/motion/vibe
│   ├── events.py           # rolling-baseline event generation
│   ├── storage.py          # CSV + SQLite + optional MQTT
│   ├── visualize.py        # live annotated overlay
│   └── pipeline.py         # orchestration loop
├── tests/
│   ├── smoke_test.py       # end-to-end check, no model/network needed
│   └── make_sample_video.py# synthetic clip for offline runs
└── data/                   # generated CSV / SQLite / heatmap / report (gitignored)
```

---

## 11. Submission checklist (maps to the rubric)

- [x] **Code + docs (20%)** - modular `src/`, documented `config.yaml`, this README, smoke test.
- [x] **Relevant info extraction (30%)** - vibe index, dwell, groups, heatmap, light, motion.
- [x] **Event generation (20%)** - 6 event types with adaptive baselines + cooldowns.
- [x] **Data storage (15%)** - CSV + SQLite (3 tables) + optional MQTT.
- [x] **Analysis & motivation (15%)** - dashboard, report charts, §1 & §9 above.
- [ ] **Raw data** - commit a sample `data/metrics.csv` / `data/events.csv` (or a link).
- [ ] **Videos** - record the 2-min explainer + the live-demo screencast.

> **Tip for the screencast:** just run `streamlit run dashboard.py` -- it starts
> the capture itself and shows the live stream next to the heatmap with the KPIs
> filling in, so the whole demo is one screen. Use `python analyze.py` afterwards
> to produce the summary charts for the explainer.
