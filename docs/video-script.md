# Video scripts

Two submissions: a <=2 min explainer (Q&A) and a demo screencast. Record each as
one screen-recording-with-voiceover pass; export two separate files.

## Pre-flight
- Tab A: README on GitHub.
- Tab B: dashboard at http://localhost:8501 (running, with data + heatmap built up).
- For the demo, click **Clear data** first so the heatmap builds up on screen.

## Video 1 - Explainer (~1:55)

**Show README top.** "This is my project for the Data Extraction from Livestreams
assignment: Public-Space Vibe Analytics. It reads a public livestream of a busy
place - a Shinjuku street camera in Tokyo - and turns the video into useful
information about the place, not just a count of people. Here's what it extracts."

**Scroll to the section 1 signal table; point at each row name:**

- **Crowd count** - how busy it is right now: people per frame, averaged per window.
- **Vibe index** - a 0-100 liveliness score: 45% how full, 35% how much motion, 20% how social (people in groups vs alone). It captures how a place feels, not just the headcount.
- **Dwell time** - how long each individual stays, from persistent track IDs. The distribution shows whether people pass through or linger.
- **Groups vs solo** - the social structure: are people together or alone.
- **Activity / motion** - movement from frame-differencing; works even when people are far and small.
- **Light state** - average brightness as a day-night and weather proxy.
- **Hotspot heatmap** - where people actually congregate in the scene.
- **Unique visitors** - total footfall, without counting the same person twice.

**Switch to the dashboard for ~8s** - point at the live frame, vibe KPI, heatmap.
"And here it all is, live: detections, the vibe index, dwell, and the heatmap."

**Switch back to README, scroll to the Big Data context diagram.** "Why useful? A
tourism board sees how lively a square is by hour, shops know when to staff up,
security gets events on sudden crowd spikes, and planners place seating from the
heatmap and dwell. And for big data, this is one camera node: it does the vision at
the edge and emits a handful of numbers per window, not raw video - cheap and
privacy-friendly. Each node writes CSV, SQLite, and MQTT, and at city scale dozens
fan into a time-series database or data lake for fleet-wide analytics. Thanks."

## Video 2 - Demo (~2-3 min)

- **Terminal:** show `streamlit run dashboard.py`. "One command - the dashboard launches the capture itself, no second terminal."
- **Live view:** "Live analysed frame with people boxed on the left, the hotspot heatmap on the right - same size for comparison, with a link to the original feed."
- **KPI row:** "These update live: vibe index, people now, peak, unique visitors, average dwell, events."
- **Charts:** "Crowd and vibe over time, solo vs groups, activity and brightness, dwell-time distribution."
- **Switch Source in the sidebar** (e.g. `samui`): "I can switch the source and it re-captures live."
- **Event log:** "When something relevant happens - a crowd spike, a quiet period, a lighting change - it's logged with a severity."
- **Open `samples/metrics_sample.csv`:** "Everything is stored to CSV and SQLite - here's the raw extracted data."
- **Close:** "That's the full pipeline live: stream, detection, feature extraction, events, storage, dashboard."
