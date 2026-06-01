# Sample extracted data

Raw output from one capture of the `shinjuku` source (a busy Shinjuku/Kabukicho
night street in Tokyo), produced by `python main.py`. These are real results, not
synthetic, and are the "raw data results" deliverable for the assignment.

| File | What it is |
|---|---|
| `metrics_sample.csv` | One row per aggregation window: crowd count, vibe index, groups/solo, dwell, brightness, light state, motion. |
| `events_sample.csv` | Generated events (crowd spikes/drops, high density, quiet periods, lighting changes, loiters) with severity and message. |
| `tracks_sample.csv` | One row per finished visitor track, with enter/exit timestamps and dwell time. |
| `heatmap_sample.png` | Accumulated foot-position heatmap (where people congregated). |
| `annotated_frame_sample.jpg` | A processed frame with detected people boxed. |

Regenerate your own with `python main.py` (writes to `data/`) or inspect them
with `python analyze.py`.
