# Implementation notes

Deeper notes that don't belong in the main [README](../README.md): a couple of
real problems hit and how they were solved, plus how the live dashboard works
under the hood.

## Debouncing `lighting_change`

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

## Why YouTube live needs the Android player client

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

### Finding your own source

- **YouTube live** - search *"live cam crossing"*, *"24/7 city live"*, *"shibuya
  live"*. Copy the watch URL; `yt-dlp` plus the Android-client fix above resolve it.
- **Direct HLS** (`.../playlist.m3u8`) - many municipal/tourism webcams; these need
  no `yt-dlp` and open straight in OpenCV. Browse <https://www.skylinewebcams.com>
  or <https://webcams24.live>.
- **Local file** - for fully offline testing/grading, use `sample` or the generator
  (`python tests/make_sample_video.py`).

Verify any candidate before relying on it (it reports live?/resolution/people-count
and saves an annotated preview):

```bash
python tests/check_stream.py --source "<URL or preset name>"
```

## How the dashboard updates live (and stays smooth)

The capture and the dashboard are separate processes that talk only through the
files in `data/` - the same loose coupling a real fleet would use. The pipeline
writes a fresh annotated frame (`latest_frame.jpg`, ~3x/second) and heatmap
(`heatmap_latest.png`, ~every 2s) using a temp-file-plus-atomic-rename, so a reader
never catches a half-written image.

On the dashboard side, each panel is an `st.fragment` with its own `run_every`, so
they refresh independently and partially instead of reloading the whole page. The
fast panels are separated from the slow one: the live view (analysed frame + heatmap,
side by side at the same size) refreshes ~1.4x/second so it looks like live video,
the KPIs every 2s, and the charts on the slower "Charts refresh" interval (default
5s). A link to the original video feed sits underneath the live view.

The frame and heatmap update by swapping the image, which is smooth. The charts,
being Vega, redraw when their fragment re-runs - a Streamlit limitation - so they
live in their own slower fragment to keep the redraw infrequent (raise "Charts
refresh" for less flicker, lower it for fresher charts). Charts have explicit heights
so a redraw never shifts the page, and both live-view images are built inside a
single fragment with equal columns so they always render at the same size.
