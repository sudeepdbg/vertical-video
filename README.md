# Reframe · AI Vertical Video Studio

Reframe converts landscape (16:9) video into vertical (9:16) video for
TikTok/Reels/Shorts. It uses YOLO-based subject detection, face detection,
optical flow, and per-mode tracking logic to decide what to keep in frame,
then smooths the resulting camera path and renders the crop with FFmpeg.

Two entry points:

- **`app.py`** — a Streamlit UI (mobile-first, light theme) for interactive use.
- **`verticalize.py`** — the processing engine. All video I/O, detection,
  tracking, smoothing, rendering, subtitle, and thumbnail logic lives here.
  It has no Streamlit dependency and can be imported/used standalone (see
  [Using the engine directly](#using-the-engine-directly)).

---

## Features

### Two workflows

| Mode | What it does |
|---|---|
| **Single Clip** | Upload one landscape video → get one 9:16 output. |
| **Auto-Clip** | Upload a long video (e.g. a 30–90 min stream/podcast) → AI scans for saliency peaks and scene arcs, proposes a set of highlight clips, then verticalizes every clip you select in one batch. |

### Tracking modes

| Mode | Best for | Notes |
|---|---|---|
| 🎯 **Subject** | General content, vlogs, product shots | YOLO person/subject tracking with optical-flow fallback and temporal-saliency fallback. Includes **Panel Mode** (see below) for multi-person layouts. |
| 👤 **Talking Head** | Single speaker to camera | OpenCV/YuNet face detection, no YOLO required. |
| 🎬 **Cinematic** | Movies, dialogue, dramas, interviews | Actor/face-priority framing with two-shot detection, shot-aware smoothing, camera-motion compensation, and headroom bias. Sports logic is fully disabled in this mode. |
| 🏀 **Sports Action** | Basketball, football, soccer, hockey | Ball-aware tracking: YOLO + ROI tracker (CSRT) + Kalman filter + HSV color-model fallback, multi-object player tracking with Hungarian assignment, play-phase detection (fast break / half-court / rebound / static), field/court color-masking. |

### Panel Mode (news panels, podcasts, interviews with 2+ people)

Auto-detects (or can be forced on) when a source has multiple people in a
stable side-by-side layout, and renders an N-person split screen instead of
tracking a single subject:

- **1–4 person layouts**: solo full-frame, 2-way split, 1+2 split, 2×2 grid
- **`split_orientation`**: `"horizontal"` (top/bottom style splits, default)
  or `"vertical"` (left/right style splits) — respected for the 2- and
  3-person layouts
- **`n_splits`**: when `split_mode="force_on"`, forces an exact slot count
  regardless of how many people detection finds in a given frame
- Speaker-focus and solo-spotlight layout weighting
- Equal head-sizing normalization across speakers
- Lower-third (text banner) awareness — crops avoid cutting into on-screen
  name banners
- Portrait/head-and-shoulders extraction mode
- Smooth layout transitions (cross-dissolve) when the person count changes,
  with holdover so a person briefly stepping out of frame doesn't trigger
  an instant re-layout

### Subtitles

- Whisper transcription (`tiny`/`base`/`small`/`medium`), burned in as
  styled subtitles (3 built-in styles: Bold White/TikTok, Yellow/Classic,
  Box/Accessible)
- Optional translation to ~25 languages via `deep-translator`
- Whisper models are cached in-process (`_get_whisper_model`) so batch runs
  don't reload the model per clip

### Thumbnails

Every rendered clip (single-clip mode, sports, cinematic, and every clip in
an Auto-Clip batch) gets a **minimum of 3 JPEG thumbnails** generated
automatically:

- The clip's usable duration (5%–95%, to skip fade-in/out) is split into
  `n` time segments so thumbnails stay spread across the whole clip.
- Within each segment, several candidate frames are sampled and scored on
  **sharpness** (Laplacian variance), **exposure sanity** (near-black/blown-out
  frames are rejected), and **saturation** — the best-scoring frame per
  segment is kept.
- This avoids landing on blurry whip-pans, cross-dissolve frames, or
  black scene-cut frames, which pure time-uniform sampling was prone to.
- Each thumbnail is individually downloadable in the UI, plus a
  "download all thumbnails (.zip)" button when there's more than one
  (single-clip mode, and per-clip in Auto-Clip mode).

See `generate_thumbnails()` / `save_clip_thumbnails()` / `_thumbnail_frame_score()`
in `verticalize.py`.

### Output & quality controls

- Resolution presets: Match source, 1080p, 720p, 540p, 480p (upscaling is
  allowed for explicit presets, never for "match source")
- Frame rate: keep source, or force 24/25/30/60 fps
- CRF quality slider (15–35) and encoder speed preset (ultrafast → slow)
- Color grading (warm/cool/vibrant/matte), vignette, sharpening
- Optional two-pass encoding path (bitrate-targeted, off by default —
  `TWO_PASS_ENCODING_ENABLED` in `verticalize.py`)
- Lower-third guard: crop never pushes the tracked subject into the bottom
  20% of frame
- Per-clip analytics: file-size reduction, camera-path smoothness %,
  input/output resolution & bitrate, CPU/RAM usage during processing
  (via `ResourceMonitor`)

---

## Requirements

```
streamlit>=1.28.0
opencv-python-headless>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
openai-whisper>=1.0.0
deep-translator>=1.11.0
imageio-ffmpeg>=0.4.9
psutil>=5.9.0
av>=12.0.0
torch>=2.0.0
scipy>=1.10.0
```

Plus **FFmpeg and ffprobe** available on `PATH` (used directly via
`subprocess`, not just through `imageio-ffmpeg`).

Everything above is optional-degrading rather than hard-required:

| Missing dependency | Effect |
|---|---|
| `ultralytics` (YOLO) | Subject/sports/panel/cinematic-person detection falls back to optical flow → temporal saliency. Talking Head mode is unaffected (uses OpenCV/YuNet, not YOLO). |
| `openai-whisper` | Subtitle burning is disabled in the UI (`whisper_available()` gates it). |
| `deep-translator` | Subtitle translation is disabled (`translation_available()` gates it). |
| `psutil` | CPU/RAM analytics report all-zero instead of erroring. |
| `scipy` | Falls back to Gaussian/EMA smoothing instead of Savitzky-Golay. |
| `torch` | Only used for CUDA device selection (`_get_device`); CPU path works without it if YOLO itself doesn't need it either. |

If your app also talks to a separate API/service layer, unrelated packages
like `fastapi`, `uvicorn`, `boto3`, `python-multipart` may belong in your
`requirements.txt` for that layer — they aren't imported anywhere in
`app.py` or `verticalize.py`.

---

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:

1. Pick **Single Clip** or **Auto-Clip**.
2. Pick a tracking mode (Subject / Talking Head / Cinematic / Sports Action).
3. Upload a landscape video (≤500 MB single-clip, ≤2000 MB auto-clip).
4. Adjust Output / Tracking / Subtitles / Advanced settings as needed.
5. **Single Clip**: click *Convert to Vertical*.
   **Auto-Clip**: click *Scan for Clips*, select the clips you want, then
   *Verticalize N Clips*.
6. Download the output video (and `.srt` if subtitles were burned), and the
   generated thumbnails render alongside the result.

---

## Using the engine directly

`verticalize.py` has no Streamlit dependency, so you can call it directly:

```python
from verticalize import process_video, process_sports_video, process_cinematic_video

meta = process_video(
    "input.mp4", "output.mp4",
    target_preset_label="720p   (720x1280  - HD)",
    tracking_mode="subject",       # "subject" | "talking_head" | "cinematic"
    burn_subtitles=True,
    whisper_model="base",
)
# meta["analytics"]   -> dict of size/smoothness/resource metrics
# meta["thumbnails"]  -> list of JPEG bytes (>= 3), already scored/selected
# meta["subtitle_path"] -> path to the .srt if one was generated (caller
#                          owns cleanup; app.py reads it, then deletes it)
```

Auto-Clip batch pipeline:

```python
from verticalize import detect_clips, process_clips_batch

clips = detect_clips("long_source.mp4", target_n_clips=8)
results = process_clips_batch("long_source.mp4", "out_dir", clips)
# each result: {"clip": ClipSegment, "output_path": str | None,
#               "analytics": {...}, "thumbnail_paths": [str, ...]}
```

`process_sports_video(...)` and `process_cinematic_video(...)` follow the
same `process_video`-shaped signature/return convention, with mode-specific
extra parameters (e.g. `sport_type`, `use_ball_tracking`, `use_kalman` for
sports; `cinematic_config` for cinematic).

---

## Architecture notes

- **Detection**: YOLO (`ultralytics`) for person/ball detection where
  available; OpenCV YuNet (with Haar cascade fallback) for faces.
- **Tracking**: mode-specific. Sports mode uses a full multi-object tracker
  (`MultiObjectSportsTracker`) with Hungarian-algorithm re-identification,
  a dedicated ball Kalman filter (`BallKalmanFilter`) with
  gravity/ground/possession modeling, an HSV ball-color appearance model as
  a last-resort fallback, and play-phase-aware camera-path smoothing.
  Cinematic mode uses face/two-shot/actor-union priority with camera-motion
  compensation. Subject mode uses YOLO → optical flow → saliency, in that
  order, with panel-mode detection layered on top.
- **Smoothing**: Savitzky-Golay (or Gaussian/EMA fallback) segmented at
  detected scene cuts, plus adaptive-window smoothing keyed to subject
  velocity (fast motion → tighter window, static shots → wider window).
- **Rendering**: raw frames are piped to FFmpeg via `subprocess` (not a
  Python video-writer library), so encoder settings (CRF, preset, color
  grade, subtitle burn-in, two-pass) are all real FFmpeg flags.
- **Resource monitoring**: `ResourceMonitor` samples `psutil` CPU/RAM for
  the current process + children during processing and reports
  avg/max CPU%, avg/max RAM, and wall-clock processing time.

---

## Changelog (recent fixes)

**v7.8** (backend, `verticalize.py`):
- Fixed a real performance regression in `generate_thumbnails`: it was
  spawning one `ffmpeg` subprocess per *candidate* frame (with defaults,
  `n=3 * candidates_per_slot=4` = 12 subprocess launches per clip — on a
  synthetic test clip this measured ~1.8s just for thumbnails, per clip).
  It now does a single sequential `FFmpegVideoReader` pass over the clip
  and scores frames as they stream by, exactly one `ffmpeg` process per
  clip regardless of `n`/`candidates_per_slot` — the same optimization
  `_detect_panel_mode` already uses for the same reason. Measured ~6x
  faster on the same test clip (1.79s → 0.28s), with thumbnail quality
  unchanged. A slower per-timestamp fallback is kept for the rare case
  where the single-pass read itself fails.

**v7.7** (backend, `verticalize.py`):
- Fixed an unbounded feedback loop in `temporal_saliency_center` — its
  frame-to-frame saliency amplification term (`sal * (1 + diff**2)`) had no
  ceiling, so on long clips it compounded until float32 overflowed to
  `inf`, then produced `NaN`. A `t < 1e-6` guard meant to catch degenerate
  cases didn't catch `NaN` (NaN comparisons are always `False`), so it fell
  through to `int(.../NaN)` and crashed with `"cannot convert float NaN to
  integer"` — surfacing both as an Auto-Clip "Scan error" and as a
  `[cinematic_tracking] Error` on longer single-clip runs. Fixed by clipping
  the amplification input, sanitizing with `np.nan_to_num`, and replacing
  the `t < 1e-6` check with an explicit `np.isfinite` check.

**v7.6** (backend, `verticalize.py`):
- Removed a duplicate `BallColorModel.reset()` definition that silently
  shadowed the real one.
- Fixed a subtitle temp-file leak in `_render_video` on `BrokenPipeError`
  (cleanup now runs in its own unconditional `finally` block).
- `_tracking_pass` now explicitly forces a scene-cut reset on a det-frame
  shape change instead of silently skipping cut detection.
- `process_clips_batch` now forwards `whisper_language` to both
  `process_video` and `process_sports_video` (previously dropped).
- `_detect_panel_mode` now logs a warning instead of silently returning
  `False` when zero probe frames could be collected.
- `ResourceMonitor` now also catches `psutil.ZombieProcess`.
- `detect_clips` SOI probing now uses the stateful `temporal_saliency_center`
  consistently with the rest of the pipeline.
- **New:** thumbnail generation (`generate_thumbnails`, `save_clip_thumbnails`,
  `_thumbnail_frame_score`) wired into every `process_*` function and
  `process_clips_batch`; quality-scored segment sampling rather than naive
  time-uniform sampling.
- `PanelModeConfig.split_orientation="vertical"` is now actually respected
  by `_render_panel_frame` (previously validated but ignored).
- `PanelModeConfig.n_splits` is now honored when `split_mode="force_on"`
  (previously validated but never consumed).
- `LayoutTransitionManager`'s default `transition_frames` now reads from
  the `PANEL_TRANSITION_FRAMES` constant instead of a separate hardcoded
  literal.
- Whisper models are now cached in-process (`_get_whisper_model`) instead
  of being reloaded from disk on every `transcribe_to_srt` call.
- Removed a dead, never-incremented `yolo_failures` variable in
  `_tracking_pass`.

**v5.2** (`app.py`): thumbnail strip displayed for single-clip output and
per-card in Auto-Clip mode, with per-thumbnail download buttons and a
"download all (.zip)" option; all `use_container_width=True` calls updated
to `width="stretch"` per Streamlit's deprecation guidance (cosmetic only —
these were warnings, not errors).

**v5.1** (`app.py` / `verticalize.py`): Cinematic Mode added.

**v5.0**: Panel Mode enhanced with N-person support, speaker focus, head
normalization, lower-third awareness, portrait extraction.

---

## Known limitations / things to be aware of

- `TWO_PASS_ENCODING_ENABLED` two-pass encoding path is implemented but not
  exposed anywhere in the Streamlit UI — it's a module-level toggle for
  programmatic use only.
- Panel Mode's 4-person layout is a fixed 2×2 grid; `split_orientation`
  doesn't change it (a symmetric grid has no meaningful "horizontal" vs.
  "vertical" alternative).
- Thumbnail scoring is intentionally lightweight (no face/YOLO detection)
  so it stays cheap to run on every clip in a large batch — it optimizes
  for sharpness/exposure/color, not "best framing of a person."
- `fastapi` / `uvicorn` / `boto3` / `python-multipart`, if present in your
  environment's `requirements.txt`, are not used by anything in this repo —
  only relevant if you're running a separate API/service layer alongside it.
