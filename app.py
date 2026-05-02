"""
app.py — Reframe · AI Vertical Video Studio
Mobile-first · Light theme · Auto-clip detection · Talking Head Mode
"""

import streamlit as st
import tempfile
import os
from verticalize import (
    process_video, get_video_info, detect_clips, process_clips_batch,
    RESOLUTION_PRESETS, SUBTITLE_STYLES, TRANSLATION_LANGUAGES,
    resolve_target_size, whisper_available, translation_available,
    ClipSegment,
)

st.set_page_config(
    page_title="Reframe",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --bg:         #f8f7f4;
  --surface:    #ffffff;
  --surf2:      #f0ede8;
  --surf3:      #e8e4dd;
  --ink:        #18150f;
  --ink2:       #5c5449;
  --ink3:       #9c9080;
  --bdr:        #e2ddd6;
  --bdr2:       #ccc6bc;
  --accent:     #e05a1a;
  --accent-l:   #fdf1eb;
  --accent-d:   #b84511;
  --green:      #1e7a4f;
  --green-l:    #edf7f1;
  --purple:     #5b3fc7;
  --purple-l:   #f1eefb;
  --amber:      #c87800;
  --amber-l:    #fdf6e7;
  --r:          12px;
  --r-sm:       8px;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--ink) !important;
}
.stApp { background: var(--bg) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

/* ── Top bar ── */
.rf-topbar {
  height: 52px;
  background: var(--surface);
  border-bottom: 1px solid var(--bdr);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  position: sticky;
  top: 0;
  z-index: 200;
}
.rf-logo {
  display: flex; align-items: center; gap: 9px;
}
.rf-logo-mark {
  width: 28px; height: 28px; border-radius: 8px;
  background: var(--ink);
  display: flex; align-items: center; justify-content: center;
}
.rf-logo-name {
  font-family: 'DM Serif Display', serif;
  font-size: 17px;
  color: var(--ink);
  letter-spacing: -0.01em;
}
.rf-tagline {
  font-size: 11px;
  color: var(--ink3);
  letter-spacing: 0.01em;
}

/* ── Mode switcher ── */
.rf-mode-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  padding: 16px 20px 0;
}
.rf-mode-btn {
  border: 1.5px solid var(--bdr2);
  border-radius: var(--r);
  padding: 12px 14px;
  background: var(--surface);
  cursor: pointer;
  transition: all 0.15s;
  text-align: left;
}
.rf-mode-btn.active {
  border-color: var(--accent);
  background: var(--accent-l);
}
.rf-mode-btn.active-purple {
  border-color: var(--purple);
  background: var(--purple-l);
}
.rf-mode-icon { font-size: 18px; margin-bottom: 5px; }
.rf-mode-title {
  font-size: 12px; font-weight: 700; color: var(--ink); margin-bottom: 2px;
}
.rf-mode-sub { font-size: 10px; color: var(--ink3); line-height: 1.4; }

/* ── Section label ── */
.rf-sec {
  font-size: 10px; font-weight: 700;
  letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--ink3); margin-bottom: 10px;
  display: flex; align-items: center; gap: 8px;
}
.rf-sec::after {
  content: ''; flex: 1; height: 1px; background: var(--bdr);
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed var(--bdr2) !important;
  border-radius: var(--r) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background: var(--accent-l) !important;
}
[data-testid="stFileUploadDropzone"] { padding: 22px 14px !important; }
[data-testid="stFileUploadDropzone"] * {
  color: var(--ink3) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 12px !important;
}

/* ── Video player ── */
[data-testid="stVideo"] { border-radius: var(--r) !important; overflow: hidden !important; }
video { border-radius: var(--r) !important; width: 100% !important; }

/* ── Metrics strip ── */
.rf-metrics {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--bdr);
  border: 1px solid var(--bdr); border-radius: var(--r); overflow: hidden;
}
.rf-m {
  background: var(--surface); padding: 10px 12px;
}
.rf-m-l {
  font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ink3); margin-bottom: 3px;
}
.rf-m-v {
  font-family: 'DM Serif Display', serif;
  font-size: 16px; color: var(--ink); letter-spacing: -0.02em;
}
.rf-m-v.accent { color: var(--accent); }
.rf-m-v.green  { color: var(--green); }

/* ── Clip cards ── */
.rf-clip-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.rf-clip-card {
  background: var(--surface);
  border: 1.5px solid var(--bdr);
  border-radius: var(--r);
  padding: 12px;
  position: relative;
  cursor: pointer;
  transition: all 0.15s;
}
.rf-clip-card:hover {
  border-color: var(--accent);
  box-shadow: 0 2px 8px rgba(224,90,26,0.1);
}
.rf-clip-card.selected {
  border-color: var(--accent);
  background: var(--accent-l);
}
.rf-clip-card.done {
  border-color: var(--green);
  background: var(--green-l);
}
.rf-clip-score {
  position: absolute; top: 10px; right: 10px;
  background: var(--ink); color: #fff;
  font-size: 10px; font-weight: 700;
  padding: 2px 7px; border-radius: 99px;
}
.rf-clip-score.high { background: var(--accent); }
.rf-clip-score.med  { background: var(--amber); }
.rf-clip-title {
  font-size: 11px; font-weight: 700; color: var(--ink); margin-bottom: 4px; padding-right: 48px;
}
.rf-clip-meta {
  font-size: 10px; color: var(--ink3); line-height: 1.5;
}
.rf-clip-dur {
  display: inline-block; background: var(--surf2);
  border: 1px solid var(--bdr); border-radius: 4px;
  font-size: 10px; font-weight: 700; color: var(--ink2);
  padding: 1px 6px; margin-top: 5px;
}
.rf-clip-soi {
  display: inline-block; background: var(--purple-l);
  border: 1px solid #c8b8f0; border-radius: 4px;
  font-size: 10px; font-weight: 600; color: var(--purple);
  padding: 1px 6px; margin-top: 5px; margin-left: 4px;
}

/* ── Badges ── */
.rf-badge-new {
  display: inline-block; background: var(--purple); color: #fff;
  font-size: 9px; font-weight: 800; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 2px 6px; border-radius: 99px;
}
.rf-badge-ai {
  display: inline-block; background: var(--accent); color: #fff;
  font-size: 9px; font-weight: 800; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 2px 6px; border-radius: 99px;
}

/* ── Callout boxes ── */
.rf-info  { background: #eef3ff; border: 1px solid #b0bef5; border-radius: var(--r-sm);
  padding: 9px 12px; font-size: 12px; color: #2a3fa0; margin-bottom: 8px; }
.rf-warn  { background: #fff8ec; border: 1px solid #f5cc80; border-radius: var(--r-sm);
  padding: 9px 12px; font-size: 12px; color: #8a5a10; margin-bottom: 8px; }
.rf-ok    { background: var(--green-l); border: 1px solid #9fd4b8; border-radius: var(--r-sm);
  padding: 9px 12px; font-size: 12px; color: var(--green); margin-bottom: 8px;
  display: flex; align-items: center; gap: 8px; font-weight: 600; }
.rf-purple-info { background: var(--purple-l); border: 1px solid #c8b8f0;
  border-radius: var(--r-sm); padding: 9px 12px; font-size: 12px;
  color: var(--purple); margin-bottom: 8px; }

/* ── Empty state ── */
.rf-empty {
  background: var(--surface); border: 2px dashed var(--bdr2);
  border-radius: var(--r); padding: 40px 20px; text-align: center;
  display: flex; flex-direction: column; align-items: center; gap: 7px;
  min-height: 180px; justify-content: center;
}
.rf-empty-icon {
  width: 44px; height: 44px; background: var(--surf2);
  border: 1.5px solid var(--bdr); border-radius: 12px;
  font-size: 20px; display: flex; align-items: center; justify-content: center;
  margin-bottom: 3px;
}
.rf-empty-h {
  font-family: 'DM Serif Display', serif; font-size: 15px; color: var(--ink3);
}
.rf-empty-s { font-size: 11px; color: var(--ink3); opacity: 0.7; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important; font-size: 13px !important;
  transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; padding: 10px 20px !important;
}
.stButton > button[kind="primary"]:hover {
  background: #000 !important; transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:disabled {
  background: var(--bdr2) !important; color: var(--ink3) !important;
  transform: none !important;
}
.stButton > button[kind="secondary"] {
  background: var(--surface) !important; color: var(--ink2) !important;
  border: 1.5px solid var(--bdr2) !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}
.stDownloadButton > button {
  background: var(--green) !important; color: #fff !important;
  border: none !important; border-radius: var(--r-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 13px !important;
  padding: 10px 18px !important; width: 100% !important;
  transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
  background: #165c3a !important; transform: translateY(-1px) !important;
}

/* ── Progress ── */
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px; }
.stProgress > div > div { background: var(--bdr) !important; border-radius: 99px; height: 3px !important; }
.stProgress > div { height: 3px !important; }

/* ── Form controls ── */
[data-baseweb="select"] > div {
  background: var(--surface) !important; border-color: var(--bdr2) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
}
[data-baseweb="select"] * { color: var(--ink) !important; }
[data-baseweb="popover"], [data-baseweb="menu"] {
  background: var(--surface) !important; border: 1px solid var(--bdr) !important;
  border-radius: 10px !important;
}
[data-baseweb="option"] { background: var(--surface) !important; color: var(--ink2) !important; font-size: 13px !important; }
[data-baseweb="option"]:hover { background: var(--accent-l) !important; color: var(--accent) !important; }

[data-baseweb="tab-list"] {
  background: var(--surf2) !important; border-radius: var(--r-sm) !important;
  padding: 3px !important; gap: 2px !important; border: none !important;
}
[data-baseweb="tab"] {
  background: transparent !important; border-radius: 6px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 12px !important;
  font-weight: 600 !important; color: var(--ink3) !important;
  padding: 6px 11px !important; border: none !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: var(--surface) !important; color: var(--ink) !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

.stSlider label { font-size: 12px !important; color: var(--ink2) !important; font-weight: 600 !important; }
.stSlider [role="slider"] { background: var(--accent) !important; border: 2px solid #fff !important; }
.stSlider [data-testid="stSliderTrackFill"] { background: var(--accent) !important; }
.stSlider > div > div { background: var(--bdr) !important; }
[data-testid="stSliderValue"] { color: var(--accent) !important; font-size: 11px !important; font-weight: 700 !important; }

[data-testid="stToggleSwitch"] > div { background: var(--bdr2) !important; }
[data-testid="stToggleSwitch"][aria-checked="true"] > div { background: var(--accent) !important; }

/* ── Footer ── */
.rf-footer {
  margin-top: 32px; padding: 12px 20px;
  border-top: 1px solid var(--bdr);
  display: flex; align-items: center; justify-content: space-between;
}
.rf-tech { display: flex; gap: 5px; flex-wrap: wrap; }
.rf-tech span {
  font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 3px 7px;
  border: 1px solid var(--bdr); border-radius: 4px; color: var(--ink3);
}

/* ── Chip ── */
.rf-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surf2); border: 1px solid var(--bdr);
  border-radius: 6px; padding: 4px 9px;
  font-size: 11px; color: var(--ink2); max-width: 100%;
}
.rf-chip strong { color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px; }

/* ── Two-panel layout ── */
.rf-panel { padding: 14px 20px; }
.rf-panel-r { padding: 14px 20px 14px 10px; }

/* Responsive: stack on mobile */
@media (max-width: 768px) {
  .rf-mode-row { grid-template-columns: 1fr 1fr; gap: 6px; padding: 12px 14px 0; }
  .rf-clip-grid { grid-template-columns: 1fr; }
  .rf-metrics   { grid-template-columns: repeat(2, 1fr); }
  .rf-panel, .rf-panel-r { padding: 12px 14px; }
}

.stCaption, small { color: var(--ink3) !important; font-size: 10px !important; }
[data-testid="stHorizontalBlock"] { gap: 10px !important; }

/* Clip select checkboxes (hidden, we use card UI) */
.rf-cb-row { display: flex; align-items: center; gap: 8px; margin-top: 5px; }
.rf-check {
  width: 16px; height: 16px; border-radius: 4px;
  border: 1.5px solid var(--bdr2); background: var(--surface);
  display: inline-flex; align-items: center; justify-content: center;
  flex-shrink: 0; cursor: pointer;
}
.rf-check.checked {
  background: var(--accent); border-color: var(--accent);
  color: #fff; font-size: 10px;
}

/* Lower-third safe zone badge */
.rf-safe-badge {
  display: inline-flex; align-items: center; gap: 4px;
  background: var(--green-l); border: 1px solid #9fd4b8;
  border-radius: 4px; padding: 2px 7px;
  font-size: 10px; font-weight: 600; color: var(--green);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        # Shared
        input_path=None, uploaded_file_name=None,
        video_info=None,
        # App mode: "single" | "autoClip"
        app_mode="single",
        tracking_mode="subject",
        # Single clip
        output_path=None,
        processing_done=False, output_bytes=None,
        srt_bytes=None, last_settings=None,
        # Auto-clip
        detected_clips=None,          # List[ClipSegment]
        selected_clip_indices=None,   # set of int
        clip_results=None,            # List[Dict]
        detecting_clips=False,
        processing_clips=False,
        scan_done=False,
        current_clip_idx=0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _cleanup():
    p = st.session_state.get("input_path")
    if p and os.path.exists(p):
        try: os.unlink(p)
        except OSError: pass
    op = st.session_state.get("output_path")
    if op and os.path.exists(op):
        try: os.unlink(op)
        except OSError: pass
    st.session_state.update(
        input_path=None, output_path=None,
        output_bytes=None, srt_bytes=None,
        video_info=None, processing_done=False,
        detected_clips=None, selected_clip_indices=None,
        clip_results=None, scan_done=False,
    )

def _new_out():
    fd, p = tempfile.mkstemp(suffix=".mp4")
    os.close(fd); os.unlink(p)
    st.session_state.output_path = p

def _invalidate_if_changed(cur):
    if (st.session_state.processing_done
            and st.session_state.last_settings != cur):
        st.session_state.processing_done = False
        st.session_state.output_bytes    = None
        st.session_state.srt_bytes       = None

_init()
_whisper_ok   = whisper_available()
_translate_ok = translation_available()


# ─────────────────────────────────────────────────────────────────────────────
#  Top bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-topbar">
  <div class="rf-logo">
    <div class="rf-logo-mark">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <rect x="1" y="1" width="5" height="12" rx="1.5" fill="white"/>
        <rect x="8" y="4" width="5" height="9" rx="1.5" fill="white" opacity="0.5"/>
      </svg>
    </div>
    <span class="rf-logo-name">Reframe</span>
  </div>
  <span class="rf-tagline">AI Vertical Video</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  App mode selector
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div style='padding:0 20px'>", unsafe_allow_html=True)
st.markdown('<div class="rf-sec">Mode</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

mc1, mc2 = st.columns(2, gap="small")
with mc1:
    if st.button("📱  Single Clip", type="secondary", use_container_width=True,
                 help="Convert one full video to 9:16 vertical"):
        st.session_state.app_mode = "single"
with mc2:
    if st.button("🎬  Auto-Clip  ✦", type="secondary", use_container_width=True,
                 help="Scan long video, detect high-engagement segments, verticalize each"):
        st.session_state.app_mode = "autoClip"

app_mode = st.session_state.app_mode

# Mode description
st.markdown("<div style='padding:0 20px;margin-top:8px;'>", unsafe_allow_html=True)
if app_mode == "single":
    st.markdown("""
    <div style='background:var(--surf2);border:1.5px solid var(--bdr);
        border-radius:var(--r);padding:10px 14px;display:flex;gap:10px;align-items:flex-start;'>
      <span style='font-size:16px'>📱</span>
      <div>
        <div style='font-size:12px;font-weight:700;color:var(--ink);margin-bottom:2px;'>Single Clip Mode</div>
        <div style='font-size:11px;color:var(--ink2);line-height:1.5;'>
          Upload any landscape video. AI tracks your subject and crops to 9:16 in one pass.
          Supports YOLOv8 subject tracking and face-locked Talking Head mode.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background:var(--purple-l);border:1.5px solid var(--purple);
        border-radius:var(--r);padding:10px 14px;display:flex;gap:10px;align-items:flex-start;'>
      <span style='font-size:16px'>🎬</span>
      <div>
        <div style='font-size:12px;font-weight:700;color:var(--purple);margin-bottom:2px;'>
          Auto-Clip Mode <span class="rf-badge-ai">AI</span>
        </div>
        <div style='font-size:11px;color:var(--ink2);line-height:1.5;'>
          Upload a 30–90 min video. AI scans for saliency peaks, detects narrative arcs
          (beginning · middle · end), identifies SOI coordinates per clip, and
          enforces the lower-third safe zone — then verticalizes each selected clip.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Settings tabs (shared config)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='padding:0 20px'>", unsafe_allow_html=True)
st.markdown('<div class="rf-sec">Tracking Mode</div>', unsafe_allow_html=True)
tm1, tm2 = st.columns(2, gap="small")
with tm1:
    if st.button("🎯  Subject Tracking", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "subject"
with tm2:
    if st.button("👤  Talking Head  ✦", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "talking_head"
tracking_mode = st.session_state.tracking_mode
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Settings accordion
with st.container():
    st.markdown("<div style='padding:0 20px'>", unsafe_allow_html=True)
    if app_mode == "single":
        tab_out, tab_trk, tab_sub, tab_adv = st.tabs(
            ["🎞 Output", "🎯 Tracking", "📝 Subtitles", "⚙ Advanced"])
    else:
        tab_out, tab_trk, tab_sub, tab_adv, tab_clip = st.tabs(
            ["🎞 Output", "🎯 Tracking", "📝 Subtitles", "⚙ Advanced", "✂️ Clips"])

    with tab_out:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        o1, o2 = st.columns(2, gap="medium")
        with o1:
            resolution_label = st.selectbox("Resolution", list(RESOLUTION_PRESETS.keys()), index=0)
            fps_label = st.selectbox(
                "Frame rate",
                ["Source (keep original)", "60 fps", "30 fps", "25 fps", "24 fps"], index=0)
        with o2:
            crf = st.slider("Quality (CRF)", 15, 35, 23, 1)
            st.caption("18 = near-lossless  ·  28 = compact")
            encoder_preset_label = st.selectbox("Speed", ["ultrafast", "fast", "medium", "slow"], index=1)
        _fps_map = {"Source (keep original)": None, "60 fps": 60.0,
                    "30 fps": 30.0, "25 fps": 25.0, "24 fps": 24.0}
        output_fps = _fps_map[fps_label]

    with tab_trk:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if tracking_mode == "talking_head":
            th1, th2 = st.columns(2, gap="medium")
            with th1:
                talking_head_bias = st.slider("Upper-third pull", 0.0, 1.0, 0.30, 0.05)
                st.caption("0 = center  ·  1 = upper third")
                smooth_window = st.slider("Smoothness", 3, 31, 21, 2)
            with th2:
                adaptive_smoothing = st.toggle("Adaptive smoothing", value=False)
                use_optical_flow   = st.toggle("Optical flow bridge", value=True)
                rule_of_thirds     = st.toggle("Horizontal rule-of-thirds", value=True)
            confidence = 0.5; scene_cut_threshold = 0.35
        else:
            t1, t2 = st.columns(2, gap="medium")
            with t1:
                adaptive_smoothing = st.toggle("Adaptive smoothing", value=True)
                smooth_window      = st.slider("Smoothness", 3, 31, 15, 2)
                confidence         = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
            with t2:
                use_optical_flow   = st.toggle("Optical flow fallback", value=True)
                rule_of_thirds     = st.toggle("Look-room / Rule-of-thirds", value=True)
                scene_cut_threshold = st.slider("Scene-cut sensitivity", 0.10, 0.60, 0.35, 0.05)
            talking_head_bias = 0.30

    with tab_sub:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if not _whisper_ok:
            st.markdown('<div class="rf-purple-info">⚠️ Install <code>openai-whisper</code> to enable subtitles.</div>',
                        unsafe_allow_html=True)
        s1, s2 = st.columns(2, gap="medium")
        with s1:
            burn_subtitles = st.toggle("Burn subtitles", value=False, disabled=not _whisper_ok)
            if not _whisper_ok: burn_subtitles = False
            translate_subtitles = st.toggle("Translate 🌐", value=False,
                disabled=(not _whisper_ok or not _translate_ok or not burn_subtitles))
            if not burn_subtitles or not _translate_ok: translate_subtitles = False
            whisper_model    = st.selectbox("Whisper model", ["tiny","base","small","medium"], index=1,
                                            disabled=not _whisper_ok)
        with s2:
            subtitle_style_name = st.selectbox("Style", list(SUBTITLE_STYLES.keys()),
                                               disabled=not _whisper_ok)
            whisper_language = st.selectbox("Audio language",
                ["Auto-detect","en","hi","es","fr","de","ja","zh","pt","ar"],
                disabled=not _whisper_ok)
            if whisper_language == "Auto-detect": whisper_language = None
            subtitle_max_chars = st.slider("Max chars/line", 20, 60, 42, 2, disabled=not _whisper_ok)
            subtitle_translate_to_label = st.selectbox("Translate to",
                list(TRANSLATION_LANGUAGES.keys()), index=0,
                disabled=(not _whisper_ok or not _translate_ok or not burn_subtitles or not translate_subtitles))
            subtitle_translate_to = TRANSLATION_LANGUAGES[subtitle_translate_to_label] or None
            if not translate_subtitles: subtitle_translate_to = None

    with tab_adv:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        a1, a2 = st.columns(2, gap="medium")
        with a1:
            audio_bitrate_label = st.selectbox("Audio bitrate", ["64k","96k","128k","192k"], index=2)
        with a2:
            yolo_weights = st.selectbox("YOLO model",
                ["yolov8n.pt","yolov8s.pt","yolov8m.pt"], index=0,
                disabled=(tracking_mode=="talking_head")) \
                if tracking_mode == "subject" else "yolov8n.pt"
        st.markdown("""
        <div class="rf-safe-badge">✓ Lower-third guard ON — subjects kept above bottom 20%</div>
        """, unsafe_allow_html=True)

    if app_mode == "autoClip":
        with tab_clip:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            cl1, cl2 = st.columns(2, gap="medium")
            with cl1:
                clip_min_dur = st.slider("Min clip duration (s)", 15, 45, 25, 5)
                clip_max_dur = st.slider("Max clip duration (s)", 30, 90, 60, 5)
            with cl2:
                clip_target_n = st.slider("Target # clips", 3, 20, 8, 1)
                st.markdown("""
                <div class="rf-info" style="margin-top:8px;">
                💡 AI detects saliency peaks, scene boundaries, and narrative arcs
                (beginning · middle · end) to select the most engaging segments.
                </div>""", unsafe_allow_html=True)
    else:
        clip_min_dur = 25; clip_max_dur = 60; clip_target_n = 8

    st.markdown("</div>", unsafe_allow_html=True)


current_settings = dict(
    app_mode=app_mode, tracking_mode=tracking_mode,
    resolution_label=resolution_label, fps_label=fps_label, crf=crf,
    encoder_preset_label=encoder_preset_label,
    smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
    confidence=confidence, use_optical_flow=use_optical_flow,
    rule_of_thirds=rule_of_thirds, scene_cut_threshold=scene_cut_threshold,
    talking_head_bias=talking_head_bias,
    burn_subtitles=burn_subtitles,
    whisper_model=whisper_model if burn_subtitles else "",
    audio_bitrate_label=audio_bitrate_label,
)
_invalidate_if_changed(current_settings)

st.markdown("<div style='height:2px;background:var(--bdr);margin:10px 0 0'></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  File upload (shared)
# ─────────────────────────────────────────────────────────────────────────────
col_src, col_out = st.columns([1, 1], gap="small")

with col_src:
    st.markdown("<div class='rf-panel'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec">Source Video</div>', unsafe_allow_html=True)

    max_mb = 2000 if app_mode == "autoClip" else 500
    uploaded_file = st.file_uploader(
        "Drop video",
        type=["mp4","mov","avi","mkv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        mb = len(uploaded_file.getvalue()) / (1024**2)
        if mb > max_mb:
            st.markdown(f'<div class="rf-warn">⚠ {mb:.1f} MB — max {max_mb} MB.</div>',
                        unsafe_allow_html=True)
            uploaded_file = None

    if (uploaded_file is not None
            and st.session_state.uploaded_file_name != uploaded_file.name):
        _cleanup()
        st.session_state.processing_done = False
        st.session_state.last_settings   = None
        st.session_state.scan_done       = False
        st.session_state.detected_clips  = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getvalue())
            st.session_state.input_path = tmp.name
        _new_out()
        st.session_state.uploaded_file_name = uploaded_file.name
        try:
            st.session_state.video_info = get_video_info(st.session_state.input_path)
        except Exception:
            st.session_state.video_info = None

    if uploaded_file is not None and st.session_state.input_path:
        info = st.session_state.video_info
        if info and not info["is_landscape"]:
            st.markdown('<div class="rf-warn">⚠ Video is already vertical.</div>',
                        unsafe_allow_html=True)
        mb_str = f"{len(uploaded_file.getvalue()) / (1024**2):.1f} MB"
        st.markdown(
            f'<div class="rf-chip"><span>🎬</span>'
            f'<strong>{uploaded_file.name}</strong>'
            f'<span style="color:var(--bdr2)">·</span>'
            f'<span>{mb_str}</span></div>',
            unsafe_allow_html=True)
        st.video(uploaded_file)

        # Metrics strip
        if info:
            dur = info["duration_seconds"]
            mins, secs = int(dur//60), int(dur%60)
            dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
            st.markdown(f"""
            <div class='rf-metrics' style='margin-top:10px;'>
              <div class='rf-m'><div class='rf-m-l'>Duration</div><div class='rf-m-v'>{dur_str}</div></div>
              <div class='rf-m'><div class='rf-m-l'>Source</div><div class='rf-m-v'>{info['width']}×{info['height']}</div></div>
              <div class='rf-m'><div class='rf-m-l'>Output</div><div class='rf-m-v accent'>{eff_w}×{eff_h}</div></div>
              <div class='rf-m'><div class='rf-m-l'>FPS</div><div class='rf-m-v'>{info['fps']:.0f}</div></div>
            </div>""", unsafe_allow_html=True)

            if app_mode == "autoClip" and dur < 60:
                st.markdown(
                    '<div class="rf-warn" style="margin-top:8px;">⚠ Auto-Clip works best on videos ≥ 2 minutes.</div>',
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Right column — output / clips
# ─────────────────────────────────────────────────────────────────────────────
with col_out:
    st.markdown("<div class='rf-panel-r'>", unsafe_allow_html=True)

    # ────────────── SINGLE CLIP OUTPUT ──────────────
    if app_mode == "single":
        st.markdown('<div class="rf-sec">Output · 9:16</div>', unsafe_allow_html=True)

        if st.session_state.processing_done and st.session_state.output_bytes:
            info = st.session_state.video_info
            out_mb = len(st.session_state.output_bytes) / (1024**2)
            st.markdown(
                f'<div class="rf-ok">✓ Done — {out_mb:.1f} MB</div>',
                unsafe_allow_html=True)
            st.video(st.session_state.output_bytes, format="video/mp4")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            stem = os.path.splitext(st.session_state.uploaded_file_name or "video")[0]
            st.download_button("↓  Download vertical video",
                data=st.session_state.output_bytes,
                file_name=f"{stem}_vertical.mp4",
                mime="video/mp4", use_container_width=True)
            if st.session_state.srt_bytes:
                st.download_button("↓  Download subtitles (.srt)",
                    data=st.session_state.srt_bytes,
                    file_name=f"{stem}.srt",
                    mime="text/plain", use_container_width=True)
        else:
            st.markdown("""
            <div class="rf-empty">
              <div class="rf-empty-icon">📱</div>
              <div class="rf-empty-h">Vertical output</div>
              <div class="rf-empty-s">appears here after conversion</div>
            </div>""", unsafe_allow_html=True)

    # ────────────── AUTO-CLIP PANEL ──────────────
    else:
        st.markdown('<div class="rf-sec">Detected Clips</div>', unsafe_allow_html=True)

        if not st.session_state.scan_done:
            st.markdown("""
            <div class="rf-empty">
              <div class="rf-empty-icon">🔍</div>
              <div class="rf-empty-h">Scan first</div>
              <div class="rf-empty-s">AI will detect high-engagement segments</div>
            </div>""", unsafe_allow_html=True)

        elif st.session_state.detected_clips:
            clips: list = st.session_state.detected_clips
            if st.session_state.selected_clip_indices is None:
                st.session_state.selected_clip_indices = set(range(len(clips)))

            sel = st.session_state.selected_clip_indices

            st.markdown(f"<div style='font-size:11px;color:var(--ink3);margin-bottom:8px;'>"
                        f"{len(clips)} clips found · {len(sel)} selected</div>",
                        unsafe_allow_html=True)

            # Clip cards
            for ci, clip in enumerate(clips):
                score_pct = int(clip.score * 100)
                score_cls = "high" if clip.score > 0.7 else ("med" if clip.score > 0.4 else "")
                is_sel    = ci in sel
                done_meta = None
                if st.session_state.clip_results:
                    done_meta = next(
                        (r for r in st.session_state.clip_results
                         if getattr(r.get("clip"), "start_sec", None) == clip.start_sec),
                        None)
                is_done = done_meta and done_meta.get("output_path") and \
                          os.path.exists(done_meta.get("output_path",""))

                card_cls = "rf-clip-card" + (" done" if is_done else (" selected" if is_sel else ""))
                dur_str  = f"{clip.duration:.0f}s"
                mins_s   = int(clip.start_sec // 60)
                secs_s   = int(clip.start_sec % 60)
                mins_e   = int(clip.end_sec // 60)
                secs_e   = int(clip.end_sec % 60)
                time_str = f"{mins_s}:{secs_s:02d} → {mins_e}:{secs_e:02d}"

                st.markdown(f"""
                <div class="{card_cls}" style="margin-bottom:8px;">
                  <span class="rf-clip-score {score_cls}">{score_pct}%</span>
                  <div class="rf-clip-title">Clip {ci+1}</div>
                  <div class="rf-clip-meta">{time_str}</div>
                  <span class="rf-clip-dur">{dur_str}</span>
                  <span class="rf-clip-soi">SOI: {clip.soi_region}</span>
                  {"<div style='margin-top:5px;font-size:10px;color:var(--green);font-weight:700;'>✓ Converted</div>" if is_done else ""}
                </div>""", unsafe_allow_html=True)

                # Toggle select
                col_tog, col_dl = st.columns([2, 1])
                with col_tog:
                    if is_done:
                        pass
                    else:
                        toggled = st.checkbox(
                            "Include" if not is_sel else "✓ Selected",
                            value=is_sel, key=f"clip_sel_{ci}")
                        if toggled != is_sel:
                            if toggled:
                                st.session_state.selected_clip_indices.add(ci)
                            else:
                                st.session_state.selected_clip_indices.discard(ci)
                            st.rerun()

                with col_dl:
                    if is_done and done_meta.get("output_path"):
                        try:
                            with open(done_meta["output_path"], "rb") as f:
                                clip_bytes = f.read()
                            st.download_button(
                                "↓ Download",
                                data=clip_bytes,
                                file_name=f"clip_{ci+1}_vertical.mp4",
                                mime="video/mp4",
                                key=f"dl_{ci}",
                                use_container_width=True)
                        except Exception:
                            pass
        else:
            st.markdown("""
            <div class="rf-empty">
              <div class="rf-empty-icon">🔍</div>
              <div class="rf-empty-h">No clips detected</div>
              <div class="rf-empty-s">try adjusting clip duration settings</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Action bar
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None and st.session_state.input_path:
    info = st.session_state.video_info
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:var(--bdr);margin:0 20px'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    can_go = bool(info and info.get("is_landscape", True))

    # ────────────── SINGLE CLIP ACTIONS ──────────────
    if app_mode == "single":
        if not st.session_state.processing_done:
            a1, a2, a3 = st.columns([4, 5, 2])
            with a1:
                go = st.button("▶  Convert to Vertical",
                               type="primary", use_container_width=True,
                               disabled=not can_go)
            with a2:
                if info:
                    eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
                    mode_t = "Talking Head" if tracking_mode == "talking_head" else "Subject"
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;margin-top:12px;'>"
                        f"{mode_t} · {eff_w}×{eff_h} · CRF {crf}</p>",
                        unsafe_allow_html=True)
            with a3:
                if st.button("Clear", type="secondary", use_container_width=True):
                    _cleanup(); st.session_state.uploaded_file_name = None; st.rerun()

            if go:
                st.session_state.last_settings = current_settings
                prog   = st.progress(0.0)
                status = st.empty()
                status.info("⚡ Starting…")
                try:
                    def _cb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(msg)

                    meta = process_video(
                        st.session_state.input_path,
                        st.session_state.output_path,
                        target_preset_label=resolution_label,
                        tracking_mode=tracking_mode,
                        talking_head_bias=talking_head_bias,
                        confidence=confidence,
                        smooth_window=smooth_window,
                        adaptive_smoothing=adaptive_smoothing,
                        use_optical_flow=use_optical_flow,
                        rule_of_thirds=rule_of_thirds,
                        scene_cut_threshold=scene_cut_threshold,
                        output_fps=output_fps,
                        crf=crf,
                        encoder_preset=encoder_preset_label,
                        audio_bitrate=audio_bitrate_label,
                        yolo_weights=yolo_weights,
                        burn_subtitles=burn_subtitles,
                        whisper_model=whisper_model,
                        whisper_language=whisper_language,
                        subtitle_style_name=subtitle_style_name,
                        subtitle_max_chars=subtitle_max_chars,
                        subtitle_translate_to=subtitle_translate_to,
                        progress_callback=_cb,
                    )
                    prog.progress(1.0)
                    out = st.session_state.output_path
                    if os.path.exists(out) and os.path.getsize(out) > 0:
                        with open(out, "rb") as f:
                            st.session_state.output_bytes = f.read()
                        srt_path = meta.get("subtitle_path")
                        if srt_path and os.path.exists(srt_path):
                            with open(srt_path, "rb") as f:
                                st.session_state.srt_bytes = f.read()
                            try: os.unlink(srt_path)
                            except OSError: pass
                        st.session_state.processing_done = True
                        status.success("✅ Done!")
                        st.rerun()
                    else:
                        status.error("Output is empty — something went wrong.")
                except Exception as exc:
                    status.error(f"Error: {exc}")

        else:
            r1, _, r2 = st.columns([2, 5, 2])
            with r1:
                if st.button("← Start over", type="secondary", use_container_width=True):
                    _cleanup(); st.session_state.uploaded_file_name = None
                    st.session_state.processing_done = False; st.rerun()
            with r2:
                if info and st.session_state.output_bytes:
                    in_mb  = len(uploaded_file.getvalue()) / (1024**2)
                    out_mb = len(st.session_state.output_bytes) / (1024**2)
                    delta  = out_mb - in_mb
                    dcol   = "var(--green)" if delta < 0 else "var(--accent)"
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;text-align:right;margin-top:12px;'>"
                        f"{out_mb:.1f} MB "
                        f"<span style='color:{dcol}'>({delta:+.1f} MB)</span></p>",
                        unsafe_allow_html=True)

    # ────────────── AUTO-CLIP ACTIONS ──────────────
    else:
        if not st.session_state.scan_done:
            # Scan button
            b1, b2, b3 = st.columns([4, 4, 2])
            with b1:
                scan_btn = st.button("🔍  Scan for Clips",
                                     type="primary", use_container_width=True,
                                     disabled=not can_go)
            with b2:
                if info:
                    dur = info["duration_seconds"]
                    est = max(10, dur * 0.05)
                    est_str = f"~{int(est//60)}m {int(est%60):02d}s" if est >= 60 else f"~{int(est)}s"
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;margin-top:12px;'>"
                        f"Scan est. {est_str}</p>", unsafe_allow_html=True)
            with b3:
                if st.button("Clear", type="secondary", use_container_width=True):
                    _cleanup(); st.session_state.uploaded_file_name = None; st.rerun()

            if scan_btn:
                prog   = st.progress(0.0)
                status = st.empty()
                status.info("🔍 Scanning video for high-engagement segments…")
                try:
                    def _scan_cb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(msg)

                    model_for_scan = None
                    clips = detect_clips(
                        st.session_state.input_path,
                        min_duration_sec=float(clip_min_dur),
                        max_duration_sec=float(clip_max_dur),
                        target_n_clips=clip_target_n,
                        model=model_for_scan,
                        confidence=confidence,
                        progress_callback=_scan_cb,
                    )
                    prog.progress(1.0)
                    st.session_state.detected_clips = clips
                    st.session_state.selected_clip_indices = set(range(len(clips)))
                    st.session_state.scan_done = True
                    status.success(f"✅ Found {len(clips)} clips!")
                    st.rerun()
                except Exception as exc:
                    status.error(f"Scan error: {exc}")

        else:
            # Scan done — show process/rescan buttons
            clips = st.session_state.detected_clips or []
            sel   = st.session_state.selected_clip_indices or set()

            if not st.session_state.clip_results:
                p1, p2, p3 = st.columns([4, 3, 2])
                with p1:
                    process_btn = st.button(
                        f"▶  Verticalize {len(sel)} Clip{'s' if len(sel)!=1 else ''}",
                        type="primary", use_container_width=True,
                        disabled=len(sel) == 0)
                with p2:
                    if st.button("🔄 Re-scan", type="secondary", use_container_width=True):
                        st.session_state.scan_done = False
                        st.session_state.detected_clips = None
                        st.rerun()
                with p3:
                    if st.button("Clear", type="secondary", use_container_width=True):
                        _cleanup(); st.session_state.uploaded_file_name = None; st.rerun()

                if process_btn and sel:
                    selected_clips = [clips[i] for i in sorted(sel)]

                    prog   = st.progress(0.0)
                    status = st.empty()
                    status.info(f"⚡ Processing {len(selected_clips)} clips…")

                    out_dir = tempfile.mkdtemp()

                    def _batch_cb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(msg)

                    try:
                        results = process_clips_batch(
                            input_path=st.session_state.input_path,
                            output_dir=out_dir,
                            clips=selected_clips,
                            target_preset_label=resolution_label,
                            tracking_mode=tracking_mode,
                            talking_head_bias=talking_head_bias,
                            confidence=confidence,
                            smooth_window=smooth_window,
                            adaptive_smoothing=adaptive_smoothing,
                            rule_of_thirds=rule_of_thirds,
                            crf=crf,
                            encoder_preset=encoder_preset_label,
                            audio_bitrate=audio_bitrate_label,
                            yolo_weights=yolo_weights,
                            burn_subtitles=burn_subtitles,
                            whisper_model=whisper_model,
                            subtitle_style_name=subtitle_style_name,
                            subtitle_max_chars=subtitle_max_chars,
                            progress_callback=_batch_cb,
                        )
                        prog.progress(1.0)
                        st.session_state.clip_results = results
                        n_ok = sum(1 for r in results if not r.get("error") and r.get("output_path"))
                        status.success(f"✅ {n_ok}/{len(results)} clips converted!")
                        st.rerun()
                    except Exception as exc:
                        status.error(f"Error: {exc}")

            else:
                # Results ready
                results = st.session_state.clip_results or []
                n_ok = sum(1 for r in results if not r.get("error"))
                st.markdown(
                    f'<div class="rf-ok">✓ {n_ok} clip{"s" if n_ok!=1 else ""} ready to download</div>',
                    unsafe_allow_html=True)

                r1, r2, r3 = st.columns([3, 3, 2])
                with r1:
                    if st.button("← New scan", type="secondary", use_container_width=True):
                        st.session_state.scan_done = False
                        st.session_state.detected_clips = None
                        st.session_state.clip_results = None
                        st.rerun()
                with r2:
                    if st.button("← New video", type="secondary", use_container_width=True):
                        _cleanup()
                        st.session_state.uploaded_file_name = None
                        st.rerun()

else:
    # Welcome state
    st.markdown("""
    <div style='padding:0 20px 44px;margin-top:16px;'>
      <div style='background:var(--surface);border:2px dashed var(--bdr);
          border-radius:var(--r);padding:48px 28px;text-align:center;'>
        <div style='font-family:"DM Serif Display",serif;font-size:clamp(1.6rem,4vw,2.4rem);
            font-weight:400;color:var(--bdr2);letter-spacing:-0.03em;
            margin-bottom:10px;line-height:1.1;'>
          Drop a video to begin.
        </div>
        <p style='font-size:12px;color:var(--ink3);margin-bottom:16px;'>
          Landscape MP4, MOV, AVI or MKV
        </p>
        <div style='display:flex;gap:6px;justify-content:center;flex-wrap:wrap;'>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--ink3);padding:4px 9px;border:1px solid var(--bdr);border-radius:4px;'>MP4</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--ink3);padding:4px 9px;border:1px solid var(--bdr);border-radius:4px;'>MOV</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--ink3);padding:4px 9px;border:1px solid var(--bdr);border-radius:4px;'>AVI</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--ink3);padding:4px 9px;border:1px solid var(--bdr);border-radius:4px;'>MKV</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-footer">
  <div class="rf-tech">
    <span>YOLOv8</span><span>OpenCV</span>
    <span>Whisper</span><span>FFmpeg</span>
  </div>
  <div style='font-size:10px;color:var(--bdr2);'>Reframe · AI Vertical Video</div>
</div>
""", unsafe_allow_html=True)
