"""
app.py  —  Reframe · AI Video Converter
Warm light theme · tight two-column layout · full video config.
"""

import streamlit as st
import tempfile
import os
from verticalize import (
    process_video, get_video_info, RESOLUTION_PRESETS, resolve_target_size
)

st.set_page_config(
    page_title="Reframe · AI Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Design tokens + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&display=swap');

:root {
  --bg:         #f5f2ed;
  --surface:    #ffffff;
  --surface2:   #eeebe4;
  --border:     #ddd8cf;
  --border2:    #ccc5b9;
  --ink:        #1c1814;
  --ink2:       #5a5048;
  --ink3:       #968b7f;
  --accent:     #c94f14;
  --accent-bg:  #fdf0e9;
  --accent-dk:  #a53e0e;
  --green:      #1a7a50;
  --green-bg:   #edf7f2;
  --red:        #b82a2a;
  --radius:     10px;
  --shadow-sm:  0 1px 2px rgba(28,24,20,.05), 0 2px 8px rgba(28,24,20,.05);
  --shadow:     0 1px 3px rgba(28,24,20,.07), 0 4px 16px rgba(28,24,20,.08);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
}
.stApp { background: var(--bg) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

/* ── Nav ─────────────────────────────────────────────────── */
.rf-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; height: 56px;
    background: rgba(245,242,237,0.94);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
}
.rf-wordmark {
    font-family: 'Fraunces', serif; font-size: 19px; font-weight: 700;
    color: var(--ink); letter-spacing: -0.02em;
    display: flex; align-items: center; gap: 8px;
}
.rf-nav-pills { display: flex; gap: 5px; }
.rf-nav-pill {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 4px 11px; border-radius: 99px;
    background: var(--surface2); color: var(--ink3); border: 1px solid var(--border);
}

/* ── Hero (compact) ──────────────────────────────────────── */
.rf-hero {
    padding: 36px 40px 28px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    justify-content: space-between; gap: 24px; flex-wrap: wrap;
}
.rf-kicker {
    font-size: 10px; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 10px;
}
.rf-headline {
    font-family: 'Fraunces', serif;
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    font-weight: 700; line-height: 1.08;
    letter-spacing: -0.03em; color: var(--ink); margin-bottom: 8px;
}
.rf-headline em { font-style: italic; color: var(--accent); }
.rf-desc { font-size: 13px; color: var(--ink2); line-height: 1.6; max-width: 400px; }
.rf-hero-stats { display: flex; gap: 28px; }
.rf-stat { text-align: right; }
.rf-stat-val {
    font-family: 'Fraunces', serif; font-size: 1.6rem;
    font-weight: 700; color: var(--ink); letter-spacing: -0.03em; line-height: 1;
}
.rf-stat-lbl { font-size: 10px; color: var(--ink3); margin-top: 2px; }

/* ── Settings tabs ──────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: var(--surface2) !important; border-radius: 8px !important;
    padding: 3px !important; gap: 2px !important; border: none !important;
}
[data-baseweb="tab"] {
    background: transparent !important; border-radius: 6px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 12px !important; font-weight: 600 !important;
    color: var(--ink3) !important; padding: 7px 14px !important;
    border: none !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--surface) !important; color: var(--ink) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] { display: none !important; }

/* ── Upload zone ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-bg) !important;
}
[data-testid="stFileUploadDropzone"] { padding: 32px 20px !important; }
[data-testid="stFileUploadDropzone"] * {
    color: var(--ink3) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stFileUploadDropzone"] svg { color: var(--border2) !important; }

/* ── Video containers: tight, no extra padding ──────────── */
[data-testid="stVideo"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    line-height: 0 !important;
}
video {
    border-radius: var(--radius) !important;
    width: 100% !important;
    display: block !important;
    margin: 0 !important;
    background: #111;
}

/* ── Section label ──────────────────────────────────────── */
.rf-sec-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink3);
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}
.rf-sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── File chip ──────────────────────────────────────────── */
.rf-file-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 10px; font-size: 11px;
    color: var(--ink2); margin-bottom: 8px; max-width: 100%;
}
.rf-file-chip strong { color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Metrics grid ───────────────────────────────────────── */
.rf-metrics {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden;
}
.rf-met { background: var(--surface); padding: 11px 13px; }
.rf-met-lbl {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ink3); margin-bottom: 3px;
}
.rf-met-val {
    font-family: 'Fraunces', serif; font-size: 16px;
    font-weight: 700; color: var(--ink); letter-spacing: -0.02em;
}
.rf-met-val.a { color: var(--accent); }

/* ── Config summary ─────────────────────────────────────── */
.rf-cfg {
    display: grid; grid-template-columns: repeat(6,1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden; margin-top: 10px;
}
.rf-cfg-cell { background: var(--surface); padding: 9px 11px; }
.rf-cfg-lbl { font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink3); margin-bottom: 2px; }
.rf-cfg-val { font-size: 12px; font-weight: 700; color: var(--ink); }
.rf-cfg-val.a { color: var(--accent); }

/* ── Banners ────────────────────────────────────────────── */
.rf-warn {
    background: #fff8f0; border: 1px solid #f0c08a;
    border-radius: 7px; padding: 9px 12px; font-size: 12px;
    color: #7a4a10; margin-bottom: 10px;
}
.rf-success-banner {
    background: var(--green-bg); border: 1px solid #9fd4b8;
    border-radius: 8px; padding: 10px 13px;
    display: flex; align-items: center; gap: 9px; margin-bottom: 10px;
}
.rf-success-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.rf-success-text { font-size: 12px; color: var(--green); font-weight: 700; }

/* ── Empty state ────────────────────────────────────────── */
.rf-empty {
    background: var(--surface2); border: 2px dashed var(--border);
    border-radius: var(--radius); padding: 48px 24px;
    text-align: center; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
}
.rf-empty-icon {
    width: 44px; height: 44px; border-radius: 12px;
    background: var(--surface); border: 1px solid var(--border);
    font-size: 20px; display: flex; align-items: center; justify-content: center;
    margin-bottom: 4px; box-shadow: var(--shadow-sm);
}
.rf-empty-h { font-family: 'Fraunces', serif; font-size: 15px; font-weight: 600; color: var(--ink3); }
.rf-empty-s { font-size: 11px; color: var(--border2); }

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 13px !important; transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; padding: 11px 24px !important;
    box-shadow: 0 2px 6px rgba(201,79,20,.22) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-dk) !important;
    box-shadow: 0 4px 16px rgba(201,79,20,.32) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--border2) !important; color: var(--ink3) !important;
    box-shadow: none !important; transform: none !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important; color: var(--ink2) !important;
    border: 1.5px solid var(--border2) !important; padding: 9px 16px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
}
.stDownloadButton > button {
    background: var(--green) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 13px !important;
    padding: 11px 24px !important; width: 100% !important;
    box-shadow: 0 2px 6px rgba(26,122,80,.18) !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: #155f3e !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(26,122,80,.28) !important;
}

/* ── Progress ───────────────────────────────────────────── */
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px; }
.stProgress > div > div { background: var(--border) !important; border-radius: 99px; height: 3px !important; }
.stProgress > div { height: 3px !important; }

/* ── Select ─────────────────────────────────────────────── */
[data-baseweb="select"] > div {
    background: var(--surface) !important; border-color: var(--border2) !important;
    border-radius: 7px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; color: var(--ink) !important;
}
[data-baseweb="select"] * { color: var(--ink) !important; }
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 9px !important;
}
[data-baseweb="option"] { background: var(--surface) !important; color: var(--ink2) !important; font-size: 13px !important; }
[data-baseweb="option"]:hover { background: var(--accent-bg) !important; color: var(--accent) !important; }

/* ── Slider ─────────────────────────────────────────────── */
.stSlider label {
    font-size: 12px !important; color: var(--ink2) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important; border: 2px solid #fff !important;
    box-shadow: 0 1px 4px rgba(201,79,20,.28) !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--border) !important; }
[data-testid="stSliderValue"] { color: var(--accent) !important; font-size: 11px !important; font-weight: 700 !important; }

/* ── Toggle ─────────────────────────────────────────────── */
[data-testid="stToggleSwitch"] > div { background: var(--border2) !important; }
[data-testid="stToggleSwitch"][aria-checked="true"] > div { background: var(--accent) !important; }
[data-testid="stToggleSwitch"] span { color: var(--ink2) !important; font-size: 12px !important; }

/* ── Alerts ─────────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; }
[data-baseweb="notification"] { border-radius: 8px !important; }
.stCaption, small { color: var(--ink3) !important; font-size: 10px !important; }

/* ── Footer ─────────────────────────────────────────────── */
.rf-footer {
    margin-top: 32px; padding: 16px 40px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
}
.rf-tech { display: flex; gap: 5px; }
.rf-tech span {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 3px 9px;
    border: 1px solid var(--border); border-radius: 4px; color: var(--ink3);
}
.rf-footer-copy { font-size: 10px; color: var(--border2); }

/* ── Column gap override ─────────────────────────────────── */
[data-testid="stHorizontalBlock"] { gap: 16px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    for k, v in dict(
        input_path=None, output_path=None,
        uploaded_file_name=None,
        processing_done=False, output_bytes=None,
        video_info=None, last_settings=None,
    ).items():
        if k not in st.session_state:
            st.session_state[k] = v

def _cleanup():
    for key in ("input_path", "output_path"):
        p = st.session_state.get(key)
        if p and os.path.exists(p):
            try: os.unlink(p)
            except OSError: pass
        st.session_state[key] = None
    st.session_state.output_bytes = None
    st.session_state.video_info   = None

def _new_out():
    fd, p = tempfile.mkstemp(suffix=".mp4")
    os.close(fd); os.unlink(p)
    st.session_state.output_path = p

def _invalidate_if_changed(cur):
    if st.session_state.processing_done and st.session_state.last_settings != cur:
        st.session_state.processing_done = False
        st.session_state.output_bytes    = None

_init()


# ─────────────────────────────────────────────────────────────────────────────
#  Nav
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-nav">
  <div class="rf-wordmark">
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <rect width="20" height="20" rx="6" fill="#c94f14"/>
      <rect x="4" y="4" width="5" height="12" rx="2" fill="white"/>
      <rect x="11" y="7" width="5" height="9" rx="2" fill="white" opacity="0.55"/>
    </svg>
    Reframe
  </div>
  <div class="rf-nav-pills">
    <span class="rf-nav-pill">YOLOv8</span>
    <span class="rf-nav-pill">OpenCV</span>
    <span class="rf-nav-pill">FFmpeg</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Hero (compact)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-hero">
  <div>
    <div class="rf-kicker">AI-Powered Vertical Video</div>
    <h1 class="rf-headline">Landscape to vertical,<br><em>automatically.</em></h1>
    <p class="rf-desc">AI tracks subjects, computes adaptive crop paths, and exports
    vertical video ready for TikTok, Reels &amp; Shorts.</p>
  </div>
  <div class="rf-hero-stats">
    <div class="rf-stat"><div class="rf-stat-val">9:16</div><div class="rf-stat-lbl">Output ratio</div></div>
    <div class="rf-stat"><div class="rf-stat-val">YOLOv8</div><div class="rf-stat-lbl">AI model</div></div>
    <div class="rf-stat"><div class="rf-stat-val">FFmpeg</div><div class="rf-stat-lbl">Pipe encode</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Settings tabs
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div style='padding:12px 40px 0'>", unsafe_allow_html=True)
    tab_out, tab_track, tab_adv = st.tabs(["🎞 Output & Quality", "🎯 Tracking", "⚙ Advanced"])

    with tab_out:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            resolution_label = st.selectbox(
                "Output resolution",
                list(RESOLUTION_PRESETS.keys()), index=0,
                help="'Match source' avoids upscaling a 480p video to 1080p.",
            )
        with c2:
            fps_label = st.selectbox(
                "Output frame rate",
                ["Source (keep original)", "60 fps", "30 fps", "25 fps", "24 fps"],
                index=0,
            )
            _fps_map = {"Source (keep original)": None, "60 fps": 60.0,
                        "30 fps": 30.0, "25 fps": 25.0, "24 fps": 24.0}
            output_fps = _fps_map[fps_label]
        with c3:
            crf = st.slider("Quality (CRF)", 15, 35, 23, 1,
                help="Lower = better quality & larger file. 18–23 = excellent.")
            st.caption("18 = near-lossless  ·  28 = compact")
        with c4:
            encoder_preset_label = st.selectbox(
                "Encode speed", ["ultrafast", "fast", "medium", "slow"], index=1,
            )

    with tab_track:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4, gap="medium")
        with t1:
            adaptive_smoothing = st.toggle("Adaptive smoothing", value=True,
                help="Automatically reduces smoothing window when subjects move fast")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            smooth_window = st.slider("Base smoothness", 3, 31, 15, 2,
                help="Used as base when adaptive is on; fixed window when off")
            st.caption("Higher → steadier  ·  Lower → snappier")
        with t2:
            confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05,
                help="Lower = detects more; higher = stricter")
            st.caption("Lower → sensitive  ·  Higher → strict")
        with t3:
            use_optical_flow = st.toggle("Optical flow fallback", value=True,
                help="Tracks motion between frames when YOLO finds nothing")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            rule_of_thirds = st.toggle("Look-room / Rule-of-thirds", value=True,
                help="Shifts crop ahead of moving subjects (look-room), or toward composition lines when still")
        with t4:
            scene_cut_threshold = st.slider("Scene-cut sensitivity", 0.10, 0.60, 0.35, 0.05,
                help="Diff % to detect a hard cut (resets tracking & smoothing)")
            st.caption("Higher → hard cuts only  ·  Lower → more cuts")

    with tab_adv:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3, gap="medium")
        with a1:
            audio_bitrate_label = st.selectbox(
                "Audio bitrate", ["64k", "96k", "128k", "192k", "256k"], index=2,
            )
        with a2:
            yolo_weights = st.selectbox(
                "YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0,
                help="nano = fastest; medium = best accuracy",
            )
        with a3:
            st.markdown("""
            <div style='background:var(--surface2);border:1px solid var(--border);
                border-radius:7px;padding:10px 12px;font-size:11px;color:var(--ink3);margin-top:22px;'>
              <strong style='color:var(--ink2);display:block;margin-bottom:4px'>Model sizes</strong>
              nano: ~6 MB · fast, good<br>small: ~22 MB · balanced<br>medium: ~50 MB · best
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Settings fingerprint for invalidation
current_settings = dict(
    resolution_label=resolution_label, fps_label=fps_label,
    crf=crf, encoder_preset_label=encoder_preset_label,
    smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
    confidence=confidence, use_optical_flow=use_optical_flow,
    rule_of_thirds=rule_of_thirds,
    scene_cut_threshold=scene_cut_threshold,
    audio_bitrate_label=audio_bitrate_label, yolo_weights=yolo_weights,
)
_invalidate_if_changed(current_settings)

st.markdown("<div style='height:1px;background:var(--border);margin:8px 0 0'></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-column work area — tight, no wasted vertical space
# ─────────────────────────────────────────────────────────────────────────────
col_src, col_out = st.columns([11, 7], gap="small")

# ── Source ──────────────────────────────────────────────────────────────────
with col_src:
    st.markdown("<div style='padding:20px 40px 0 40px'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec-label">Source · Landscape</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop video here",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        mb = len(uploaded_file.getvalue()) / (1024 ** 2)
        if mb > 500:
            st.markdown(f'<div class="rf-warn">⚠ {mb:.1f} MB — max 500 MB.</div>',
                        unsafe_allow_html=True)
            uploaded_file = None

    if (uploaded_file is not None and
            st.session_state.uploaded_file_name != uploaded_file.name):
        _cleanup()
        st.session_state.processing_done = False
        st.session_state.last_settings   = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.getvalue())
            st.session_state.input_path = tmp_in.name
        _new_out()
        st.session_state.uploaded_file_name = uploaded_file.name
        try:
            st.session_state.video_info = get_video_info(st.session_state.input_path)
        except Exception:
            st.session_state.video_info = None

    if uploaded_file is not None and st.session_state.input_path:
        info = st.session_state.video_info

        if info and not info["is_landscape"]:
            st.markdown('<div class="rf-warn">⚠ Video is already vertical — upload a landscape video.</div>',
                        unsafe_allow_html=True)

        mb_str = f"{len(uploaded_file.getvalue()) / (1024**2):.1f} MB"
        st.markdown(
            f'<div class="rf-file-chip"><span>🎬</span>'
            f'<strong>{uploaded_file.name}</strong>'
            f'<span style="color:var(--border2)">·</span>'
            f'<span>{mb_str}</span></div>',
            unsafe_allow_html=True
        )
        st.video(uploaded_file)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Output ──────────────────────────────────────────────────────────────────
with col_out:
    st.markdown("<div style='padding:20px 40px 0 8px'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec-label">Output · Vertical</div>', unsafe_allow_html=True)

    if st.session_state.processing_done and st.session_state.output_bytes:
        info = st.session_state.video_info
        out_mb = len(st.session_state.output_bytes) / (1024 ** 2)

        # Resolve what actual output size was used (for display)
        if info:
            eff_tw, eff_th = resolve_target_size(
                resolution_label, info["width"], info["height"], RESOLUTION_PRESETS
            ) if resolution_label in RESOLUTION_PRESETS else (
                RESOLUTION_PRESETS.get(resolution_label, (1080, 1920))
            )
        else:
            eff_tw, eff_th = 1080, 1920

        st.markdown(
            f'<div class="rf-success-banner"><div class="rf-success-dot"></div>'
            f'<div class="rf-success-text">Done — {eff_tw}×{eff_th} · {out_mb:.1f} MB</div></div>',
            unsafe_allow_html=True
        )
        st.video(st.session_state.output_bytes, format="video/mp4")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        stem = os.path.splitext(st.session_state.uploaded_file_name or "video")[0]
        st.download_button(
            label="↓  Download vertical video",
            data=st.session_state.output_bytes,
            file_name=f"{stem}_vertical.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.markdown("""
        <div class="rf-empty">
          <div class="rf-empty-icon">📱</div>
          <div class="rf-empty-h">Vertical output</div>
          <div class="rf-empty-s">appears here after conversion</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics + Config summary + Action bar
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None and st.session_state.input_path:
    info = st.session_state.video_info
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:var(--border);margin:0 40px'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if info:
        dur = info["duration_seconds"]
        mins, secs = int(dur // 60), int(dur % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        est_sec = max(10, dur * 0.6 + 8)
        est_min = est_sec / 60
        est_str = (f"~{int(est_min)}m {int(est_sec%60):02d}s"
                   if est_min >= 1 else f"~{int(est_sec)}s")
        fps_display = fps_label if fps_label != "Source (keep original)" else f"{info['fps']:.0f} fps"

        # Compute effective output resolution
        eff_tw, eff_th = resolve_target_size(
            resolution_label, info["width"], info["height"], RESOLUTION_PRESETS
        )
        was_clamped = (eff_tw != RESOLUTION_PRESETS.get(resolution_label, (0,0))[0]
                       and resolution_label != "Match source (recommended)")

        st.markdown(f"""
        <div style='padding:0 40px;margin-bottom:10px;'>
        <div class='rf-metrics'>
          <div class='rf-met'><div class='rf-met-lbl'>Duration</div>
            <div class='rf-met-val'>{dur_str}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Source</div>
            <div class='rf-met-val'>{info['width']}×{info['height']}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>FPS</div>
            <div class='rf-met-val'>{fps_display}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Output</div>
            <div class='rf-met-val a'>{eff_tw}×{eff_th}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Est. time</div>
            <div class='rf-met-val'>{est_str}</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        if was_clamped:
            st.markdown(
                f'<div style="padding:0 40px;margin-bottom:6px;">'
                f'<div class="rf-warn">ℹ Output clamped to {eff_tw}×{eff_th} — '
                f'source is only {info["width"]}×{info["height"]} '
                f'(upscaling disabled).</div></div>',
                unsafe_allow_html=True
            )

        crf_lbl = "Near-lossless" if crf <= 18 else ("Balanced" if crf <= 24 else "Compact")
        smooth_lbl = f"Adaptive ({smooth_window} base)" if adaptive_smoothing else str(smooth_window)
        st.markdown(f"""
        <div style='padding:0 40px;margin-bottom:16px;'>
        <div class='rf-cfg'>
          <div class='rf-cfg-cell'><div class='rf-cfg-lbl'>CRF {crf}</div>
            <div class='rf-cfg-val a'>{crf_lbl}</div></div>
          <div class='rf-cfg-cell'><div class='rf-cfg-lbl'>Encode</div>
            <div class='rf-cfg-val'>{encoder_preset_label}</div></div>
          <div class='rf-cfg-cell'><div class='rf-cfg-lbl'>Smoothing</div>
            <div class='rf-cfg-val'>{smooth_lbl}</div></div>
          <div class='rf-cfg-cell'><div class='rf-cfg-lbl'>Confidence</div>
            <div class='rf-cfg-val'>{confidence:.2f}</div></div>
          <div class='rf-cfg-cell'><div class='rf-cfg-lbl'>Audio</div>
            <div class='rf-cfg-val'>{audio_bitrate_label}</div></div>
          <div class='rf-cfg-cell'><div class='rf-cfg-lbl'>Model</div>
            <div class='rf-cfg-val'>{yolo_weights.replace(".pt","")}</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Action row
    _, action_col, __ = st.columns([0.05, 10, 0.05])
    with action_col:
        if not st.session_state.processing_done:
            btn_c, gap_c, clr_c = st.columns([3, 6, 1.5])
            can_go = bool(info and info.get("is_landscape", True))
            with btn_c:
                go = st.button("▶  Convert to vertical",
                               type="primary", use_container_width=True,
                               disabled=not can_go)
            with gap_c:
                if info:
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;margin-top:13px;'>"
                        f"Output {eff_tw}×{eff_th} · CRF {crf} · "
                        f"{'Adaptive smooth' if adaptive_smoothing else f'Smooth {smooth_window}'} · "
                        f"Audio {audio_bitrate_label}</p>",
                        unsafe_allow_html=True,
                    )
            with clr_c:
                if st.button("Clear", type="secondary", use_container_width=True):
                    _cleanup()
                    st.session_state.uploaded_file_name = None
                    st.rerun()

            if go:
                st.session_state.last_settings = current_settings
                prog = st.progress(0.0)
                status = st.empty()
                status.info("⚡ Starting…")
                try:
                    def _cb(v: float, msg: str = ""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(msg)

                    process_video(
                        st.session_state.input_path,
                        st.session_state.output_path,
                        target_preset_label=resolution_label,
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
                        progress_callback=_cb,
                    )
                    prog.progress(1.0)
                    out = st.session_state.output_path
                    if os.path.exists(out) and os.path.getsize(out) > 0:
                        with open(out, "rb") as f:
                            st.session_state.output_bytes = f.read()
                        st.session_state.processing_done = True
                        status.success("Done! Download below.")
                        st.rerun()
                    else:
                        status.error("Output is empty — something went wrong.")
                except Exception as exc:
                    status.error(f"Error: {exc}")

        else:
            rst_c, _, sz_c = st.columns([2, 5, 2])
            with rst_c:
                if st.button("← Start over", type="secondary", use_container_width=True):
                    _cleanup()
                    st.session_state.uploaded_file_name = None
                    st.session_state.processing_done    = False
                    st.rerun()
            with sz_c:
                if info and st.session_state.output_bytes:
                    in_mb     = len(uploaded_file.getvalue()) / (1024**2)
                    out_mb    = len(st.session_state.output_bytes) / (1024**2)
                    delta     = out_mb - in_mb
                    delta_col = "var(--green)" if delta < 0 else "var(--red)"
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;"
                        f"text-align:right;margin-top:13px;'>"
                        f"Output {out_mb:.1f} MB "
                        f"<span style='color:{delta_col}'>({delta:+.1f} MB)</span></p>",
                        unsafe_allow_html=True,
                    )

# ─────────────────────────────────────────────────────────────────────────────
#  Empty state
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='padding:0 40px 48px;margin-top:20px;'>
      <div style='background:var(--surface);border:2px dashed var(--border);
          border-radius:14px;padding:64px 40px;text-align:center;'>
        <div style='font-family:Fraunces,serif;font-size:2.6rem;font-weight:700;
            color:var(--border);letter-spacing:-0.04em;margin-bottom:10px;line-height:1.05;'>
          Drop a video<br>to begin.
        </div>
        <p style='font-size:13px;color:var(--ink3);margin-bottom:18px;'>
          Landscape MP4, MOV, AVI, or MKV · up to 500 MB
        </p>
        <div style='display:flex;gap:6px;justify-content:center;flex-wrap:wrap;'>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
              color:var(--ink3);padding:4px 10px;border:1px solid var(--border);border-radius:4px;'>MP4</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
              color:var(--ink3);padding:4px 10px;border:1px solid var(--border);border-radius:4px;'>MOV</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
              color:var(--ink3);padding:4px 10px;border:1px solid var(--border);border-radius:4px;'>AVI</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
              color:var(--ink3);padding:4px 10px;border:1px solid var(--border);border-radius:4px;'>MKV</span>
          <span style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
              color:var(--ink3);padding:4px 10px;border:1px solid var(--border);border-radius:4px;'>max 500 MB</span>
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
    <span>YOLOv8</span><span>OpenCV</span><span>FFmpeg pipe</span><span>Streamlit</span>
  </div>
  <div class="rf-footer-copy">Reframe · AI Video Converter</div>
</div>
""", unsafe_allow_html=True)
