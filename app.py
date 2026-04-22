"""
app.py  —  Reframe · AI Video Converter
Warm light theme · balanced two-column layout · full video config.
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&display=swap');

:root {
  --bg:        #f5f2ed;
  --surface:   #ffffff;
  --surface2:  #eeebe4;
  --border:    #ddd8cf;
  --border2:   #ccc5b9;
  --ink:       #1c1814;
  --ink2:      #5a5048;
  --ink3:      #968b7f;
  --accent:    #c94f14;
  --accent-bg: #fdf0e9;
  --accent-dk: #a53e0e;
  --green:     #1a7a50;
  --green-bg:  #edf7f2;
  --red:       #b82a2a;
  --r:         10px;
  --sh:        0 1px 3px rgba(28,24,20,.07), 0 3px 12px rgba(28,24,20,.07);
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

/* Nav */
.rf-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; height: 54px;
    background: rgba(245,242,237,0.94); backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
}
.rf-wordmark {
    font-family: 'Fraunces', serif; font-size: 18px; font-weight: 700;
    color: var(--ink); letter-spacing: -0.02em;
    display: flex; align-items: center; gap: 8px;
}
.rf-nav-pills { display: flex; gap: 5px; }
.rf-nav-pill {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 99px;
    background: var(--surface2); color: var(--ink3); border: 1px solid var(--border);
}

/* Hero */
.rf-hero {
    padding: 30px 40px 22px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    justify-content: space-between; gap: 20px; flex-wrap: wrap;
}
.rf-kicker {
    font-size: 10px; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 8px;
}
.rf-headline {
    font-family: 'Fraunces', serif;
    font-size: clamp(1.6rem, 2.4vw, 2.2rem);
    font-weight: 700; line-height: 1.1;
    letter-spacing: -0.03em; color: var(--ink); margin-bottom: 6px;
}
.rf-headline em { font-style: italic; color: var(--accent); }
.rf-desc { font-size: 12px; color: var(--ink2); line-height: 1.6; max-width: 380px; }
.rf-hero-stats { display: flex; gap: 24px; }
.rf-stat { text-align: right; }
.rf-stat-val {
    font-family: 'Fraunces', serif; font-size: 1.4rem;
    font-weight: 700; color: var(--ink); letter-spacing: -0.02em; line-height: 1;
}
.rf-stat-lbl { font-size: 10px; color: var(--ink3); margin-top: 2px; }

/* Tabs */
[data-baseweb="tab-list"] {
    background: var(--surface2) !important; border-radius: 7px !important;
    padding: 3px !important; gap: 2px !important; border: none !important;
}
[data-baseweb="tab"] {
    background: transparent !important; border-radius: 5px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 12px !important; font-weight: 600 !important;
    color: var(--ink3) !important; padding: 6px 12px !important; border: none !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--surface) !important; color: var(--ink) !important;
    box-shadow: 0 1px 3px rgba(28,24,20,.08) !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* Upload */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: var(--r) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-bg) !important;
}
[data-testid="stFileUploadDropzone"] { padding: 28px 16px !important; }
[data-testid="stFileUploadDropzone"] * {
    color: var(--ink3) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 12px !important;
}
[data-testid="stFileUploadDropzone"] svg { color: var(--border2) !important; }

/* Video — no gap, natural aspect */
[data-testid="stVideo"] {
    border-radius: var(--r) !important;
    overflow: hidden !important;
    display: block !important;
    line-height: 0 !important;
}
video {
    border-radius: var(--r) !important;
    width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 0 !important;
    background: #0a0a0a;
}

/* Section label */
.rf-sec-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink3);
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}
.rf-sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* File chip */
.rf-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 9px; font-size: 11px;
    color: var(--ink2); margin-bottom: 8px; max-width: 100%;
}
.rf-chip strong { color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 240px; }

/* Metrics */
.rf-metrics {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--r); overflow: hidden;
}
.rf-met { background: var(--surface); padding: 10px 12px; }
.rf-met-lbl { font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink3); margin-bottom: 3px; }
.rf-met-val { font-family: 'Fraunces', serif; font-size: 15px; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
.rf-met-val.a { color: var(--accent); }

/* Config summary */
.rf-cfg {
    display: grid; grid-template-columns: repeat(6,1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-top: 8px;
}
.rf-cfg-cell { background: var(--surface); padding: 8px 10px; }
.rf-cfg-lbl { font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink3); margin-bottom: 2px; }
.rf-cfg-val { font-size: 12px; font-weight: 700; color: var(--ink); }
.rf-cfg-val.a { color: var(--accent); }

/* Alerts */
.rf-warn {
    background: #fff8f0; border: 1px solid #f0c08a; border-radius: 7px;
    padding: 8px 11px; font-size: 12px; color: #7a4a10; margin-bottom: 8px;
}
.rf-info {
    background: #f0f5ff; border: 1px solid #b0c4f0; border-radius: 7px;
    padding: 8px 11px; font-size: 12px; color: #2040a0; margin-bottom: 8px;
}
.rf-success {
    background: var(--green-bg); border: 1px solid #9fd4b8;
    border-radius: 8px; padding: 9px 12px;
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
}
.rf-success-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.rf-success-text { font-size: 12px; color: var(--green); font-weight: 700; }

/* Empty state */
.rf-empty {
    background: var(--surface2); border: 2px dashed var(--border);
    border-radius: var(--r); padding: 40px 20px;
    text-align: center; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 7px;
    min-height: 200px;
}
.rf-empty-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: var(--surface); border: 1px solid var(--border);
    font-size: 18px; display: flex; align-items: center; justify-content: center;
    margin-bottom: 3px; box-shadow: var(--sh);
}
.rf-empty-h { font-family: 'Fraunces', serif; font-size: 14px; font-weight: 600; color: var(--ink3); }
.rf-empty-s { font-size: 11px; color: var(--border2); }

/* Buttons */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 13px !important; transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; padding: 10px 22px !important;
    box-shadow: 0 2px 6px rgba(201,79,20,.22) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-dk) !important;
    box-shadow: 0 4px 14px rgba(201,79,20,.32) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--border2) !important; color: var(--ink3) !important;
    box-shadow: none !important; transform: none !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important; color: var(--ink2) !important;
    border: 1.5px solid var(--border2) !important; padding: 8px 14px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
}
.stDownloadButton > button {
    background: var(--green) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 13px !important;
    padding: 10px 22px !important; width: 100% !important;
    box-shadow: 0 2px 6px rgba(26,122,80,.18) !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: #155f3e !important; transform: translateY(-1px) !important;
}

/* Progress */
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px; }
.stProgress > div > div { background: var(--border) !important; border-radius: 99px; height: 3px !important; }
.stProgress > div { height: 3px !important; }

/* Select */
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

/* Slider */
.stSlider label { font-size: 12px !important; color: var(--ink2) !important; font-weight: 600 !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important; border: 2px solid #fff !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--border) !important; }
[data-testid="stSliderValue"] { color: var(--accent) !important; font-size: 11px !important; font-weight: 700 !important; }

/* Toggle */
[data-testid="stToggleSwitch"] > div { background: var(--border2) !important; }
[data-testid="stToggleSwitch"][aria-checked="true"] > div { background: var(--accent) !important; }
[data-testid="stToggleSwitch"] span { color: var(--ink2) !important; font-size: 12px !important; }

/* Alerts */
.stAlert { border-radius: 8px !important; }
.stCaption, small { color: var(--ink3) !important; font-size: 10px !important; }

/* Footer */
.rf-footer {
    margin-top: 28px; padding: 14px 40px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
}
.rf-tech { display: flex; gap: 5px; }
.rf-tech span {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 3px 8px;
    border: 1px solid var(--border); border-radius: 4px; color: var(--ink3);
}
.rf-footer-copy { font-size: 10px; color: var(--border2); }

/* Column gap */
[data-testid="stHorizontalBlock"] { gap: 12px !important; }
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
    if (st.session_state.processing_done
            and st.session_state.last_settings != cur):
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
#  Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-hero">
  <div>
    <div class="rf-kicker">AI-Powered Vertical Video</div>
    <h1 class="rf-headline">Landscape to vertical,<br><em>automatically.</em></h1>
    <p class="rf-desc">AI tracks subjects, computes adaptive crop paths,
    and exports vertical video ready for TikTok, Reels &amp; Shorts.</p>
  </div>
  <div class="rf-hero-stats">
    <div class="rf-stat"><div class="rf-stat-val">9:16</div><div class="rf-stat-lbl">Output ratio</div></div>
    <div class="rf-stat"><div class="rf-stat-val">YOLOv8</div><div class="rf-stat-lbl">AI detection</div></div>
    <div class="rf-stat"><div class="rf-stat-val">FFmpeg</div><div class="rf-stat-lbl">Encode</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Settings tabs
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div style='padding:10px 40px 0'>", unsafe_allow_html=True)
    tab_out, tab_track, tab_adv = st.tabs(
        ["🎞 Output & Quality", "🎯 Tracking", "⚙ Advanced"]
    )

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
                help="18–23 = excellent. 28+ = smaller file.")
            st.caption("18 = near-lossless  ·  28 = compact")
        with c4:
            encoder_preset_label = st.selectbox(
                "Encode speed",
                ["ultrafast", "fast", "medium", "slow"], index=1,
            )

    with tab_track:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4, gap="medium")
        with t1:
            adaptive_smoothing = st.toggle("Adaptive smoothing", value=True,
                help="Auto-reduces window when subjects move fast")
            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
            smooth_window = st.slider("Base smoothness", 3, 31, 15, 2)
            st.caption("Higher → steady  ·  Lower → snappy")
        with t2:
            confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
            st.caption("Lower → sensitive  ·  Higher → strict")
        with t3:
            use_optical_flow = st.toggle("Optical flow fallback", value=True)
            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
            rule_of_thirds = st.toggle("Look-room / Rule-of-thirds", value=True,
                help="Shifts crop ahead of moving subjects; rule-of-thirds when still")
        with t4:
            scene_cut_threshold = st.slider("Scene-cut sensitivity", 0.10, 0.60, 0.35, 0.05)
            st.caption("Higher → hard cuts  ·  Lower → more cuts")

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
                border-radius:7px;padding:9px 11px;font-size:11px;color:var(--ink3);margin-top:20px;'>
              <strong style='color:var(--ink2);display:block;margin-bottom:3px'>Model sizes</strong>
              nano: ~6 MB · fast, good<br>small: ~22 MB · balanced<br>medium: ~50 MB · best
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

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

st.markdown("<div style='height:1px;background:var(--border);margin:6px 0 0'></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-column work area — EQUAL columns so videos match height
# ─────────────────────────────────────────────────────────────────────────────
col_src, col_out = st.columns(2, gap="small")

# ── Source ──────────────────────────────────────────────────────────────────
with col_src:
    st.markdown("<div style='padding:16px 40px 0 40px'>", unsafe_allow_html=True)
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

    if (uploaded_file is not None
            and st.session_state.uploaded_file_name != uploaded_file.name):
        _cleanup()
        st.session_state.processing_done = False
        st.session_state.last_settings   = None
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
            st.markdown(
                '<div class="rf-warn">⚠ Video is already vertical — upload a landscape video.</div>',
                unsafe_allow_html=True
            )

        mb_str = f"{len(uploaded_file.getvalue()) / (1024**2):.1f} MB"
        st.markdown(
            f'<div class="rf-chip"><span>🎬</span>'
            f'<strong>{uploaded_file.name}</strong>'
            f'<span style="color:var(--border2)">·</span>'
            f'<span>{mb_str}</span></div>',
            unsafe_allow_html=True
        )
        st.video(uploaded_file)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Output ──────────────────────────────────────────────────────────────────
with col_out:
    st.markdown("<div style='padding:16px 40px 0 12px'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec-label">Output · Vertical</div>', unsafe_allow_html=True)

    if st.session_state.processing_done and st.session_state.output_bytes:
        info = st.session_state.video_info
        out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
        if info:
            eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
        else:
            eff_w, eff_h = 1080, 1920
        st.markdown(
            f'<div class="rf-success"><div class="rf-success-dot"></div>'
            f'<div class="rf-success-text">Done — {eff_w}×{eff_h} · {out_mb:.1f} MB</div></div>',
            unsafe_allow_html=True
        )
        st.video(st.session_state.output_bytes, format="video/mp4")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
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
#  Metrics + Config + Action bar
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None and st.session_state.input_path:
    info = st.session_state.video_info
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:var(--border);margin:0 40px'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if info:
        dur = info["duration_seconds"]
        mins, secs = int(dur // 60), int(dur % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        est_sec = max(10, dur * 0.6 + 8)
        est_str = (f"~{int(est_sec//60)}m {int(est_sec%60):02d}s"
                   if est_sec >= 60 else f"~{int(est_sec)}s")
        fps_display = (fps_label if fps_label != "Source (keep original)"
                       else f"{info['fps']:.0f} fps")
        eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
        clamped = (resolution_label != "Match source (no upscale)"
                   and (eff_h < RESOLUTION_PRESETS[resolution_label][1]
                        or eff_w < RESOLUTION_PRESETS[resolution_label][0]))

        st.markdown(f"""
        <div style='padding:0 40px;margin-bottom:8px;'>
        <div class='rf-metrics'>
          <div class='rf-met'><div class='rf-met-lbl'>Duration</div>
            <div class='rf-met-val'>{dur_str}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Source</div>
            <div class='rf-met-val'>{info['width']}×{info['height']}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>FPS</div>
            <div class='rf-met-val'>{fps_display}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Output</div>
            <div class='rf-met-val a'>{eff_w}×{eff_h}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Est. time</div>
            <div class='rf-met-val'>{est_str}</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        if clamped:
            st.markdown(
                f'<div style="padding:0 40px;margin-bottom:6px;">'
                f'<div class="rf-info">ℹ Output clamped to {eff_w}×{eff_h} — '
                f'source is {info["width"]}×{info["height"]} (upscaling disabled).</div></div>',
                unsafe_allow_html=True
            )

        crf_lbl   = "Near-lossless" if crf <= 18 else ("Balanced" if crf <= 24 else "Compact")
        smooth_lbl = f"Adaptive ({smooth_window})" if adaptive_smoothing else str(smooth_window)
        st.markdown(f"""
        <div style='padding:0 40px;margin-bottom:14px;'>
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
    _, act, __ = st.columns([0.05, 10, 0.05])
    with act:
        if not st.session_state.processing_done:
            bc, gc, cc = st.columns([3, 6, 1.5])
            can_go = bool(info and info.get("is_landscape", True))
            with bc:
                go = st.button("▶  Convert to vertical",
                               type="primary", use_container_width=True,
                               disabled=not can_go)
            with gc:
                if info:
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;margin-top:12px;'>"
                        f"{eff_w}×{eff_h} · CRF {crf} · "
                        f"{'Adaptive' if adaptive_smoothing else 'Fixed'} smooth · "
                        f"Audio {audio_bitrate_label}</p>",
                        unsafe_allow_html=True,
                    )
            with cc:
                if st.button("Clear", type="secondary", use_container_width=True):
                    _cleanup()
                    st.session_state.uploaded_file_name = None
                    st.rerun()

            if go:
                st.session_state.last_settings = current_settings
                prog   = st.progress(0.0)
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
            rc, _, sc = st.columns([2, 5, 2])
            with rc:
                if st.button("← Start over", type="secondary", use_container_width=True):
                    _cleanup()
                    st.session_state.uploaded_file_name = None
                    st.session_state.processing_done    = False
                    st.rerun()
            with sc:
                if info and st.session_state.output_bytes:
                    in_mb   = len(uploaded_file.getvalue()) / (1024**2)
                    out_mb  = len(st.session_state.output_bytes) / (1024**2)
                    delta   = out_mb - in_mb
                    dcol    = "var(--green)" if delta < 0 else "var(--red)"
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:11px;"
                        f"text-align:right;margin-top:12px;'>"
                        f"Output {out_mb:.1f} MB "
                        f"<span style='color:{dcol}'>({delta:+.1f} MB)</span></p>",
                        unsafe_allow_html=True,
                    )

# ─────────────────────────────────────────────────────────────────────────────
#  Empty state
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='padding:0 40px 44px;margin-top:16px;'>
      <div style='background:var(--surface);border:2px dashed var(--border);
          border-radius:12px;padding:56px 40px;text-align:center;'>
        <div style='font-family:Fraunces,serif;font-size:2.4rem;font-weight:700;
            color:var(--border);letter-spacing:-0.04em;margin-bottom:10px;line-height:1.05;'>
          Drop a video to begin.
        </div>
        <p style='font-size:13px;color:var(--ink3);margin-bottom:16px;'>
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
    <span>YOLOv8</span><span>OpenCV</span><span>FFmpeg</span><span>Streamlit</span>
  </div>
  <div class="rf-footer-copy">Reframe · AI Video Converter</div>
</div>
""", unsafe_allow_html=True)
