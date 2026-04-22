"""
app.py  —  Reframe · AI Video Converter
Light theme, full video config controls.
"""

import streamlit as st
import tempfile
import os
from verticalize import process_video, get_video_info, RESOLUTION_PRESETS

st.set_page_config(
    page_title="Reframe · AI Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — warm light editorial theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&display=swap');

:root {
  --bg:        #f7f4ef;
  --surface:   #ffffff;
  --surface2:  #f2ede5;
  --border:    #e3dbd0;
  --border2:   #d0c8bc;
  --ink:       #1c1814;
  --ink2:      #5a5048;
  --ink3:      #9a8f84;
  --accent:    #d4581a;
  --accent-bg: #fdf1ea;
  --accent-dk: #b04010;
  --green:     #1a7a50;
  --green-bg:  #edf7f2;
  --red:       #c03030;
  --radius:    12px;
  --shadow:    0 1px 3px rgba(28,24,20,.06), 0 4px 16px rgba(28,24,20,.06);
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
    padding: 0 52px; height: 60px;
    background: rgba(247,244,239,0.92);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
}
.rf-wordmark {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 20px; font-weight: 700; color: var(--ink);
    letter-spacing: -0.02em; display: flex; align-items: center; gap: 8px;
}
.rf-pill-nav { display: flex; gap: 6px; }
.rf-pill {
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 5px 12px; border-radius: 99px;
    background: var(--surface2); color: var(--ink3); border: 1px solid var(--border);
}

/* Hero */
.rf-hero {
    padding: 56px 52px 40px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: flex-end;
    justify-content: space-between; gap: 32px; flex-wrap: wrap;
}
.rf-kicker {
    font-size: 11px; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 14px;
}
.rf-headline {
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 700; line-height: 1.05;
    letter-spacing: -0.03em; color: var(--ink); margin-bottom: 12px;
}
.rf-headline em { font-style: italic; color: var(--accent); }
.rf-desc { font-size: 14px; color: var(--ink2); line-height: 1.65; max-width: 440px; }
.rf-hero-stats { display: flex; gap: 32px; flex-wrap: wrap; }
.rf-stat { text-align: right; }
.rf-stat-val {
    font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 600;
    color: var(--ink); letter-spacing: -0.03em; line-height: 1;
}
.rf-stat-lbl { font-size: 11px; color: var(--ink3); margin-top: 3px; }

/* Section label */
.rf-sec-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink3);
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.rf-sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Upload */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-bg) !important;
}
[data-testid="stFileUploadDropzone"] { padding: 44px 24px !important; }
[data-testid="stFileUploadDropzone"] * {
    color: var(--ink3) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stFileUploadDropzone"] svg { color: var(--border2) !important; }

/* Param summary grid */
.rf-param-grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    margin-bottom: 20px;
}
.rf-param-cell { background: var(--surface); padding: 12px 14px; }
.rf-param-name { font-size: 10px; color: var(--ink3); margin-bottom: 3px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
.rf-param-val  { font-size: 13px; color: var(--ink); font-weight: 700; }
.rf-param-val.a { color: var(--accent); }

/* File chip */
.rf-file-chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 12px; font-size: 12px;
    color: var(--ink2); margin-bottom: 12px; max-width: 100%;
}
.rf-file-chip strong { color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Alerts */
.rf-alert-warn {
    background: #fff8f0; border: 1px solid #f0c08a;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #8a5010; margin-bottom: 12px;
}
.rf-alert-error {
    background: #fff0f0; border: 1px solid #f0a0a0;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: var(--red); margin-bottom: 12px;
}
.rf-success-banner {
    background: var(--green-bg); border: 1px solid #a0d4b8;
    border-radius: 10px; padding: 12px 16px;
    display: flex; align-items: center; gap: 10px; margin-bottom: 16px;
}
.rf-success-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.rf-success-text { font-size: 13px; color: var(--green); font-weight: 600; }

/* Empty state */
.rf-empty {
    background: var(--surface2); border: 2px dashed var(--border);
    border-radius: var(--radius); padding: 64px 32px;
    text-align: center; min-height: 300px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
}
.rf-empty-icon {
    width: 52px; height: 52px; border-radius: 14px;
    background: var(--surface); border: 1px solid var(--border);
    font-size: 22px; display: flex; align-items: center; justify-content: center;
    margin-bottom: 4px; box-shadow: var(--shadow);
}
.rf-empty-h { font-family: 'Fraunces', serif; font-size: 16px; font-weight: 600; color: var(--ink3); }
.rf-empty-s { font-size: 12px; color: var(--border2); }

/* Metrics row */
.rf-metrics {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden; margin-bottom: 20px;
}
.rf-met { background: var(--surface); padding: 14px 16px; }
.rf-met-lbl { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink3); margin-bottom: 4px; }
.rf-met-val { font-family: 'Fraunces', serif; font-size: 18px; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
.rf-met-val.a { color: var(--accent); }

/* Buttons */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border-radius: 9px !important; font-weight: 700 !important;
    font-size: 14px !important; transition: all 0.16s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; padding: 13px 28px !important;
    box-shadow: 0 2px 8px rgba(212,88,26,.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-dk) !important;
    box-shadow: 0 4px 18px rgba(212,88,26,.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--border2) !important; color: var(--ink3) !important;
    box-shadow: none !important; transform: none !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important; color: var(--ink2) !important;
    border: 1.5px solid var(--border2) !important; padding: 11px 20px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
}

.stDownloadButton > button {
    background: var(--green) !important; color: #fff !important;
    border: none !important; border-radius: 9px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 13px 28px !important; width: 100% !important;
    box-shadow: 0 2px 8px rgba(26,122,80,.2) !important;
    transition: all 0.16s ease !important;
}
.stDownloadButton > button:hover {
    background: #155f3e !important;
    box-shadow: 0 4px 18px rgba(26,122,80,.3) !important;
    transform: translateY(-1px) !important;
}

/* Progress */
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px; }
.stProgress > div > div { background: var(--border) !important; border-radius: 99px; height: 4px !important; }
.stProgress > div { height: 4px !important; }

/* Select */
[data-baseweb="select"] > div {
    background: var(--surface) !important; border-color: var(--border2) !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; color: var(--ink) !important;
}
[data-baseweb="select"] * { color: var(--ink) !important; }
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-baseweb="option"] { background: var(--surface) !important; color: var(--ink2) !important; }
[data-baseweb="option"]:hover { background: var(--accent-bg) !important; color: var(--accent) !important; }

/* Slider */
.stSlider label {
    font-size: 12px !important; color: var(--ink2) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important; border: 2px solid #fff !important;
    box-shadow: 0 1px 4px rgba(212,88,26,.3) !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--border) !important; }
[data-testid="stSliderValue"] {
    color: var(--accent) !important; font-size: 12px !important; font-weight: 700 !important;
}

/* Toggle */
[data-testid="stToggleSwitch"] > div { background: var(--border2) !important; }
[data-testid="stToggleSwitch"][aria-checked="true"] > div { background: var(--accent) !important; }
[data-testid="stToggleSwitch"] span { color: var(--ink2) !important; font-size: 13px !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: var(--surface2) !important; border-radius: 8px !important; padding: 4px !important; gap: 2px !important; }
[data-baseweb="tab"] {
    background: transparent !important; border-radius: 6px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important; color: var(--ink3) !important;
    padding: 8px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--surface) !important; color: var(--ink) !important;
    box-shadow: var(--shadow) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; }
[data-baseweb="notification"] { border-radius: 10px !important; }

/* Video */
video { border-radius: 10px !important; width: 100% !important; background: #111; display: block; }

/* Caption */
.stCaption, small { color: var(--ink3) !important; font-size: 11px !important; }

/* Footer */
.rf-footer {
    margin-top: 40px; padding: 20px 52px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
}
.rf-tech { display: flex; gap: 6px; }
.rf-tech span {
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 10px;
    border: 1px solid var(--border); border-radius: 4px; color: var(--ink3);
}
.rf-footer-copy { font-size: 11px; color: var(--border2); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        input_path=None, output_path=None,
        uploaded_file_name=None,
        processing_done=False, output_bytes=None,
        video_info=None,
        last_settings=None,
    )
    for k, v in defaults.items():
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

def _new_output_path():
    fd, p = tempfile.mkstemp(suffix=".mp4")
    os.close(fd); os.unlink(p)
    st.session_state.output_path = p

def _invalidate_if_changed(current: dict):
    if (st.session_state.processing_done and
            st.session_state.last_settings != current):
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
      <rect width="20" height="20" rx="6" fill="#d4581a"/>
      <rect x="4" y="4" width="5" height="12" rx="2" fill="white"/>
      <rect x="11" y="7" width="5" height="9" rx="2" fill="white" opacity="0.55"/>
    </svg>
    Reframe
  </div>
  <div class="rf-pill-nav">
    <span class="rf-pill">YOLOv8</span>
    <span class="rf-pill">OpenCV</span>
    <span class="rf-pill">FFmpeg</span>
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
    <p class="rf-desc">Upload landscape footage. AI tracks subjects frame-by-frame,
    computes a smooth crop path, and exports a vertical video ready for TikTok,
    Reels &amp; Shorts.</p>
  </div>
  <div class="rf-hero-stats">
    <div class="rf-stat"><div class="rf-stat-val">9:16</div><div class="rf-stat-lbl">Output ratio</div></div>
    <div class="rf-stat"><div class="rf-stat-val">1080p</div><div class="rf-stat-lbl">Full HD</div></div>
    <div class="rf-stat"><div class="rf-stat-val">YOLOv8</div><div class="rf-stat-lbl">AI model</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Settings tabs
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div style='padding:0 52px'>", unsafe_allow_html=True)

    tab_output, tab_tracking, tab_advanced = st.tabs(
        ["🎞  Output & Quality", "🎯  Tracking", "⚙  Advanced"]
    )

    with tab_output:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            resolution_label = st.selectbox(
                "Output resolution",
                list(RESOLUTION_PRESETS.keys()), index=0,
                help="Width × Height of output vertical video",
            )
            target_size = RESOLUTION_PRESETS[resolution_label]
        with c2:
            fps_label = st.selectbox(
                "Output frame rate",
                ["Source (keep original)", "60 fps", "30 fps", "25 fps", "24 fps"],
                index=0, help="Reduce to shrink file size.",
            )
            _fps_map = {"Source (keep original)": None, "60 fps": 60.0,
                        "30 fps": 30.0, "25 fps": 25.0, "24 fps": 24.0}
            output_fps = _fps_map[fps_label]
        with c3:
            crf = st.slider("Quality (CRF)", 15, 35, 23, 1,
                help="Lower = better quality, larger file. 18–23 excellent.")
            st.caption("18 = near-lossless  ·  28 = compact")
        with c4:
            encoder_preset_label = st.selectbox(
                "Encode speed",
                ["ultrafast", "fast", "medium", "slow"], index=1,
                help="Slower encodes compress slightly better at same CRF.",
            )

    with tab_tracking:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4, gap="medium")
        with t1:
            smooth_window = st.slider("Path smoothness", 3, 31, 15, 2,
                help="Higher = steadier pan; lower = snappier follow")
            st.caption("Higher → steadier  ·  Lower → responsive")
        with t2:
            confidence = st.slider("Detection confidence", 0.10, 0.95, 0.50, 0.05,
                help="Minimum YOLO confidence to accept a detection")
            st.caption("Lower → sensitive  ·  Higher → strict")
        with t3:
            use_optical_flow = st.toggle("Motion tracking (optical flow)", value=True,
                help="Estimate motion when no subject is detected")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            rule_of_thirds = st.toggle("Rule-of-thirds framing", value=True,
                help="Nudge crop toward cinematic composition lines")
        with t4:
            scene_cut_threshold = st.slider("Scene-cut sensitivity", 0.10, 0.60, 0.35, 0.05,
                help="Frame diff % to declare a scene cut (resets tracking)")
            st.caption("Higher → hard cuts only  ·  Lower → more cuts")

    with tab_advanced:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3, gap="medium")
        with a1:
            audio_bitrate_label = st.selectbox(
                "Audio bitrate",
                ["64k", "96k", "128k", "192k", "256k"], index=2,
                help="AAC audio bitrate in output file",
            )
        with a2:
            yolo_weights = st.selectbox(
                "YOLO model",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0,
                help="Nano (fastest) → Small → Medium (most accurate)",
            )
        with a3:
            st.markdown("""
            <div style='background:var(--surface2);border:1px solid var(--border);
                border-radius:8px;padding:12px 14px;font-size:12px;
                color:var(--ink3);margin-top:24px;'>
              <strong style='color:var(--ink2)'>Model sizes</strong><br>
              nano: ~6 MB · fast, good accuracy<br>
              small: ~22 MB · balanced<br>
              medium: ~50 MB · best accuracy
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Settings dict for invalidation
current_settings = dict(
    resolution_label=resolution_label, fps_label=fps_label,
    crf=crf, encoder_preset_label=encoder_preset_label,
    smooth_window=smooth_window, confidence=confidence,
    use_optical_flow=use_optical_flow, rule_of_thirds=rule_of_thirds,
    scene_cut_threshold=scene_cut_threshold,
    audio_bitrate_label=audio_bitrate_label, yolo_weights=yolo_weights,
)
_invalidate_if_changed(current_settings)

st.markdown("<div style='height:2px;background:var(--border);margin:0'></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-column work area
# ─────────────────────────────────────────────────────────────────────────────
col_src, col_out = st.columns(2, gap="large")

with col_src:
    st.markdown("<div style='padding:28px 52px 0 52px'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec-label">Source · Landscape</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop video here",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        mb = len(uploaded_file.getvalue()) / (1024 ** 2)
        if mb > 500:
            st.markdown(
                f'<div class="rf-alert-error">File is {mb:.1f} MB — max 500 MB.</div>',
                unsafe_allow_html=True
            )
            uploaded_file = None

    if (uploaded_file is not None and
            st.session_state.uploaded_file_name != uploaded_file.name):
        _cleanup()
        st.session_state.processing_done = False
        st.session_state.last_settings   = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.getvalue())
            st.session_state.input_path = tmp_in.name

        _new_output_path()
        st.session_state.uploaded_file_name = uploaded_file.name

        try:
            st.session_state.video_info = get_video_info(st.session_state.input_path)
        except Exception:
            st.session_state.video_info = None

    if uploaded_file is not None and st.session_state.input_path:
        info = st.session_state.video_info

        if info and not info["is_landscape"]:
            st.markdown(
                '<div class="rf-alert-warn">This video is already vertical — '
                'please upload a landscape (wider than tall) video.</div>',
                unsafe_allow_html=True
            )

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


with col_out:
    st.markdown("<div style='padding:28px 52px 0 52px'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec-label">Output · Vertical</div>', unsafe_allow_html=True)

    if st.session_state.processing_done and st.session_state.output_bytes:
        out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
        tw, th = target_size
        st.markdown(
            f'<div class="rf-success-banner"><div class="rf-success-dot"></div>'
            f'<div class="rf-success-text">Conversion complete — '
            f'{tw}×{th} · {out_mb:.1f} MB</div></div>',
            unsafe_allow_html=True
        )
        st.video(st.session_state.output_bytes, format="video/mp4")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
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
          <div class="rf-empty-s">will appear here after conversion</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics + Action bar
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None and st.session_state.input_path:
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:var(--border);margin:0 52px'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    info = st.session_state.video_info

    if info:
        dur = info["duration_seconds"]
        mins, secs = int(dur // 60), int(dur % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        est_sec = max(10, dur * 0.6 + 8)
        est_min = est_sec / 60
        est_str = (f"~{int(est_min)}m {int(est_sec%60):02d}s"
                   if est_min >= 1 else f"~{int(est_sec)}s")
        fps_display = fps_label if fps_label != "Source (keep original)" else f"{info['fps']:.0f} fps"

        st.markdown(f"""
        <div style='padding:0 52px;margin-bottom:16px;'>
        <div class='rf-metrics'>
          <div class='rf-met'><div class='rf-met-lbl'>Duration</div>
            <div class='rf-met-val'>{dur_str}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Source res.</div>
            <div class='rf-met-val'>{info['width']}×{info['height']}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Frame rate</div>
            <div class='rf-met-val'>{fps_display}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Output</div>
            <div class='rf-met-val a'>{target_size[0]}×{target_size[1]}</div></div>
          <div class='rf-met'><div class='rf-met-lbl'>Est. time</div>
            <div class='rf-met-val'>{est_str}</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        crf_lbl = "Near-lossless" if crf <= 18 else ("Balanced" if crf <= 24 else "Compact")
        st.markdown(f"""
        <div style='padding:0 52px;margin-bottom:20px;'>
        <div class='rf-param-grid'>
          <div class='rf-param-cell'><div class='rf-param-name'>Quality CRF {crf}</div>
            <div class='rf-param-val a'>{crf_lbl}</div></div>
          <div class='rf-param-cell'><div class='rf-param-name'>Encode speed</div>
            <div class='rf-param-val'>{encoder_preset_label}</div></div>
          <div class='rf-param-cell'><div class='rf-param-name'>Path smoothness</div>
            <div class='rf-param-val'>{smooth_window}</div></div>
          <div class='rf-param-cell'><div class='rf-param-name'>AI confidence</div>
            <div class='rf-param-val'>{confidence:.2f}</div></div>
          <div class='rf-param-cell'><div class='rf-param-name'>Audio bitrate</div>
            <div class='rf-param-val'>{audio_bitrate_label}</div></div>
          <div class='rf-param-cell'><div class='rf-param-name'>YOLO model</div>
            <div class='rf-param-val'>{yolo_weights}</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Action row
    _, action_col, __ = st.columns([0.06, 10, 0.06])
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
                    tw, th = target_size
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:12px;"
                        f"margin-top:15px;font-family:Plus Jakarta Sans,sans-serif'>"
                        f"{tw}×{th} · CRF {crf} · {encoder_preset_label} · "
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
                progress_bar = st.progress(0.0)
                status_text  = st.empty()
                status_text.info("⚡ Starting…")
                try:
                    def _cb(prog: float, msg: str = ""):
                        progress_bar.progress(min(prog, 1.0))
                        if msg:
                            status_text.info(msg)

                    process_video(
                        st.session_state.input_path,
                        st.session_state.output_path,
                        target_size=target_size,
                        confidence=confidence,
                        smooth_window=smooth_window,
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
                    progress_bar.progress(1.0)
                    out = st.session_state.output_path
                    if os.path.exists(out) and os.path.getsize(out) > 0:
                        with open(out, "rb") as f:
                            st.session_state.output_bytes = f.read()
                        st.session_state.processing_done = True
                        status_text.success("Done! Download your video below.")
                        st.rerun()
                    else:
                        status_text.error("Output file is empty — something went wrong.")
                except Exception as exc:
                    status_text.error(f"Error: {exc}")

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
                    in_mb      = len(uploaded_file.getvalue()) / (1024**2)
                    out_mb     = len(st.session_state.output_bytes) / (1024**2)
                    delta      = out_mb - in_mb
                    delta_col  = "var(--green)" if delta < 0 else "var(--red)"
                    st.markdown(
                        f"<p style='color:var(--ink3);font-size:12px;"
                        f"text-align:right;margin-top:15px;'>"
                        f"Output: {out_mb:.1f} MB "
                        f"<span style='color:{delta_col}'>({delta:+.1f} MB)</span></p>",
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
#  Empty state (no file uploaded)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='padding:0 52px 56px;margin-top:24px;'>
      <div style='background:var(--surface);border:2px dashed var(--border);
          border-radius:16px;padding:72px 48px;text-align:center;'>
        <div style='font-family:Fraunces,serif;font-size:2.8rem;font-weight:700;
            color:var(--border);letter-spacing:-0.04em;margin-bottom:12px;line-height:1.05;'>
          Drop a video<br>to begin.
        </div>
        <p style='font-size:14px;color:var(--ink3);margin-bottom:20px;'>
          Landscape MP4, MOV, AVI, or MKV · up to 500 MB
        </p>
        <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
          <span style='font-size:11px;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:var(--ink3);padding:5px 12px;
              border:1px solid var(--border);border-radius:4px;'>MP4</span>
          <span style='font-size:11px;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:var(--ink3);padding:5px 12px;
              border:1px solid var(--border);border-radius:4px;'>MOV</span>
          <span style='font-size:11px;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:var(--ink3);padding:5px 12px;
              border:1px solid var(--border);border-radius:4px;'>AVI</span>
          <span style='font-size:11px;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:var(--ink3);padding:5px 12px;
              border:1px solid var(--border);border-radius:4px;'>MKV</span>
          <span style='font-size:11px;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:var(--ink3);padding:5px 12px;
              border:1px solid var(--border);border-radius:4px;'>max 500 MB</span>
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
    <span>FFmpeg</span><span>Streamlit</span>
  </div>
  <div class="rf-footer-copy">Reframe · AI Video Converter</div>
</div>
""", unsafe_allow_html=True)
