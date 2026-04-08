"""
app.py  —  Reframe · AI Video Converter
Streamlit frontend for verticalize.py
"""

import streamlit as st
import tempfile
import os
import math
from verticalize import process_video, get_video_info, extract_thumbnail, RESOLUTION_PRESETS

st.set_page_config(
    page_title="Reframe · AI Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system: warm editorial light theme ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Reset & base ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #faf8f4 !important;
    color: #1a1714;
}
.stApp { background-color: #faf8f4 !important; }

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #f2ede6 !important;
    border-right: 1px solid #e0d9cf !important;
}
section[data-testid="stSidebar"] > div { padding-top: 2rem; }

/* ── Typography ───────────────────────────────── */
h1 {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #1a1714 !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
h2, h3 {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    color: #2d2925 !important;
}
.stMarkdown p { color: #3d3832; line-height: 1.65; }

/* ── Uploader ─────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 2px dashed #c9bfb0 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #c0692a !important; }
[data-testid="stFileUploadDropzone"] * { color: #7a6f64 !important; }

/* ── Primary button ───────────────────────────── */
.stButton > button[kind="primary"] {
    background: #c0692a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.01em;
    padding: 12px 28px !important;
    box-shadow: 0 2px 8px rgba(192,105,42,0.25) !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #a8581f !important;
    box-shadow: 0 4px 16px rgba(192,105,42,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ─────────────────────────── */
.stButton > button[kind="secondary"] {
    background: #fff !important;
    color: #5a5048 !important;
    border: 1.5px solid #c9bfb0 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #c0692a !important;
    color: #c0692a !important;
}

/* ── Download button ──────────────────────────── */
.stDownloadButton > button {
    background: #1a6b45 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    box-shadow: 0 2px 8px rgba(26,107,69,0.22) !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: #155737 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(26,107,69,0.3) !important;
}

/* ── Progress bar ─────────────────────────────── */
.stProgress > div > div > div {
    background: #c0692a !important;
    border-radius: 99px;
}
.stProgress > div > div {
    background: #e8e0d6 !important;
    border-radius: 99px;
}

/* ── Alerts ───────────────────────────────────── */
.stAlert { border-radius: 10px !important; }
[data-baseweb="notification"] { border-radius: 10px !important; }

/* ── Select / Slider ──────────────────────────── */
[data-baseweb="select"] > div {
    background: #fff !important;
    border-color: #c9bfb0 !important;
    border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #c0692a !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: #c0692a !important;
}

/* ── Toggle ───────────────────────────────────── */
[data-testid="stToggleSwitch"][aria-checked="true"] > div {
    background: #c0692a !important;
}

/* ── Metrics ──────────────────────────────────── */
[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e0d9cf;
    border-radius: 12px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #7a6f64 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] {
    font-family: 'Libre Baskerville', serif !important;
    color: #1a1714 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Video ────────────────────────────────────── */
video {
    border-radius: 10px;
    width: 100% !important;
    background: #000;
}

/* ── Divider ──────────────────────────────────── */
hr { border-color: #e0d9cf !important; margin: 1.5rem 0 !important; }

/* ── Caption ──────────────────────────────────── */
.stCaption, small { color: #9a8f84 !important; font-size: 12px !important; }

/* ── Expander ─────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e0d9cf !important;
    border-radius: 10px !important;
    background: #fff !important;
}

/* ── Custom components ────────────────────────── */
.info-card {
    background: #fff;
    border: 1px solid #e0d9cf;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
    font-size: 13.5px;
    color: #3d3832;
}
.step-num {
    flex-shrink: 0;
    width: 24px; height: 24px;
    background: #c0692a;
    color: #fff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    margin-top: 1px;
}
.video-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #9a8f84;
    margin-bottom: 8px;
}
.stat-pill {
    display: inline-block;
    background: #f2ede6;
    border: 1px solid #e0d9cf;
    border-radius: 99px;
    padding: 3px 11px;
    font-size: 12px;
    color: #5a5048;
    margin: 2px 4px 2px 0;
}
.empty-state {
    background: #fff;
    border: 2px dashed #d8d0c4;
    border-radius: 16px;
    padding: 60px 32px;
    text-align: center;
}
.tag {
    display: inline-block;
    background: #f2ede6;
    border: 1px solid #e0d9cf;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    color: #7a6f64;
    margin: 2px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Libre Baskerville,serif;font-size:1.4rem;"
        "margin-bottom:2px'>Reframe</h2>"
        "<p style='color:#9a8f84;font-size:13px;margin-top:0'>AI Video Converter</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**How it works**")
    for i, (title, sub) in enumerate([
        ("Upload landscape video",   "MP4, MOV, AVI, MKV · max 500 MB"),
        ("AI detects subjects",       "People, vehicles, animals — one pass"),
        ("Smooth crop is computed",   "Tracking + scene cuts + rule-of-thirds"),
        ("Download your reel",        "H.264 · browser-ready · with audio"),
    ], 1):
        st.markdown(
            f'<div class="step-row"><div class="step-num">{i}</div>'
            f'<div><strong>{title}</strong><br>'
            f'<span style="color:#9a8f84;font-size:12px">{sub}</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Output resolution**")
    resolution_label = st.selectbox(
        "Resolution",
        list(RESOLUTION_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    target_size = RESOLUTION_PRESETS[resolution_label]

    st.markdown("---")
    st.markdown("**Tracking settings**")
    smooth_window = st.slider(
        "Smoothness", min_value=3, max_value=31, value=15, step=2,
        help="Higher = steadier pan · Lower = snappier tracking",
    )
    st.caption("Higher → steadier  ·  Lower → responsive")

    confidence = st.slider(
        "Detection confidence", min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="Lower detects more but may add false positives",
    )
    st.caption("Lower → sensitive  ·  Higher → strict")

    st.markdown("---")
    st.markdown("**Advanced**")
    use_optical_flow = st.toggle("Motion tracking (optical flow)", value=True,
        help="Tracks movement when AI detection finds nothing")
    rule_of_thirds = st.toggle("Rule-of-thirds framing", value=True,
        help="Nudges crop toward cinematic 1/3 composition lines")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:11px;color:#c9bfb0;text-align:center'>"
        "YOLOv8 · OpenCV · FFmpeg</p>",
        unsafe_allow_html=True,
    )


# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        input_path=None,
        output_path=None,
        uploaded_file_name=None,
        processing_done=False,
        output_bytes=None,
        video_info=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _cleanup():
    for key in ("input_path", "output_path"):
        p = st.session_state.get(key)
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
        st.session_state[key] = None
    st.session_state.output_bytes = None
    st.session_state.video_info   = None


_init()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>AI Video Reframer</h1>"
    "<p style='color:#7a6f64;font-size:15px;margin-top:4px;margin-bottom:28px'>"
    "Convert landscape videos to vertical format for "
    "<span class='tag'>TikTok</span> "
    "<span class='tag'>Reels</span> "
    "<span class='tag'>Shorts</span>"
    "</p>",
    unsafe_allow_html=True,
)


# ── File uploader ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a video here, or click to browse",
    type=["mp4", "mov", "avi", "mkv"],
    help="Landscape (wider than tall) videos only. Max 500 MB.",
)

# Validate size
if uploaded_file is not None:
    mb = len(uploaded_file.getvalue()) / (1024 ** 2)
    if mb > 500:
        st.error(f"⚠️ File is {mb:.1f} MB — please upload a file under 500 MB.")
        uploaded_file = None

# New upload → reset state & write temp file
if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    _cleanup()
    st.session_state.processing_done = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        st.session_state.input_path = tmp_in.name

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    os.unlink(out_path)
    st.session_state.output_path = out_path
    st.session_state.uploaded_file_name = uploaded_file.name

    # Read video metadata for display
    try:
        st.session_state.video_info = get_video_info(st.session_state.input_path)
    except Exception:
        st.session_state.video_info = None


# ── Main UI ─────────────────────────────────────────────────────────────────────
if uploaded_file is not None and st.session_state.input_path:

    st.markdown("---")

    # ── Video info strip ───────────────────────────────────────────────
    info = st.session_state.video_info
    if info:
        dur = info["duration_seconds"]
        mins, secs = int(dur // 60), int(dur % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

        # Estimate processing time: ~0.5× real-time for short, ~1× for longer
        est_sec = max(10, dur * 0.6 + 8)
        est_min = est_sec / 60
        est_str = f"~{int(est_min)}m {int(est_sec % 60):02d}s" if est_min >= 1 else f"~{int(est_sec)}s"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Duration",    dur_str)
        c2.metric("Resolution",  f"{info['width']}×{info['height']}")
        c3.metric("Frame rate",  f"{info['fps']:.0f} fps")
        c4.metric("Output",      f"{target_size[0]}×{target_size[1]}")
        c5.metric("Est. time",   est_str)

        st.markdown("")

    # ── Preview columns ────────────────────────────────────────────────
    col_orig, col_gap, col_vert = st.columns([11, 1, 6])

    with col_orig:
        st.markdown('<p class="video-label">▶ Original · Landscape</p>',
                    unsafe_allow_html=True)
        st.video(uploaded_file)
        mb = len(uploaded_file.getvalue()) / (1024 ** 2)
        st.caption(f"{uploaded_file.name}  ·  {mb:.1f} MB")

    with col_vert:
        if st.session_state.processing_done and st.session_state.output_bytes:
            st.markdown(
                '<p class="video-label" style="color:#1a6b45">📱 Vertical · Ready</p>',
                unsafe_allow_html=True,
            )
            st.video(st.session_state.output_bytes, format="video/mp4")
            out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
            st.caption(f"Output: {out_mb:.1f} MB  ·  {target_size[0]}×{target_size[1]}")
        else:
            st.markdown('<p class="video-label">📱 Vertical · Preview</p>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="empty-state" style="min-height:240px;padding:40px 20px">'
                '<div style="font-size:40px;margin-bottom:10px">📱</div>'
                '<p style="color:#9a8f84;font-size:13px;margin:0">'
                "Your vertical video<br>will appear here"
                "</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Action area ────────────────────────────────────────────────────
    if not st.session_state.processing_done:
        a_col, b_col = st.columns([3, 5])
        with a_col:
            go = st.button("🎬  Convert to Vertical", type="primary",
                           use_container_width=True)
        with b_col:
            if info:
                st.markdown(
                    f"<p style='color:#9a8f84;font-size:13px;margin-top:10px'>"
                    f"Output: <strong>{target_size[0]}×{target_size[1]}</strong>"
                    f" · Est. {est_str}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

        if go:
            progress_bar  = st.progress(0.0)
            status_text   = st.empty()
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
                    progress_callback=_cb,
                )

                progress_bar.progress(1.0)
                out = st.session_state.output_path
                if os.path.exists(out) and os.path.getsize(out) > 0:
                    with open(out, "rb") as f:
                        st.session_state.output_bytes = f.read()
                    st.session_state.processing_done = True
                    status_text.success("✅ Conversion complete!")
                    st.rerun()
                else:
                    status_text.error("❌ Output file is empty — something went wrong.")

            except Exception as exc:
                status_text.error(f"❌ {exc}")

    # ── Download + actions ─────────────────────────────────────────────
    if st.session_state.processing_done and st.session_state.output_bytes:
        st.success("🎉 Your vertical video is ready to download!")

        dl_col, meta_col, reset_col = st.columns([4, 4, 2])

        with dl_col:
            stem = os.path.splitext(uploaded_file.name)[0]
            st.download_button(
                label="📥  Download Vertical Video",
                data=st.session_state.output_bytes,
                file_name=f"{stem}_vertical.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        with meta_col:
            in_mb  = len(uploaded_file.getvalue()) / (1024 ** 2)
            out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
            delta  = out_mb - in_mb
            st.metric(
                "File size",
                f"{out_mb:.1f} MB",
                f"{delta:+.1f} MB vs input",
            )

        with reset_col:
            if st.button("🗑️  Start over", type="secondary",
                         use_container_width=True):
                _cleanup()
                st.session_state.uploaded_file_name = None
                st.session_state.processing_done    = False
                st.rerun()

    elif not st.session_state.processing_done:
        _, reset_col = st.columns([9, 2])
        with reset_col:
            if st.button("✕  Clear", type="secondary", use_container_width=True):
                _cleanup()
                st.session_state.uploaded_file_name = None
                st.rerun()

else:
    # ── Empty state ────────────────────────────────────────────────────
    st.markdown(
        '<div class="empty-state" style="margin-top:32px">'
        '<div style="font-size:52px;margin-bottom:16px">🎬</div>'
        '<h3 style="font-family:Libre Baskerville,serif;color:#2d2925;'
        'margin-bottom:8px">Drop a video to get started</h3>'
        '<p style="color:#7a6f64;font-size:15px;max-width:440px;margin:0 auto 20px">'
        "Upload a landscape video and convert it to vertical format "
        "with AI-powered subject tracking."
        "</p>"
        '<span class="tag">MP4</span>'
        '<span class="tag">MOV</span>'
        '<span class="tag">AVI</span>'
        '<span class="tag">MKV</span>'
        '<span class="tag">≤ 500 MB</span>'
        "</div>",
        unsafe_allow_html=True,
    )
