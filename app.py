import streamlit as st
import tempfile
import os
from verticalize import process_video

st.set_page_config(
    page_title="Reframe · AI Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * {
    color: #b0b4c8 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e8eaf0 !important;
    font-family: 'Syne', sans-serif;
}

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #ffffff !important; letter-spacing: -0.5px; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #e8eaf0 !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #13161e;
    border: 2px dashed #2a2d3e;
    border-radius: 16px;
    padding: 8px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #5b6cff;
}
[data-testid="stFileUploader"] * { color: #b0b4c8 !important; }

/* Buttons */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5b6cff 0%, #8b5cf6 100%) !important;
    color: white !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    box-shadow: 0 4px 20px rgba(91, 108, 255, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(91, 108, 255, 0.5) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e2130 !important;
    color: #b0b4c8 !important;
    border: 1px solid #2a2d3e !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #252a3a !important;
    color: #e8eaf0 !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(16, 185, 129, 0.5) !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #5b6cff, #8b5cf6) !important;
    border-radius: 99px;
}
.stProgress > div > div {
    background: #1e2130 !important;
    border-radius: 99px;
}

/* Info / success / error messages */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
[data-testid="stNotification"] {
    background: #13161e !important;
    border-radius: 10px !important;
}

/* Video player */
video {
    border-radius: 12px;
    width: 100% !important;
}

/* Sliders */
.stSlider > div > div > div { background: #5b6cff !important; }

/* Caption */
.stCaption { color: #6b7280 !important; }

/* Divider */
hr { border-color: #1e2130 !important; }

/* Metric */
[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; }
[data-testid="stMetricValue"] { color: #e8eaf0 !important; font-family: 'Syne', sans-serif !important; }

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #5b6cff, #8b5cf6);
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin-right: 10px;
    flex-shrink: 0;
}
.step-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    color: #b0b4c8;
    font-size: 14px;
}

/* Panel card */
.panel-card {
    background: #13161e;
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Video label */
.video-label {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 10px;
}
.video-label-active {
    color: #8b5cf6;
}

/* Tag pill */
.pill {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2a2d3e;
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 12px;
    color: #6b7280;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 Reframe")
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-top:-8px;'>AI-powered vertical video</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### How it works")
    for i, step in enumerate([
        ("Upload a landscape video", "MP4, MOV, AVI, MKV — up to 500 MB"),
        ("AI analyses subjects", "Detects people, objects, motion"),
        ("Smart crop is computed", "Rule-of-thirds + smooth tracking"),
        ("Download your reel", "Ready for TikTok, Reels, Shorts"),
    ], 1):
        st.markdown(f"""
        <div class="step-row">
            <span class="step-badge">{i}</span>
            <span><strong style="color:#e8eaf0">{step[0]}</strong><br>
            <span style="font-size:12px;color:#6b7280">{step[1]}</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    smooth_window = st.slider(
        "Tracking smoothness",
        min_value=3, max_value=31, value=15, step=2,
        help="Higher = steadier camera pan. Lower = snappier tracking.",
    )
    st.caption("Higher → steadier  |  Lower → more responsive")

    confidence = st.slider(
        "Detection sensitivity",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="Lower detects more subjects but may pick up false positives.",
    )
    st.caption("Lower → more sensitive  |  Higher → stricter")

    use_optical_flow = st.toggle("Motion tracking (optical flow)", value=True,
        help="Track moving objects even when AI detection misses them.")
    rule_of_thirds = st.toggle("Rule-of-thirds framing", value=True,
        help="Nudge the crop toward cinematic rule-of-thirds composition.")

    st.markdown("---")
    st.markdown("<p style='font-size:12px;color:#3d4260;text-align:center'>Powered by YOLOv8 + OpenCV</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "input_path": None,
        "output_path": None,
        "uploaded_file_name": None,
        "processing_done": False,
        "progress_value": 0.0,
        "processing_status": "",
        "output_bytes": None,  # ← key fix: cache bytes for st.video()
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _cleanup_temp_files():
    for key in ("input_path", "output_path"):
        path = st.session_state.get(key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        st.session_state[key] = None
    st.session_state["output_bytes"] = None


_init_state()


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom:32px">
    <h1 style="font-size:2.6rem;margin-bottom:4px">AI Video Reframer</h1>
    <p style="color:#6b7280;font-size:16px;margin:0">
        Convert landscape videos to vertical format — optimised for
        <span class="pill">TikTok</span><span class="pill">Reels</span><span class="pill">Shorts</span>
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
ALLOWED_TYPES = ["mp4", "mov", "avi", "mkv"]
MAX_FILE_MB = 500

uploaded_file = st.file_uploader(
    "Drop your video here, or click to browse",
    type=ALLOWED_TYPES,
    help=f"Horizontal (landscape) videos only. Max {MAX_FILE_MB} MB.",
    label_visibility="visible",
)

if uploaded_file is not None:
    file_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
    if file_mb > MAX_FILE_MB:
        st.error(f"⚠️ File is {file_mb:.1f} MB — please upload a file under {MAX_FILE_MB} MB.")
        uploaded_file = None

# Detect new upload
if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    _cleanup_temp_files()
    st.session_state.processing_done = False
    st.session_state.progress_value = 0.0
    st.session_state.processing_status = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        st.session_state.input_path = tmp_in.name

    tmp_out_fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_out_fd)
    # Remove the empty stub so process_video can write fresh
    if os.path.exists(tmp_out_path):
        os.unlink(tmp_out_path)
    st.session_state.output_path = tmp_out_path
    st.session_state.uploaded_file_name = uploaded_file.name


# ---------------------------------------------------------------------------
# Main UI (only when a file is loaded)
# ---------------------------------------------------------------------------
if uploaded_file is not None and st.session_state.input_path:

    st.markdown("---")

    # ── Video preview columns ────────────────────────────────────────────
    col_orig, col_spacer, col_vert = st.columns([5, 1, 3])

    with col_orig:
        st.markdown('<p class="video-label">▶ Original · Landscape 16:9</p>', unsafe_allow_html=True)
        st.video(uploaded_file)
        file_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
        st.caption(f"📦 {uploaded_file.name}  ·  {file_mb:.1f} MB")

    with col_vert:
        if st.session_state.processing_done and st.session_state.output_bytes:
            st.markdown('<p class="video-label video-label-active">📱 Vertical 9:16 — Ready</p>', unsafe_allow_html=True)
            # ── KEY FIX: pass bytes directly — avoids stale path issues ──
            st.video(st.session_state.output_bytes, format="video/mp4")
            out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
            st.caption(f"📦 Output: {out_mb:.1f} MB")
        else:
            st.markdown('<p class="video-label">📱 Vertical 9:16</p>', unsafe_allow_html=True)
            st.markdown("""
            <div style="
                background:#13161e;
                border:2px dashed #1e2130;
                border-radius:12px;
                min-height:280px;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                gap:12px;
                color:#3d4260;
                font-size:14px;
                padding:24px;
                text-align:center;
            ">
                <span style="font-size:36px">📱</span>
                <span>Your vertical video will<br>appear here after conversion</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Action row ───────────────────────────────────────────────────────
    if not st.session_state.processing_done:
        btn_col, _ = st.columns([3, 5])
        with btn_col:
            convert_clicked = st.button(
                "🎬 Convert to Vertical",
                type="primary",
                use_container_width=True,
            )

        if convert_clicked:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            status_text.info("⚡ Starting AI analysis…")

            try:
                def update_progress(progress: float, message: str = ""):
                    progress_bar.progress(min(progress, 1.0))
                    st.session_state.progress_value = progress
                    display_msg = message or (
                        "🔍 Detecting subjects…" if progress < 0.3
                        else f"🎥 Rendering… {int(progress * 100)}%" if progress < 0.9
                        else "🎵 Finalising audio…"
                    )
                    status_text.info(display_msg)

                process_video(
                    st.session_state.input_path,
                    st.session_state.output_path,
                    confidence=confidence,
                    smooth_window=smooth_window,
                    use_optical_flow=use_optical_flow,
                    rule_of_thirds=rule_of_thirds,
                    progress_callback=update_progress,
                )

                progress_bar.progress(1.0)

                out_path = st.session_state.output_path
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    # ── KEY FIX: read bytes into session state immediately ──
                    with open(out_path, "rb") as f:
                        st.session_state.output_bytes = f.read()

                    st.session_state.processing_done = True
                    st.session_state.progress_value = 1.0
                    status_text.success("✅ Conversion complete!")
                    st.rerun()
                else:
                    status_text.error("❌ Output file is empty — processing may have failed.")

            except Exception as exc:
                status_text.error(f"❌ {exc}")
                st.error(f"Processing failed: {exc}")

    # ── Download section ─────────────────────────────────────────────────
    if st.session_state.processing_done and st.session_state.output_bytes:
        st.success("🎉 Conversion complete — your vertical video is ready!")

        dl_col, info_col, reset_col = st.columns([4, 3, 2])

        with dl_col:
            base_name = os.path.splitext(uploaded_file.name)[0]
            st.download_button(
                label="📥 Download Vertical Video",
                data=st.session_state.output_bytes,
                file_name=f"{base_name}_vertical.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        with info_col:
            out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
            in_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
            st.metric("Output size", f"{out_mb:.1f} MB", f"{out_mb - in_mb:+.1f} MB vs input")

        with reset_col:
            if st.button("🗑️ Start Over", type="secondary", use_container_width=True):
                _cleanup_temp_files()
                st.session_state.uploaded_file_name = None
                st.session_state.processing_done = False
                st.session_state.progress_value = 0.0
                st.session_state.processing_status = ""
                st.rerun()

    elif not st.session_state.processing_done:
        # Reset button before conversion
        _, reset_col = st.columns([8, 2])
        with reset_col:
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                _cleanup_temp_files()
                st.session_state.uploaded_file_name = None
                st.session_state.processing_done = False
                st.rerun()

else:
    # ── Empty state ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background:#13161e;
        border:2px dashed #1e2130;
        border-radius:20px;
        padding:64px 32px;
        text-align:center;
        margin-top:24px;
    ">
        <div style="font-size:56px;margin-bottom:16px">🎬</div>
        <h3 style="color:#e8eaf0;margin-bottom:8px">Drop a video to get started</h3>
        <p style="color:#6b7280;font-size:15px;max-width:420px;margin:0 auto">
            Upload a landscape (16:9) video above and convert it to a
            vertical format ready for social media — powered by YOLOv8 AI.
        </p>
        <div style="margin-top:24px">
            <span class="pill">MP4</span>
            <span class="pill">MOV</span>
            <span class="pill">AVI</span>
            <span class="pill">MKV</span>
            <span class="pill">Up to 500 MB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
