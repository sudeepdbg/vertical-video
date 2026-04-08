import streamlit as st
import tempfile
import os
from verticalize import process_video

st.set_page_config(
    page_title="Video Reframer",
    page_icon="🎬",
    layout="wide",
)

# ── Light theme enforcement (no dark mode) ──────────────────────────────────
st.markdown("""
    <style>
    /* Force light background throughout */
    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }
    /* Clean card-style containers */
    .video-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #6366f1;
    }
    /* Button styling */
    .stButton > button {
        background-color: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #4f46e5;
    }
    /* Subtle header styling */
    h1, h2, h3 {
        color: #111827;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 AI Video Reframer")
st.markdown(
    "Convert horizontal (16:9) videos to vertical (9:16) with AI-powered subject tracking"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app uses AI to detect and track subjects in your video,
        automatically cropping to vertical format.

        **Supported formats:** MP4, MOV, AVI, MKV  
        **Max file size:** 500 MB  
        **Best for:** Short videos under 2 minutes for optimal performance
        """
    )

    st.header("🔄 How it works")
    st.markdown(
        """
        1. Upload a horizontal video
        2. AI detects people and tracks their movement
        3. Video is cropped to vertical format following the subject
        4. Download your vertical video ready for social media
        """
    )

    st.header("⚙️ Advanced settings")
    smooth_window = st.slider(
        "Smoothing window (frames)",
        min_value=3,
        max_value=31,
        value=15,
        step=2,
        help="Larger values produce steadier pans but slower reactions to movement.",
    )
    confidence = st.slider(
        "Detection confidence",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Lower values detect more people but may cause false positives.",
    )


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "input_path": None,
        "output_path": None,
        "uploaded_file_name": None,
        "processing_done": False,
        "progress_value": 0.0,
        "processing_status": "",
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


_init_state()

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
ALLOWED_TYPES = ["mp4", "mov", "avi", "mkv"]
MAX_FILE_MB = 500

uploaded_file = st.file_uploader(
    "📁 Choose a video file",
    type=ALLOWED_TYPES,
    help=f"Upload a horizontal video (16:9) for conversion to vertical format (9:16). Max {MAX_FILE_MB} MB.",
)

# Detect a new upload and (re)create temp files
if uploaded_file is not None:
    file_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
    if file_mb > MAX_FILE_MB:
        st.error(f"⚠️ File is {file_mb:.1f} MB — please upload a file under {MAX_FILE_MB} MB.")
        uploaded_file = None

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
    os.unlink(tmp_out_path)
    st.session_state.output_path = tmp_out_path
    st.session_state.uploaded_file_name = uploaded_file.name

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
if uploaded_file is not None and st.session_state.input_path:
    
    # ── Video preview section (always visible when file uploaded) ─────────
    st.markdown('<div class="video-card">', unsafe_allow_html=True)
    
    # Parallel playback: Original | Vertical (side-by-side)
    col_orig, col_vert = st.columns(2)
    
    with col_orig:
        st.markdown("##### 📺 Original Video (16:9)")
        st.video(uploaded_file)
    
    with col_vert:
        st.markdown("##### 📱 Vertical Video (9:16)")
        if st.session_state.processing_done and st.session_state.output_path:
            out_path = st.session_state.output_path
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                st.video(out_path)
            else:
                st.info("⏳ Processing… vertical video will appear here")
        else:
            st.info("✨ Click 'Convert' to generate your vertical video")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Convert button & progress section ────────────────────────────────
    if not st.session_state.processing_done:
        if st.button("🎬 Convert to Vertical", type="primary", use_container_width=True):
            st.session_state.processing_done = False
            st.session_state.progress_value = 0.0
            
            # Progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.info("🔍 Initializing AI detection…")
            
            try:
                def update_progress(value: float):
                    progress_bar.progress(min(value, 1.0))
                    st.session_state.progress_value = value
                    if value < 0.3:
                        status_text.info("🔍 Detecting subjects…")
                    elif value < 0.9:
                        status_text.info(f"🎥 Rendering frames… {int(value * 100)}%")
                    else:
                        status_text.info("🎵 Finalizing audio…")
                
                process_video(
                    st.session_state.input_path,
                    st.session_state.output_path,
                    confidence=confidence,
                    smooth_window=smooth_window,
                    progress_callback=update_progress,
                )
                
                progress_bar.progress(1.0)
                status_text.success("✅ Conversion complete!")
                st.session_state.processing_done = True
                st.session_state.progress_value = 1.0
                
                # Auto-rerun to show the result
                st.rerun()
                
            except Exception as exc:
                status_text.error(f"❌ Error: {exc}")
                st.error(f"Processing failed: {exc}")
    
    # ── Download section (shown after successful processing) ─────────────
    if st.session_state.processing_done:
        out_path = st.session_state.output_path
        if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            st.success("🎉 Your vertical video is ready!")
            
            # Download button with file info
            file_size_mb = os.path.getsize(out_path) / (1024 ** 2)
            st.caption(f"📦 Output size: {file_size_mb:.1f} MB")
            
            with open(out_path, "rb") as f:
                st.download_button(
                    label="📥 Download Vertical Video",
                    data=f,
                    file_name=f"vertical_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True,
                )
    
    # ── Clear/reset button ───────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear & Start Over", type="secondary"):
        _cleanup_temp_files()
        st.session_state.uploaded_file_name = None
        st.session_state.processing_done = False
        st.session_state.progress_value = 0.0
        st.session_state.processing_status = ""
        st.rerun()

else:
    # ── Empty state (no file uploaded) ───────────────────────────────────
    st.info("👆 Upload a video above to get started")
