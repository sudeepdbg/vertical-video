import streamlit as st
import tempfile
import os
from verticalize import process_video

st.set_page_config(
    page_title="Video Reframer",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AI Video Reframer")
st.markdown(
    "Convert horizontal (16:9) videos to vertical (9:16) with AI-powered subject tracking"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app uses AI to detect and track subjects in your video,
        automatically cropping to vertical format.

        **Supported formats:** MP4, MOV, AVI, MKV  
        **Max file size:** 500 MB  
        **Best for:** Short videos under 2 minutes for optimal performance
        """
    )

    st.header("How it works")
    st.markdown(
        """
        1. Upload a horizontal video
        2. AI detects people and tracks their movement
        3. Video is cropped to vertical format following the subject
        4. Download your vertical video ready for social media
        """
    )

    st.header("Advanced settings")
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
    "Choose a video file",
    type=ALLOWED_TYPES,
    help=f"Upload a horizontal video (16:9) for conversion to vertical format (9:16). Max {MAX_FILE_MB} MB.",
)

# Detect a new upload and (re)create temp files
if uploaded_file is not None:
    file_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
    if file_mb > MAX_FILE_MB:
        st.error(f"File is {file_mb:.1f} MB — please upload a file under {MAX_FILE_MB} MB.")
        uploaded_file = None

if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    _cleanup_temp_files()
    st.session_state.processing_done = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        st.session_state.input_path = tmp_in.name

    # Only create the output path; do NOT pre-create the file so moviepy can write freely
    tmp_out_fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_out_fd)
    os.unlink(tmp_out_path)          # Remove the empty file; moviepy will create it
    st.session_state.output_path = tmp_out_path

    st.session_state.uploaded_file_name = uploaded_file.name

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
if uploaded_file is not None and st.session_state.input_path:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Video (16:9)")
        st.video(uploaded_file)

    # ----- Convert button -----
    if st.button("🎬 Convert to Vertical", type="primary"):
        st.session_state.processing_done = False

        with st.status("Processing video…", expanded=True) as status:
            st.write("🔍 Detecting subjects…")
            progress_bar = st.progress(0.0)

            def update_progress(value: float):
                progress_bar.progress(value)
                if value < 0.9:
                    status.update(label=f"🎥 Rendering frames… {value*100:.0f}%")
                else:
                    status.update(label="🎵 Muxing audio…")

            try:
                process_video(
                    st.session_state.input_path,
                    st.session_state.output_path,
                    confidence=confidence,
                    smooth_window=smooth_window,
                    progress_callback=update_progress,
                )
                progress_bar.progress(1.0)
                status.update(label="✅ Processing complete!", state="complete")
                st.session_state.processing_done = True

            except Exception as exc:
                status.update(label=f"❌ Processing failed: {exc}", state="error")
                st.error(f"Error: {exc}")

    # ----- Show result (persists after button click) -----
    if st.session_state.processing_done:
        out_path = st.session_state.output_path
        if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with col2:
                st.subheader("Vertical Video (9:16)")
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Vertical Video",
                        data=f,
                        file_name="vertical_video.mp4",
                        mime="video/mp4",
                    )
        else:
            st.error("Output video file is empty or missing — please try again.")

    # ----- Clear button -----
    if st.button("🗑️ Clear uploaded video"):
        _cleanup_temp_files()
        st.session_state.uploaded_file_name = None
        st.session_state.processing_done = False
        st.rerun()
