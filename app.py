import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from verticalize import process_video, ProcessingError

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎬 AI Video Reframer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
    .stButton>button:disabled { background: #4a5568; transform: none; }
    .success-box { padding: 1rem; border-radius: 8px; background: #10b98120; border: 1px solid #10b981; }
    .error-box { padding: 1rem; border-radius: 8px; background: #ef444420; border: 1px solid #ef4444; }
    .video-container { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    .header-glow { text-shadow: 0 0 20px rgba(102,126,234,0.5); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<h1 class="header-glow">🎬 AI Video Reframer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#94a3b8;font-size:1.1rem">Convert horizontal (16:9) videos to vertical (9:16) with AI-powered subject tracking</p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses YOLOv8 AI to detect and track subjects in your video, "
        "automatically cropping to vertical format optimized for social media."
    )
    
    st.markdown("### 📋 Requirements")
    st.markdown("""
    - **Formats**: MP4, MOV, AVI, MKV
    - **Max Size**: 500 MB
    - **Recommended**: Videos under 2 minutes
    - **Aspect Ratio**: Works best with 16:9 source videos
    """)
    
    st.markdown("### 🔄 How It Works")
    st.markdown("""
    1. 📤 Upload your horizontal video
    2. 🎯 AI detects people & tracks movement
    3. ✂️ Video is smart-cropped to 9:16
    4. 🎵 Audio is preserved and synced
    5. ⬇️ Download your vertical video
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    
    smooth_window = st.slider(
        "🎚️ Smoothing Window",
        min_value=3, max_value=31, value=15, step=2,
        help="Higher = smoother camera movement, Lower = faster subject tracking"
    )
    
    confidence = st.slider(
        "🎯 Detection Confidence",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="Lower detects more subjects but may include false positives"
    )
    
    sample_rate = st.selectbox(
        "⚡ Processing Speed",
        options=["Fast (every 2 sec)", "Balanced (every 1 sec)", "Precise (every frame)"],
        index=1,
        help="Faster = quicker processing, Precise = better tracking for fast movement"
    )
    
    # Map selection to sample interval
    sample_map = {"Fast (every 2 sec)": 2, "Balanced (every 1 sec)": 1, "Precise (every frame)": 0}
    sample_interval = sample_map[sample_rate]

# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "input_path": None,
        "output_path": None,
        "uploaded_file_name": None,
        "processing_done": False,
        "error_message": None,
        "temp_files": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def cleanup_temp_files():
    """Clean up all temporary files created during processing."""
    for path in st.session_state.get("temp_files", []):
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
    st.session_state["temp_files"] = []
    st.session_state["input_path"] = None
    st.session_state["output_path"] = None

init_session_state()

# ---------------------------------------------------------------------------
# File Upload Section
# ---------------------------------------------------------------------------
ALLOWED_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_MB = 500

st.markdown("### 📤 Upload Your Video")

uploaded_file = st.file_uploader(
    "Choose a horizontal video file",
    type=ALLOWED_TYPES,
    help=f"Supported: {', '.join(ALLOWED_TYPES).upper()} | Max: {MAX_FILE_MB} MB",
    label_visibility="collapsed"
)

# Handle new file upload
if uploaded_file is not None:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 ** 2)
    
    if file_size_mb > MAX_FILE_MB:
        st.error(f"❌ File is {file_size_mb:.1f} MB — please upload a file under {MAX_FILE_MB} MB.")
        uploaded_file = None
        st.stop()
    
    # New file detected - reset state and create temp files
    if st.session_state.uploaded_file_name != uploaded_file.name:
        cleanup_temp_files()
        st.session_state.processing_done = False
        st.session_state.error_message = None
        
        # Create input temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_in:
            tmp_in.write(uploaded_file.getvalue())
            st.session_state.input_path = tmp_in.name
            st.session_state.temp_files.append(tmp_in.name)
        
        # Create output path (file will be created by processing)
        output_fd, output_path = tempfile.mkstemp(suffix="_vertical.mp4")
        os.close(output_fd)
        st.session_state.output_path = output_path
        st.session_state.temp_files.append(output_path)
        
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

# ---------------------------------------------------------------------------
# Main Processing Area
# ---------------------------------------------------------------------------
if uploaded_file is not None and st.session_state.input_path:
    
    # Preview Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🔲 Original Video (16:9)")
        st.video(uploaded_file)
    
    with col2:
        st.markdown("#### 📱 Target Format (9:16)")
        # Show placeholder for vertical format
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea20, #764ba220);
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            color: #94a3b8;
        ">
            <div style="font-size: 3rem;">📱</div>
            <p><strong>Vertical Video</strong></p>
            <p style="font-size: 0.9rem;">1080 × 1920<br>9:16 Aspect Ratio</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Video info
        st.markdown("##### 📊 File Info")
        st.markdown(f"""
        - **Name**: {uploaded_file.name}
        - **Size**: {file_size_mb:.1f} MB
        - **Format**: {uploaded_file.name.split('.')[-1].upper()}
        """)
    
    st.divider()
    
    # Convert Button
    if not st.session_state.processing_done:
        if st.button("🎬 Convert to Vertical", type="primary", use_container_width=True):
            st.session_state.error_message = None
            
            with st.status("🔄 Processing your video...", expanded=True) as status:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value: float, message: str = ""):
                    progress_bar.progress(min(value, 1.0))
                    if message:
                        status_text.markdown(f"*{message}*")
                
                try:
                    # Determine sample interval based on FPS and user selection
                    fps_estimate = 30  # Default, will be detected in processing
                    if sample_interval == 0:
                        detection_interval = 1  # Every frame
                    elif sample_interval == 1:
                        detection_interval = max(1, int(fps_estimate))
                    else:
                        detection_interval = max(1, int(fps_estimate * 2))
                    
                    update_progress(0.05, "🔍 Initializing AI model...")
                    
                    process_video(
                        input_path=st.session_state.input_path,
                        output_path=st.session_state.output_path,
                        confidence=confidence,
                        smooth_window=smooth_window,
                        sample_interval=detection_interval,
                        progress_callback=update_progress,
                    )
                    
                    update_progress(1.0, "✅ Processing complete!")
                    status.update(label="✨ Success! Your vertical video is ready.", state="complete")
                    st.session_state.processing_done = True
                    
                except ProcessingError as e:
                    status.update(label=f"❌ Error: {e}", state="error")
                    st.session_state.error_message = str(e)
                    st.error(f"Processing failed: {e}")
                    
                except Exception as e:
                    status.update(label=f"❌ Unexpected error", state="error")
                    st.session_state.error_message = f"An unexpected error occurred: {type(e).__name__}"
                    st.error(f"Unexpected error: {e}")
                    import traceback
                    st.exception(e)
    
    # Results Section (shown after successful processing)
    if st.session_state.processing_done:
        out_path = st.session_state.output_path
        
        if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:  # >1KB
            st.success("✨ Your vertical video is ready!")
            
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                st.markdown("#### 📱 Vertical Video (9:16)")
                st.video(out_path)
            
            with col_result2:
                st.markdown("##### ⬇️ Download")
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Vertical Video",
                        data=f,
                        file_name=f"vertical_{Path(uploaded_file.name).stem}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        type="primary"
                    )
                
                st.markdown("##### 📏 Output Specs")
                st.markdown("""
                - **Resolution**: 1080 × 1920
                - **Aspect Ratio**: 9:16 (Vertical)
                - **Format**: MP4 (H.264 + AAC)
                - **Audio**: Preserved from original
                """)
        else:
            st.error("❌ Output file is missing or empty. Please try again with a different video.")
    
    # Error Display
    if st.session_state.error_message:
        st.markdown(f'<div class="error-box">⚠️ {st.session_state.error_message}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Action Buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔄 Convert Another Video", use_container_width=True):
            cleanup_temp_files()
            st.session_state.uploaded_file_name = None
            st.session_state.processing_done = False
            st.session_state.error_message = None
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear Current Video", use_container_width=True):
            cleanup_temp_files()
            st.session_state.uploaded_file_name = None
            st.session_state.processing_done = False
            st.session_state.error_message = None
            st.rerun()
    
    with col_btn3:
        if st.button("🧹 Clear All Temp Files", use_container_width=True):
            cleanup_temp_files()
            st.success("✅ Temporary files cleared!")

# ---------------------------------------------------------------------------
# Footer / Empty State
# ---------------------------------------------------------------------------
else:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 16px;
        border: 1px dashed rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
        <h3 style="color: white; margin-bottom: 0.5rem;">Ready to Create Vertical Videos?</h3>
        <p style="color: #94a3b8; margin-bottom: 1.5rem;">
            Upload a horizontal video above to get started with AI-powered reframing.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="background: #667eea30; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem;">🎯 AI Subject Tracking</span>
            <span style="background: #667eea30; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem;">✂️ Smart Cropping</span>
            <span style="background: #667eea30; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem;">🎵 Audio Preserved</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cleanup on Script Rerun
# ---------------------------------------------------------------------------
# Note: Streamlit automatically handles cleanup on session end,
# but we can add a final cleanup for orphaned temp files if needed.
