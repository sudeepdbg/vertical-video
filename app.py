import streamlit as st
import tempfile
import os
from pathlib import Path
from verticalize import VideoVerticalizer
import time

st.set_page_config(
    page_title="Video Reframer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize verticalizer once (cached)
@st.cache_resource
def get_verticalizer():
    return VideoVerticalizer(use_gpu=False)  # Set to True if GPU available

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 AI Video Reframer")
st.markdown("Convert horizontal (16:9) videos to vertical (9:16) with AI-powered subject tracking")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    sample_interval = st.slider(
        "Detection Interval (frames)",
        min_value=5,
        max_value=60,
        value=15,
        help="Lower = more accurate but slower. Higher = faster but may miss movements."
    )
    
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Higher = fewer false detections but may miss subjects"
    )
    
    st.divider()
    
    st.header("ℹ️ Information")
    st.markdown("""
    **Supported formats:** MP4, MOV, AVI, MKV
    
    **Recommended:** 
    - Videos under 2 minutes
    - Resolution: 720p or 1080p
    - Clear subject visibility
    
    **Processing time:** ~1-2x video duration
    """)

# Initialize session state
if 'input_path' not in st.session_state:
    st.session_state.input_path = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def cleanup_temp_files():
    """Clean up temporary files."""
    for path_key in ['input_path', 'output_path']:
        path = st.session_state.get(path_key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                st.session_state[path_key] = None
            except Exception as e:
                st.error(f"Error cleaning up {path}: {e}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'mov', 'avi', 'mkv'],
    help="Upload a horizontal video (16:9) for conversion to vertical format (9:16)"
)

# Handle new upload
if uploaded_file is not None:
    # Check file size (limit to 100MB)
    if uploaded_file.size > 100 * 1024 * 1024:
        st.error("File size exceeds 100MB limit. Please upload a smaller video.")
        uploaded_file = None
    elif st.session_state.uploaded_file_name != uploaded_file.name:
        # Clean up old files
        cleanup_temp_files()
        
        # Create new temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
            tmp_input.write(uploaded_file.getvalue())
            st.session_state.input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            st.session_state.output_path = tmp_output.name
        
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.processing_complete = False
        st.rerun()

# Main content
if uploaded_file is not None and st.session_state.input_path:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Original Video (16:9)")
        st.video(uploaded_file)
        st.caption(f"Size: {uploaded_file.size / (1024*1024):.1f} MB")
    
    # Process button
    if not st.session_state.processing_complete:
        if st.button("🎬 Convert to Vertical", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Get verticalizer
                verticalizer = get_verticalizer()
                
                status_text.text("Initializing AI model...")
                
                # Process video
                success = verticalizer.process_video(
                    input_path=st.session_state.input_path,
                    output_path=st.session_state.output_path,
                    sample_interval=sample_interval,
                    target_size=(1080, 1920),
                    confidence=confidence,
                    progress_callback=lambda p: progress_bar.progress(p)
                )
                
                if success and os.path.exists(st.session_state.output_path):
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    st.session_state.processing_complete = True
                    st.rerun()
                else:
                    raise Exception("Video processing failed")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                status_text.text("Processing failed")
                progress_bar.empty()
    
    # Show results
    elif st.session_state.processing_complete and os.path.exists(st.session_state.output_path):
        output_size = os.path.getsize(st.session_state.output_path)
        
        with col2:
            st.subheader("📱 Vertical Video (9:16)")
            st.video(st.session_state.output_path)
            
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Conversion Complete!</strong><br>
                Output size: {output_size / (1024*1024):.1f} MB
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            with open(st.session_state.output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Vertical Video",
                    data=f,
                    file_name=f"vertical_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )
        
        # Reset button
        if st.button("🔄 Convert Another Video", use_container_width=True):
            cleanup_temp_files()
            st.session_state.uploaded_file_name = None
            st.session_state.processing_complete = False
            st.rerun()

else:
    # Empty state
    st.info("👆 Upload a video to get started")
    
    st.markdown("""
    ### How it works:
    1. **Upload** your horizontal video (16:9 aspect ratio)
    2. **AI Detection** - YOLOv8 detects and tracks people in the video
    3. **Smart Cropping** - Video is cropped to 9:16 following the main subject
    4. **Download** - Get your vertical video ready for Instagram Reels, TikTok, etc.
    
    ### Tips for best results:
    - Ensure the main subject is clearly visible
    - Avoid very fast camera movements
    - Good lighting improves detection accuracy
    """)

# Footer
st.divider()
st.caption("Powered by YOLOv8 • Built with Streamlit")
