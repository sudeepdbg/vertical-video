import streamlit as st
import tempfile
import os
from pathlib import Path
from verticalize import VideoVerticalizer
import time
from datetime import datetime

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
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        color: #004085;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
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
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None

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
        st.error("❌ File size exceeds 100MB limit. Please upload a smaller video.")
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
        st.session_state.processing_error = None
        st.rerun()

# Main content
if uploaded_file is not None and st.session_state.input_path:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Original Video (16:9)")
        st.video(uploaded_file)
        
        # Show file info
        file_size_mb = uploaded_file.size / (1024*1024)
        st.caption(f"📁 {uploaded_file.name} • {file_size_mb:.1f} MB")
    
    # Process button
    if not st.session_state.processing_complete:
        st.markdown("---")
        
        # Show settings summary
        st.markdown(f"""
        <div class="info-box">
            <strong>⚙️ Processing Settings:</strong><br>
            • Detection Interval: {sample_interval} frames<br>
            • Confidence Threshold: {confidence}<br>
            • Output Resolution: 1080×1920 (9:16)
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎬 Convert to Vertical", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            try:
                # Get verticalizer
                verticalizer = get_verticalizer()
                
                status_text.text("⏳ Initializing AI model...")
                time.sleep(0.5)  # Small delay for UX
                
                status_text.text("🔍 Analyzing video and detecting subjects...")
                
                # Process video
                success = verticalizer.process_video(
                    input_path=st.session_state.input_path,
                    output_path=st.session_state.output_path,
                    sample_interval=sample_interval,
                    target_size=(1080, 1920),
                    confidence=confidence,
                    progress_callback=lambda p: progress_bar.progress(p)
                )
                
                processing_time = time.time() - start_time
                
                if success and os.path.exists(st.session_state.output_path):
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    # Show processing stats
                    output_size = os.path.getsize(st.session_state.output_path)
                    st.success(f"""
                        ✅ **Video converted successfully!**<br>
                        ⏱️ Processing time: {processing_time:.1f} seconds<br>
                        📦 Output size: {output_size / (1024*1024):.1f} MB
                    """)
                    
                    st.session_state.processing_complete = True
                    st.rerun()
                else:
                    raise Exception("Video processing failed - output file not created")
                    
            except Exception as e:
                error_msg = str(e)
                st.session_state.processing_error = error_msg
                st.error(f"❌ Error: {error_msg}")
                status_text.text("❌ Processing failed")
                progress_bar.empty()
                
                # Show troubleshooting tips
                st.markdown("""
                **Troubleshooting tips:**
                - Ensure the video file is not corrupted
                - Try reducing the video resolution
                - Check if you have enough disk space
                - Try with a shorter video first
                """)
    
    # Show results
    elif st.session_state.processing_complete and os.path.exists(st.session_state.output_path):
        output_size = os.path.getsize(st.session_state.output_path)
        
        with col2:
            st.subheader("📱 Vertical Video (9:16)")
            st.video(st.session_state.output_path)
            
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Conversion Complete!</strong><br>
                📦 Output size: {output_size / (1024*1024):.1f} MB<br>
                🎬 Resolution: 1080×1920<br>
                📱 Ready for: TikTok, Instagram Reels, YouTube Shorts
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
        st.markdown("---")
        if st.button("🔄 Convert Another Video", use_container_width=True):
            cleanup_temp_files()
            st.session_state.uploaded_file_name = None
            st.session_state.processing_complete = False
            st.session_state.processing_error = None
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
    - ✅ Ensure the main subject is clearly visible
    - ✅ Avoid very fast camera movements
    - ✅ Good lighting improves detection accuracy
    - ✅ Keep videos under 2 minutes for faster processing
    """)

# Footer
st.divider()
st.caption("Powered by YOLOv8 • Built with Streamlit • Processing happens locally on your machine")
