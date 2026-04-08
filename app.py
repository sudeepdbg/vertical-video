import streamlit as st
import tempfile
import os
from verticalize import process_video

st.set_page_config(
    page_title="Video Reframer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video Reframer")
st.markdown("Convert horizontal (16:9) videos to vertical (9:16) with AI-powered subject tracking")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses AI to detect and track subjects in your video, automatically cropping to vertical format.
    
    **Supported formats:** MP4, MOV, AVI, MKV
    
    **Best for:** Short videos under 2 minutes for optimal performance
    """)
    
    st.header("How it works")
    st.markdown("""
    1. Upload a horizontal video
    2. AI detects people and tracks their movement
    3. Video is cropped to vertical format following the subject
    4. Download your vertical video ready for social media
    """)

# Initialize session state for temp file paths
if 'input_path' not in st.session_state:
    st.session_state.input_path = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'mov', 'avi', 'mkv'],
    help="Upload a horizontal video (16:9) for conversion to vertical format (9:16)"
)

# Reset temp files if a new file is uploaded
if uploaded_file is not None and (st.session_state.uploaded_file_name != uploaded_file.name):
    # Clean up old temp files if they exist
    if st.session_state.input_path and os.path.exists(st.session_state.input_path):
        os.unlink(st.session_state.input_path)
    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        os.unlink(st.session_state.output_path)
    # Create new temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        st.session_state.input_path = tmp_input.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
        st.session_state.output_path = tmp_output.name
    st.session_state.uploaded_file_name = uploaded_file.name

if uploaded_file is not None and st.session_state.input_path:
    # Display uploaded video
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Video (16:9)")
        st.video(uploaded_file)
    
    # Process button
    if st.button("🎬 Convert to Vertical", type="primary"):
        with st.status("Processing video...", expanded=True) as status:
            progress_bar = st.progress(0)
            
            try:
                # Call your processing function
                process_video(st.session_state.input_path, st.session_state.output_path)
                
                progress_bar.progress(100)
                status.update(label="Processing complete!", state="complete")
                
                # Display result
                with col2:
                    st.subheader("Vertical Video (9:16)")
                    # Check if output file exists and has content
                    if os.path.exists(st.session_state.output_path) and os.path.getsize(st.session_state.output_path) > 0:
                        st.video(st.session_state.output_path)
                        # Download button
                        with open(st.session_state.output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Vertical Video",
                                data=f,
                                file_name="vertical_video.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("Output video file is empty or missing.")
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                status.update(label=f"Processing failed: {str(e)}", state="error")

    # Optional: Add a button to clear temp files manually
    if st.button("🗑️ Clear uploaded video"):
        if st.session_state.input_path and os.path.exists(st.session_state.input_path):
            os.unlink(st.session_state.input_path)
        if st.session_state.output_path and os.path.exists(st.session_state.output_path):
            os.unlink(st.session_state.output_path)
        st.session_state.input_path = None
        st.session_state.output_path = None
        st.session_state.uploaded_file_name = None
        st.rerun()
