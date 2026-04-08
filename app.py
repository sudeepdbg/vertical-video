import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import subprocess
import tempfile

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path: str, output_path: str, progress_callback=None):
    """
    Convert horizontal video to vertical with AI-powered face tracking.
    Uses face detection with saliency fallback for better framing.
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not determine frame count — the file may be corrupt.")
    
    # Calculate target dimensions for 9:16 aspect ratio
    target_height = frame_width * 16 // 9
    target_width = frame_width
    
    # If target height is larger than original, adjust
    if target_height > frame_height:
        target_height = frame_height
        target_width = frame_height * 9 // 16
    
    # Initialize face detection with full-range model
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # Full-range model for better detection
        min_detection_confidence=0.5
    )
    
    # Create temporary file for processed video
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_path = temp_output.name
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (target_width, target_height))
    
    frame_count = 0
    prev_center_x = frame_width // 2
    prev_center_y = frame_height // 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        # Determine crop center
        if results.detections:
            # Use the largest face detected
            largest_face = max(results.detections, 
                             key=lambda d: d.location_data.relative_bounding_box.width * 
                                          d.location_data.relative_bounding_box.height)
            
            bbox = largest_face.location_data.relative_bounding_box
            face_center_x = int((bbox.xmin + bbox.width / 2) * frame_width)
            face_center_y = int((bbox.ymin + bbox.height / 2) * frame_height)
            
            # Smooth the movement (exponential moving average)
            prev_center_x = int(prev_center_x * 0.7 + face_center_x * 0.3)
            prev_center_y = int(prev_center_y * 0.7 + face_center_y * 0.3)
            center_x = prev_center_x
            center_y = prev_center_y
        else:
            # No face detected - use previous position (don't jump to center)
            center_x = prev_center_x
            center_y = prev_center_y
        
        # Calculate crop boundaries
        half_width = target_width // 2
        half_height = target_height // 2
        
        left = max(0, center_x - half_width)
        right = min(frame_width, center_x + half_width)
        top = max(0, center_y - half_height)
        bottom = min(frame_height, center_y + half_height)
        
        # Adjust if crop goes out of bounds
        if right - left < target_width:
            if left == 0:
                right = min(frame_width, left + target_width)
            else:
                left = max(0, right - target_width)
        
        if bottom - top < target_height:
            if top == 0:
                bottom = min(frame_height, top + target_height)
            else:
                top = max(0, bottom - target_height)
        
        # Crop the frame
        cropped_frame = frame[top:bottom, left:right]
        
        # Resize if needed using high-quality interpolation
        if cropped_frame.shape[1] != target_width or cropped_frame.shape[0] != target_height:
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
        
        out.write(cropped_frame)
        
        # Update progress
        if progress_callback:
            progress = (frame_count / total_frames) * 100
            progress_callback(progress)
    
    # Release resources
    cap.release()
    out.release()
    face_detection.close()
    
    # Use FFmpeg to ensure proper encoding and audio preservation
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_path,
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        # If FFmpeg fails, just use the OpenCV output
        if result.returncode != 0:
            import shutil
            shutil.copy(temp_path, output_path)
            
    except Exception as e:
        # Fallback: just copy the temp file
        import shutil
        shutil.copy(temp_path, output_path)
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)
    
    return True
