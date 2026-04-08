import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoVerticalizer:
    """AI-powered video reframer with subject tracking."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', use_gpu: bool = False):
        """Initialize with YOLO model."""
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model once for reuse."""
        try:
            device = 0 if self.use_gpu else 'cpu'
            self.model = YOLO(self.model_path)
            self.model.to(device)
            logger.info(f"Model loaded on {'GPU' if self.use_gpu else 'CPU'}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def smooth_centers(self, centers: List[Tuple[int, int]], window: int = 7) -> List[Tuple[int, int]]:
        """Apply moving average to subject center coordinates."""
        if not centers:
            return []
        
        smoothed = []
        for i in range(len(centers)):
            start = max(0, i - window // 2)
            end = min(len(centers), i + window // 2 + 1)
            window_centers = centers[start:end]
            
            avg_x = int(np.mean([c[0] for c in window_centers]))
            avg_y = int(np.mean([c[1] for c in window_centers]))
            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def detect_largest_person(self, frame: np.ndarray, confidence: float = 0.5) -> Optional[Tuple[int, int]]:
        """Return (x, y) center of the largest person in frame."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        results = self.model(frame, verbose=False, conf=confidence)[0]
        persons = []
        
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                
                if conf_score > confidence:
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    persons.append((x1, y1, x2, y2, area))
        
        if not persons:
            return None
        
        # Get largest person by area
        largest = max(persons, key=lambda b: b[4])
        center_x = (largest[0] + largest[2]) // 2
        center_y = (largest[1] + largest[3]) // 2
        
        return (center_x, center_y)
    
    def interpolate_centers(self, detected_centers: List[Tuple[int, int]], 
                           detected_indices: List[int], 
                           total_frames: int) -> List[Tuple[int, int]]:
        """Linearly interpolate centers for frames without detection."""
        if not detected_centers:
            return [(0, 0)] * total_frames
        
        if len(detected_centers) == 1:
            return [detected_centers[0]] * total_frames
        
        all_centers = []
        center_idx = 0
        
        for i in range(total_frames):
            if i <= detected_indices[0]:
                all_centers.append(detected_centers[0])
            elif i >= detected_indices[-1]:
                all_centers.append(detected_centers[-1])
            else:
                # Find the segment this frame belongs to
                while center_idx < len(detected_indices) - 1:
                    if detected_indices[center_idx] <= i <= detected_indices[center_idx + 1]:
                        start_idx = detected_indices[center_idx]
                        end_idx = detected_indices[center_idx + 1]
                        start_center = detected_centers[center_idx]
                        end_center = detected_centers[center_idx + 1]
                        
                        ratio = (i - start_idx) / (end_idx - start_idx)
                        cx = int(start_center[0] + ratio * (end_center[0] - start_center[0]))
                        cy = int(start_center[1] + ratio * (end_center[1] - start_center[1]))
                        all_centers.append((cx, cy))
                        break
                    center_idx += 1
                else:
                    all_centers.append(detected_centers[-1])
        
        return all_centers
    
    def process_video(self, input_path: str, output_path: str, 
                     sample_interval: int = 15, 
                     target_size: Tuple[int, int] = (1080, 1920),
                     confidence: float = 0.5,
                     progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Convert horizontal video to vertical format with AI tracking.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            sample_interval: Frame sampling interval for detection
            target_size: Target dimensions (width, height)
            confidence: Detection confidence threshold
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
        
        Returns:
            bool: True if successful, False otherwise
        """
        cap = None
        temp_video_path = None
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames == 0 or fps == 0:
                raise ValueError("Invalid video properties")
            
            logger.info(f"Video: {orig_w}x{orig_h}, {fps}fps, {total_frames} frames")
            
            # Phase 1: Detect subject centers on sampled frames
            logger.info("Phase 1: Detecting subjects...")
            detected_centers = []
            detected_indices = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    center = self.detect_largest_person(frame, confidence)
                    if center:
                        detected_centers.append(center)
                        detected_indices.append(frame_idx)
                
                frame_idx += 1
                if progress_callback:
                    progress_callback(0.1 * (frame_idx / total_frames))
            
            cap.release()
            
            # Handle no detection case
            if not detected_centers:
                logger.warning("No persons detected, using center of frame")
                detected_centers = [(orig_w // 2, orig_h // 2)]
                detected_indices = [0]
            
            logger.info(f"Detected {len(detected_centers)} frames with subjects")
            
            # Phase 2: Interpolate and smooth
            logger.info("Phase 2: Interpolating and smoothing...")
            all_centers = self.interpolate_centers(detected_centers, detected_indices, total_frames)
            all_centers = self.smooth_centers(all_centers, window=7)
            
            # Phase 3: Calculate crop dimensions
            target_aspect = target_size[0] / target_size[1]
            crop_h = orig_h
            crop_w = int(crop_h * target_aspect)
            
            if crop_w > orig_w:
                crop_w = orig_w
                crop_h = int(crop_w / target_aspect)
            
            logger.info(f"Crop size: {crop_w}x{crop_h}")
            
            # Phase 4: Create cropped video
            logger.info("Phase 3: Creating vertical video...")
            temp_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, target_size)
            
            if not out.isOpened():
                raise ValueError("Failed to create VideoWriter")
            
            cap = cv2.VideoCapture(input_path)
            frame_num = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                center_x, center_y = all_centers[frame_num]
                
                # Calculate crop boundaries
                left = center_x - crop_w // 2
                top = center_y - crop_h // 2
                
                # Ensure within bounds
                left = max(0, min(left, orig_w - crop_w))
                top = max(0, min(top, orig_h - crop_h))
                right = left + crop_w
                bottom = top + crop_h
                
                # Crop and resize
                cropped = frame[top:bottom, left:right]
                resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
                out.write(resized)
                
                frame_num += 1
                if progress_callback:
                    progress_callback(0.5 + 0.4 * (frame_num / total_frames))
            
            cap.release()
            out.release()
            
            # Phase 5: Add audio using ffmpeg (faster than moviepy)
            logger.info("Phase 4: Adding audio...")
            if progress_callback:
                progress_callback(0.95)
            
            self._add_audio_ffmpeg(temp_video_path, input_path, output_path)
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"✅ Saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return False
        
        finally:
            # Cleanup
            if cap and cap.isOpened():
                cap.release()
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
    
    def _add_audio_ffmpeg(self, video_path: str, original_path: str, output_path: str):
        """Add audio from original video using ffmpeg."""
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', original_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg error: {e.stderr}")
            # Fallback: copy video without audio
            import shutil
            shutil.copy(video_path, output_path)
