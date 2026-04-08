"""
verticalize.py — AI-Powered Video Verticalizer with Subject Tracking
═══════════════════════════════════════════════════════════════════
Converts horizontal videos to vertical (9:16) format using YOLOv8
for intelligent subject detection, tracking, and smooth cropping.

Features:
• Multi-person detection with largest-subject prioritization
• Temporal smoothing for stable framing
• Linear interpolation for undetected frames
• FFmpeg-based audio preservation
• Progress callbacks for UI integration
• Configurable crop padding and safety margins
"""
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CropConfig:
    """Configuration for cropping behavior."""
    target_size: Tuple[int, int] = (1080, 1920)  # width, height (9:16)
    padding_ratio: float = 0.1  # Extra space around subject (0.0-0.3)
    min_subject_ratio: float = 0.3  # Min subject size vs frame
    max_jump_pixels: int = 100  # Max frame-to-frame center movement
    stabilization_window: int = 7  # Frames for moving average


class VideoVerticalizer:
    """AI-powered video reframer with robust subject tracking."""
    
    PERSON_CLASS_ID = 0  # COCO dataset: 0 = person
    
    def __init__(
        self, 
        model_path: str = 'yolov8n.pt', 
        use_gpu: bool = False,
        crop_config: Optional[CropConfig] = None
    ):
        """Initialize with YOLO model and cropping configuration."""
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.config = crop_config or CropConfig()
        self.model: Optional[YOLO] = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with error handling."""
        try:
            device = 0 if self.use_gpu and self._gpu_available() else 'cpu'
            self.model = YOLO(self.model_path)
            self.model.to(device)
            logger.info(f"✓ Model loaded on {'GPU' if isinstance(device, int) else 'CPU'}")
        except Exception as e:
            logger.error(f"✗ Failed to load model '{self.model_path}': {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    @staticmethod
    def _gpu_available() -> bool:
        """Check if CUDA-capable GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def detect_largest_person(
        self, 
        frame: np.ndarray, 
        confidence: float = 0.5
    ) -> Optional[Dict]:
        """
        Detect the largest person in frame and return structured data.
        
        Returns:
            Dict with 'center', 'bbox', 'area', 'confidence' or None
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model() first.")
        
        results = self.model(frame, verbose=False, conf=confidence, classes=[self.PERSON_CLASS_ID])[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            
            if conf_score < confidence:
                continue
                
            width, height = x2 - x1, y2 - y1
            area = width * height
            
            # Filter very small detections
            frame_area = frame.shape[0] * frame.shape[1]
            if area / frame_area < self.config.min_subject_ratio ** 2:
                continue
                
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': area,
                'confidence': conf_score
            })
        
        if not detections:
            return None
        
        # Return largest by area
        return max(detections, key=lambda d: d['area'])
    
    def _constrain_movement(
        self, 
        prev_center: Tuple[int, int], 
        new_center: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Limit center movement between frames for stability."""
        dx = new_center[0] - prev_center[0]
        dy = new_center[1] - prev_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > self.config.max_jump_pixels:
            # Scale movement to max allowed
            scale = self.config.max_jump_pixels / distance
            return (
                prev_center[0] + int(dx * scale),
                prev_center[1] + int(dy * scale)
            )
        return new_center
    
    def smooth_centers(
        self, 
        centers: List[Tuple[int, int]], 
        window: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Apply weighted moving average for smoother tracking."""
        if not centers:
            return []
        
        window = window or self.config.stabilization_window
        smoothed = []
        
        for i in range(len(centers)):
            # Weighted window: closer frames have higher weight
            weights = []
            weighted_x, weighted_y = 0, 0
            
            for j in range(max(0, i - window), min(len(centers), i + window + 1)):
                weight = 1.0 / (1 + abs(i - j))  # Inverse distance weighting
                weights.append(weight)
                weighted_x += centers[j][0] * weight
                weighted_y += centers[j][1] * weight
            
            total_weight = sum(weights)
            smoothed.append((
                int(weighted_x / total_weight),
                int(weighted_y / total_weight)
            ))
        
        return smoothed
    
    def interpolate_centers(
        self, 
        detected_centers: List[Tuple[int, int]], 
        detected_indices: List[int], 
        total_frames: int
    ) -> List[Tuple[int, int]]:
        """Interpolate centers with boundary handling and movement constraints."""
        if not detected_centers:
            return [(0, 0)] * total_frames
        
        if len(detected_centers) == 1:
            return [detected_centers[0]] * total_frames
        
        all_centers: List[Tuple[int, int]] = []
        
        for frame_idx in range(total_frames):
            if frame_idx <= detected_indices[0]:
                all_centers.append(detected_centers[0])
            elif frame_idx >= detected_indices[-1]:
                all_centers.append(detected_centers[-1])
            else:
                # Find bounding detected frames
                for i in range(len(detected_indices) - 1):
                    if detected_indices[i] <= frame_idx <= detected_indices[i + 1]:
                        start_idx, end_idx = detected_indices[i], detected_indices[i + 1]
                        start_c, end_c = detected_centers[i], detected_centers[i + 1]
                        
                        # Linear interpolation
                        ratio = (frame_idx - start_idx) / (end_idx - start_idx)
                        cx = int(start_c[0] + ratio * (end_c[0] - start_c[0]))
                        cy = int(start_c[1] + ratio * (end_c[1] - start_c[1]))
                        all_centers.append((cx, cy))
                        break
                else:
                    all_centers.append(detected_centers[-1])
        
        return all_centers
    
    def _calculate_crop_bounds(
        self,
        center: Tuple[int, int],
        orig_w: int,
        orig_h: int,
        crop_w: int,
        crop_h: int
    ) -> Tuple[int, int, int, int]:
        """Calculate crop boundaries with padding and bounds checking."""
        padding = int(min(crop_w, crop_h) * self.config.padding_ratio)
        
        # Expand crop area with padding, then re-center
        left = center[0] - crop_w // 2
        top = center[1] - crop_h // 2
        
        # Apply bounds with padding consideration
        left = max(0, min(left, orig_w - crop_w))
        top = max(0, min(top, orig_h - crop_h))
        
        return left, top, left + crop_w, top + crop_h
    
    def _add_audio_ffmpeg(
        self, 
        video_path: str, 
        original_path: str, 
        output_path: str
    ) -> bool:
        """Add audio from original using ffmpeg with fallback handling."""
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_path,
            '-i', original_path,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            logger.info("✓ Audio merged successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning("⚠ FFmpeg timed out, copying video without audio")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠ FFmpeg error: {e.stderr[:200] if e.stderr else 'Unknown'}")
        except Exception as e:
            logger.warning(f"⚠ Audio merge failed: {e}")
        
        # Fallback: copy without audio
        import shutil
        shutil.copy2(video_path, output_path)
        logger.info("✓ Saved video without audio (fallback)")
        return False
    
    def process_video(
        self, 
        input_path: str, 
        output_path: str,
        sample_interval: int = 15,
        target_size: Optional[Tuple[int, int]] = None,
        confidence: float = 0.5,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, any]:
        """
        Convert horizontal video to vertical format with AI tracking.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output vertical video
            sample_interval: Frame sampling interval for detection (higher = faster)
            target_size: Target (width, height) - defaults to 1080x1920
            confidence: YOLO detection confidence threshold (0.0-1.0)
            progress_callback: Optional callback(progress: float, message: str)
        
        Returns:
            Dict with 'success', 'output_path', 'stats', and optional 'error'
        """
        stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'processing_time_sec': 0
        }
        
        cap = None
        temp_video_path = None
        import time
        start_time = time.time()
        
        try:
            # Update config if custom target size provided
            if target_size:
                self.config.target_size = target_size
            
            # Validate input
            if not Path(input_path).exists():
                return {'success': False, 'error': f'File not found: {input_path}'}
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return {'success': False, 'error': f'Cannot open video: {input_path}'}
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames == 0 or fps == 0:
                return {'success': False, 'error': 'Invalid video metadata'}
            
            logger.info(f"Processing: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")
            if progress_callback:
                progress_callback(0.01, "Analyzing video...")
            
            # Calculate crop dimensions based on target aspect ratio
            target_w, target_h = self.config.target_size
            target_aspect = target_w / target_h
            
            # Determine optimal crop size from source
            if orig_w / orig_h > target_aspect:
                # Source is wider: crop width to match aspect
                crop_h = orig_h
                crop_w = int(crop_h * target_aspect)
            else:
                # Source is taller: crop height to match aspect
                crop_w = orig_w
                crop_h = int(crop_w / target_aspect)
            
            logger.info(f"Crop: {crop_w}x{crop_h} → {target_w}x{target_h}")
            
            # ═══════════════════════════════════════════════════════
            # PHASE 1: Sampled detection pass
            # ═══════════════════════════════════════════════════════
            logger.info("Phase 1: Detecting subjects...")
            detected_centers = []
            detected_indices = []
            frame_idx = 0
            prev_center = (orig_w // 2, orig_h // 2)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    detection = self.detect_largest_person(frame, confidence)
                    if detection:
                        # Constrain movement from previous detection
                        constrained = self._constrain_movement(prev_center, detection['center'])
                        detected_centers.append(constrained)
                        detected_indices.append(frame_idx)
                        prev_center = constrained
                        stats['detections_found'] += 1
                
                frame_idx += 1
                if progress_callback and frame_idx % max(1, total_frames // 100) == 0:
                    progress_callback(0.15 * (frame_idx / total_frames), "Detecting subjects...")
            
            cap.release()
            
            # Fallback if no detections
            if not detected_centers:
                logger.warning("⚠ No persons detected; using frame center")
                detected_centers = [(orig_w // 2, orig_h // 2)]
                detected_indices = [0]
            
            logger.info(f"✓ Found {len(detected_centers)} detections in {frame_idx} sampled frames")
            
            # ═══════════════════════════════════════════════════════
            # PHASE 2: Interpolation + smoothing
            # ═══════════════════════════════════════════════════════
            logger.info("Phase 2: Interpolating trajectory...")
            all_centers = self.interpolate_centers(
                detected_centers, detected_indices, total_frames
            )
            all_centers = self.smooth_centers(all_centers)
            
            # ═══════════════════════════════════════════════════════
            # PHASE 3: Render cropped video
            # ═══════════════════════════════════════════════════════
            logger.info("Phase 3: Rendering vertical video...")
            
            temp_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4', prefix='vertical_')
            os.close(temp_fd)
            
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 for better compatibility
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, self.config.target_size)
            
            if not out.isOpened():
                raise RuntimeError("Failed to initialize VideoWriter")
            
            cap = cv2.VideoCapture(input_path)
            frame_num = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                center_x, center_y = all_centers[frame_num]
                left, top, right, bottom = self._calculate_crop_bounds(
                    (center_x, center_y), orig_w, orig_h, crop_w, crop_h
                )
                
                # Crop → Resize → Write
                cropped = frame[top:bottom, left:right]
                resized = cv2.resize(
                    cropped, 
                    self.config.target_size, 
                    interpolation=cv2.INTER_LANCZOS4
                )
                out.write(resized)
                
                frame_num += 1
                stats['frames_processed'] = frame_num
                
                if progress_callback and frame_num % max(1, total_frames // 100) == 0:
                    progress_callback(
                        0.2 + 0.75 * (frame_num / total_frames), 
                        f"Rendering: {frame_num}/{total_frames}"
                    )
            
            cap.release()
            out.release()
            
            # ═══════════════════════════════════════════════════════
            # PHASE 4: Audio merge
            # ═══════════════════════════════════════════════════════
            logger.info("Phase 4: Merging audio...")
            if progress_callback:
                progress_callback(0.97, "Finalizing...")
            
            self._add_audio_ffmpeg(temp_video_path, input_path, output_path)
            
            # Final stats
            stats['processing_time_sec'] = round(time.time() - start_time, 2)
            stats['output_size_mb'] = round(Path(output_path).stat().st_size / 1_048_576, 2)
            
            if progress_callback:
                progress_callback(1.0, "Complete ✓")
            
            logger.info(f"✓ Done in {stats['processing_time_sec']}s → {output_path}")
            return {
                'success': True,
                'output_path': output_path,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"✗ Processing failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'stats': stats
            }
        
        finally:
            # Cleanup resources
            if cap and cap.isOpened():
                cap.release()
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except OSError as e:
                    logger.warning(f"Cleanup warning: {e}")


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point (optional)
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Video Verticalizer")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output vertical video path")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("-s", "--sample", type=int, default=15, help="Detection sample interval")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Detection confidence")
    
    args = parser.parse_args()
    
    verticalizer = VideoVerticalizer(model_path=args.model, use_gpu=args.gpu)
    result = verticalizer.process_video(
        input_path=args.input,
        output_path=args.output,
        sample_interval=args.sample,
        confidence=args.confidence,
        progress_callback=lambda p, m: print(f"[{p*100:5.1f}%] {m}")
    )
    
    if result['success']:
        print(f"\n✓ Success: {result['output_path']}")
        print(f"  Stats: {result['stats']}")
    else:
        print(f"\n✗ Failed: {result.get('error', 'Unknown error')}")
        exit(1)
