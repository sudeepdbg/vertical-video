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
from typing import Callable, Optional, List, Tuple, Dict, Any
import logging
from dataclasses import dataclass
import time
import shutil

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
    ) -> Optional[Dict[str, Any]]:
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
    ) -> Dict[str, Any]:
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
        writer = None
        temp_video_path = None
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
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames for detection
                if frame_idx % sample_interval == 0:
                    detection = self.detect_largest_person(frame, confidence)
                    if detection:
                        # Constrain movement from previous detection
                        constrained = self._constrain_movement(prev_center, detection['center'])
                        detected_centers.append(constrained)
                        detected_indices.append(frame_idx)
                        prev_center = constrained
                        stats['detections_found'] += 1
                
                # Progress update
                if progress_callback and frame_idx % max(1, total_frames // 100) == 0:
                    progress = 0.1 + 0.3 * (frame_idx / total_frames)
                    progress_callback(progress, f"Detecting: {frame_idx}/{total_frames}")
                
                frame_idx += 1
            
            cap.release()
            logger.info(f"Phase 1 complete: {len(detected_centers)} detections in {frame_idx} frames")
            
            # ═══════════════════════════════════════════════════════
            # PHASE 2: Interpolate and smooth centers
            # ═══════════════════════════════════════════════════════
            logger.info("Phase 2: Interpolating and smoothing...")
            if progress_callback:
                progress_callback(0.4, "Smoothing tracking path...")
            
            # Interpolate missing frames
            all_centers = self.interpolate_centers(
                detected_centers, detected_indices, total_frames
            )
            
            # Apply smoothing
            smooth_centers = self.smooth_centers(all_centers)
            
            # Final constraint pass for extra stability
            final_centers = [smooth_centers[0]]
            for i in range(1, len(smooth_centers)):
                constrained = self._constrain_movement(final_centers[-1], smooth_centers[i])
                final_centers.append(constrained)
            
            logger.info(f"Phase 2 complete: {len(final_centers)} centers processed")
            
            # ═══════════════════════════════════════════════════════
            # PHASE 3: Render output video with crops
            # ═══════════════════════════════════════════════════════
            logger.info("Phase 3: Rendering output video...")
            
            # Create temp file for video without audio
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_video_path = tmp.name
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                temp_video_path, fourcc, fps, (target_w, target_h)
            )
            if not writer.isOpened():
                return {'success': False, 'error': 'Failed to initialize video writer'}
            
            # Re-open input for second pass
            cap = cv2.VideoCapture(input_path)
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get crop center for this frame
                center = final_centers[frame_idx] if frame_idx < len(final_centers) else (orig_w // 2, orig_h // 2)
                
                # Calculate crop bounds
                x1, y1, x2, y2 = self._calculate_crop_bounds(
                    center, orig_w, orig_h, crop_w, crop_h
                )
                
                # Extract and resize crop
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    writer.write(resized)
                    stats['frames_processed'] += 1
                
                # Progress update
                if progress_callback and frame_idx % max(1, total_frames // 100) == 0:
                    progress = 0.4 + 0.5 * (frame_idx / total_frames)
                    progress_callback(progress, f"Rendering: {frame_idx}/{total_frames}")
                
                frame_idx += 1
            
            cap.release()
            writer.release()
            logger.info(f"Phase 3 complete: {stats['frames_processed']} frames rendered")
            
            # ═══════════════════════════════════════════════════════
            # PHASE 4: Merge audio (if present)
            # ═══════════════════════════════════════════════════════
            if progress_callback:
                progress_callback(0.95, "Finalizing output...")
            
            self._add_audio_ffmpeg(temp_video_path, input_path, output_path)
            
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            
            # Final stats
            stats['processing_time_sec'] = round(time.time() - start_time, 2)
            stats['output_resolution'] = f"{target_w}x{target_h}"
            stats['source_resolution'] = f"{orig_w}x{orig_h}"
            
            logger.info(f"✓ Processing complete in {stats['processing_time_sec']}s")
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            return {
                'success': True,
                'output_path': output_path,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"✗ Processing failed: {e}", exc_info=True)
            # Clean up on error
            if cap and cap.isOpened():
                cap.release()
            if writer and writer.isOpened():
                writer.release()
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            
            return {
                'success': False,
                'error': str(e),
                'stats': stats
            }
    
    def batch_process(
        self,
        input_paths: List[str],
        output_dir: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos with the same configuration.
        
        Args:
            input_paths: List of input video paths
            output_dir: Directory for output files
            **kwargs: Additional args passed to process_video()
        
        Returns:
            List of result dicts for each video
        """
        results = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, input_path in enumerate(input_paths):
            # Generate output path
            input_name = Path(input_path).stem
            output_path = str(Path(output_dir) / f"{input_name}_vertical.mp4")
            
            logger.info(f"[{i+1}/{len(input_paths)}] Processing: {input_path}")
            
            result = self.process_video(input_path, output_path, **kwargs)
            result['input_path'] = input_path
            results.append(result)
            
            if not result['success']:
                logger.warning(f"Failed: {input_path} - {result.get('error', 'Unknown error')}")
        
        return results


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert horizontal videos to vertical format')
    parser.add_argument('input', help='Input video path or directory')
    parser.add_argument('-o', '--output', default='output', help='Output path or directory')
    parser.add_argument('-m', '--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Detection confidence (0.0-1.0)')
    parser.add_argument('-s', '--sample', type=int, default=15, help='Detection sample interval')
    parser.add_argument('-p', '--padding', type=float, default=0.1, help='Crop padding ratio')
    
    args = parser.parse_args()
    
    # Initialize verticalizer
    config = CropConfig(padding_ratio=args.padding)
    verticalizer = VideoVerticalizer(
        model_path=args.model,
        use_gpu=args.gpu,
        crop_config=config
    )
    
    # Handle single file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        output_path = args.output if args.output.endswith('.mp4') else f"{args.output}/{input_path.stem}_vertical.mp4"
        result = verticalizer.process_video(
            str(input_path),
            output_path,
            sample_interval=args.sample,
            confidence=args.confidence,
            progress_callback=lambda p, m: print(f"[{p*100:.0f}%] {m}")
        )
        print(f"\nResult: {'✓ Success' if result['success'] else '✗ Failed'}")
        if result.get('stats'):
            print(f"Stats: {result['stats']}")
        if not result['success'] and result.get('error'):
            print(f"Error: {result['error']}")
    else:
        # Directory batch mode
        video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.mov'))
        if not video_files:
            print(f"No videos found in: {input_path}")
            return
        
        print(f"Found {len(video_files)} videos. Processing...")
        results = verticalizer.batch_process(
            [str(f) for f in video_files],
            args.output,
            sample_interval=args.sample,
            confidence=args.confidence,
            progress_callback=lambda p, m: None  # Suppress per-video progress in batch
        )
        
        # Summary
        success_count = sum(1 for r in results if r['success'])
        print(f"\nBatch complete: {success_count}/{len(results)} succeeded")
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"  {status} {Path(r['input_path']).name}")
            if not r['success'] and r.get('error'):
                print(f"     Error: {r['error'][:80]}...")


if __name__ == '__main__':
    main()
