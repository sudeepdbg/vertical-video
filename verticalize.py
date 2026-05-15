"""
verticalize.py — Clean Vertical Video Converter v4.0
─────────────────────────────────────────────────────
FIXES:
- Forces 9:16 aspect ratio (no more 404x720 distortions)
- Single-pass color grading via FFmpeg only
- Simplified tracking: Kalman + EMA anchor only
- No Ken Burns, no dissolve, no optical flow fallbacks
- Expanded crop margins for fast basketball motion
- Proper error handling and logging
"""
import subprocess, sys, os, tempfile, math
from typing import Optional, Tuple, List
import cv2, numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ProcessingError(Exception):
    pass

# ─── Constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS_ID = 0
MAX_FILE_SIZE_MB = 2000
LOWER_THIRD_GUARD = 0.75  # Keep subjects above bottom 25%

# Camera smoothing
MAX_PX_PER_FRAME = 3.0  # Increased for fast basketball motion
CAMERA_ALPHA_MAX = 0.12
TARGET_EMA_ALPHA = 0.30

# Kalman filter (more aggressive for sports)
KALMAN_PROCESS_NOISE_POS = 6.0
KALMAN_PROCESS_NOISE_VEL = 3.0
KALMAN_MEASUREMENT_NOISE = 180.0
KALMAN_MAX_VELOCITY_PX = 120.0

RESOLUTION_PRESETS = {
    "1080p": (1080, 1920),
    "720p":  (720, 1280),
    "540p":  (540, 960),
}

COLOR_GRADES = {
    "none":   None,
    "warm":   "eq=brightness=0.02:saturation=1.12:gamma_r=1.05:gamma_b=0.95",
    "cool":   "eq=brightness=0.01:saturation=1.08:gamma_r=0.95:gamma_b=1.05",
    "vibrant":"eq=brightness=0.0:saturation=1.25:contrast=1.05",
}

# ─── FFmpeg Helpers ─────────────────────────────────────────────────────────────
def _check_ffmpeg():
    for tool in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise ProcessingError(f"{tool} not found or not responding. Install FFmpeg.")

def _get_video_info(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 4:
            raise ProcessingError("ffprobe output malformed")
        width = int(lines[0])
        height = int(lines[1])
        fps_str = lines[2]
        duration = float(lines[3]) if lines[3] else 0.0
        
        # Parse FPS
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str) if fps_str else 30.0
            
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "total_frames": int(duration * fps) if duration > 0 else 0
        }
    except Exception as e:
        raise ProcessingError(f"Failed to read video info: {e}")

def _has_audio(path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15, check=True
        )
        return "audio" in result.stdout.lower()
    except:
        return False

# ─── YOLO Model ─────────────────────────────────────────────────────────────────
_model_cache = {}

def _get_model(weights="yolov8n.pt"):
    if not YOLO_AVAILABLE:
        print("⚠️  YOLO not available. Using saliency fallback.", file=sys.stderr)
        return None
    if weights in _model_cache:
        return _model_cache[weights]
    try:
        model = YOLO(weights)
        _model_cache[weights] = model
        return model
    except Exception as e:
        print(f"⚠️  Failed to load YOLO: {e}", file=sys.stderr)
        return None

def detect_persons(frame, model, conf_thresh=0.45):
    """Detect persons and return bounding boxes in original frame coordinates."""
    if model is None:
        return []
    
    try:
        results = model(frame, verbose=False, conf=conf_thresh)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return []
        
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception as e:
        print(f"Detection error: {e}", file=sys.stderr)
        return []

def filter_persons(boxes, frame_w, frame_h):
    """Filter out tiny, edge, or lower-third persons."""
    min_w = frame_w * 0.05
    min_h = frame_h * 0.12
    edge_margin = frame_w * 0.08
    lower_guard = frame_h * LOWER_THIRD_GUARD
    
    filtered = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        
        # Skip too small
        if w < min_w or h < min_h:
            continue
        # Skip near edges
        if cx < edge_margin or cx > frame_w - edge_margin:
            continue
        # Skip lower third (scoreboard area)
        if y1 > lower_guard:
            continue
            
        filtered.append((x1, y1, x2, y2))
    
    return filtered

def get_union_box(boxes):
    """Get union bounding box of all person boxes."""
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)

def saliency_center(frame):
    """Fallback: find center of most active region using Laplacian."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.GaussianBlur(np.abs(lap), (31, 31), 0)
    
    # Ignore borders
    margin = max(1, int(w * 0.05))
    lap[:margin, :] = lap[-margin:, :] = lap[:, :margin] = lap[:, -margin:] = 0
    
    total = lap.sum()
    if total < 1e-6:
        return w // 2, h // 2
    
    ys, xs = np.mgrid[0:h, 0:w]
    cx = int((xs * lap).sum() / total)
    cy = int((ys * lap).sum() / total)
    return cx, cy

# ─── Kalman Tracker ─────────────────────────────────────────────────────────────
class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE_POS
        self.kalman.processNoiseCov[2, 2] = KALMAN_PROCESS_NOISE_VEL
        self.kalman.processNoiseCov[3, 3] = KALMAN_PROCESS_NOISE_VEL
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        
        self.state = np.zeros((4, 1), np.float32)
        self.initialized = False
        
    def update(self, cx, cy):
        """Update tracker with new measurement."""
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        
        if not self.initialized:
            self.state[:2] = measurement
            self.state[2:] = 0
            self.kalman.statePost = self.state.copy()
            self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0
            self.initialized = True
            return cx, cy
        
        # Predict
        prediction = self.kalman.predict()
        
        # Correct
        corrected = self.kalman.correct(measurement)
        
        # Clamp velocity
        vx, vy = corrected[2, 0], corrected[3, 0]
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > KALMAN_MAX_VELOCITY_PX:
            scale = KALMAN_MAX_VELOCITY_PX / speed
            corrected[2, 0] = vx * scale
            corrected[3, 0] = vy * scale
        
        self.state = corrected
        return int(self.state[0, 0]), int(self.state[1, 0])
    
    def predict(self):
        """Predict next position without measurement."""
        if not self.initialized:
            return None
        prediction = self.kalman.predict()
        return int(prediction[0, 0]), int(prediction[1, 0])

# ─── Camera Anchor (EMA Smoothing) ──────────────────────────────────────────────
class CameraAnchor:
    def __init__(self):
        self.cx = None
        self.cy = None
        self.tx = None  # Target X
        self.ty = None  # Target Y
        
    def set_target(self, cx, cy):
        """Set new target with EMA smoothing."""
        if self.tx is None:
            self.tx = float(cx)
            self.ty = float(cy)
        else:
            alpha = TARGET_EMA_ALPHA
            self.tx = self.tx * (1 - alpha) + cx * alpha
            self.ty = self.ty * (1 - alpha) + cy * alpha
    
    def step(self, max_move=MAX_PX_PER_FRAME, alpha_max=CAMERA_ALPHA_MAX):
        """Move camera toward target with speed limit."""
        if self.cx is None:
            self.cx = self.tx or 0.0
            self.cy = self.ty or 0.0
            return int(self.cx), int(self.cy)
        
        if self.tx is None:
            return int(self.cx), int(self.cy)
        
        dx = self.tx - self.cx
        dy = self.ty - self.cy
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 1e-6:
            return int(self.cx), int(self.cy)
        
        # Adaptive alpha: slower when far, faster when close
        alpha = min(max_move / dist, alpha_max)
        self.cx += alpha * dx
        self.cy += alpha * dy
        
        return int(self.cx), int(self.cy)

# ─── Main Processing Function ───────────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    resolution_label: str = "720p",
    confidence: float = 0.45,
    color_grade: str = "none",
    sharpen_strength: float = 0.0,
    progress_callback=None
):
    """
    Convert landscape video to vertical 9:16 format.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        resolution_label: "1080p", "720p", or "540p"
        confidence: YOLO detection confidence threshold
        color_grade: "none", "warm", "cool", or "vibrant"
        sharpen_strength: 0.0 to 1.0
        progress_callback: Optional function(v, msg) for progress updates
    """
    
    def _progress(v, msg=""):
        if progress_callback:
            try:
                progress_callback(min(max(v, 0.0), 1.0), msg)
            except:
                pass
    
    # Validate inputs
    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")
    
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File too large: {file_size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB")
    
    # Get video info
    info = _get_video_info(input_path)
    orig_w, orig_h = info["width"], info["height"]
    fps = info["fps"]
    duration = info["duration"]
    total_frames = info["total_frames"]
    
    if orig_w <= orig_h:
        raise ProcessingError("Video is already vertical or square.")
    
    # Resolve target resolution (FORCE 9:16)
    if resolution_label not in RESOLUTION_PRESETS:
        resolution_label = "720p"
    target_w, target_h = RESOLUTION_PRESETS[resolution_label]
    
    # Calculate crop dimensions from source (maintain 9:16 aspect ratio)
    source_aspect = orig_w / orig_h
    target_aspect = target_w / target_h  # Should be 9/16 = 0.5625
    
    if source_aspect > target_aspect:
        # Source is wider than target: crop width
        crop_h = orig_h
        crop_w = int(round(crop_h * target_aspect))
    else:
        # Source is taller than target: crop height
        crop_w = orig_w
        crop_h = int(round(crop_w / target_aspect))
    
    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    
    _progress(0.05, f"Target: {target_w}x{target_h}, Crop: {crop_w}x{crop_h}")
    
    # Load YOLO model
    model = _get_model("yolov8n.pt")
    
    # Initialize trackers
    kalman = KalmanTracker()
    anchor = CameraAnchor()
    
    # Initial position: center of frame
    init_cx, init_cy = orig_w // 2, orig_h // 2
    kalman.update(init_cx, init_cy)
    anchor.set_target(init_cx, init_cy)
    
    # Prepare FFmpeg encoder
    # Use FFmpeg for color grading to avoid Python artifacts
    vf_filters = []
    if color_grade in COLOR_GRADES and COLOR_GRADES[color_grade]:
        vf_filters.append(COLOR_GRADES[color_grade])
    if sharpen_strength > 0:
        # Simple unsharp mask via FFmpeg
        vf_filters.append(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={sharpen_strength * 0.8}")
    
    vf_string = ",".join(vf_filters) if vf_filters else None
    
    # Build FFmpeg command
    ffmpeg_cmd = ["ffmpeg", "-y", "-hwaccel", "none"]
    
    # Input raw frames
    ffmpeg_cmd += [
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{orig_w}x{orig_h}", "-r", str(fps), "-i", "pipe:0"
    ]
    
    # Audio
    if _has_audio(input_path):
        ffmpeg_cmd += ["-i", input_path, "-map", "0:v:0", "-map", "1:a:0?",
                      "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    else:
        ffmpeg_cmd += ["-an"]
    
    # Video filters
    if vf_string:
        ffmpeg_cmd += ["-vf", vf_string]
    
    # Output
    ffmpeg_cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_path
    ]
    
    _progress(0.10, "Starting encoding...")
    
    # Start FFmpeg process
    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        raise ProcessingError(f"Failed to start FFmpeg: {e}")
    
    # Process frames
    frame_bytes = orig_w * orig_h * 3
    half_crop_w = crop_w // 2
    half_crop_h = crop_h // 2
    
    sample_interval = max(1, int(fps * 0.5))  # Detect every 0.5 seconds
    fi = 0
    last_boxes = []
    
    try:
        with open(input_path, 'rb') as f:
            # Read video using FFmpeg decoder for reliability
            decode_cmd = ["ffmpeg", "-y", "-hwaccel", "none", "-i", input_path,
                         "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-vf", f"scale={orig_w}:{orig_h}", "pipe:1"]
            decoder = subprocess.Popen(
                decode_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            while fi < total_frames:
                # Read frame
                frame_data = decoder.stdout.read(frame_bytes)
                if len(frame_data) != frame_bytes:
                    break
                
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(orig_h, orig_w, 3)
                
                # Detection on sample frames
                if fi % sample_interval == 0:
                    boxes = detect_persons(frame, model, confidence)
                    boxes = filter_persons(boxes, orig_w, orig_h)
                    
                    if boxes:
                        union = get_union_box(boxes)
                        if union:
                            ux1, uy1, ux2, uy2 = union
                            ucx = (ux1 + ux2) // 2
                            ucy = (uy1 + uy2) // 2
                            
                            # Update Kalman
                            kx, ky = kalman.update(ucx, ucy)
                            anchor.set_target(kx, ky)
                            last_boxes = boxes
                        else:
                            kalman.predict()
                    else:
                        # Fallback to saliency
                        scx, scy = saliency_center(frame)
                        kx, ky = kalman.update(scx, scy)
                        anchor.set_target(kx, ky)
                else:
                    # Non-sample frames: predict only
                    pred = kalman.predict()
                    if pred:
                        anchor.set_target(pred[0], pred[1])
                
                # Get camera position
                cur_cx, cur_cy = anchor.step()
                
                # Clamp to valid range
                cur_cx = max(half_crop_w, min(cur_cx, orig_w - half_crop_w))
                cur_cy = max(half_crop_h, min(cur_cy, orig_h - half_crop_h))
                
                # Crop frame
                left = cur_cx - half_crop_w
                top = cur_cy - half_crop_h
                cropped = frame[top:top+crop_h, left:left+crop_w]
                
                # Resize to target if needed
                if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
                    cropped = cv2.resize(
                        cropped, (target_w, target_h),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                
                # Write to FFmpeg
                try:
                    proc.stdin.write(cropped.tobytes())
                except BrokenPipeError:
                    break
                
                fi += 1
                
                # Progress update
                if fi % max(1, total_frames // 20) == 0:
                    pct = 0.10 + 0.85 * (fi / total_frames)
                    _progress(pct, f"Processing frame {fi}/{total_frames}")
        
        decoder.wait()
        
    except Exception as e:
        proc.stdin.close()
        proc.wait()
        raise ProcessingError(f"Frame processing error: {e}")
    
    # Close encoder
    _progress(0.95, "Finalizing...")
    try:
        proc.stdin.close()
        stderr_data = proc.stderr.read().decode('utf-8', errors='ignore')
        proc.wait(timeout=30)
        
        if proc.returncode != 0:
            raise ProcessingError(f"FFmpeg failed:\n{stderr_data}")
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ProcessingError("Output file is empty or missing.")
            
    except subprocess.TimeoutExpired:
        proc.terminate()
        raise ProcessingError("FFmpeg timed out during shutdown.")
    except Exception as e:
        raise ProcessingError(f"Encoder shutdown failed: {e}")
    
    _progress(1.0, "Done!")
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Output: {output_path} ({output_size_mb:.1f} MB)", file=sys.stderr)
    
    return {
        "output_path": output_path,
        "resolution": f"{target_w}x{target_h}",
        "duration": duration,
        "frames_processed": fi
    }

# ─── CLI Interface ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert landscape video to vertical 9:16")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("-r", "--resolution", choices=["1080p", "720p", "540p"],
                       default="720p", help="Output resolution")
    parser.add_argument("-c", "--confidence", type=float, default=0.45,
                       help="YOLO detection confidence (0.0-1.0)")
    parser.add_argument("-g", "--grade", choices=["none", "warm", "cool", "vibrant"],
                       default="none", help="Color grade")
    parser.add_argument("-s", "--sharpen", type=float, default=0.0,
                       help="Sharpen strength (0.0-1.0)")
    
    args = parser.parse_args()
    
    try:
        result = process_video(
            input_path=args.input,
            output_path=args.output,
            resolution_label=args.resolution,
            confidence=args.confidence,
            color_grade=args.grade,
            sharpen_strength=args.sharpen,
            progress_callback=lambda v, m: print(f"\r[{int(v*100):3d}%] {m}", end="", flush=True)
        )
        print(f"\n✅ Success! Output saved to: {result['output_path']}")
    except ProcessingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(130)
