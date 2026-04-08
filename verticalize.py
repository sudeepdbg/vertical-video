import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import sys
import subprocess
import shutil
from typing import Optional, Callable, List, Tuple, Union

# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class ProcessingError(Exception):
    pass

# ---------------------------------------------------------------------------
# MoviePy Import (compatible with 1.x and 2.x)
# ---------------------------------------------------------------------------
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 0
DEFAULT_TARGET_SIZE = (1080, 1920)
MAX_FILE_SIZE_MB = 500
MIN_FRAME_DIMENSION = 240

# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
_model_cache = {}

def _get_model(weights: str = "yolov8n.pt") -> YOLO:
    if weights not in _model_cache:
        try:
            _model_cache[weights] = YOLO(weights)
        except Exception as e:
            raise ProcessingError(f"Failed to load YOLO model '{weights}': {e}")
    return _model_cache[weights]

# ---------------------------------------------------------------------------
# Detection Helpers
# ---------------------------------------------------------------------------
def detect_largest_person(frame: np.ndarray, model: YOLO, confidence: float = 0.5) -> Optional[Tuple[int, int]]:
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"⚠️ Detection error: {e}", file=sys.stderr)
        return None
    best_center = None
    best_area = 0
    if results.boxes is not None:
        for box in results.boxes:
            if int(box.cls[0]) != PERSON_CLASS_ID:
                continue
            if float(box.conf[0]) < confidence:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return best_center

def saliency_center(frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIMENSION or h < MIN_FRAME_DIMENSION:
        return (w // 2, h // 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    _, dark = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(bright, dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    border = max(1, int(w * 0.05))
    mask[:, :border] = 0
    mask[:, w - border:] = 0
    mask[:border, :] = 0
    mask[h - border:, :] = 0
    col_proj = np.count_nonzero(mask, axis=0)
    row_proj = np.count_nonzero(mask, axis=1)
    if col_proj.max() < 1 or row_proj.max() < 1:
        return (w // 2, h // 2)
    nonzero_cols = np.where(col_proj > 0)[0]
    nonzero_rows = np.where(row_proj > 0)[0]
    cx = int((nonzero_cols[0] + nonzero_cols[-1]) // 2)
    cy = int((nonzero_rows[0] + nonzero_rows[-1]) // 2)
    return (cx, cy)

# ---------------------------------------------------------------------------
# Center Smoothing & Interpolation
# ---------------------------------------------------------------------------
def smooth_centers(centers: List[Tuple[int, int]], window: int = 7) -> List[Tuple[int, int]]:
    if not centers or len(centers) < 3:
        return centers.copy() if centers else []
    n = len(centers)
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    half = window // 2
    if half == 0:
        return centers
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / (half / 2 + 1e-6)) ** 2)
    kernel /= kernel.sum()
    xs_padded = np.pad(xs, half, mode="reflect")
    ys_padded = np.pad(ys, half, mode="reflect")
    xs_smooth = np.convolve(xs_padded, kernel, mode="valid")
    ys_smooth = np.convolve(ys_padded, kernel, mode="valid")
    return [(int(x), int(y)) for x, y in zip(xs_smooth, ys_smooth)]

def interpolate_centers(detected_centers: List[Tuple[int, int]], detected_indices: List[int], total_frames: int) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        return []
    if not detected_centers or not detected_indices:
        return [(0, 0)] * total_frames
    all_centers = []
    for frame_idx in range(total_frames):
        if frame_idx <= detected_indices[0]:
            all_centers.append(detected_centers[0])
            continue
        if frame_idx >= detected_indices[-1]:
            all_centers.append(detected_centers[-1])
            continue
        prev_idx = 0
        for i, idx in enumerate(detected_indices):
            if idx > frame_idx:
                prev_idx = i - 1
                break
        next_idx = prev_idx + 1
        if next_idx >= len(detected_indices):
            all_centers.append(detected_centers[-1])
            continue
        t = (frame_idx - detected_indices[prev_idx]) / max(detected_indices[next_idx] - detected_indices[prev_idx], 1)
        cx = int(detected_centers[prev_idx][0] + t * (detected_centers[next_idx][0] - detected_centers[prev_idx][0]))
        cy = int(detected_centers[prev_idx][1] + t * (detected_centers[next_idx][1] - detected_centers[prev_idx][1]))
        all_centers.append((cx, cy))
    while len(all_centers) < total_frames:
        all_centers.append(all_centers[-1] if all_centers else (0, 0))
    return all_centers[:total_frames]

# ---------------------------------------------------------------------------
# Crop Geometry
# ---------------------------------------------------------------------------
def calculate_crop_dimensions(orig_w: int, orig_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h
    if orig_ratio > target_ratio:
        crop_h = orig_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        crop_w = orig_w
        crop_h = int(round(crop_w / target_ratio))
    crop_w = min(crop_w, orig_w)
    crop_h = min(crop_h, orig_h)
    return crop_w, crop_h

# ---------------------------------------------------------------------------
# Faststart helper (using ffmpeg directly)
# ---------------------------------------------------------------------------
def _add_faststart(input_file: str, output_file: str) -> None:
    """Rewrite MP4 with moov atom at the beginning for streaming."""
    # Use a temporary file to avoid overwriting issues
    temp_out = output_file + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-c", "copy",
        "-movflags", "+faststart",
        temp_out
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        shutil.move(temp_out, output_file)
    except subprocess.CalledProcessError as e:
        raise ProcessingError(f"Faststart failed: {e.stderr}")

# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------
def process_video(
    input_path: str,
    output_path: str,
    sample_interval: Optional[int] = None,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    confidence: float = 0.5,
    smooth_window: int = 15,
    yolo_weights: str = "yolov8n.pt",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:
    def _progress(value: float, message: str = ""):
        if progress_callback:
            try:
                progress_callback(min(max(value, 0.0), 1.0), message)
            except Exception:
                pass

    # --- Validate input ---
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")
    file_size_mb = os.path.getsize(input_path) / (1024 ** 2)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File size {file_size_mb:.1f} MB exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # --- Open video ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ProcessingError(f"Cannot open video file: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        cap.release()
        raise ProcessingError("Could not read video metadata - file may be corrupt.")
    if orig_w <= orig_h:
        cap.release()
        raise ProcessingError("Video appears to be vertical already. This tool is designed for horizontal (16:9) videos.")
    if sample_interval is None or sample_interval == 0:
        sample_interval = max(1, int(fps / 2))
    target_w, target_h = target_size
    crop_w, crop_h = calculate_crop_dimensions(orig_w, orig_h, target_w, target_h)
    _progress(0.02, f"📐 Calculated crop: {crop_w}×{crop_h} → {target_w}×{target_h}")

    # --- Phase 1: Detection ---
    _progress(0.05, "🔍 Loading AI model...")
    try:
        model = _get_model(yolo_weights)
    except Exception as e:
        cap.release()
        raise ProcessingError(f"Failed to load detection model: {e}")

    detected_centers = []
    detected_indices = []
    saliency_fallback = []
    saliency_frames = []
    _progress(0.10, f"🔎 Analyzing video ({total_frames} frames)...")
    frame_idx = 0
    frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            center = detect_largest_person(frame, model, confidence)
            if center:
                detected_centers.append(center)
                detected_indices.append(frame_idx)
            else:
                sc = saliency_center(frame)
                saliency_fallback.append(sc)
                saliency_frames.append(frame_idx)
        frames_processed += 1
        if frames_processed % max(1, total_frames // 20) == 0:
            _progress(0.10 + 0.30 * (frames_processed / total_frames), f"🔎 Analyzed {frames_processed}/{total_frames} frames...")
        frame_idx += 1
    cap.release()

    if not detected_centers:
        if saliency_fallback:
            detected_centers = saliency_fallback
            detected_indices = saliency_frames
            _progress(0.40, "🎨 Using content-based tracking (no people detected)")
        else:
            detected_centers = [(orig_w // 2, orig_h // 2)]
            detected_indices = [0]
            _progress(0.40, "📍 Using center-point tracking")

    # --- Phase 2: Interpolate & smooth ---
    _progress(0.45, "📈 Calculating smooth tracking path...")
    all_centers = interpolate_centers(detected_centers, detected_indices, total_frames)
    all_centers = smooth_centers(all_centers, window=smooth_window)
    if len(all_centers) < total_frames:
        all_centers.extend([all_centers[-1]] * (total_frames - len(all_centers)))
    all_centers = all_centers[:total_frames]
    _progress(0.50, "✂️ Rendering vertical video...")

    # --- Phase 3: Render frames ---
    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video_path = tmp.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_w, target_h))
        if not out.isOpened():
            raise ProcessingError("Failed to initialize video encoder.")
        cap = cv2.VideoCapture(input_path)
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cx, cy = all_centers[frame_num]
            left = cx - crop_w // 2
            top = cy - crop_h // 2
            left = max(0, min(left, orig_w - crop_w))
            top = max(0, min(top, orig_h - crop_h))
            cropped = frame[top:top + crop_h, left:left + crop_w]
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
            frame_num += 1
            if frame_num % max(1, total_frames // 20) == 0:
                prog = 0.50 + 0.40 * (frame_num / total_frames)
                _progress(prog, f"✂️ Rendered {frame_num}/{total_frames} frames...")
        cap.release()
        out.release()
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) < 1000:
            raise ProcessingError("Failed to create video output.")

        # --- Phase 4: Add audio and faststart ---
        _progress(0.90, "🎵 Adding audio...")
        # Create final output with audio using moviepy
        try:
            video_clip = VideoFileClip(temp_video_path)
            source_clip = VideoFileClip(input_path)
            if source_clip.audio is not None:
                # Set audio without trimming – moviepy will handle duration automatically
                final_clip = video_clip.set_audio(source_clip.audio)
            else:
                final_clip = video_clip
            # Write video with audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_final:
                temp_with_audio = tmp_final.name
            final_clip.write_videofile(
                temp_with_audio,
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                threads=4,
                logger=None
            )
            final_clip.close()
            source_clip.close()
            video_clip.close()
            # Apply faststart to the file with audio
            _add_faststart(temp_with_audio, output_path)
            os.unlink(temp_with_audio)
        except Exception as e:
            print(f"⚠️ Audio muxing failed ({e}), using video-only output", file=sys.stderr)
            # Use the video-only temp file, add faststart directly
            _add_faststart(temp_video_path, output_path)

        _progress(1.0, "✅ Complete!")
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ProcessingError("Failed to create final output file.")
        print(f"✅ Successfully saved: {output_path}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except OSError:
                pass
