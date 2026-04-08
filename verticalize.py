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


def detect_any_objects(frame: np.ndarray, model: YOLO, confidence: float = 0.4) -> Optional[Tuple[int, int]]:
    """Detect any prominent objects (not just people) and return weighted center."""
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"⚠️ Detection error: {e}", file=sys.stderr)
        return None

    if results.boxes is None or len(results.boxes) == 0:
        return None

    total_weight = 0.0
    weighted_cx = 0.0
    weighted_cy = 0.0

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        weight = area * conf
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        weighted_cx += cx * weight
        weighted_cy += cy * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return (int(weighted_cx / total_weight), int(weighted_cy / total_weight))


def optical_flow_center(prev_gray: np.ndarray, curr_gray: np.ndarray, orig_w: int, orig_h: int) -> Optional[Tuple[int, int]]:
    """Use optical flow to find regions of motion — great for non-person content."""
    if prev_gray is None or curr_gray is None:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Ignore border noise
        border = max(1, int(orig_w * 0.05))
        magnitude[:, :border] = 0
        magnitude[:, orig_w - border:] = 0
        magnitude[:border, :] = 0
        magnitude[orig_h - border:, :] = 0

        if magnitude.max() < 0.5:
            return None

        # Weighted centroid of motion
        total = magnitude.sum()
        if total == 0:
            return None
        ys, xs = np.mgrid[0:orig_h, 0:orig_w]
        cx = int((xs * magnitude).sum() / total)
        cy = int((ys * magnitude).sum() / total)
        return (cx, cy)
    except Exception:
        return None


def saliency_center(frame: np.ndarray) -> Tuple[int, int]:
    """Content-aware fallback: finds the visually interesting region."""
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIMENSION or h < MIN_FRAME_DIMENSION:
        return (w // 2, h // 2)

    # Use multiple saliency cues
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Laplacian for edges / texture richness
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.abs(lap)
    lap = cv2.GaussianBlur(lap.astype(np.float32), (31, 31), 0)

    # Colour saturation map
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    sat = cv2.GaussianBlur(sat, (31, 31), 0)

    # Combine
    saliency = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)

    # Suppress borders
    border = max(1, int(w * 0.05))
    saliency[:, :border] = 0
    saliency[:, w - border:] = 0
    saliency[:border, :] = 0
    saliency[h - border:, :] = 0

    if saliency.max() < 1e-6:
        return (w // 2, h // 2)

    total = saliency.sum()
    ys, xs = np.mgrid[0:h, 0:w]
    cx = int((xs * saliency).sum() / total)
    cy = int((ys * saliency).sum() / total)
    return (cx, cy)


# ---------------------------------------------------------------------------
# Rule-of-thirds bias
# ---------------------------------------------------------------------------
def apply_rule_of_thirds_bias(cx: int, cy: int, orig_w: int, orig_h: int,
                               crop_w: int, crop_h: int, bias_strength: float = 0.3) -> Tuple[int, int]:
    """Nudge crop center toward the nearest rule-of-thirds line."""
    thirds_x = [orig_w // 3, 2 * orig_w // 3]
    thirds_y = [orig_h // 3, 2 * orig_h // 3]

    nearest_tx = min(thirds_x, key=lambda x: abs(x - cx))
    nearest_ty = min(thirds_y, key=lambda y: abs(y - cy))

    new_cx = int(cx + bias_strength * (nearest_tx - cx))
    new_cy = int(cy + bias_strength * (nearest_ty - cy))

    # Keep within bounds so crop stays in frame
    half_w, half_h = crop_w // 2, crop_h // 2
    new_cx = max(half_w, min(new_cx, orig_w - half_w))
    new_cy = max(half_h, min(new_cy, orig_h - half_h))

    return (new_cx, new_cy)


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
# Faststart helper
# ---------------------------------------------------------------------------
def _add_faststart(input_file: str, output_file: str) -> None:
    """Rewrite MP4 with moov atom at the beginning for streaming."""
    temp_out = output_file + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-c", "copy",
        "-movflags", "+faststart",
        temp_out
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        shutil.move(temp_out, output_file)
    except subprocess.CalledProcessError as e:
        # If faststart fails just copy the file as-is
        print(f"⚠️ Faststart failed ({e.stderr}), copying as-is", file=sys.stderr)
        shutil.copy2(input_file, output_file)
    finally:
        if os.path.exists(temp_out):
            try:
                os.unlink(temp_out)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Re-encode to H.264 baseline for browser compatibility
# ---------------------------------------------------------------------------
def _reencode_for_browser(input_file: str, output_file: str, fps: float) -> None:
    """
    Re-encode using ffmpeg to H.264 baseline + AAC, with faststart.
    This ensures the video plays inline in browsers / Streamlit st.video().
    """
    temp_out = output_file + ".browser.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",       # required for browser compatibility
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-r", str(fps),
        temp_out
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        shutil.move(temp_out, output_file)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Browser re-encode failed: {e.stderr}", file=sys.stderr)
        if os.path.exists(temp_out):
            os.unlink(temp_out)
        # Fall back to faststart-only
        _add_faststart(input_file, output_file)


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
    use_optical_flow: bool = True,
    rule_of_thirds: bool = True,
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
        raise ProcessingError("Video appears to be vertical already. This tool is designed for horizontal (landscape) videos.")

    if sample_interval is None or sample_interval == 0:
        sample_interval = max(1, int(fps / 2))

    target_w, target_h = target_size
    crop_w, crop_h = calculate_crop_dimensions(orig_w, orig_h, target_w, target_h)
    _progress(0.02, f"📐 Crop window: {crop_w}×{crop_h} → output {target_w}×{target_h}")

    # --- Phase 1: Load model ---
    _progress(0.05, "🔍 Loading AI model...")
    try:
        model = _get_model(yolo_weights)
    except Exception as e:
        cap.release()
        raise ProcessingError(f"Failed to load detection model: {e}")

    # --- Phase 2: Detection pass ---
    detected_centers = []
    detected_indices = []
    fallback_centers = []
    fallback_indices = []

    _progress(0.10, f"🔎 Analysing {total_frames} frames for subjects & motion...")
    frame_idx = 0
    frames_processed = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Priority 1: Person detection
            center = detect_largest_person(frame, model, confidence)

            # Priority 2: Any object detection
            if center is None:
                center = detect_any_objects(frame, model, confidence * 0.8)

            # Priority 3: Optical flow (motion)
            if center is None and use_optical_flow:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                curr_gray_small = cv2.resize(curr_gray, (orig_w // 2, orig_h // 2))
                if prev_gray is not None:
                    flow_center = optical_flow_center(prev_gray, curr_gray_small, orig_w // 2, orig_h // 2)
                    if flow_center is not None:
                        center = (flow_center[0] * 2, flow_center[1] * 2)
                prev_gray = curr_gray_small
            elif use_optical_flow:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.resize(curr_gray, (orig_w // 2, orig_h // 2))

            if center is not None:
                detected_centers.append(center)
                detected_indices.append(frame_idx)
            else:
                # Priority 4: Saliency fallback
                sc = saliency_center(frame)
                fallback_centers.append(sc)
                fallback_indices.append(frame_idx)

        frames_processed += 1
        if frames_processed % max(1, total_frames // 20) == 0:
            _progress(0.10 + 0.30 * (frames_processed / total_frames),
                      f"🔎 Analysed {frames_processed}/{total_frames} frames...")
        frame_idx += 1

    cap.release()

    # Merge: use detected where available, saliency where not
    if not detected_centers:
        if fallback_centers:
            detected_centers = fallback_centers
            detected_indices = fallback_indices
            _progress(0.40, "🎨 Using content-aware tracking (no subjects detected)")
        else:
            detected_centers = [(orig_w // 2, orig_h // 2)]
            detected_indices = [0]
            _progress(0.40, "📍 Using centre-point tracking")
    else:
        # Merge saliency into gaps if there are large undetected spans
        if fallback_centers:
            merged_centers = detected_centers + fallback_centers
            merged_indices = detected_indices + fallback_indices
            sorted_pairs = sorted(zip(merged_indices, merged_centers))
            detected_indices = [p[0] for p in sorted_pairs]
            detected_centers = [p[1] for p in sorted_pairs]

    # --- Phase 3: Interpolate & smooth ---
    _progress(0.45, "📈 Computing smooth tracking path...")
    all_centers = interpolate_centers(detected_centers, detected_indices, total_frames)
    all_centers = smooth_centers(all_centers, window=smooth_window)

    # Apply rule-of-thirds bias to each center
    if rule_of_thirds:
        all_centers = [
            apply_rule_of_thirds_bias(cx, cy, orig_w, orig_h, crop_w, crop_h, bias_strength=0.25)
            for (cx, cy) in all_centers
        ]

    if len(all_centers) < total_frames:
        all_centers.extend([all_centers[-1]] * (total_frames - len(all_centers)))
    all_centers = all_centers[:total_frames]

    _progress(0.50, "✂️ Rendering vertical frames...")

    # --- Phase 4: Render frames ---
    temp_video_path = None
    temp_with_audio = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_w, target_h))
        if not out.isOpened():
            raise ProcessingError("Failed to initialise video encoder.")

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
                prog = 0.50 + 0.38 * (frame_num / total_frames)
                _progress(prog, f"✂️ Rendered {frame_num}/{total_frames} frames...")
        cap.release()
        out.release()

        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) < 1000:
            raise ProcessingError("Rendered video file is empty or missing.")

        # --- Phase 5: Mux audio + browser-compatible encode ---
        _progress(0.88, "🎵 Muxing audio & encoding for browser playback...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_final:
            temp_with_audio = tmp_final.name

        audio_muxed = False
        try:
            video_clip = VideoFileClip(temp_video_path)
            source_clip = VideoFileClip(input_path)
            if source_clip.audio is not None:
                final_clip = video_clip.set_audio(source_clip.audio)
            else:
                final_clip = video_clip
            final_clip.write_videofile(
                temp_with_audio,
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                threads=4,
                logger=None,
                ffmpeg_params=["-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p"]
            )
            final_clip.close()
            source_clip.close()
            video_clip.close()
            audio_muxed = True
        except Exception as e:
            print(f"⚠️ Audio muxing failed ({e}), re-encoding video only", file=sys.stderr)

        _progress(0.93, "📦 Finalising (faststart for streaming)...")

        source_for_encode = temp_with_audio if audio_muxed else temp_video_path

        # Always re-encode to baseline H.264 + faststart for browser compatibility
        _reencode_for_browser(source_for_encode, output_path, fps)

        _progress(1.0, "✅ Done!")

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ProcessingError("Final output file is empty or missing.")

        print(f"✅ Saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    finally:
        for p in [temp_video_path, temp_with_audio]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
