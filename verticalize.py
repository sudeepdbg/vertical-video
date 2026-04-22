"""
verticalize.py
──────────────
Convert landscape video to vertical format using AI subject tracking.

Dependencies: opencv-python, ultralytics, numpy
Audio muxing:  FFmpeg subprocess only
"""

import bisect
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import sys
import subprocess
import shutil
from typing import Optional, Callable, List, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class ProcessingError(Exception):
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSON_CLASS_ID    = 0
HIGH_PRIO_CLASSES  = {0, 2, 3, 5, 7, 15, 16}   # person, car, motorbike, bus, truck, cat, dog
DEFAULT_TARGET_SIZE: Tuple[int, int] = (1080, 1920)
MAX_FILE_SIZE_MB   = 500
MIN_FRAME_DIMENSION = 240
MAX_FRAMES_GUARD   = 216_000  # 1 hour @ 60 fps safety cap

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "1080p  (1080×1920 — Full HD)":  (1080, 1920),
    "720p   (720×1280  — HD)":       (720,  1280),
    "540p   (540×960   — SD)":       (540,  960),
    "480p   (480×854   — Low)":      (480,  854),
}

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def _check_ffmpeg() -> None:
    """Verify both ffmpeg and ffprobe are on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True,
                           capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(
                f"{tool} not found. Please install FFmpeg (includes ffprobe) "
                "and ensure it is on your PATH."
            )


def _ffmpeg_mux_audio_and_encode(
    video_only_path: str,
    source_audio_path: str,
    output_path: str,
    fps: float,
    duration_seconds: float,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
) -> None:
    """
    Single FFmpeg call:
      - Renders video + original audio
      - Encodes to H.264 baseline + AAC
      - Trims to exact video duration
      - Writes moov atom first (faststart) for browser streaming
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_only_path,
        "-i", source_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", encoder_preset,
        "-crf", str(crf),
        "-profile:v", "baseline",
        "-level",   "3.1",
        "-pix_fmt", "yuv420p",
        "-r",       str(fps),
        "-t",       str(duration_seconds),
        "-c:a",     "aac",
        "-b:a",     audio_bitrate,
        "-ac",      "2",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise ProcessingError(f"FFmpeg encode failed:\n{e.stderr[-2000:]}")


def _ffmpeg_encode_video_only(
    video_only_path: str,
    output_path: str,
    fps: float,
    crf: int = 23,
    encoder_preset: str = "fast",
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_only_path,
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-preset", encoder_preset,
        "-crf", str(crf),
        "-profile:v", "baseline",
        "-level",   "3.1",
        "-pix_fmt", "yuv420p",
        "-r",       str(fps),
        "-an",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise ProcessingError(f"FFmpeg video-only encode failed:\n{e.stderr[-2000:]}")


def _ffmpeg_has_audio(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "audio" in result.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------
def get_video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ProcessingError(f"Cannot open video: {path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "total_frames": min(total_frames, MAX_FRAMES_GUARD),
        "width": w,
        "height": h,
        "duration_seconds": duration,
        "is_landscape": w > h,
    }


def extract_thumbnail(path: str, time_seconds: float = 1.0) -> Optional[bytes]:
    """Return JPEG bytes of the frame at time_seconds, or None on failure."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_seconds * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# Model loader — returns a plain YOLO; caller should use @st.cache_resource
# ---------------------------------------------------------------------------
_model_cache: Dict[str, YOLO] = {}

def _get_model(weights: str = "yolov8n.pt") -> YOLO:
    if weights not in _model_cache:
        try:
            _model_cache[weights] = YOLO(weights)
        except Exception as e:
            raise ProcessingError(f"Failed to load YOLO model '{weights}': {e}")
    return _model_cache[weights]


# ---------------------------------------------------------------------------
# Combined single-pass detection  (person + object in one YOLO call)
# ---------------------------------------------------------------------------
def detect_subject_center(
    frame: np.ndarray,
    model: YOLO,
    confidence: float = 0.5,
) -> Optional[Tuple[int, int]]:
    """
    One YOLO inference pass.
    Priority: persons → high-priority objects (cars, animals…) → everything else.
    Returns area-×-confidence weighted centroid, or None.
    """
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"⚠️ Detection error: {e}", file=sys.stderr)
        return None

    if results.boxes is None or len(results.boxes) == 0:
        return None

    person_pool: List[Tuple[float, int, int]] = []
    hiprio_pool: List[Tuple[float, int, int]] = []
    all_pool:    List[Tuple[float, int, int]] = []

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area   = (x2 - x1) * (y2 - y1)
        weight = area * conf
        cx     = (x1 + x2) // 2
        cy     = (y1 + y2) // 2
        entry  = (weight, cx, cy)
        if cls == PERSON_CLASS_ID:
            person_pool.append(entry)
        elif cls in HIGH_PRIO_CLASSES:
            hiprio_pool.append(entry)
        all_pool.append(entry)

    pool = person_pool or hiprio_pool or all_pool
    if not pool:
        return None

    total_w = sum(e[0] for e in pool)
    if total_w == 0:
        return None
    cx = int(sum(e[0] * e[1] for e in pool) / total_w)
    cy = int(sum(e[0] * e[2] for e in pool) / total_w)
    return (cx, cy)


# ---------------------------------------------------------------------------
# Optical flow motion center
# ---------------------------------------------------------------------------
def optical_flow_center(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> Optional[Tuple[int, int]]:
    if prev_gray is None or curr_gray is None:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Mask out 4% border to ignore edge noise
        b = max(1, int(orig_w * 0.04))
        mag[:, :b] = mag[:, orig_w - b:] = 0
        mag[:b, :] = mag[orig_h - b:, :] = 0
        if mag.max() < 0.8:
            return None
        total = mag.sum()
        if total == 0:
            return None
        ys, xs = np.mgrid[0:orig_h, 0:orig_w]
        return (int((xs * mag).sum() / total), int((ys * mag).sum() / total))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Saliency fallback
# ---------------------------------------------------------------------------
def saliency_center(frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIMENSION or h < MIN_FRAME_DIMENSION:
        return (w // 2, h // 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32),
        (31, 31), 0)
    sat  = cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32),
        (31, 31), 0)
    sal  = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    b = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w - b:] = sal[:b, :] = sal[h - b:, :] = 0
    total = sal.sum()
    if total < 1e-6:
        return (w // 2, h // 2)
    ys, xs = np.mgrid[0:h, 0:w]
    return (int((xs * sal).sum() / total), int((ys * sal).sum() / total))


# ---------------------------------------------------------------------------
# Scene-change detection
# ---------------------------------------------------------------------------
def is_scene_change(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    threshold: float = 0.35,
) -> bool:
    if prev_gray is None:
        return False
    try:
        return float(cv2.absdiff(prev_gray, curr_gray).mean()) / 255.0 > threshold
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rule-of-thirds bias
# ---------------------------------------------------------------------------
def apply_rule_of_thirds_bias(
    cx: int, cy: int,
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
    bias: float = 0.25,
) -> Tuple[int, int]:
    tx = min([orig_w // 3, 2 * orig_w // 3], key=lambda x: abs(x - cx))
    ty = min([orig_h // 3, 2 * orig_h // 3], key=lambda y: abs(y - cy))
    nx = int(cx + bias * (tx - cx))
    ny = int(cy + bias * (ty - cy))
    hw, hh = crop_w // 2, crop_h // 2
    return (max(hw, min(nx, orig_w - hw)), max(hh, min(ny, orig_h - hh)))


# ---------------------------------------------------------------------------
# Smoothing  (fixed: odd window enforced; correct segment boundary handling)
# ---------------------------------------------------------------------------
def smooth_centers(
    centers: List[Tuple[int, int]],
    window: int = 15,
    scene_cuts: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """Gaussian-smooth crop path; hard-reset at scene-cut boundaries."""
    if not centers or len(centers) < 3:
        return centers.copy() if centers else []

    # Ensure window is odd
    window = window if window % 2 == 1 else window + 1

    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    cut_set = set(scene_cuts or [])

    def _smooth_segment(x_seg: np.ndarray, y_seg: np.ndarray, w: int):
        if len(x_seg) < 3:
            return x_seg.copy(), y_seg.copy()
        # Make w odd and at most len-2
        w = min(w, len(x_seg) - 1)
        w = w if w % 2 == 1 else w - 1
        if w < 3:
            return x_seg.copy(), y_seg.copy()
        h2 = w // 2
        sigma = h2 / 2.0 + 1e-6
        k = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
        k /= k.sum()
        xp = np.pad(x_seg, h2, mode="reflect")
        yp = np.pad(y_seg, h2, mode="reflect")
        return (np.convolve(xp, k, mode="valid")[:len(x_seg)],
                np.convolve(yp, k, mode="valid")[:len(y_seg)])

    if not cut_set:
        xs_s, ys_s = _smooth_segment(xs, ys, window)
        return [(int(x), int(y)) for x, y in zip(xs_s, ys_s)]

    result_x, result_y = xs.copy(), ys.copy()
    cuts = sorted(cut_set)
    boundaries = [0] + cuts + [len(centers)]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 3:
            continue
        xs_s, ys_s = _smooth_segment(xs[s:e], ys[s:e], window)
        result_x[s:e] = xs_s
        result_y[s:e] = ys_s

    return [(int(x), int(y)) for x, y in zip(result_x, result_y)]


# ---------------------------------------------------------------------------
# Interpolation  (fixed: O(n log n) via bisect instead of O(n²))
# ---------------------------------------------------------------------------
def interpolate_centers(
    detected_centers: List[Tuple[int, int]],
    detected_indices: List[int],
    total_frames: int,
) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        return []
    if not detected_centers:
        return [(0, 0)] * total_frames

    result: List[Tuple[int, int]] = []
    n = len(detected_indices)

    for fi in range(total_frames):
        if fi <= detected_indices[0]:
            result.append(detected_centers[0])
            continue
        if fi >= detected_indices[-1]:
            result.append(detected_centers[-1])
            continue

        # bisect_right gives insertion point → right neighbor index
        right = bisect.bisect_right(detected_indices, fi)
        left  = right - 1

        if right >= n:
            result.append(detected_centers[-1])
            continue

        span = max(detected_indices[right] - detected_indices[left], 1)
        t    = (fi - detected_indices[left]) / span
        cx   = int(detected_centers[left][0] + t * (detected_centers[right][0] - detected_centers[left][0]))
        cy   = int(detected_centers[left][1] + t * (detected_centers[right][1] - detected_centers[left][1]))
        result.append((cx, cy))

    # Safety pad
    while len(result) < total_frames:
        result.append(result[-1] if result else (0, 0))
    return result[:total_frames]


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------
def calculate_crop_dimensions(
    orig_w: int, orig_h: int,
    target_w: int, target_h: int,
) -> Tuple[int, int]:
    ratio = target_w / target_h
    if (orig_w / orig_h) > ratio:
        crop_h = orig_h
        crop_w = int(round(crop_h * ratio))
    else:
        crop_w = orig_w
        crop_h = int(round(crop_w / ratio))
    return min(crop_w, orig_w), min(crop_h, orig_h)


# ---------------------------------------------------------------------------
# Main processing function
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
    scene_cut_threshold: float = 0.35,
    output_fps: Optional[float] = None,       # None = keep source fps
    crf: int = 23,                             # H.264 quality: 0=lossless, 51=worst
    encoder_preset: str = "fast",             # ultrafast/fast/medium/slow
    audio_bitrate: str = "128k",              # AAC audio bitrate
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:

    def _p(v: float, msg: str = ""):
        if progress_callback:
            try:
                progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception:
                pass

    # ── Validate ────────────────────────────────────────────────────────
    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")
    if os.path.getsize(input_path) / (1024 ** 2) > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    info = get_video_info(input_path)
    fps, total_frames = info["fps"], info["total_frames"]
    orig_w, orig_h    = info["width"], info["height"]
    duration_sec      = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ProcessingError("Could not read video metadata — file may be corrupt.")
    if not info["is_landscape"]:
        raise ProcessingError(
            "Video is already vertical/square. This tool converts landscape → portrait."
        )

    if not sample_interval:
        sample_interval = max(1, int(fps / 2))

    # Determine output frame rate
    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps

    target_w, target_h = target_size
    crop_w, crop_h = calculate_crop_dimensions(orig_w, orig_h, target_w, target_h)
    _p(0.02, f"📐 Crop window {crop_w}×{crop_h} → output {target_w}×{target_h}")

    # ── Load model ──────────────────────────────────────────────────────
    _p(0.05, "🤖 Loading AI model…")
    model = _get_model(yolo_weights)

    # ── Phase 1: Detection pass ─────────────────────────────────────────
    _p(0.08, f"🔎 Analysing {total_frames} frames…")

    # Separate YOLO detections from saliency fallbacks to avoid polluting path
    detected_centers: List[Tuple[int, int]] = []
    detected_indices: List[int]             = []
    saliency_centers: List[Tuple[int, int]] = []
    saliency_indices: List[int]             = []
    scene_cuts:       List[int]             = []

    cap            = cv2.VideoCapture(input_path)
    prev_gray_full: Optional[np.ndarray] = None
    # optical flow uses its own prev frame, updated every sampled frame
    prev_gray_flow: Optional[np.ndarray] = None
    frame_idx = 0
    report_every = max(1, total_frames // 25)

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Scene-change detection (every frame — very cheap)
        if is_scene_change(prev_gray_full, curr_gray_full, scene_cut_threshold):
            scene_cuts.append(frame_idx)
            prev_gray_flow = None  # reset optical flow across scene cuts
        prev_gray_full = curr_gray_full

        if frame_idx % sample_interval == 0:
            center = detect_subject_center(frame, model, confidence)

            if center is None and use_optical_flow:
                small = cv2.resize(curr_gray_full, (orig_w // 2, orig_h // 2))
                if prev_gray_flow is not None:
                    fc = optical_flow_center(prev_gray_flow, small,
                                             orig_w // 2, orig_h // 2)
                    if fc is not None:
                        center = (fc[0] * 2, fc[1] * 2)
                # Always update flow prev for sampled frames
                prev_gray_flow = small

            if center is not None:
                detected_centers.append(center)
                detected_indices.append(frame_idx)
            else:
                saliency_centers.append(saliency_center(frame))
                saliency_indices.append(frame_idx)

        frame_idx += 1
        if frame_idx % report_every == 0:
            _p(0.08 + 0.32 * (frame_idx / total_frames),
               f"🔎 Analysed {frame_idx}/{total_frames} frames…")

    cap.release()
    _p(0.40,
       f"📍 {len(detected_centers)} AI anchors · "
       f"{len(saliency_centers)} saliency fills · "
       f"{len(scene_cuts)} scene cuts")

    # Use only YOLO/flow detections for the primary path.
    # Fill gaps with saliency only where no detection exists.
    if not detected_centers:
        # Full fallback: use saliency for everything
        detected_centers = saliency_centers or [(orig_w // 2, orig_h // 2)]
        detected_indices = saliency_indices  or [0]
    else:
        # Merge saliency into gaps between detections (do NOT add if nearby detection exists)
        gap_threshold = sample_interval * 4
        for si, sc in zip(saliency_indices, saliency_centers):
            nearest_dist = min(abs(si - di) for di in detected_indices)
            if nearest_dist > gap_threshold:
                detected_indices.append(si)
                detected_centers.append(sc)
        # Re-sort by frame index
        pairs = sorted(zip(detected_indices, detected_centers))
        detected_indices = [p[0] for p in pairs]
        detected_centers = [p[1] for p in pairs]

    # ── Phase 2: Path computation ────────────────────────────────────────
    _p(0.42, "📈 Computing smooth tracking path…")
    all_centers = interpolate_centers(detected_centers, detected_indices, total_frames)
    all_centers = smooth_centers(all_centers, window=smooth_window, scene_cuts=scene_cuts)

    if rule_of_thirds:
        all_centers = [
            apply_rule_of_thirds_bias(cx, cy, orig_w, orig_h, crop_w, crop_h, 0.25)
            for cx, cy in all_centers
        ]

    # Guarantee length matches total_frames
    if len(all_centers) < total_frames:
        all_centers += [all_centers[-1]] * (total_frames - len(all_centers))
    all_centers = all_centers[:total_frames]

    # ── Phase 3: Render frames to temp file ──────────────────────────────
    _p(0.46, "✂️ Rendering vertical frames…")
    temp_video: Optional[str] = None
    temp_final: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
            temp_video = tmp.name

        # Use MJPG for intermediate — avoids mp4v container fragility
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_video, fourcc, render_fps, (target_w, target_h))
        if not writer.isOpened():
            raise ProcessingError("Failed to initialise video encoder.")

        cap = cv2.VideoCapture(input_path)
        report_every_render = max(1, total_frames // 25)
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cx, cy = all_centers[frame_num]
            left = max(0, min(cx - crop_w // 2, orig_w - crop_w))
            top  = max(0, min(cy - crop_h // 2, orig_h - crop_h))
            cropped = frame[top:top + crop_h, left:left + crop_w]
            writer.write(cv2.resize(cropped, (target_w, target_h),
                                    interpolation=cv2.INTER_LANCZOS4))
            if (frame_num + 1) % report_every_render == 0:
                _p(0.46 + 0.40 * ((frame_num + 1) / total_frames),
                   f"✂️ Rendered {frame_num + 1}/{total_frames} frames…")
        cap.release()
        writer.release()

        if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 1000:
            raise ProcessingError("Rendered video file is empty.")

        # ── Phase 4: FFmpeg — audio mux + browser encode ─────────────────
        _p(0.87, "🎵 Muxing audio & encoding with FFmpeg…")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp2:
            temp_final = tmp2.name

        if _ffmpeg_has_audio(input_path):
            _ffmpeg_mux_audio_and_encode(
                temp_video, input_path, temp_final,
                render_fps, duration_sec,
                crf=crf, encoder_preset=encoder_preset,
                audio_bitrate=audio_bitrate,
            )
        else:
            _ffmpeg_encode_video_only(
                temp_video, temp_final, render_fps,
                crf=crf, encoder_preset=encoder_preset,
            )

        if not os.path.exists(temp_final) or os.path.getsize(temp_final) < 1000:
            raise ProcessingError("FFmpeg produced an empty output file.")

        shutil.move(temp_final, output_path)
        temp_final = None  # moved — don't delete in finally

        _p(1.0, "✅ Done!")
        print(
            f"✅  {output_path}  "
            f"({os.path.getsize(output_path) / 1024 ** 2:.1f} MB)",
            flush=True,
        )

    finally:
        for p in (temp_video, temp_final):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
