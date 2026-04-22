"""
verticalize.py
──────────────
Convert landscape video to vertical format using AI subject tracking.

Key improvements over previous version:
  - FFmpeg rawvideo pipe — eliminates heavy MJPG intermediate file
  - Upscale guard — output resolution capped to source resolution
  - Multi-subject union framing — keeps ALL detected subjects in crop window
  - Motion-aware adaptive smoothing — fast subjects get shorter window
  - Look-room bias — crops ahead of subject movement direction
  - Velocity-weighted rule-of-thirds — bias scales with stillness

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

# Adaptive smoothing: velocity (px/frame) → window size mapping
# High velocity = short window (snappy); low velocity = long window (steady)
VELOCITY_SMOOTH_MAP: List[Tuple[float, int]] = [
    (0.0,  31),   # static subject
    (5.0,  25),
    (15.0, 17),
    (30.0, 11),
    (60.0,  7),   # fast-moving subject
    (120.0, 3),   # very fast (car, sports)
]

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Match source (recommended)": (0, 0),          # sentinel: filled at runtime
    "1080p  (1080×1920 — Full HD)":  (1080, 1920),
    "720p   (720×1280  — HD)":       (720,  1280),
    "540p   (540×960   — SD)":       (540,  960),
    "480p   (480×854   — Low)":      (480,  854),
}

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True,
                           capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(
                f"{tool} not found. Please install FFmpeg and ensure it is on your PATH."
            )


def _ffmpeg_pipe_encode(
    input_pipe,               # open binary pipe (stdin source)
    source_audio_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    duration_seconds: float,
    has_audio: bool,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
) -> None:
    """
    Single FFmpeg process reading rawvideo frames from stdin.
    Eliminates the heavy MJPG intermediate file entirely.
    """
    audio_args = ["-i", source_audio_path,
                  "-map", "0:v:0", "-map", "1:a:0?",
                  "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"] if has_audio \
                 else ["-map", "0:v:0", "-an"]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ] + (["-i", source_audio_path] if has_audio else []) + [
        "-map", "0:v:0",
    ] + (["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
         if has_audio else ["-an"]) + [
        "-c:v", "libx264",
        "-preset", encoder_preset,
        "-crf", str(crf),
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-t", str(duration_seconds),
        "-movflags", "+faststart",
        output_path,
    ]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def _ffmpeg_has_audio(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", path,
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
# Upscale-safe target resolution
# ---------------------------------------------------------------------------
def resolve_target_size(
    preset_label: str,
    orig_w: int,
    orig_h: int,
    presets: Dict[str, Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Returns final (target_w, target_h) never exceeding source dimensions.
    'Match source' preset chooses the largest 9:16 that fits inside orig_h.
    """
    tw, th = presets[preset_label]

    if tw == 0 and th == 0:
        # Match source: largest 9:16 crop that fits inside the source frame
        # crop_h = orig_h, crop_w = orig_h * 9/16  ≤ orig_w
        crop_w = int(orig_h * 9 / 16)
        if crop_w > orig_w:
            crop_w = orig_w
            th = int(crop_w * 16 / 9)
            tw = crop_w
        else:
            tw, th = crop_w, orig_h
        # round to even
        tw = tw - (tw % 2)
        th = th - (th % 2)
        return tw, th

    # Hard cap: never upscale beyond source height
    max_h = orig_h
    max_w = int(max_h * tw / th)
    if tw > max_w or th > max_h:
        scale = min(max_w / tw, max_h / th)
        tw = int(tw * scale) - (int(tw * scale) % 2)
        th = int(th * scale) - (int(th * scale) % 2)

    return tw, th


# ---------------------------------------------------------------------------
# Model loader
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
# Detection result dataclass (named tuple for speed)
# ---------------------------------------------------------------------------
from collections import namedtuple
DetectionResult = namedtuple("DetectionResult", [
    "center_x", "center_y",   # weighted centroid of ALL selected subjects
    "union_x1", "union_y1",   # bounding box union of ALL selected subjects
    "union_x2", "union_y2",
    "subject_count",           # how many subjects contributed
])


def detect_subjects(
    frame: np.ndarray,
    model: YOLO,
    confidence: float = 0.5,
) -> Optional[DetectionResult]:
    """
    Runs one YOLO pass. Returns DetectionResult with:
      - weighted centroid (for smooth tracking)
      - bounding-box UNION of all relevant subjects (for framing)
    Priority: persons → high-priority classes → everything else.
    """
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"⚠️ Detection error: {e}", file=sys.stderr)
        return None

    if results.boxes is None or len(results.boxes) == 0:
        return None

    # Collect boxes by priority pool
    person_boxes:  List[Tuple[float, int, int, int, int]] = []  # (weight, x1,y1,x2,y2)
    hiprio_boxes:  List[Tuple[float, int, int, int, int]] = []
    all_boxes:     List[Tuple[float, int, int, int, int]] = []

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area   = max(1, (x2 - x1) * (y2 - y1))
        weight = area * conf
        entry  = (weight, x1, y1, x2, y2)
        if cls == PERSON_CLASS_ID:
            person_boxes.append(entry)
        elif cls in HIGH_PRIO_CLASSES:
            hiprio_boxes.append(entry)
        all_boxes.append(entry)

    pool = person_boxes or hiprio_boxes or all_boxes
    if not pool:
        return None

    total_w = sum(e[0] for e in pool)
    if total_w == 0:
        return None

    # Weighted centroid
    cx = int(sum(e[0] * ((e[1] + e[3]) / 2) for e in pool) / total_w)
    cy = int(sum(e[0] * ((e[2] + e[4]) / 2) for e in pool) / total_w)

    # Union bounding box — ensures ALL subjects remain visible
    ux1 = min(e[1] for e in pool)
    uy1 = min(e[2] for e in pool)
    ux2 = max(e[3] for e in pool)
    uy2 = max(e[4] for e in pool)

    return DetectionResult(cx, cy, ux1, uy1, ux2, uy2, len(pool))


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
# Multi-subject framing: ensure union bbox fits inside crop window
# ---------------------------------------------------------------------------
def frame_for_union(
    ux1: int, uy1: int, ux2: int, uy2: int,
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
) -> Tuple[int, int]:
    """
    Given the union bounding box of all subjects, return a crop center
    that keeps as much of the union bbox visible as possible.
    If union is wider/taller than crop, centers on union centroid.
    """
    ucx = (ux1 + ux2) // 2
    ucy = (uy1 + uy2) // 2
    union_w = ux2 - ux1
    union_h = uy2 - uy1

    # If all subjects fit inside crop, use union centroid directly
    if union_w <= crop_w and union_h <= crop_h:
        cx, cy = ucx, ucy
    else:
        # Scale-to-fit: center on centroid, subjects at edges are clipped
        cx, cy = ucx, ucy

    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(cx, orig_w - hw))
    cy = max(hh, min(cy, orig_h - hh))
    return cx, cy


# ---------------------------------------------------------------------------
# Motion-aware look-room bias (replaces dumb rule-of-thirds)
# ---------------------------------------------------------------------------
def apply_look_room_bias(
    cx: int, cy: int,
    vx: float, vy: float,          # velocity vector (px / frame)
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
    speed: float,                  # magnitude of velocity
    rot_bias: float = 0.25,        # rule-of-thirds bias when still
    look_room_max: float = 0.18,   # max fractional look-room offset
) -> Tuple[int, int]:
    """
    When subject is moving: shift crop ahead of motion direction (look-room).
    When subject is still:  apply light rule-of-thirds bias.
    Blends between the two based on speed.
    """
    hw, hh = crop_w // 2, crop_h // 2

    # ── Look-room (motion-based) ────────────────────────────────────
    look_strength = min(speed / 40.0, 1.0)   # saturates at 40 px/frame
    if look_strength > 0.05 and speed > 0.5:
        norm = speed + 1e-6
        look_x = int(cx + (vx / norm) * look_room_max * crop_w * look_strength)
        look_y = int(cy + (vy / norm) * look_room_max * crop_h * look_strength)
    else:
        look_x, look_y = cx, cy

    # ── Rule-of-thirds (stillness-based) ────────────────────────────
    still_strength = max(0.0, 1.0 - look_strength * 2)
    if still_strength > 0.01:
        tx = min([orig_w // 3, 2 * orig_w // 3], key=lambda x: abs(x - cx))
        ty = min([orig_h // 3, 2 * orig_h // 3], key=lambda y: abs(y - cy))
        rot_x = int(cx + rot_bias * still_strength * (tx - cx))
        rot_y = int(cy + rot_bias * still_strength * (ty - cy))
    else:
        rot_x, rot_y = cx, cy

    # Blend
    nx = int(look_x * look_strength + rot_x * (1 - look_strength))
    ny = int(look_y * look_strength + rot_y * (1 - look_strength))

    return (max(hw, min(nx, orig_w - hw)), max(hh, min(ny, orig_h - hh)))


# ---------------------------------------------------------------------------
# Adaptive smoothing window from velocity
# ---------------------------------------------------------------------------
def velocity_to_window(speed: float) -> int:
    """Interpolate smoothing window from VELOCITY_SMOOTH_MAP."""
    table = VELOCITY_SMOOTH_MAP
    if speed <= table[0][0]:
        return table[0][1]
    if speed >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        v0, w0 = table[i]
        v1, w1 = table[i + 1]
        if v0 <= speed <= v1:
            t = (speed - v0) / (v1 - v0)
            w = int(w0 + t * (w1 - w0))
            return w if w % 2 == 1 else w + 1
    return 15


# ---------------------------------------------------------------------------
# Adaptive smoothing (per-segment, variable window)
# ---------------------------------------------------------------------------
def smooth_centers_adaptive(
    centers: List[Tuple[int, int]],
    velocities: List[float],       # per-frame speed in px/frame
    base_window: int = 15,
    scene_cuts: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """
    Gaussian smooth with a window that adapts to subject velocity.
    Fast segments get shorter windows; slow segments get longer windows.
    """
    if not centers or len(centers) < 3:
        return centers.copy() if centers else []

    n = len(centers)
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    cut_set = set(scene_cuts or [])

    def _smooth_seg(x_s, y_s, speeds):
        seg_len = len(x_s)
        if seg_len < 3:
            return x_s.copy(), y_s.copy()
        # Compute per-frame window sizes from velocity
        windows = np.array([velocity_to_window(float(s)) for s in speeds])
        # Clamp to segment length
        windows = np.clip(windows, 3, seg_len - 1 if seg_len > 3 else 3)
        # Make odd
        windows = np.where(windows % 2 == 0, windows + 1, windows)

        # Use median window for the segment (keep it simple, still adaptive)
        w = int(np.median(windows))
        w = w if w % 2 == 1 else w + 1
        w = min(w, seg_len - 1)
        if w < 3:
            return x_s.copy(), y_s.copy()

        h2 = w // 2
        sigma = h2 / 2.0 + 1e-6
        k = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
        k /= k.sum()
        xp = np.pad(x_s, h2, mode="reflect")
        yp = np.pad(y_s, h2, mode="reflect")
        return (np.convolve(xp, k, mode="valid")[:seg_len],
                np.convolve(yp, k, mode="valid")[:seg_len])

    vels = np.array(velocities[:n], dtype=float)
    if len(vels) < n:
        vels = np.pad(vels, (0, n - len(vels)), mode="edge")

    if not cut_set:
        xs_s, ys_s = _smooth_seg(xs, ys, vels)
        return [(int(x), int(y)) for x, y in zip(xs_s, ys_s)]

    result_x, result_y = xs.copy(), ys.copy()
    cuts = sorted(cut_set)
    boundaries = [0] + cuts + [n]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 3:
            continue
        xs_s, ys_s = _smooth_seg(xs[s:e], ys[s:e], vels[s:e])
        result_x[s:e] = xs_s
        result_y[s:e] = ys_s

    return [(int(x), int(y)) for x, y in zip(result_x, result_y)]


# ---------------------------------------------------------------------------
# Interpolation  O(n log n)
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
# Per-frame velocity computation
# ---------------------------------------------------------------------------
def compute_velocities(centers: List[Tuple[int, int]], window: int = 5) -> List[float]:
    """Compute smoothed per-frame speed (px/frame)."""
    n = len(centers)
    if n < 2:
        return [0.0] * n
    speeds = [0.0]
    for i in range(1, n):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        speeds.append(float(np.sqrt(dx*dx + dy*dy)))
    # Smooth speeds with a short box filter
    w = min(window, n)
    kernel = np.ones(w) / w
    smoothed = np.convolve(speeds, kernel, mode="same")
    return smoothed.tolist()


# ---------------------------------------------------------------------------
# Per-frame velocity vector (for look-room direction)
# ---------------------------------------------------------------------------
def compute_velocity_vectors(
    centers: List[Tuple[int, int]], lookahead: int = 4
) -> List[Tuple[float, float]]:
    """
    Returns (vx, vy) per frame using lookahead for stability.
    Uses forward difference where possible, backward at end.
    """
    n = len(centers)
    vecs: List[Tuple[float, float]] = []
    for i in range(n):
        j = min(i + lookahead, n - 1)
        k = max(i - lookahead, 0)
        if j > k:
            vx = (centers[j][0] - centers[k][0]) / (j - k)
            vy = (centers[j][1] - centers[k][1]) / (j - k)
        else:
            vx, vy = 0.0, 0.0
        vecs.append((vx, vy))
    return vecs


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_video(
    input_path: str,
    output_path: str,
    sample_interval: Optional[int] = None,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    target_preset_label: str = "",          # if set, used with resolve_target_size
    confidence: float = 0.5,
    smooth_window: int = 15,                # base window; overridden by adaptive
    adaptive_smoothing: bool = True,        # enable velocity-adaptive smoothing
    yolo_weights: str = "yolov8n.pt",
    use_optical_flow: bool = True,
    rule_of_thirds: bool = True,
    scene_cut_threshold: float = 0.35,
    output_fps: Optional[float] = None,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
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

    # ── Resolve target size (with upscale guard) ─────────────────────────
    if target_preset_label and target_preset_label in RESOLUTION_PRESETS:
        target_w, target_h = resolve_target_size(
            target_preset_label, orig_w, orig_h, RESOLUTION_PRESETS
        )
    else:
        target_w, target_h = target_size
        # Upscale guard: never produce output taller than source
        if target_h > orig_h:
            scale = orig_h / target_h
            target_h = int(orig_h - (orig_h % 2))
            target_w = int(target_w * scale)
            target_w = target_w - (target_w % 2)
            _p(0.01, f"⚠️ Clamped output to {target_w}×{target_h} (source is {orig_w}×{orig_h})")

    if not sample_interval:
        sample_interval = max(1, int(fps / 2))

    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps

    crop_w, crop_h = calculate_crop_dimensions(orig_w, orig_h, target_w, target_h)
    _p(0.02, f"📐 Crop {crop_w}×{crop_h} from {orig_w}×{orig_h} → output {target_w}×{target_h}")

    # ── Load model ──────────────────────────────────────────────────────
    _p(0.05, "🤖 Loading AI model…")
    model = _get_model(yolo_weights)

    # ── Phase 1: Detection pass ─────────────────────────────────────────
    _p(0.08, f"🔎 Analysing {total_frames} frames…")

    detected_centers:  List[Tuple[int, int]] = []
    detected_indices:  List[int]             = []
    detected_unions:   List[Tuple[int,int,int,int]] = []  # (ux1,uy1,ux2,uy2)
    saliency_centers:  List[Tuple[int, int]] = []
    saliency_indices:  List[int]             = []
    scene_cuts:        List[int]             = []

    cap            = cv2.VideoCapture(input_path)
    prev_gray_full: Optional[np.ndarray] = None
    prev_gray_flow: Optional[np.ndarray] = None
    frame_idx = 0
    report_every = max(1, total_frames // 25)

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_scene_change(prev_gray_full, curr_gray_full, scene_cut_threshold):
            scene_cuts.append(frame_idx)
            prev_gray_flow = None
        prev_gray_full = curr_gray_full

        if frame_idx % sample_interval == 0:
            det = detect_subjects(frame, model, confidence)
            center: Optional[Tuple[int, int]] = None
            union: Optional[Tuple[int,int,int,int]] = None

            if det is not None:
                # Use union-bbox framing instead of raw centroid
                center = frame_for_union(
                    det.union_x1, det.union_y1, det.union_x2, det.union_y2,
                    orig_w, orig_h, crop_w, crop_h,
                )
                union = (det.union_x1, det.union_y1, det.union_x2, det.union_y2)
            elif use_optical_flow:
                small = cv2.resize(curr_gray_full, (orig_w // 2, orig_h // 2))
                if prev_gray_flow is not None:
                    fc = optical_flow_center(prev_gray_flow, small,
                                             orig_w // 2, orig_h // 2)
                    if fc is not None:
                        center = (fc[0] * 2, fc[1] * 2)
                prev_gray_flow = small

            if center is not None:
                detected_centers.append(center)
                detected_indices.append(frame_idx)
                if union:
                    detected_unions.append(union)
                else:
                    detected_unions.append((center[0], center[1], center[0], center[1]))
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

    # Merge saliency into genuine gaps
    if not detected_centers:
        detected_centers = saliency_centers or [(orig_w // 2, orig_h // 2)]
        detected_indices = saliency_indices  or [0]
        detected_unions  = [(c[0], c[1], c[0], c[1]) for c in detected_centers]
    else:
        gap_threshold = sample_interval * 4
        for si, sc in zip(saliency_indices, saliency_centers):
            nearest_dist = min(abs(si - di) for di in detected_indices)
            if nearest_dist > gap_threshold:
                detected_indices.append(si)
                detected_centers.append(sc)
                detected_unions.append((sc[0], sc[1], sc[0], sc[1]))
        pairs = sorted(zip(detected_indices, detected_centers, detected_unions))
        detected_indices = [p[0] for p in pairs]
        detected_centers = [p[1] for p in pairs]
        detected_unions  = [p[2] for p in pairs]

    # ── Phase 2: Path computation ────────────────────────────────────────
    _p(0.42, "📈 Computing adaptive tracking path…")
    all_centers = interpolate_centers(detected_centers, detected_indices, total_frames)

    # Compute velocities for adaptive smoothing and look-room
    velocities = compute_velocities(all_centers, window=5)
    vel_vecs   = compute_velocity_vectors(all_centers, lookahead=4)

    if adaptive_smoothing:
        all_centers = smooth_centers_adaptive(
            all_centers, velocities, base_window=smooth_window, scene_cuts=scene_cuts
        )
    else:
        # Fixed-window fallback
        from functools import reduce
        all_centers = _smooth_fixed(all_centers, smooth_window, scene_cuts)

    # Re-compute velocities after smoothing (for framing)
    velocities = compute_velocities(all_centers, window=3)
    vel_vecs   = compute_velocity_vectors(all_centers, lookahead=3)

    # Apply look-room + rule-of-thirds bias
    if rule_of_thirds:
        framed = []
        for i, (cx, cy) in enumerate(all_centers):
            vx, vy = vel_vecs[i]
            speed  = velocities[i]
            framed.append(apply_look_room_bias(
                cx, cy, vx, vy, orig_w, orig_h, crop_w, crop_h, speed,
            ))
        all_centers = framed

    # Final clamp
    hw, hh = crop_w // 2, crop_h // 2
    all_centers = [
        (max(hw, min(cx, orig_w - hw)), max(hh, min(cy, orig_h - hh)))
        for cx, cy in all_centers
    ]

    if len(all_centers) < total_frames:
        all_centers += [all_centers[-1]] * (total_frames - len(all_centers))
    all_centers = all_centers[:total_frames]

    # ── Phase 3: Render via FFmpeg pipe (no intermediate file) ───────────
    _p(0.46, "✂️ Rendering & encoding via FFmpeg pipe…")
    has_audio = _ffmpeg_has_audio(input_path)

    temp_final: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_final = tmp.name

        ffproc = _ffmpeg_pipe_encode(
            None,                    # stdin handled below
            input_path,
            temp_final,
            width=target_w, height=target_h,
            fps=render_fps,
            duration_seconds=duration_sec,
            has_audio=has_audio,
            crf=crf,
            encoder_preset=encoder_preset,
            audio_bitrate=audio_bitrate,
        )

        cap = cv2.VideoCapture(input_path)
        report_every_render = max(1, total_frames // 40)
        ffmpeg_error = b""

        try:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                cx, cy = all_centers[frame_num]
                left = max(0, min(cx - crop_w // 2, orig_w - crop_w))
                top  = max(0, min(cy - crop_h // 2, orig_h - crop_h))
                cropped = frame[top:top + crop_h, left:left + crop_w]
                if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
                    cropped = cv2.resize(cropped, (target_w, target_h),
                                         interpolation=cv2.INTER_LANCZOS4)
                try:
                    ffproc.stdin.write(cropped.tobytes())
                except BrokenPipeError:
                    ffmpeg_error = ffproc.stderr.read()
                    raise ProcessingError(
                        f"FFmpeg pipe closed unexpectedly:\n"
                        f"{ffmpeg_error.decode('utf-8', errors='replace')[-1500:]}"
                    )

                if (frame_num + 1) % report_every_render == 0:
                    _p(0.46 + 0.42 * ((frame_num + 1) / total_frames),
                       f"✂️ {frame_num + 1}/{total_frames} frames…")
        finally:
            cap.release()
            ffproc.stdin.close()

        _, stderr_bytes = ffproc.communicate()
        if ffproc.returncode != 0:
            raise ProcessingError(
                f"FFmpeg encode failed (rc={ffproc.returncode}):\n"
                f"{stderr_bytes.decode('utf-8', errors='replace')[-2000:]}"
            )

        if not os.path.exists(temp_final) or os.path.getsize(temp_final) < 1000:
            raise ProcessingError("FFmpeg produced an empty output file.")

        shutil.move(temp_final, output_path)
        temp_final = None

        _p(1.0, "✅ Done!")
        print(f"✅  {output_path}  ({os.path.getsize(output_path) / 1024**2:.1f} MB)", flush=True)

    finally:
        if temp_final and os.path.exists(temp_final):
            try:
                os.unlink(temp_final)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Fallback fixed-window smoother (used when adaptive_smoothing=False)
# ---------------------------------------------------------------------------
def _smooth_fixed(
    centers: List[Tuple[int, int]],
    window: int = 15,
    scene_cuts: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    if not centers or len(centers) < 3:
        return centers.copy() if centers else []
    window = window if window % 2 == 1 else window + 1
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    cut_set = set(scene_cuts or [])

    def _seg(x_s, y_s):
        seg_len = len(x_s)
        if seg_len < 3:
            return x_s.copy(), y_s.copy()
        w = min(window, seg_len - 1)
        w = w if w % 2 == 1 else w - 1
        if w < 3:
            return x_s.copy(), y_s.copy()
        h2 = w // 2
        sigma = h2 / 2.0 + 1e-6
        k = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
        k /= k.sum()
        return (np.convolve(np.pad(x_s, h2, "reflect"), k, "valid")[:seg_len],
                np.convolve(np.pad(y_s, h2, "reflect"), k, "valid")[:seg_len])

    if not cut_set:
        xs_s, ys_s = _seg(xs, ys)
        return [(int(x), int(y)) for x, y in zip(xs_s, ys_s)]

    rx, ry = xs.copy(), ys.copy()
    cuts = sorted(cut_set)
    boundaries = [0] + cuts + [len(centers)]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 3:
            continue
        xs_s, ys_s = _seg(xs[s:e], ys[s:e])
        rx[s:e] = xs_s
        ry[s:e] = ys_s
    return [(int(x), int(y)) for x, y in zip(rx, ry)]
