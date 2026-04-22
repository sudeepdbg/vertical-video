"""
verticalize.py
──────────────
Convert landscape video to vertical format using AI subject tracking.

Improvements over original:
  • Upscale guard — output capped to source resolution
  • Multi-subject union framing — all detected subjects stay in crop
  • Motion-aware adaptive smoothing — velocity-driven window
  • Look-room bias — crop leads subject's direction of travel
  • Reliable render via MJPG temp → FFmpeg (pipe approach abandoned:
    too fragile under memory pressure on shared hosts)
  • ffprobe verified at startup
  • O(n log n) interpolation via bisect

Dependencies: opencv-python, ultralytics, numpy, ffmpeg (system)
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
from collections import namedtuple
from typing import Optional, Callable, List, Tuple, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
class ProcessingError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
PERSON_CLASS_ID   = 0
HIGH_PRIO_CLASSES = {0, 2, 3, 5, 7, 15, 16}   # person, car, motorbike, bus, truck, cat, dog
DEFAULT_TARGET_SIZE: Tuple[int, int] = (1080, 1920)
MAX_FILE_SIZE_MB  = 500
MIN_FRAME_DIM     = 240
MAX_FRAMES_GUARD  = 216_000  # 1 hr @ 60 fps

# Adaptive smoothing: velocity (px/frame) → Gaussian window size
# Sorted ascending by velocity. Values are ODD.
VELOCITY_SMOOTH_TABLE: List[Tuple[float, int]] = [
    (0.0,   31),
    (5.0,   25),
    (15.0,  17),
    (30.0,  11),
    (60.0,   7),
    (120.0,  3),
]

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Match source (no upscale)":     (0, 0),      # sentinel; resolved at runtime
    "1080p  (1080×1920 — Full HD)":  (1080, 1920),
    "720p   (720×1280  — HD)":       (720,  1280),
    "540p   (540×960   — SD)":       (540,  960),
    "480p   (480×854   — Low)":      (480,  854),
}


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True,
                           capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(
                f"{tool} not found. Install FFmpeg and add it to PATH."
            )


def _ffmpeg_encode(
    video_path: str,
    audio_source: Optional[str],
    output_path: str,
    fps: float,
    duration: float,
    crf: int = 23,
    preset: str = "fast",
    audio_bitrate: str = "128k",
) -> None:
    """
    Re-encode rendered frames (MJPG .avi) → H.264 .mp4, optionally muxing audio.
    Reliable: uses a temp file, not a pipe. No BrokenPipe risk.
    """
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if audio_source:
        cmd += ["-i", audio_source]
    cmd += [
        "-map", "0:v:0",
    ]
    if audio_source:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-t", str(duration),
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ProcessingError(
            f"FFmpeg failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
        )


def _has_audio(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "audio" in r.stdout
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Video metadata
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ProcessingError(f"Cannot open video: {path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": min(nf, MAX_FRAMES_GUARD),
        "width": w,
        "height": h,
        "duration_seconds": nf / fps if fps > 0 else 0.0,
        "is_landscape": w > h,
    }


def extract_thumbnail(path: str, t: float = 1.0) -> Optional[bytes]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ─────────────────────────────────────────────────────────────────────────────
#  Upscale-safe resolution resolver
# ─────────────────────────────────────────────────────────────────────────────
def resolve_target_size(
    label: str,
    orig_w: int,
    orig_h: int,
) -> Tuple[int, int]:
    """
    Returns (target_w, target_h) guaranteed ≤ source dimensions.
    'Match source' → largest exact 9:16 that fits inside the source.
    Other presets  → clamped down if source is too small.
    """
    tw, th = RESOLUTION_PRESETS[label]

    if tw == 0 and th == 0:
        # Match source: fit 9:16 inside orig_h (portrait axis)
        cw = int(orig_h * 9 / 16)
        if cw > orig_w:
            cw = orig_w
            ch = int(cw * 16 / 9)
        else:
            ch = orig_h
        tw = cw - (cw % 2)
        th = ch - (ch % 2)
        return tw, th

    # Clamp: never upscale
    if th > orig_h:
        scale = orig_h / th
        tw = int(tw * scale)
        th = int(orig_h)
    if tw > orig_w:
        scale = orig_w / tw
        tw = int(orig_w)
        th = int(th * scale)
    # Ensure even
    tw = tw - (tw % 2)
    th = th - (th % 2)
    return max(tw, 2), max(th, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  YOLO model cache
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, YOLO] = {}

def _get_model(weights: str = "yolov8n.pt") -> YOLO:
    if weights not in _model_cache:
        try:
            _model_cache[weights] = YOLO(weights)
        except Exception as e:
            raise ProcessingError(f"Failed to load model '{weights}': {e}")
    return _model_cache[weights]


# ─────────────────────────────────────────────────────────────────────────────
#  Detection
# ─────────────────────────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult",
    ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"])


def detect_subjects(
    frame: np.ndarray,
    model: YOLO,
    confidence: float = 0.5,
) -> Optional[DetectionResult]:
    """
    One YOLO pass. Returns weighted centroid + union bbox of all subjects.
    Priority: persons > high-prio objects > everything else.
    """
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"⚠ Detection: {e}", file=sys.stderr)
        return None

    if results.boxes is None or len(results.boxes) == 0:
        return None

    person_pool: List[Tuple[float, int, int, int, int]] = []
    hiprio_pool: List[Tuple[float, int, int, int, int]] = []
    all_pool:    List[Tuple[float, int, int, int, int]] = []

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area   = max(1, (x2 - x1) * (y2 - y1))
        weight = area * conf
        entry  = (weight, x1, y1, x2, y2)
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

    cx = int(sum(e[0] * (e[1] + e[3]) / 2 for e in pool) / total_w)
    cy = int(sum(e[0] * (e[2] + e[4]) / 2 for e in pool) / total_w)
    return DetectionResult(
        cx, cy,
        min(e[1] for e in pool), min(e[2] for e in pool),
        max(e[3] for e in pool), max(e[4] for e in pool),
        len(pool),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-subject framing: keep union bbox in frame
# ─────────────────────────────────────────────────────────────────────────────
def frame_for_union(
    ux1: int, uy1: int, ux2: int, uy2: int,
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
) -> Tuple[int, int]:
    ucx = (ux1 + ux2) // 2
    ucy = (uy1 + uy2) // 2
    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(ucx, orig_w - hw))
    cy = max(hh, min(ucy, orig_h - hh))
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
#  Optical flow fallback
# ─────────────────────────────────────────────────────────────────────────────
def optical_flow_center(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    w: int,
    h: int,
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
        b = max(1, int(w * 0.04))
        mag[:, :b] = mag[:, w - b:] = 0
        mag[:b, :] = mag[h - b:, :] = 0
        if mag.max() < 0.8:
            return None
        total = mag.sum()
        if total == 0:
            return None
        ys, xs = np.mgrid[0:h, 0:w]
        return (int((xs * mag).sum() / total), int((ys * mag).sum() / total))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Saliency fallback
# ─────────────────────────────────────────────────────────────────────────────
def saliency_center(frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIM or h < MIN_FRAME_DIM:
        return w // 2, h // 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31, 31), 0)
    sat  = cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (31, 31), 0)
    sal  = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    b = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w - b:] = sal[:b, :] = sal[h - b:, :] = 0
    total = sal.sum()
    if total < 1e-6:
        return w // 2, h // 2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / total), int((ys * sal).sum() / total)


# ─────────────────────────────────────────────────────────────────────────────
#  Scene-change detection
# ─────────────────────────────────────────────────────────────────────────────
def is_scene_change(
    prev: Optional[np.ndarray],
    curr: np.ndarray,
    threshold: float = 0.35,
) -> bool:
    if prev is None:
        return False
    try:
        return float(cv2.absdiff(prev, curr).mean()) / 255.0 > threshold
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Look-room + rule-of-thirds bias  (motion-aware)
# ─────────────────────────────────────────────────────────────────────────────
def apply_framing_bias(
    cx: int, cy: int,
    vx: float, vy: float,
    speed: float,
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
    look_room_frac: float = 0.18,
    rot_bias: float = 0.25,
) -> Tuple[int, int]:
    """
    Fast subject  → shift crop ahead of motion (look-room).
    Still subject → nudge toward nearest rule-of-thirds line.
    Blends smoothly between the two based on speed.
    """
    hw, hh = crop_w // 2, crop_h // 2

    look_strength = min(speed / 40.0, 1.0)

    # Look-room component
    if look_strength > 0.05:
        norm = speed + 1e-9
        lx = int(cx + (vx / norm) * look_room_frac * crop_w * look_strength)
        ly = int(cy + (vy / norm) * look_room_frac * crop_h * look_strength)
    else:
        lx, ly = cx, cy

    # Rule-of-thirds component (only when still)
    still = max(0.0, 1.0 - look_strength * 2)
    if still > 0.01:
        tx = min([orig_w // 3, 2 * orig_w // 3], key=lambda x: abs(x - cx))
        ty = min([orig_h // 3, 2 * orig_h // 3], key=lambda y: abs(y - cy))
        rx = int(cx + rot_bias * still * (tx - cx))
        ry = int(cy + rot_bias * still * (ty - cy))
    else:
        rx, ry = cx, cy

    # Blend
    nx = int(lx * look_strength + rx * (1.0 - look_strength))
    ny = int(ly * look_strength + ry * (1.0 - look_strength))

    return (max(hw, min(nx, orig_w - hw)), max(hh, min(ny, orig_h - hh)))


# ─────────────────────────────────────────────────────────────────────────────
#  Velocity helpers
# ─────────────────────────────────────────────────────────────────────────────
def _compute_speeds(centers: List[Tuple[int, int]], smooth: int = 5) -> List[float]:
    n = len(centers)
    if n < 2:
        return [0.0] * n
    raw = [0.0]
    for i in range(1, n):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        raw.append(float(np.sqrt(dx * dx + dy * dy)))
    w = min(smooth, n)
    out = np.convolve(raw, np.ones(w) / w, mode="same")
    return out.tolist()


def _compute_vel_vecs(
    centers: List[Tuple[int, int]], look: int = 4
) -> List[Tuple[float, float]]:
    n = len(centers)
    vecs = []
    for i in range(n):
        j = min(i + look, n - 1)
        k = max(i - look, 0)
        span = j - k
        if span > 0:
            vecs.append((
                (centers[j][0] - centers[k][0]) / span,
                (centers[j][1] - centers[k][1]) / span,
            ))
        else:
            vecs.append((0.0, 0.0))
    return vecs


def _velocity_to_window(speed: float) -> int:
    tbl = VELOCITY_SMOOTH_TABLE
    if speed <= tbl[0][0]:
        return tbl[0][1]
    if speed >= tbl[-1][0]:
        return tbl[-1][1]
    for i in range(len(tbl) - 1):
        v0, w0 = tbl[i]
        v1, w1 = tbl[i + 1]
        if v0 <= speed <= v1:
            t = (speed - v0) / (v1 - v0 + 1e-9)
            w = int(w0 + t * (w1 - w0))
            return w if w % 2 == 1 else w + 1
    return 15


# ─────────────────────────────────────────────────────────────────────────────
#  Smoothing (adaptive + fixed-window fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _gauss_smooth_segment(
    xs: np.ndarray, ys: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 3:
        return xs.copy(), ys.copy()
    w = min(window, n - 1)
    w = w if w % 2 == 1 else w - 1
    if w < 3:
        return xs.copy(), ys.copy()
    h2 = w // 2
    sigma = h2 / 2.0 + 1e-9
    k = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(xs, h2, mode="reflect")
    yp = np.pad(ys, h2, mode="reflect")
    return (
        np.convolve(xp, k, mode="valid")[:n],
        np.convolve(yp, k, mode="valid")[:n],
    )


def smooth_centers(
    centers: List[Tuple[int, int]],
    speeds: List[float],
    base_window: int = 15,
    adaptive: bool = True,
    scene_cuts: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    if not centers or len(centers) < 3:
        return centers.copy() if centers else []

    n   = len(centers)
    xs  = np.array([c[0] for c in centers], dtype=float)
    ys  = np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n:
        spd = np.pad(spd, (0, n - len(spd)), mode="edge")

    cut_set = set(scene_cuts or [])
    boundaries = [0] + sorted(cut_set) + [n]

    rx, ry = xs.copy(), ys.copy()
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 3:
            continue
        if adaptive:
            med_speed = float(np.median(spd[s:e]))
            w = _velocity_to_window(med_speed)
        else:
            w = base_window
        xs_s, ys_s = _gauss_smooth_segment(xs[s:e], ys[s:e], w)
        rx[s:e] = xs_s
        ry[s:e] = ys_s

    return [(int(x), int(y)) for x, y in zip(rx, ry)]


# ─────────────────────────────────────────────────────────────────────────────
#  Interpolation O(n log n)
# ─────────────────────────────────────────────────────────────────────────────
def interpolate_centers(
    centers: List[Tuple[int, int]],
    indices: List[int],
    total: int,
) -> List[Tuple[int, int]]:
    if total <= 0:
        return []
    if not centers:
        return [(0, 0)] * total

    n      = len(indices)
    result = []
    for fi in range(total):
        if fi <= indices[0]:
            result.append(centers[0])
            continue
        if fi >= indices[-1]:
            result.append(centers[-1])
            continue
        r = bisect.bisect_right(indices, fi)
        l = r - 1
        if r >= n:
            result.append(centers[-1])
            continue
        span = max(indices[r] - indices[l], 1)
        t  = (fi - indices[l]) / span
        cx = int(centers[l][0] + t * (centers[r][0] - centers[l][0]))
        cy = int(centers[l][1] + t * (centers[r][1] - centers[l][1]))
        result.append((cx, cy))

    while len(result) < total:
        result.append(result[-1] if result else (0, 0))
    return result[:total]


# ─────────────────────────────────────────────────────────────────────────────
#  Crop geometry
# ─────────────────────────────────────────────────────────────────────────────
def calculate_crop_dims(
    orig_w: int, orig_h: int,
    target_w: int, target_h: int,
) -> Tuple[int, int]:
    ratio = target_w / target_h
    if (orig_w / orig_h) > ratio:
        ch = orig_h
        cw = int(round(ch * ratio))
    else:
        cw = orig_w
        ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    sample_interval: Optional[int] = None,
    target_preset_label: str = "Match source (no upscale)",
    confidence: float = 0.45,
    smooth_window: int = 15,
    adaptive_smoothing: bool = True,
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

    # ── Validate ─────────────────────────────────────────────────────────
    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path) / 1024 ** 2 > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    info = get_video_info(input_path)
    fps, total_frames = info["fps"], info["total_frames"]
    orig_w, orig_h    = info["width"], info["height"]
    duration          = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ProcessingError("Corrupt or unreadable video file.")
    if not info["is_landscape"]:
        raise ProcessingError(
            "Video is already vertical/square — please upload a landscape video."
        )

    # Resolve output size (with upscale guard)
    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS \
          else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    _p(0.01, f"📐 Output {target_w}×{target_h} (source {orig_w}×{orig_h})")

    if not sample_interval:
        sample_interval = max(1, int(fps / 2))

    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)

    # ── Load model ────────────────────────────────────────────────────────
    _p(0.05, "🤖 Loading AI model…")
    model = _get_model(yolo_weights)

    # ── Phase 1: Detection ────────────────────────────────────────────────
    _p(0.08, f"🔎 Analysing {total_frames} frames…")

    det_centers: List[Tuple[int, int]] = []
    det_indices: List[int]             = []
    sal_centers: List[Tuple[int, int]] = []
    sal_indices: List[int]             = []
    scene_cuts:  List[int]             = []

    cap           = cv2.VideoCapture(input_path)
    prev_gray:    Optional[np.ndarray] = None
    prev_flow:    Optional[np.ndarray] = None
    frame_idx = 0
    report_n  = max(1, total_frames // 25)

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_scene_change(prev_gray, curr_gray, scene_cut_threshold):
            scene_cuts.append(frame_idx)
            prev_flow = None
        prev_gray = curr_gray

        if frame_idx % sample_interval == 0:
            det    = detect_subjects(frame, model, confidence)
            center = None

            if det is not None:
                center = frame_for_union(
                    det.ux1, det.uy1, det.ux2, det.uy2,
                    orig_w, orig_h, crop_w, crop_h,
                )
            elif use_optical_flow:
                small = cv2.resize(curr_gray, (orig_w // 2, orig_h // 2))
                if prev_flow is not None:
                    fc = optical_flow_center(prev_flow, small, orig_w // 2, orig_h // 2)
                    if fc is not None:
                        center = (fc[0] * 2, fc[1] * 2)
                prev_flow = small

            if center is not None:
                det_centers.append(center)
                det_indices.append(frame_idx)
            else:
                sal_centers.append(saliency_center(frame))
                sal_indices.append(frame_idx)

        frame_idx += 1
        if frame_idx % report_n == 0:
            _p(0.08 + 0.32 * (frame_idx / total_frames),
               f"🔎 {frame_idx}/{total_frames} frames…")

    cap.release()
    _p(0.40, f"📍 {len(det_centers)} AI anchors · {len(scene_cuts)} scene cuts")

    # Merge saliency into gaps
    if not det_centers:
        det_centers = sal_centers or [(orig_w // 2, orig_h // 2)]
        det_indices = sal_indices  or [0]
    else:
        gap = sample_interval * 4
        for si, sc in zip(sal_indices, sal_centers):
            if min(abs(si - di) for di in det_indices) > gap:
                det_indices.append(si)
                det_centers.append(sc)
        pairs = sorted(zip(det_indices, det_centers))
        det_indices = [p[0] for p in pairs]
        det_centers = [p[1] for p in pairs]

    # ── Phase 2: Path ─────────────────────────────────────────────────────
    _p(0.42, "📈 Computing crop path…")
    all_centers = interpolate_centers(det_centers, det_indices, total_frames)
    speeds      = _compute_speeds(all_centers)
    all_centers = smooth_centers(
        all_centers, speeds, base_window=smooth_window,
        adaptive=adaptive_smoothing, scene_cuts=scene_cuts,
    )

    # Re-derive velocity after smoothing for framing bias
    speeds   = _compute_speeds(all_centers, smooth=3)
    vel_vecs = _compute_vel_vecs(all_centers, look=3)

    if rule_of_thirds:
        framed = []
        for i, (cx, cy) in enumerate(all_centers):
            vx, vy = vel_vecs[i]
            framed.append(apply_framing_bias(
                cx, cy, vx, vy, speeds[i],
                orig_w, orig_h, crop_w, crop_h,
            ))
        all_centers = framed

    # Final boundary clamp
    hw, hh = crop_w // 2, crop_h // 2
    all_centers = [
        (max(hw, min(cx, orig_w - hw)), max(hh, min(cy, orig_h - hh)))
        for cx, cy in all_centers
    ]
    if len(all_centers) < total_frames:
        all_centers += [all_centers[-1]] * (total_frames - len(all_centers))
    all_centers = all_centers[:total_frames]

    # ── Phase 3: Render to temp .avi ─────────────────────────────────────
    _p(0.46, "✂️ Rendering frames…")
    temp_avi:   Optional[str] = None
    temp_mp4:   Optional[str] = None

    try:
        fd, temp_avi = tempfile.mkstemp(suffix=".avi")
        os.close(fd)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_avi, fourcc, render_fps, (target_w, target_h))
        if not writer.isOpened():
            raise ProcessingError("cv2.VideoWriter failed to open.")

        cap = cv2.VideoCapture(input_path)
        report_n2 = max(1, total_frames // 40)

        for fn in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cx, cy = all_centers[fn]
            left = max(0, min(cx - crop_w // 2, orig_w - crop_w))
            top  = max(0, min(cy - crop_h // 2, orig_h - crop_h))
            crop = frame[top:top + crop_h, left:left + crop_w]
            if crop.shape[1] != target_w or crop.shape[0] != target_h:
                crop = cv2.resize(crop, (target_w, target_h),
                                  interpolation=cv2.INTER_LANCZOS4)
            writer.write(crop)
            if (fn + 1) % report_n2 == 0:
                _p(0.46 + 0.40 * ((fn + 1) / total_frames),
                   f"✂️ {fn + 1}/{total_frames} frames…")

        cap.release()
        writer.release()

        if not os.path.exists(temp_avi) or os.path.getsize(temp_avi) < 1000:
            raise ProcessingError("Rendered .avi is empty.")

        # ── Phase 4: FFmpeg encode ────────────────────────────────────────
        _p(0.87, "🎵 Encoding with FFmpeg…")
        fd, temp_mp4 = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        audio_src = input_path if _has_audio(input_path) else None
        _ffmpeg_encode(
            temp_avi, audio_src, temp_mp4,
            fps=render_fps, duration=duration,
            crf=crf, preset=encoder_preset,
            audio_bitrate=audio_bitrate,
        )

        if not os.path.exists(temp_mp4) or os.path.getsize(temp_mp4) < 1000:
            raise ProcessingError("FFmpeg produced empty output.")

        shutil.move(temp_mp4, output_path)
        temp_mp4 = None  # moved — don't delete

        _p(1.0, "✅ Done!")
        print(f"✅ {output_path} ({os.path.getsize(output_path)/1024**2:.1f} MB)")

    finally:
        for p in (temp_avi, temp_mp4):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
