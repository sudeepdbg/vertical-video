"""
verticalize.py
──────────────
Convert landscape video to 9:16 vertical with AI subject tracking.

Modes:
  • Subject tracking  — YOLOv8 + optical flow
  • Talking Head Mode — DNN/Haar face detector, upper-third framing
  • Auto-clip detect  — scan long video, find high-engagement segments
  • Lower-third guard — subjects kept above bottom 20% of vertical frame

Dependencies: opencv-python, ultralytics, numpy, ffmpeg (system)
Optional:     openai-whisper, deep-translator
"""

from __future__ import annotations

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
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
class ProcessingError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
PERSON_CLASS_ID   = 0
HIGH_PRIO_CLASSES = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB  = 2000
MIN_FRAME_DIM     = 240
MAX_FRAMES_GUARD  = 1_080_000

LOWER_THIRD_GUARD = 0.80

VELOCITY_SMOOTH_TABLE: List[Tuple[float, int]] = [
    (0.0,   51), (3.0,   45), (8.0,   37),
    (15.0,  27), (30.0,  19), (60.0,  13), (120.0,  7),
]

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Match source (no upscale)":    (0, 0),
    "1080p  (1080×1920 — Full HD)": (1080, 1920),
    "720p   (720×1280  — HD)":      (720,  1280),
    "540p   (540×960   — SD)":      (540,  960),
    "480p   (480×854   — Low)":     (480,  854),
}

SUBTITLE_STYLES: Dict[str, Dict[str, Any]] = {
    "Bold White (TikTok)": {
        "fontsize": 18, "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000", "outline": 2,
        "bold": 1, "shadow": 0, "back_color": "&H00000000",
        "margin_v": 80,
    },
    "Yellow (Classic)": {
        "fontsize": 16, "primary_color": "&H0000FFFF",
        "outline_color": "&H00000000", "outline": 2,
        "bold": 1, "shadow": 1, "back_color": "&H00000000",
        "margin_v": 80,
    },
    "Box (Accessible)": {
        "fontsize": 15, "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000", "outline": 0,
        "bold": 0, "shadow": 0, "back_color": "&H80000000",
        "margin_v": 80,
    },
}

TRANSLATION_LANGUAGES: Dict[str, str] = {
    "None (keep original)": "",
    "French 🇫🇷":           "fr",
    "German 🇩🇪":           "de",
    "Spanish 🇪🇸":          "es",
    "Italian 🇮🇹":          "it",
    "Portuguese 🇵🇹":       "pt",
    "Dutch 🇳🇱":            "nl",
    "Polish 🇵🇱":           "pl",
    "Russian 🇷🇺":          "ru",
    "Japanese 🇯🇵":         "ja",
    "Korean 🇰🇷":           "ko",
    "Chinese (Simplified) 🇨🇳": "zh-CN",
    "Arabic 🇸🇦":           "ar",
    "Hindi 🇮🇳":            "hi",
    "Turkish 🇹🇷":          "tr",
    "Indonesian 🇮🇩":       "id",
    "Swedish 🇸🇪":          "sv",
    "Norwegian 🇳🇴":        "no",
    "Danish 🇩🇰":           "da",
    "Finnish 🇫🇮":          "fi",
    "Greek 🇬🇷":            "el",
    "Hebrew 🇮🇱":           "iw",
    "Thai 🇹🇭":             "th",
    "Vietnamese 🇻🇳":       "vi",
    "Malay 🇲🇾":            "ms",
    "Ukrainian 🇺🇦":        "uk",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Clip segment
# ─────────────────────────────────────────────────────────────────────────────
class ClipSegment:
    """Represents a detected high-engagement segment."""
    def __init__(self, start_sec: float, end_sec: float, score: float,
                 soi_region: str = "center", peak_frame: int = 0, title: str = ""):
        self.start_sec  = start_sec
        self.end_sec    = end_sec
        self.score      = score
        self.soi_region = soi_region
        self.peak_frame = peak_frame
        self.title      = title
        self.duration   = end_sec - start_sec

    def __repr__(self) -> str:
        return (f"<Clip {self.start_sec:.1f}s–{self.end_sec:.1f}s "
                f"dur={self.duration:.1f}s score={self.score:.2f} soi={self.soi_region}>")


# ─────────────────────────────────────────────────────────────────────────────
#  Optional dependency checks
# ─────────────────────────────────────────────────────────────────────────────
def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def translation_available() -> bool:
    try:
        import deep_translator  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{tool} not found. Install FFmpeg and add it to PATH.")


def _has_audio(path: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return "audio" in r.stdout
    except Exception:
        return False


def _get_video_duration_ffprobe(path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip())
    except Exception:
        return None


def _extract_audio_wav(video_path: str, wav_path: str) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(wav_path)


def _trim_video(input_path: str, output_path: str,
                start_sec: float, end_sec: float) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        "-reset_timestamps", "1",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(output_path)


def _ffmpeg_encode_pipe(
    frame_iter,
    frame_w: int,
    frame_h: int,
    n_frames: int,
    audio_source: Optional[str],
    output_path: str,
    fps: float,
    crf: int = 23,
    preset: str = "fast",
    audio_bitrate: str = "128k",
    subtitle_path: Optional[str] = None,
    subtitle_style: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Encode by piping raw BGR frames directly to FFmpeg stdin.
    No intermediate AVI — eliminates the MJPG/pixel-format bug entirely.
    """
    vf_chain: List[str] = []

    if subtitle_path and os.path.exists(subtitle_path):
        style = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        fs    = style.get("fontsize", 18)
        pc    = style.get("primary_color", "&H00FFFFFF")
        oc    = style.get("outline_color", "&H00000000")
        ol    = style.get("outline", 2)
        bold  = style.get("bold", 1)
        shad  = style.get("shadow", 0)
        bc    = style.get("back_color", "&H00000000")
        mv    = style.get("margin_v", 80)
        srt_esc = subtitle_path.replace("\\", "/").replace(":", "\\:")
        force_style = (
            f"Fontsize={fs},PrimaryColour={pc},OutlineColour={oc},"
            f"Outline={ol},Bold={bold},Shadow={shad},"
            f"BackColour={bc},MarginV={mv},Alignment=2"
        )
        vf_chain.append(f"subtitles='{srt_esc}':force_style='{force_style}'")

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{frame_w}x{frame_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_source:
        cmd += ["-i", audio_source]

    cmd += ["-map", "0:v:0"]
    if audio_source:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2",
                "-shortest"]
    else:
        cmd += ["-an"]

    if vf_chain:
        cmd += ["-vf", ",".join(vf_chain)]

    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frame_iter():
            if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
                frame = cv2.resize(frame, (frame_w, frame_h),
                                   interpolation=cv2.INTER_LANCZOS4)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise ProcessingError(
            f"FFmpeg pipe failed (rc={proc.returncode}):\n{stderr.decode()[-2000:]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Video metadata
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ProcessingError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ffprobe_dur = _get_video_duration_ffprobe(path)
    duration = ffprobe_dur if ffprobe_dur and ffprobe_dur > 0 else (nf / fps if fps > 0 else 0.0)
    total_frames = min(int(duration * fps), MAX_FRAMES_GUARD)

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": w, "height": h,
        "duration_seconds": duration,
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
#  Resolution resolver (upscale guard)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    """
    Compute 9:16 output dimensions that:
      - Never exceed the source dimensions (no upscale)
      - Are always even (H.264 requirement)
      - Are at least 128×128 (encoder minimum)
      - Fit entirely within orig_w × orig_h as a 9:16 crop window
    """
    TARGET_RATIO = 9 / 16   # width / height for vertical video
    MIN_DIM = 128

    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))

    if tw == 0 and th == 0:
        # "Match source" — derive the largest 9:16 rectangle that fits in orig
        # Option A: constrain by height → cw = orig_h * 9/16
        cw_a = int(orig_h * TARGET_RATIO)
        ch_a = orig_h
        # Option B: constrain by width → ch = orig_w / (9/16)
        ch_b = int(orig_w / TARGET_RATIO)
        cw_b = orig_w
        # Pick whichever fits inside source dimensions
        if cw_a <= orig_w:
            cw, ch = cw_a, ch_a
        else:
            cw, ch = cw_b, min(ch_b, orig_h)
    else:
        # Preset requested — scale down proportionally if source is smaller
        if th > orig_h:
            scale = orig_h / th
            tw = int(tw * scale)
            th = orig_h
        if tw > orig_w:
            scale = orig_w / tw
            tw = orig_w
            th = int(th * scale)
        cw, ch = tw, th

    # Enforce even dimensions
    cw = max(cw - (cw % 2), MIN_DIM)
    ch = max(ch - (ch % 2), MIN_DIM)

    # Final sanity: the crop window must actually fit inside source
    # If cw > orig_w or ch > orig_h, scale down preserving 9:16
    if cw > orig_w or ch > orig_h:
        scale = min(orig_w / cw, orig_h / ch)
        cw = max(int(cw * scale) & ~1, MIN_DIM)
        ch = max(int(ch * scale) & ~1, MIN_DIM)

    return cw, ch


# ─────────────────────────────────────────────────────────────────────────────
#  YOLO model cache
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}


def _get_model(weights: str = "yolov8n.pt") -> Any:
    if weights not in _model_cache:
        try:
            _model_cache[weights] = YOLO(weights)
        except Exception as e:
            raise ProcessingError(f"Failed to load '{weights}': {e}")
    return _model_cache[weights]


# ─────────────────────────────────────────────────────────────────────────────
#  Face detection  (DNN → Haar fallback)
# ─────────────────────────────────────────────────────────────────────────────
_face_net = None
_haar_cascade = None

_FACE_PROTO = "deploy.prototxt"
_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"


def _load_face_net() -> Optional[Any]:
    global _face_net
    if _face_net is not None:
        return _face_net
    if os.path.exists(_FACE_PROTO) and os.path.exists(_FACE_MODEL):
        try:
            _face_net = cv2.dnn.readNetFromCaffe(_FACE_PROTO, _FACE_MODEL)
            return _face_net
        except Exception:
            pass
    return None


def _get_haar() -> Optional[Any]:
    global _haar_cascade
    if _haar_cascade is not None:
        return _haar_cascade
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(path):
        casc = cv2.CascadeClassifier(path)
        if not casc.empty():
            _haar_cascade = casc
            return _haar_cascade
    return None


def detect_faces(frame: np.ndarray,
                 confidence_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
    """Return list of (x1,y1,x2,y2) face bboxes, largest first."""
    h, w = frame.shape[:2]
    net = _load_face_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False)
        net.setInput(blob)
        dets = net.forward()
        faces: List[Tuple[int, int, int, int]] = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < confidence_thresh:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
        faces.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
        return faces

    haar = _get_haar()
    if haar is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw = haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(max(30, w // 20), max(30, h // 20)))
    if len(raw) == 0:
        return []
    faces2 = [(x, y, x + bw, y + bh) for (x, y, bw, bh) in raw]
    faces2.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
    return faces2


# ─────────────────────────────────────────────────────────────────────────────
#  Lower-third guard
# ─────────────────────────────────────────────────────────────────────────────
def _apply_lower_third_guard(cy: int, crop_h: int, subject_cy_src: int,
                              orig_h: int) -> int:
    hh = crop_h // 2
    max_cy = subject_cy_src - int((1.0 - LOWER_THIRD_GUARD) * crop_h) + hh
    max_cy = min(max_cy, orig_h - hh)
    return min(cy, max_cy)


# ─────────────────────────────────────────────────────────────────────────────
#  SOI region label
# ─────────────────────────────────────────────────────────────────────────────
def _soi_region_label(cx: int, cy: int, w: int, h: int) -> str:
    col = "left" if cx < w // 3 else ("right" if cx > 2 * w // 3 else "center")
    row = "upper" if cy < h // 3 else ("lower" if cy > 2 * h // 3 else "mid")
    if row == "mid" and col == "center":
        return "center"
    if row == "mid":
        return col
    return f"{row}-{col}"


# ─────────────────────────────────────────────────────────────────────────────
#  Talking-head crop center
# ─────────────────────────────────────────────────────────────────────────────
def talking_head_center(
    faces: List[Tuple[int, int, int, int]],
    orig_w: int,
    orig_h: int,
    crop_w: int,
    crop_h: int,
    upper_third_bias: float = 0.30,
) -> Optional[Tuple[int, int]]:
    if not faces:
        return None

    ux1 = min(f[0] for f in faces)
    uy1 = min(f[1] for f in faces)
    ux2 = max(f[2] for f in faces)
    uy2 = max(f[3] for f in faces)

    face_cx = (ux1 + ux2) // 2
    face_cy = (uy1 + uy2) // 2

    target_cy = face_cy + crop_h // 6
    cy = int(face_cy * (1 - upper_third_bias) + target_cy * upper_third_bias)
    cx = face_cx

    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(cx, orig_w - hw))
    cy = max(hh, min(cy, orig_h - hh))

    cy = _apply_lower_third_guard(cy, crop_h, face_cy, orig_h)
    cy = max(hh, min(cy, orig_h - hh))

    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
#  Subject detection (YOLO)
# ─────────────────────────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult",
    ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"])


def detect_subjects(
    frame: np.ndarray,
    model: Any,
    confidence: float = 0.45,
) -> Optional[DetectionResult]:
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
        weight = max(1, (x2 - x1) * (y2 - y1)) * conf
        entry  = (weight, x1, y1, x2, y2)
        if cls == PERSON_CLASS_ID:
            person_pool.append(entry)
        elif cls in HIGH_PRIO_CLASSES:
            hiprio_pool.append(entry)
        all_pool.append(entry)

    pool = person_pool or hiprio_pool or all_pool
    if not pool:
        return None
    tw = sum(e[0] for e in pool)
    if tw == 0:
        return None
    cx = int(sum(e[0] * (e[1] + e[3]) / 2 for e in pool) / tw)
    cy = int(sum(e[0] * (e[2] + e[4]) / 2 for e in pool) / tw)
    return DetectionResult(
        cx, cy,
        min(e[1] for e in pool), min(e[2] for e in pool),
        max(e[3] for e in pool), max(e[4] for e in pool),
        len(pool),
    )


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
    cy = _apply_lower_third_guard(cy, crop_h, ucy, orig_h)
    cy = max(hh, min(cy, orig_h - hh))
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
#  Optical flow / saliency
# ─────────────────────────────────────────────────────────────────────────────
def optical_flow_center(
    prev: np.ndarray, curr: np.ndarray, w: int, h: int
) -> Optional[Tuple[int, int]]:
    if prev is None or curr is None:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        b = max(1, int(w * 0.04))
        mag[:, :b] = mag[:, w - b:] = mag[:b, :] = mag[h - b:, :] = 0
        if mag.max() < 0.8:
            return None
        t = mag.sum()
        if t == 0:
            return None
        ys, xs = np.mgrid[0:h, 0:w]
        return int((xs * mag).sum() / t), int((ys * mag).sum() / t)
    except Exception:
        return None


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
    b    = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w - b:] = sal[:b, :] = sal[h - b:, :] = 0
    t = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t)


def is_scene_change(
    prev: Optional[np.ndarray], curr: np.ndarray, threshold: float = 0.35
) -> bool:
    if prev is None:
        return False
    try:
        return float(cv2.absdiff(prev, curr).mean()) / 255.0 > threshold
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Framing bias (look-room + rule-of-thirds)
# ─────────────────────────────────────────────────────────────────────────────
def apply_framing_bias(
    cx: int, cy: int, vx: float, vy: float, speed: float,
    orig_w: int, orig_h: int, crop_w: int, crop_h: int,
    look_room_frac: float = 0.12,
    rot_bias: float = 0.15,
) -> Tuple[int, int]:
    hw, hh = crop_w // 2, crop_h // 2
    look = min(speed / 60.0, 1.0)
    if look > 0.05:
        n  = speed + 1e-9
        lx = int(cx + (vx / n) * look_room_frac * crop_w * look)
        ly = int(cy + (vy / n) * look_room_frac * crop_h * look)
    else:
        lx, ly = cx, cy
    still = max(0.0, 1.0 - look * 3)
    if still > 0.01:
        tx = min([orig_w // 3, 2 * orig_w // 3], key=lambda x: abs(x - cx))
        ty = min([orig_h // 3, 2 * orig_h // 3], key=lambda y: abs(y - cy))
        rx = int(cx + rot_bias * still * (tx - cx))
        ry = int(cy + rot_bias * still * (ty - cy))
    else:
        rx, ry = cx, cy
    nx = int(lx * look + rx * (1.0 - look))
    ny = int(ly * look + ry * (1.0 - look))
    return max(hw, min(nx, orig_w - hw)), max(hh, min(ny, orig_h - hh))


# ─────────────────────────────────────────────────────────────────────────────
#  Velocity helpers
# ─────────────────────────────────────────────────────────────────────────────
def _compute_speeds(centers: List[Tuple[int, int]], smooth: int = 9) -> List[float]:
    n = len(centers)
    if n < 2:
        return [0.0] * n
    raw = [0.0] + [
        float(np.sqrt((centers[i][0] - centers[i-1][0]) ** 2 +
                      (centers[i][1] - centers[i-1][1]) ** 2))
        for i in range(1, n)]
    w = min(smooth, n)
    return np.convolve(raw, np.ones(w) / w, mode="same").tolist()


def _compute_vel_vecs(centers: List[Tuple[int, int]], look: int = 6) -> List[Tuple[float, float]]:
    n = len(centers)
    out = []
    for i in range(n):
        j = min(i + look, n - 1)
        k = max(i - look, 0)
        span = j - k
        if span > 0:
            out.append(((centers[j][0] - centers[k][0]) / span,
                        (centers[j][1] - centers[k][1]) / span))
        else:
            out.append((0.0, 0.0))
    return out


def _vel_to_window(speed: float) -> int:
    t = VELOCITY_SMOOTH_TABLE
    if speed <= t[0][0]:  return t[0][1]
    if speed >= t[-1][0]: return t[-1][1]
    for i in range(len(t) - 1):
        v0, w0 = t[i]; v1, w1 = t[i + 1]
        if v0 <= speed <= v1:
            tt = (speed - v0) / (v1 - v0 + 1e-9)
            w  = int(w0 + tt * (w1 - w0))
            return w if w % 2 == 1 else w + 1
    return 27


# ─────────────────────────────────────────────────────────────────────────────
#  Smoothing
# ─────────────────────────────────────────────────────────────────────────────
def _gauss_seg(xs: np.ndarray, ys: np.ndarray,
               window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 3:
        return xs.copy(), ys.copy()
    w = min(window, n - 1)
    w = w if w % 2 == 1 else w - 1
    if w < 3:
        return xs.copy(), ys.copy()
    h2    = w // 2
    sigma = h2 / 2.5 + 1e-9
    k     = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
    k    /= k.sum()
    sx = np.convolve(np.pad(xs, h2, "edge"), k, "valid")[:n]
    sy = np.convolve(np.pad(ys, h2, "edge"), k, "valid")[:n]
    return sx, sy


def smooth_centers(
    centers: List[Tuple[int, int]],
    speeds: List[float],
    base_window: int = 27,
    adaptive: bool = True,
    scene_cuts: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    if not centers or len(centers) < 3:
        return list(centers) if centers else []
    n   = len(centers)
    xs  = np.array([c[0] for c in centers], dtype=float)
    ys  = np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n:
        spd = np.pad(spd, (0, n - len(spd)), mode="edge")

    cuts   = set(scene_cuts or [])
    bounds = [0] + sorted(cuts) + [n]
    rx, ry = xs.copy(), ys.copy()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 3:
            continue
        w = _vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window
        w = max(w, 13)
        xs_s, ys_s = _gauss_seg(xs[s:e], ys[s:e], w)
        rx[s:e] = xs_s
        ry[s:e] = ys_s
    return [(int(x), int(y)) for x, y in zip(rx, ry)]


# ─────────────────────────────────────────────────────────────────────────────
#  Cubic Hermite interpolation
# ─────────────────────────────────────────────────────────────────────────────
def _cubic_hermite(p0: float, p1: float, m0: float, m1: float, t: float) -> float:
    t2 = t * t; t3 = t2 * t
    return ((2*t3 - 3*t2 + 1)*p0 + (t3 - 2*t2 + t)*m0
            + (-2*t3 + 3*t2)*p1 + (t3 - t2)*m1)


def interpolate_centers(
    centers: List[Tuple[int, int]],
    indices: List[int],
    total: int,
) -> List[Tuple[int, int]]:
    """Cubic Hermite spline interpolation for ultra-smooth camera paths."""
    if total <= 0: return []
    if not centers: return [(0, 0)] * total
    if len(centers) == 1: return [centers[0]] * total

    n = len(indices)
    xs = [float(c[0]) for c in centers]
    ys = [float(c[1]) for c in centers]

    def _tangents(vals: List[float]) -> List[float]:
        m = [0.0] * len(vals)
        for i in range(len(vals)):
            if i == 0:
                m[i] = vals[1] - vals[0] if len(vals) > 1 else 0.0
            elif i == len(vals) - 1:
                m[i] = vals[-1] - vals[-2]
            else:
                m[i] = 0.5 * (vals[i+1] - vals[i-1])
        return m

    mx = _tangents(xs)
    my = _tangents(ys)

    result: List[Tuple[int, int]] = []
    for fi in range(total):
        if fi <= indices[0]:
            result.append(centers[0]); continue
        if fi >= indices[-1]:
            result.append(centers[-1]); continue

        r = bisect.bisect_right(indices, fi)
        l = r - 1
        if r >= n:
            result.append(centers[-1]); continue

        span = max(indices[r] - indices[l], 1)
        t    = (fi - indices[l]) / span

        nx = int(_cubic_hermite(xs[l], xs[r], mx[l]*span, mx[r]*span, t))
        ny = int(_cubic_hermite(ys[l], ys[r], my[l]*span, my[r]*span, t))
        result.append((nx, ny))

    while len(result) < total:
        result.append(result[-1] if result else (0, 0))
    return result[:total]


# ─────────────────────────────────────────────────────────────────────────────
#  Crop geometry
# ─────────────────────────────────────────────────────────────────────────────
def calculate_crop_dims(orig_w: int, orig_h: int, tw: int, th: int) -> Tuple[int, int]:
    ratio = tw / th
    if (orig_w / orig_h) > ratio:
        ch = orig_h; cw = int(round(ch * ratio))
    else:
        cw = orig_w; ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ─────────────────────────────────────────────────────────────────────────────
#  Whisper → SRT
# ─────────────────────────────────────────────────────────────────────────────
def _seconds_to_srt_time(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"


def transcribe_to_srt(
    video_path: str,
    srt_path: str,
    whisper_model: str = "base",
    language: Optional[str] = None,
    max_chars_per_line: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    if not whisper_available():
        return False

    import whisper  # type: ignore

    _p(0.0, "🎙️ Extracting audio…")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    try:
        if not _extract_audio_wav(video_path, wav_path):
            return False

        _p(0.2, f"📝 Transcribing with Whisper ({whisper_model})…")
        model = whisper.load_model(whisper_model)
        opts: Dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if language:
            opts["language"] = language
        result = model.transcribe(wav_path, **opts)

        _p(0.85, "✍️ Writing subtitles…")

        lines: List[str] = []
        idx   = 1
        words: List[Dict[str, Any]] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word":  w["word"].strip(),
                    "start": w["start"],
                    "end":   w["end"],
                })

        buf: List[Dict[str, Any]] = []
        buf_len = 0

        def flush_buf() -> None:
            nonlocal idx, buf, buf_len
            if not buf: return
            text  = " ".join(w["word"] for w in buf)
            start = _seconds_to_srt_time(buf[0]["start"])
            end   = _seconds_to_srt_time(buf[-1]["end"])
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            idx += 1
            buf    = []
            buf_len = 0

        for w in words:
            wlen = len(w["word"]) + 1
            if buf_len + wlen > max_chars_per_line and buf:
                flush_buf()
            buf.append(w)
            buf_len += wlen
        flush_buf()

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        _p(1.0, f"✅ {len(lines)} subtitle lines written")
        return True

    except Exception as e:
        print(f"Whisper transcription failed: {e}", file=sys.stderr)
        return False
    finally:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except OSError: pass


def translate_srt(
    srt_path: str,
    target_language: str,
    source_language: str = "auto",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    if not translation_available() or not target_language:
        return bool(not target_language)

    try:
        from deep_translator import GoogleTranslator  # type: ignore
    except ImportError:
        return False

    try:
        import re
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r"\n\n+", content.strip())
        translated_blocks: List[str] = []
        translator = GoogleTranslator(source=source_language, target=target_language)

        for i, block in enumerate(blocks):
            ls = block.strip().splitlines()
            if len(ls) < 3:
                translated_blocks.append(block)
                continue
            text = " ".join(ls[2:])
            try:
                translated = translator.translate(text) or text
            except Exception:
                translated = text
            translated_blocks.append(f"{ls[0]}\n{ls[1]}\n{translated}")
            if i % 10 == 0:
                _p(i / max(len(blocks), 1), f"🌐 Translating… {i}/{len(blocks)}")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_blocks) + "\n")

        _p(1.0, f"✅ Translated {len(translated_blocks)} subtitle blocks to [{target_language}]")
        return True

    except Exception as e:
        print(f"Translation failed: {e}", file=sys.stderr)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Frame saliency score (for clip detection)
# ─────────────────────────────────────────────────────────────────────────────
def _frame_saliency_score(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_score = min(lap_var / 3000.0, 1.0)

    motion_score = 0.0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
        motion_score = min(float(diff.mean()) / 30.0, 1.0)

    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_score = min(float(hsv[:, :, 1].mean()) / 128.0, 1.0)

    return 0.4 * motion_score + 0.4 * lap_score + 0.2 * sat_score


def _compute_frame_scores(
    input_path: str,
    fps: float,
    total_frames: int,
    sample_every: int = 15,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[np.ndarray, List[int]]:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    scores:      List[float] = []
    scene_cuts:  List[int]   = []
    prev_gray:   Optional[np.ndarray] = None
    prev_frame:  Optional[np.ndarray] = None
    frame_idx    = 0
    report_n     = max(1, total_frames // 20)

    cap = cv2.VideoCapture(input_path)
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % sample_every == 0:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff_ratio = float(cv2.absdiff(prev_gray, curr_gray).mean()) / 255.0
                if diff_ratio > 0.30:
                    scene_cuts.append(frame_idx)
            scores.append(_frame_saliency_score(frame, prev_frame))
            prev_gray  = curr_gray
            prev_frame = frame.copy()

        if frame_idx % report_n == 0:
            _p(frame_idx / total_frames, f"Scanning {frame_idx}/{total_frames}…")

        frame_idx += 1

    cap.release()
    return np.array(scores, dtype=float), scene_cuts


# ─────────────────────────────────────────────────────────────────────────────
#  Clip detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_clips(
    input_path: str,
    min_duration_sec: float = 25.0,
    max_duration_sec: float = 65.0,
    target_n_clips: int = 10,
    model: Optional[Any] = None,
    confidence: float = 0.45,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[ClipSegment]:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    info = get_video_info(input_path)
    fps          = info["fps"]
    total_frames = info["total_frames"]
    duration     = info["duration_seconds"]
    orig_w, orig_h = info["width"], info["height"]

    sample_every = max(1, int(fps))

    _p(0.0, "🔍 Scanning for engagement peaks…")
    scores, scene_cuts_frames = _compute_frame_scores(
        input_path, fps, total_frames,
        sample_every=sample_every,
        progress_callback=lambda v, m: _p(v * 0.45, m),
    )

    if len(scores) == 0:
        return []

    _p(0.45, "📊 Computing narrative arcs…")

    window = max(5, int(30 / (sample_every / fps)))
    if len(scores) >= window:
        smooth_scores = np.convolve(scores, np.ones(window) / window, mode="same")
    else:
        smooth_scores = scores.copy()

    if smooth_scores.max() > 0:
        smooth_scores = smooth_scores / smooth_scores.max()

    min_gap_samples = max(1, int(min_duration_sec * fps / sample_every))

    peaks: List[int] = []
    for i in range(1, len(smooth_scores) - 1):
        wh  = min_gap_samples // 2
        lo  = max(0, i - wh)
        hi  = min(len(smooth_scores), i + wh + 1)
        if smooth_scores[i] == smooth_scores[lo:hi].max() and smooth_scores[i] > 0.3:
            if not peaks or i - peaks[-1] > min_gap_samples // 2:
                peaks.append(i)

    peaks.sort(key=lambda i: smooth_scores[i], reverse=True)
    peaks = peaks[:target_n_clips * 2]

    def _arc_boundaries(peak_sample: int) -> Tuple[float, float]:
        peak_sec = peak_sample * sample_every / fps
        raw_start = max(0.0, peak_sec - max_duration_sec * 0.4)
        raw_end   = min(duration, raw_start + max_duration_sec)

        for sc in reversed(scene_cuts_frames):
            sc_sec = sc / fps
            if 0 < peak_sec - sc_sec < 15.0:
                raw_start = max(0.0, sc_sec - 1.0)
                break

        for sc in scene_cuts_frames:
            sc_sec = sc / fps
            if 0 < sc_sec - peak_sec < 15.0:
                raw_end = min(duration, sc_sec + 0.5)
                break

        clip_dur = raw_end - raw_start
        if clip_dur < min_duration_sec:
            raw_end = min(duration, raw_start + min_duration_sec)
        elif clip_dur > max_duration_sec:
            center    = (raw_start + raw_end) / 2
            raw_start = max(0.0, center - max_duration_sec / 2)
            raw_end   = min(duration, raw_start + max_duration_sec)

        return raw_start, raw_end

    clip_candidates: List[Tuple[float, float, float]] = []
    for peak_i in peaks:
        start, end  = _arc_boundaries(peak_i)
        peak_score  = float(smooth_scores[peak_i])
        overlaps    = any(
            min(end, ce) - max(start, cs) > min_duration_sec * 0.5
            for cs, ce, _ in clip_candidates
        )
        if not overlaps:
            clip_candidates.append((start, end, peak_score))

    clip_candidates.sort(key=lambda x: x[2], reverse=True)
    clip_candidates = clip_candidates[:target_n_clips]
    clip_candidates.sort(key=lambda x: x[0])

    _p(0.55, "🎯 Detecting SOI per clip…")

    segments: List[ClipSegment] = []
    for ci, (start_sec, end_sec, score) in enumerate(clip_candidates):
        _p(0.55 + 0.35 * (ci / max(len(clip_candidates), 1)),
           f"🎯 Clip {ci+1}/{len(clip_candidates)}…")

        soi_region = "center"
        soi_xs:  List[int] = []
        soi_ys:  List[int] = []
        n_samples = min(8, max(2, int(end_sec - start_sec)))
        sample_times = np.linspace(start_sec + 1, end_sec - 1, n_samples)

        cap = cv2.VideoCapture(input_path)
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret: continue

            if model is not None:
                try:
                    res = model(frame, verbose=False, conf=confidence)[0]
                    if res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            soi_xs.append((x1 + x2) // 2)
                            soi_ys.append((y1 + y2) // 2)
                except Exception:
                    pass
            else:
                scx, scy = saliency_center(frame)
                soi_xs.append(scx)
                soi_ys.append(scy)
        cap.release()

        if soi_xs:
            soi_region = _soi_region_label(
                int(np.median(soi_xs)), int(np.median(soi_ys)), orig_w, orig_h)

        mins_s = int(start_sec // 60)
        secs_s = int(start_sec % 60)
        mins_e = int(end_sec // 60)
        secs_e = int(end_sec % 60)
        title  = f"Clip {ci+1}  ({mins_s}:{secs_s:02d} – {mins_e}:{secs_e:02d})"

        segments.append(ClipSegment(
            start_sec=start_sec, end_sec=end_sec, score=score,
            soi_region=soi_region,
            peak_frame=int(sample_times[len(sample_times) // 2] * fps),
            title=title,
        ))

    _p(1.0, f"✅ Found {len(segments)} clips")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
#  EMA polish pass
# ─────────────────────────────────────────────────────────────────────────────
def _ema_polish(centers: List[Tuple[int, int]], alpha: float = 0.12) -> List[Tuple[int, int]]:
    """Exponential moving average forward+backward pass (zero-phase)."""
    if len(centers) < 3:
        return centers
    n = len(centers)

    fx = [float(centers[0][0])]
    fy = [float(centers[0][1])]
    for i in range(1, n):
        fx.append(alpha * centers[i][0] + (1 - alpha) * fx[-1])
        fy.append(alpha * centers[i][1] + (1 - alpha) * fy[-1])

    rx = [fx[-1]]
    ry = [fy[-1]]
    for i in range(n - 2, -1, -1):
        rx.append(alpha * fx[i] + (1 - alpha) * rx[-1])
        ry.append(alpha * fy[i] + (1 - alpha) * ry[-1])
    rx.reverse(); ry.reverse()

    return [(int(x), int(y)) for x, y in zip(rx, ry)]


# ─────────────────────────────────────────────────────────────────────────────
#  Codec compatibility guard — auto-transcode AV1/HEVC/etc. to H.264
# ─────────────────────────────────────────────────────────────────────────────
def _get_video_codec(path: str) -> str:
    """Return the video codec name via ffprobe (lowercase)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip().lower()
    except Exception:
        return ""


_OPENCV_UNSUPPORTED = {"av1", "libaom-av1", "hevc", "vp9", "vp8"}
_transcode_tmp_registry: List[str] = []


def _ensure_opencv_compat(
    input_path: str,
    result_meta: Dict[str, Any],
    _p: Callable,
) -> str:
    """
    If the source codec is not reliably decodable by OpenCV on this platform,
    transcode to H.264 in a temp file and return the new path.
    The original input_path is returned unchanged if no transcode is needed.
    """
    codec = _get_video_codec(input_path)
    if codec not in _OPENCV_UNSUPPORTED:
        return input_path

    _p(0.005, f"\u2699\ufe0f Transcoding {codec.upper()} \u2192 H.264 for OpenCV compatibility\u2026")
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    _transcode_tmp_registry.append(tmp_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
        raise ProcessingError(
            f"Failed to transcode {codec.upper()} input to H.264:\n"
            f"{r.stderr[-1500:]}"
        )

    _p(0.01, "\u2705 Transcode done \u2014 processing H.264 copy")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
#  Single-video verticalization
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    target_preset_label: str = "Match source (no upscale)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    sample_interval: Optional[int] = None,
    confidence: float = 0.45,
    use_optical_flow: bool = True,
    smooth_window: int = 27,
    adaptive_smoothing: bool = True,
    rule_of_thirds: bool = True,
    scene_cut_threshold: float = 0.35,
    output_fps: Optional[float] = None,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
    yolo_weights: str = "yolov8n.pt",
    burn_subtitles: bool = False,
    whisper_model: str = "base",
    whisper_language: Optional[str] = None,
    subtitle_style_name: str = "Bold White (TikTok)",
    subtitle_max_chars: int = 42,
    subtitle_translate_to: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Convert one landscape video to 9:16.  Returns metadata dict."""

    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception: pass

    result_meta: Dict[str, Any] = {
        "output_path":    output_path,
        "subtitle_path":  None,
        "clamped":        False,
        "effective_size": (0, 0),
        "duration":       0.0,
    }

    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path) / 1024 ** 2 > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # ── Auto-transcode if OpenCV cannot decode the source codec (e.g. AV1) ──
    input_path = _ensure_opencv_compat(input_path, result_meta, _p)

    info = get_video_info(input_path)
    fps          = info["fps"]
    total_frames = info["total_frames"]
    orig_w, orig_h = info["width"], info["height"]
    duration     = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ProcessingError("Corrupt or unreadable video.")
    if not info["is_landscape"]:
        raise ProcessingError("Video is already vertical — upload a landscape video.")

    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS \
        else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)

    req_w, req_h = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update(clamped=clamped,
                       effective_size=(target_w, target_h),
                       duration=duration)

    _p(0.01, f"📐 Output {target_w}×{target_h}")

    if not sample_interval:
        sample_interval = max(1, int(fps / 3))

    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)

    # ── Whisper ─────────────────────────────────────────────────────────
    srt_path: Optional[str] = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "🎙️ Transcribing…")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt")
        os.close(srt_fd)
        ok = transcribe_to_srt(
            input_path, srt_path,
            whisper_model=whisper_model,
            language=whisper_language,
            max_chars_per_line=subtitle_max_chars,
            progress_callback=lambda v, m: _p(0.02 + v * 0.08, m),
        )
        if not ok:
            if os.path.exists(srt_path):
                os.unlink(srt_path)
            srt_path = None
        else:
            if subtitle_translate_to:
                translate_srt(
                    srt_path,
                    target_language=subtitle_translate_to,
                    progress_callback=lambda v, m: _p(0.10 + v * 0.05, m),
                )
            result_meta["subtitle_path"] = srt_path

    # ── Load model ───────────────────────────────────────────────────────
    start_pct = 0.10
    model_obj: Optional[Any] = None
    if tracking_mode == "subject":
        _p(start_pct, "🤖 Loading YOLO model…")
        model_obj = _get_model(yolo_weights)
    else:
        _p(start_pct, "👤 Loading face detector…")
        if _get_haar() is None and _load_face_net() is None:
            raise ProcessingError(
                "No face detector available. Reinstall opencv-python.")

    # ── Detection pass ───────────────────────────────────────────────────
    _p(start_pct + 0.02, f"🔎 Analysing {total_frames} frames…")

    det_centers: List[Tuple[int, int]] = []
    det_indices: List[int]             = []
    sal_centers: List[Tuple[int, int]] = []
    sal_indices: List[int]             = []
    scene_cuts:  List[int]             = []

    prev_gray: Optional[np.ndarray] = None
    prev_flow: Optional[np.ndarray] = None
    frame_idx  = 0
    report_n   = max(1, total_frames // 25)
    det_end    = 0.42

    last_det_center: Optional[Tuple[int, int]] = None
    det_dropout_count = 0
    MAX_DROPOUT = int(fps * 1.5)

    cap = cv2.VideoCapture(input_path)
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret: break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_scene_change(prev_gray, curr_gray, scene_cut_threshold):
            scene_cuts.append(frame_idx)
            prev_flow = None
            det_dropout_count = 0
        prev_gray = curr_gray

        if frame_idx % sample_interval == 0:
            center: Optional[Tuple[int, int]] = None

            if tracking_mode == "talking_head":
                faces = detect_faces(frame, confidence_thresh=0.5)
                if faces:
                    center = talking_head_center(
                        faces, orig_w, orig_h, crop_w, crop_h, talking_head_bias)
                    det_dropout_count = 0
                elif use_optical_flow:
                    small = cv2.resize(curr_gray, (orig_w // 2, orig_h // 2))
                    if prev_flow is not None:
                        fc = optical_flow_center(prev_flow, small, orig_w // 2, orig_h // 2)
                        if fc is not None:
                            center = (fc[0] * 2, fc[1] * 2)
                    prev_flow = small
                    det_dropout_count += sample_interval
            else:
                det = detect_subjects(frame, model_obj, confidence)
                if det is not None:
                    center = frame_for_union(
                        det.ux1, det.uy1, det.ux2, det.uy2,
                        orig_w, orig_h, crop_w, crop_h)
                    last_det_center = center
                    det_dropout_count = 0
                elif use_optical_flow:
                    small = cv2.resize(curr_gray, (orig_w // 2, orig_h // 2))
                    if prev_flow is not None:
                        fc = optical_flow_center(prev_flow, small, orig_w // 2, orig_h // 2)
                        if fc is not None:
                            center = (fc[0] * 2, fc[1] * 2)
                    prev_flow = small
                    det_dropout_count += sample_interval

            if center is not None:
                det_centers.append(center)
                det_indices.append(frame_idx)
            else:
                if last_det_center is not None and det_dropout_count < MAX_DROPOUT:
                    det_centers.append(last_det_center)
                    det_indices.append(frame_idx)
                else:
                    sal_centers.append(saliency_center(frame))
                    sal_indices.append(frame_idx)

        frame_idx += 1
        if frame_idx % report_n == 0:
            pct = start_pct + 0.02 + (det_end - start_pct - 0.02) * (frame_idx / total_frames)
            _p(pct, f"🔎 {frame_idx}/{total_frames}…")

    cap.release()
    _p(det_end, f"📍 {len(det_centers)} anchors · {len(scene_cuts)} cuts")

    if not det_centers:
        det_centers = sal_centers or [(orig_w // 2, orig_h // 2)]
        det_indices = sal_indices  or [0]
    else:
        gap = sample_interval * 6
        for si, sc_center in zip(sal_indices, sal_centers):
            if not det_indices or min(abs(si - di) for di in det_indices) > gap:
                det_indices.append(si)
                det_centers.append(sc_center)
        pairs = sorted(zip(det_indices, det_centers))
        det_indices = [p[0] for p in pairs]
        det_centers = [p[1] for p in pairs]

    # ── Crop path ────────────────────────────────────────────────────────
    _p(0.43, "📈 Computing crop path…")

    all_centers = interpolate_centers(det_centers, det_indices, total_frames)
    speeds      = _compute_speeds(all_centers, smooth=11)

    if rule_of_thirds and tracking_mode != "talking_head":
        vel_vecs = _compute_vel_vecs(all_centers, look=6)
        biased   = [
            apply_framing_bias(cx, cy, vx, vy, speeds[i],
                               orig_w, orig_h, crop_w, crop_h)
            for i, ((cx, cy), (vx, vy)) in enumerate(zip(all_centers, vel_vecs))
        ]
        all_centers = biased
    elif tracking_mode == "talking_head" and rule_of_thirds:
        hw2, hh2 = crop_w // 2, crop_h // 2
        framed = []
        for cx, cy in all_centers:
            tx = min([orig_w // 3, 2 * orig_w // 3], key=lambda x: abs(x - cx))
            nx = int(cx + 0.10 * (tx - cx))
            nx = max(hw2, min(nx, orig_w - hw2))
            framed.append((nx, cy))
        all_centers = framed

    speeds      = _compute_speeds(all_centers, smooth=11)
    all_centers = smooth_centers(
        all_centers, speeds,
        base_window=smooth_window,
        adaptive=adaptive_smoothing,
        scene_cuts=scene_cuts,
    )

    all_centers = _ema_polish(all_centers, alpha=0.08)

    hw, hh = crop_w // 2, crop_h // 2
    all_centers = [
        (max(hw, min(cx, orig_w - hw)), max(hh, min(cy, orig_h - hh)))
        for cx, cy in all_centers
    ]
    all_centers += [all_centers[-1]] * max(0, total_frames - len(all_centers))
    all_centers  = all_centers[:total_frames]

    # ── Render pass — pipe frames directly to FFmpeg (no temp AVI) ───────
    _p(0.46, "✂️ Rendering & encoding…")

    style = SUBTITLE_STYLES.get(subtitle_style_name,
                                SUBTITLE_STYLES["Bold White (TikTok)"])

    # We need to do two passes over the video: one to collect frames for the
    # pipe. We use a generator that reads and crops on-the-fly.
    cap_render = cv2.VideoCapture(input_path)
    frames_written = 0
    rpt_n = max(1, total_frames // 40)

    # Count frames first (needed for n_frames param), then render via generator
    def _frame_generator():
        nonlocal frames_written
        _cap = cv2.VideoCapture(input_path)
        fn = 0
        while fn < total_frames:
            ret, frame = _cap.read()
            if not ret:
                break
            cx, cy = all_centers[fn]
            left   = max(0, min(cx - crop_w // 2, orig_w - crop_w))
            top    = max(0, min(cy - crop_h // 2, orig_h - crop_h))
            crop   = frame[top:top + crop_h, left:left + crop_w]
            if crop.shape[1] != target_w or crop.shape[0] != target_h:
                crop = cv2.resize(crop, (target_w, target_h),
                                  interpolation=cv2.INTER_LANCZOS4)
            yield crop
            frames_written += 1
            fn += 1
            if fn % rpt_n == 0:
                _p(0.46 + 0.40 * (fn / total_frames), f"✂️ {fn}/{total_frames}…")
        _cap.release()

    cap_render.release()  # not used — generator opens its own cap

    temp_mp4: Optional[str] = None
    try:
        fd, temp_mp4 = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        _ffmpeg_encode_pipe(
            frame_iter=_frame_generator,
            frame_w=target_w,
            frame_h=target_h,
            n_frames=total_frames,
            audio_source=input_path if _has_audio(input_path) else None,
            output_path=temp_mp4,
            fps=render_fps,
            crf=crf,
            preset=encoder_preset,
            audio_bitrate=audio_bitrate,
            subtitle_path=srt_path,
            subtitle_style=style,
        )

        if not os.path.exists(temp_mp4) or os.path.getsize(temp_mp4) < 1000:
            raise ProcessingError("FFmpeg produced empty output.")

        shutil.move(temp_mp4, output_path)
        temp_mp4 = None

        _p(1.0, "✅ Done!")
        print(f"✅  {output_path}  ({os.path.getsize(output_path)/1024**2:.1f} MB)",
              file=sys.stderr)
        return result_meta

    finally:
        if temp_mp4 and os.path.exists(temp_mp4):
            try: os.unlink(temp_mp4)
            except OSError: pass
        # Clean up SRT if it was temporary
        if srt_path and not result_meta.get("subtitle_path"):
            if os.path.exists(srt_path):
                try: os.unlink(srt_path)
                except OSError: pass


# ─────────────────────────────────────────────────────────────────────────────
#  Batch clip pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_clips_batch(
    input_path: str,
    output_dir: str,
    clips: List[ClipSegment],
    target_preset_label: str = "720p   (720×1280  — HD)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    confidence: float = 0.45,
    smooth_window: int = 27,
    adaptive_smoothing: bool = True,
    use_optical_flow: bool = True,
    rule_of_thirds: bool = True,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
    yolo_weights: str = "yolov8n.pt",
    burn_subtitles: bool = False,
    whisper_model: str = "base",
    subtitle_style_name: str = "Bold White (TikTok)",
    subtitle_max_chars: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Trim each ClipSegment from the source file, then verticalize.
    Returns a list of result-meta dicts (one per clip).
    """
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for i, clip in enumerate(clips):
        base_pct = i / max(len(clips), 1)
        next_pct = (i + 1) / max(len(clips), 1)

        _p(base_pct, f"✂️ Processing clip {i+1}/{len(clips)}…")

        trimmed_path: Optional[str] = None
        out_path: Optional[str] = None

        try:
            fd, trimmed_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            if not _trim_video(input_path, trimmed_path, clip.start_sec, clip.end_sec):
                results.append({"clip": clip, "output_path": None,
                                 "error": "trim failed"})
                continue

            safe_name = f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s"
            out_path  = os.path.join(output_dir, f"{safe_name}_vertical.mp4")

            def clip_cb(v: float, msg: str = "") -> None:
                _p(base_pct + v * (next_pct - base_pct), msg)

            meta = process_video(
                trimmed_path, out_path,
                target_preset_label=target_preset_label,
                tracking_mode=tracking_mode,
                talking_head_bias=talking_head_bias,
                confidence=confidence,
                smooth_window=smooth_window,
                adaptive_smoothing=adaptive_smoothing,
                use_optical_flow=use_optical_flow,
                rule_of_thirds=rule_of_thirds,
                crf=crf,
                encoder_preset=encoder_preset,
                audio_bitrate=audio_bitrate,
                yolo_weights=yolo_weights,
                burn_subtitles=burn_subtitles,
                whisper_model=whisper_model,
                subtitle_style_name=subtitle_style_name,
                subtitle_max_chars=subtitle_max_chars,
                progress_callback=clip_cb,
            )
            meta["clip"] = clip
            results.append(meta)

        except Exception as exc:
            err_msg = str(exc)
            print(f"❌ Clip {i+1} failed: {err_msg}", file=sys.stderr)
            results.append({
                "clip": clip,
                "output_path": out_path,
                "error": err_msg,
            })
        finally:
            if trimmed_path and os.path.exists(trimmed_path):
                try: os.unlink(trimmed_path)
                except OSError: pass

    n_ok  = sum(1 for r in results if not r.get("error"))
    n_err = len(results) - n_ok

    if n_ok == 0 and results:
        first_err = next((r["error"] for r in results if r.get("error")), "unknown error")
        raise ProcessingError(
            f"All {n_err} clip(s) failed. First error: {first_err}"
        )

    if n_err:
        _p(1.0, f"⚠️ Batch done — {n_ok} ok, {n_err} failed")
    else:
        _p(1.0, f"✅ Batch complete — {n_ok}/{len(results)} clips done")
    return results
