"""
verticalize.py — AI Vertical Video Converter v3.7
─────────────────────────────────────────────────
COMPLETE REWRITE focused on stability and smoothness.

Key architectural changes from v3.6:
  1. SINGLE motion tracker — no more Kalman + Anchor fighting
  2. Heavy temporal smoothing on crop position (0.5s time constant)
  3. Layout decisions are sticky — once chosen, locked for min 2s
  4. Scene cuts use multi-frame consensus, not single-frame threshold
  5. Non-sample frames interpolate crop position, never re-run layout
  6. Simplified panel rendering — no complex group tracking
  7. Basketball-optimized defaults (high fps, fast motion tolerance)
"""

from __future__ import annotations
import subprocess, sys, os, tempfile, math, hashlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import cv2, numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class ProcessingError(Exception):
    pass


# ── Constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS_ID = 0
MAX_FILE_SIZE_MB = 2000
MAX_FRAMES_GUARD = 1_080_000

# Tracking smoothing — time constant in seconds
SMOOTH_TIME_CONSTANT = 0.5  # 0.5 second smoothing on crop center

# Layout lock duration in seconds
LAYOUT_LOCK_SECONDS = 2.0

# Scene cut detection — multi-frame consensus
SCENE_CUT_WINDOW = 5  # frames
SCENE_CUT_THRESHOLD = 0.40  # higher = less sensitive

# Panel layout constants
LAYOUT_SINGLE = "single"
LAYOUT_DUO = "duo"
LAYOUT_TRIO = "trio"

PANEL_DIVIDER_PX = 4
PANEL_DIVIDER_COLOR = (20, 20, 20)

# Resolution presets
RESOLUTION_PRESETS = {
    "Match source (no upscale)":    (0, 0),
    "1080p  (1080x1920 - Full HD)": (1080, 1920),
    "720p   (720x1280  - HD)":      (720, 1280),
    "540p   (540x960   - SD)":      (540, 960),
    "480p   (480x854   - Low)":     (480, 854),
}

SUBTITLE_STYLES = {
    "Bold White (TikTok)": {
        "fontsize": 18, "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000", "outline": 2,
        "bold": 1, "shadow": 0, "back_color": "&H00000000", "margin_v": 80},
    "Yellow (Classic)": {
        "fontsize": 16, "primary_color": "&H0000FFFF",
        "outline_color": "&H00000000", "outline": 2,
        "bold": 1, "shadow": 1, "back_color": "&H00000000", "margin_v": 80},
    "Box (Accessible)": {
        "fontsize": 15, "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000", "outline": 0,
        "bold": 0, "shadow": 0, "back_color": "&H80000000", "margin_v": 80},
}

COLOR_GRADES = ("none", "warm", "cool", "vibrant", "matte")


# ── Smooth Tracker ─────────────────────────────────────────────────────────────
class SmoothTracker:
    """
    Single unified tracker with exponential smoothing.
    No Kalman, no separate anchor — just smooth position + velocity.
    """
    def __init__(self, time_constant_sec: float = 0.5):
        self.time_constant = time_constant_sec
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._vx = 0.0
        self._vy = 0.0
        self._alpha = 0.0  # set per-frame based on dt
        self._last_t = 0.0

    def set_alpha(self, fps: float):
        """Compute smoothing alpha from fps and time constant."""
        dt = 1.0 / fps
        # Exponential smoothing: alpha = 1 - exp(-dt / tau)
        self._alpha = 1.0 - math.exp(-dt / self.time_constant)

    def reset(self, x: float, y: float):
        """Hard reset to position."""
        self._x = float(x)
        self._y = float(y)
        self._vx = 0.0
        self._vy = 0.0

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with new measurement, return smoothed position."""
        if self._x is None:
            self.reset(x, y)
            return x, y

        # Velocity estimate
        self._vx = (x - self._x) * self._alpha
        self._vy = (y - self._y) * self._alpha

        # Exponential smooth toward measurement
        self._x += (x - self._x) * self._alpha
        self._y += (y - self._y) * self._alpha

        return self._x, self._y

    def predict(self, frames_ahead: int = 1) -> Tuple[float, float]:
        """Predict position ahead by N frames using velocity."""
        if self._x is None:
            return 0.0, 0.0
        return self._x + self._vx * frames_ahead, self._y + self._vy * frames_ahead

    @property
    def position(self) -> Tuple[float, float]:
        if self._x is None:
            return 0.0, 0.0
        return self._x, self._y


# ── Scene Cut Detector ─────────────────────────────────────────────────────────
class SceneCutDetector:
    """
    Multi-frame consensus scene cut detection.
    Requires N consecutive frames above threshold.
    """
    def __init__(self, threshold: float = 0.40, window: int = 5):
        self.threshold = threshold
        self.window = window
        self._recent: deque = deque(maxlen=window)
        self._prev_gray: Optional[np.ndarray] = None

    def feed(self, gray: np.ndarray) -> bool:
        """Returns True if scene cut detected."""
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            self._recent.clear()
            return False

        diff = float(cv2.absdiff(self._prev_gray, gray).mean()) / 255.0
        self._recent.append(diff > self.threshold)
        self._prev_gray = gray.copy()

        # Scene cut = all recent frames show high diff (sustained change)
        if len(self._recent) == self.window and all(self._recent):
            self._recent.clear()
            return True
        return False

    def reset(self):
        self._prev_gray = None
        self._recent.clear()


# ── Layout State Machine ───────────────────────────────────────────────────────
class LayoutState:
    """
    Sticky layout with minimum hold time.
    """
    def __init__(self, fps: float, lock_seconds: float = 2.0):
        self.current = LAYOUT_SINGLE
        self._locked_frames = 0
        self._lock_duration = int(lock_seconds * fps)
        self._fps = fps

    def propose(self, layout: str) -> str:
        """Propose a new layout. Returns actual layout (may be locked)."""
        if self._locked_frames > 0:
            self._locked_frames -= 1
            return self.current

        if layout != self.current:
            self.current = layout
            self._locked_frames = self._lock_duration

        return self.current

    def force(self, layout: str):
        """Force immediate layout change (e.g., on scene cut)."""
        self.current = layout
        self._locked_frames = self._lock_duration


# ── Vignette ───────────────────────────────────────────────────────────────────
_vignette_cache: Dict[Tuple, np.ndarray] = {}

def _build_vignette(w, h, strength=0.55, falloff=1.8):
    key = (w, h, round(strength, 3), round(falloff, 3))
    if key in _vignette_cache:
        return _vignette_cache[key]
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.sqrt(xg**2 + yg**2)
    dist /= dist.max() + 1e-6
    mask = np.clip(1.0 - strength * (dist**falloff), 0.0, 1.0)[:, :, np.newaxis]
    _vignette_cache[key] = mask
    return mask

def apply_vignette(frame, strength=0.55):
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    return (frame.astype(np.float32) * _build_vignette(w, h, strength)).clip(0, 255).astype(np.uint8)


# ── Sharpen ────────────────────────────────────────────────────────────────────
def apply_sharpen(frame, strength=0.6, radius=1):
    if strength <= 0:
        return frame
    ksize = radius * 2 + 1
    return cv2.addWeighted(frame, 1 + strength,
                           cv2.GaussianBlur(frame, (ksize, ksize), 0), -strength, 0)


# ── Color grade LUT ────────────────────────────────────────────────────────────
_lut_cache: Dict[str, np.ndarray] = {}

def _build_lut(grade: str) -> np.ndarray:
    if grade in _lut_cache:
        return _lut_cache[grade]
    x = np.arange(256, dtype=np.float32)
    if grade == "warm":
        r = np.clip(x * 1.06 + 5, 0, 255)
        g = np.clip(x * 1.02 + 2, 0, 255)
        b = np.clip(x * 0.92 - 4, 0, 255)
    elif grade == "cool":
        r = np.clip(x * 0.92 - 4, 0, 255)
        g = np.clip(x * 1.01 + 1, 0, 255)
        b = np.clip(x * 1.07 + 6, 0, 255)
    elif grade == "vibrant":
        def sc(v):
            n = v / 255
            s = n * n * (3 - 2 * n)
            return np.clip((n * 0.6 + s * 0.4) * 255, 0, 255)
        r, g, b = sc(x * 1.04), sc(x * 1.02), sc(x)
    elif grade == "matte":
        r = np.clip(x * 0.88 + 18, 0, 255)
        g = np.clip(x * 0.86 + 16, 0, 255)
        b = np.clip(x * 0.84 + 22, 0, 255)
    else:
        r = g = b = x.copy()
    lut = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(256, 1, 3)
    _lut_cache[grade] = lut
    return lut

def apply_color_grade(frame, grade="none"):
    if not grade or grade == "none":
        return frame
    return cv2.LUT(frame, _build_lut(grade))


# ── Ken Burns ──────────────────────────────────────────────────────────────────
def apply_ken_burns(frame, frame_idx, fps, max_zoom=1.04, period=8.0):
    if max_zoom <= 1.0:
        return frame
    t = (frame_idx / max(fps, 1)) % period
    scale = 1.0 + (max_zoom - 1.0) * 0.5 * (1 - math.cos(2 * math.pi * t / period))
    if abs(scale - 1.0) < 1e-4:
        return frame
    h, w = frame.shape[:2]
    nw, nh = max(int(w / scale), 2), max(int(h / scale), 2)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(frame[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LINEAR)


# ── FFmpeg Video Reader ────────────────────────────────────────────────────────
class FFmpegVideoReader:
    def __init__(self, path, width, height, seek_sec=0.0,
                 n_frames=None, scale_w=None, scale_h=None):
        self.path = path
        self.width = width
        self.height = height
        self.seek_sec = seek_sec
        self.n_frames = n_frames
        self.out_w = scale_w or width
        self.out_h = scale_h or height
        self._proc = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover = b""

    def _open(self):
        cmd = ["ffmpeg"]
        if self.seek_sec > 0:
            cmd += ["-ss", str(self.seek_sec)]
        cmd += ["-i", self.path, "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-vf", f"scale={self.out_w}:{self.out_h}"]
        if self.n_frames is not None:
            cmd += ["-vframes", str(self.n_frames)]
        cmd += ["pipe:1"]

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=max(self._frame_bytes * 4, 1 << 20))
            test = self._proc.stdout.read(self._frame_bytes)
            if len(test) == self._frame_bytes:
                self._leftover = test
                return
        except Exception:
            pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")

    def close(self):
        if self._proc:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.wait()
            except Exception:
                pass
            self._proc = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        if not self._proc:
            self._open()
        buf = self._leftover
        self._leftover = b""
        fb = self._frame_bytes
        while True:
            needed = fb - len(buf)
            while needed > 0:
                chunk = self._proc.stdout.read(needed)
                if not chunk:
                    return
                buf += chunk
                needed -= len(chunk)
            yield np.frombuffer(buf[:fb], dtype=np.uint8).reshape(self.out_h, self.out_w, 3)
            buf = buf[fb:]


def _read_frame_at(path, width, height, t_sec, scale_w=None, scale_h=None):
    r = FFmpegVideoReader(path, width, height, seek_sec=t_sec,
                          n_frames=1, scale_w=scale_w, scale_h=scale_h)
    with r:
        frames = list(r)
    return frames[0] if frames else None


# ── FFmpeg Helpers ─────────────────────────────────────────────────────────────
def _check_ffmpeg():
    for t in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([t, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{t} not found. Install FFmpeg.")

def _has_audio(path) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15)
        return "audio" in r.stdout
    except Exception:
        return False

def _extract_audio_wav(vpath, wpath) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", vpath, "-ar", "16000", "-ac", "1", "-f", "wav", wpath],
        capture_output=True)
    return r.returncode == 0 and os.path.exists(wpath)

def _trim_video(inp, out, start, end) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-hwaccel", "none",
         "-ss", str(start), "-to", str(end), "-i", inp,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
         "-c:a", "aac", "-b:a", "128k",
         "-avoid_negative_ts", "make_zero", "-reset_timestamps", "1", out],
        capture_output=True)
    return r.returncode == 0 and os.path.exists(out)


# ── Encoder ────────────────────────────────────────────────────────────────────
def _open_ffmpeg_encoder(output_path, width, height, fps, audio_source,
                          crf=23, preset="fast", audio_bitrate="128k",
                          subtitle_path=None, subtitle_style=None):
    cmd = ["ffmpeg", "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    has_aud = audio_source and _has_audio(audio_source)
    if has_aud:
        cmd += ["-hwaccel", "none", "-i", audio_source]
    vf = []
    if subtitle_path and os.path.exists(subtitle_path):
        s = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        sesc = subtitle_path.replace("\\", "/").replace(":", "\:")
        force = (f"Fontsize={s.get('fontsize',18)},"
                 f"PrimaryColour={s.get('primary_color','&H00FFFFFF')},"
                 f"OutlineColour={s.get('outline_color','&H00000000')},"
                 f"Outline={s.get('outline',2)},Bold={s.get('bold',1)},"
                 f"Shadow={s.get('shadow',0)},BackColour={s.get('back_color','&H00000000')},"
                 f"MarginV={s.get('margin_v',80)},Alignment=2")
        vf.append(f"subtitles='{sesc}':force_style='{force}'")
    cmd += ["-map", "0:v:0"]
    if has_aud:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    else:
        cmd += ["-an"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
            "-shortest", "-movflags", "+faststart", output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _close_ffmpeg_encoder(proc, output_path):
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        err = proc.stderr.read(4000).decode(errors="replace")
    except Exception:
        err = ""
    proc.wait()
    if proc.returncode != 0:
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")


# ── Video Metadata ─────────────────────────────────────────────────────────────
def get_video_info(path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
           "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    kv: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    w = int(kv.get("width", 0) or 0)
    h = int(kv.get("height", 0) or 0)
    try:
        num, den = kv.get("r_frame_rate", "30/1").split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    dur = float(kv.get("duration", 0.0) or 0.0)
    if dur <= 0:
        nb = int(kv.get("nb_frames", 0) or 0)
        dur = nb / fps if fps > 0 and nb > 0 else 0.0
    if w == 0 or h == 0:
        raise ProcessingError(f"Cannot read dimensions: {path}")
    return {"fps": fps, "total_frames": min(int(dur * fps), MAX_FRAMES_GUARD),
            "width": w, "height": h, "duration_seconds": dur, "is_landscape": w > h}


# ── Resolution Helpers ─────────────────────────────────────────────────────────
def resolve_target_size(label, orig_w, orig_h):
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        cw = int(orig_h * 9 / 16)
        if cw > orig_w:
            cw = orig_w
            ch = int(cw * 16 / 9)
        else:
            ch = orig_h
        return cw - (cw % 2), ch - (ch % 2)
    if th > orig_h:
        scale = orig_h / th
        tw = int(tw * scale)
        th = int(orig_h)
    if tw > orig_w:
        scale = orig_w / tw
        tw = int(orig_w)
        th = int(th * scale)
    return max(tw - (tw % 2), 2), max(th - (th % 2), 2)

def calculate_crop_dims(orig_w, orig_h, tw, th):
    ratio = tw / th
    if (orig_w / orig_h) > ratio:
        ch = orig_h
        cw = int(round(ch * ratio))
    else:
        cw = orig_w
        ch = int(round(cw / ratio))
    cw = min(cw - (cw % 2), orig_w)
    ch = min(ch - (ch % 2), orig_h)
    return max(cw, 2), max(ch, 2)


# ── YOLO Model Cache ───────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}

def _get_model(weights="yolov8n.pt"):
    if not _YOLO_AVAILABLE:
        return None
    if weights in _model_cache:
        return _model_cache[weights]
    try:
        m = _YOLO(weights)
        _model_cache[weights] = m
        return m
    except Exception as e:
        print(f"YOLO unavailable: {e}", file=sys.stderr)
        return None


# ── Subject Detection ──────────────────────────────────────────────────────────
def detect_subjects(frame, model, confidence=0.45) -> Optional[Tuple[float, float, List[Tuple[int, int, int, int]]]]:
    """
    Returns: (center_x, center_y, list_of_boxes) or None.
    """
    if model is None:
        return None
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception:
        return None
    if results.boxes is None or len(results.boxes) == 0:
        return None

    boxes = []
    weights = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w = x2 - x1
        h = y2 - y1
        # Filter: must be reasonable size and not at extreme edges
        fw, fh = frame.shape[1], frame.shape[0]
        if w < fw * 0.03 or h < fh * 0.08:
            continue
        cx = (x1 + x2) / 2
        if cx < fw * 0.05 or cx > fw * 0.95:
            continue
        boxes.append((x1, y1, x2, y2))
        # Weight by confidence and size
        weights.append(conf * w * h)

    if not boxes:
        return None

    # Weighted center
    tw = sum(weights)
    cx = sum(w * ((b[0] + b[2]) / 2) for w, b in zip(weights, boxes)) / tw
    cy = sum(w * ((b[1] + b[3]) / 2) for w, b in zip(weights, boxes)) / tw

    return cx, cy, boxes


# ── Saliency Center ────────────────────────────────────────────────────────────
def saliency_center(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31, 31), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1].astype(np.float32), (31, 31), 0)
    sal = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    # Mask edges
    b = max(1, int(w * 0.05))
    sal[:, :b] = 0
    sal[:, w-b:] = 0
    sal[:b, :] = 0
    sal[h-b:, :] = 0
    t = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t)


# ── Layout Classification ──────────────────────────────────────────────────────
def classify_layout(boxes, fw, fh) -> str:
    """Classify layout based on person boxes."""
    n = len(boxes)
    if n <= 1:
        return LAYOUT_SINGLE

    # Sort by x center
    centers = sorted([(b[0] + b[2]) / 2 for b in boxes])

    # Check if widely separated (duo split)
    if n == 2:
        gap = centers[1] - centers[0]
        if gap > fw * 0.55:
            return LAYOUT_DUO
        return LAYOUT_SINGLE

    # For 3+, check spread
    spread = centers[-1] - centers[0]
    if spread > fw * 0.75 and n >= 3:
        return LAYOUT_TRIO
    if spread > fw * 0.50 and n >= 2:
        return LAYOUT_DUO
    return LAYOUT_SINGLE


# ── Panel Rendering ────────────────────────────────────────────────────────────
def render_single(frame, crop_cx, crop_cy, crop_w, crop_h, out_w, out_h):
    """Render single-panel crop."""
    h, w = frame.shape[:2]
    left = int(max(0, min(crop_cx - crop_w // 2, w - crop_w)))
    top = int(max(0, min(crop_cy - crop_h // 2, h - crop_h)))
    crop = frame[top:top+crop_h, left:left+crop_w]
    if crop.shape[1] != out_w or crop.shape[0] != out_h:
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return crop

def render_duo(frame, boxes, out_w, out_h, color_grade="none"):
    """Render two-panel split."""
    h, w = frame.shape[:2]
    # Sort boxes by x center
    sorted_boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)

    div = PANEL_DIVIDER_PX
    sh_top = ((out_h - div) // 2) & ~1
    sh_bot = out_h - sh_top - div

    panels = []
    for i, box in enumerate(sorted_boxes[:2]):
        bh = sh_top if i == 0 else sh_bot
        # Tight crop around box with padding
        pad = 1.6
        bw = box[2] - box[0]
        bh_src = box[3] - box[1]
        src_h = int(bh_src * pad)
        src_w = int(src_h * (out_w / bh))
        src_w = max(src_w, bw)
        src_h = max(src_h, bh_src)

        ucx = (box[0] + box[2]) // 2
        ucy = (box[1] + box[3]) // 2

        x0 = max(0, min(ucx - src_w // 2, w - src_w))
        y0 = max(0, min(ucy - src_h // 2, h - src_h))
        x1 = min(x0 + src_w, w)
        y1 = min(y0 + src_h, h)
        x0 = max(0, x1 - src_w)
        y0 = max(0, y1 - src_h)

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            crop = frame
        panel = cv2.resize(crop, (out_w, bh), interpolation=cv2.INTER_LANCZOS4)
        if color_grade and color_grade != "none":
            panel = apply_color_grade(panel, color_grade)
        panels.append(panel)

    # If only one panel, duplicate
    while len(panels) < 2:
        panels.append(panels[-1] if panels else np.zeros((sh_bot, out_w, 3), dtype=np.uint8))

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    y = 0
    canvas[y:y+sh_top, :] = panels[0]
    y += sh_top
    canvas[y:y+div, :] = PANEL_DIVIDER_COLOR
    y += div
    canvas[y:y+sh_bot, :] = panels[1]
    return canvas

def render_trio(frame, boxes, out_w, out_h, color_grade="none"):
    """Render three-panel layout."""
    h, w = frame.shape[:2]
    sorted_boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)

    div = PANEL_DIVIDER_PX
    sh_main = int((out_h - 2 * div) * 0.58) & ~1
    sh_side = ((out_h - sh_main - 2 * div) // 2) & ~1

    # Main = largest box
    main_box = max(sorted_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    side_boxes = [b for b in sorted_boxes if b is not main_box][:2]

    # Main panel
    pad = 1.5
    bw = main_box[2] - main_box[0]
    bh_src = main_box[3] - main_box[1]
    src_h = int(bh_src * pad)
    src_w = int(src_h * (out_w / sh_main))
    src_w = max(src_w, bw)
    src_h = max(src_h, bh_src)

    ucx = (main_box[0] + main_box[2]) // 2
    ucy = (main_box[1] + main_box[3]) // 2
    x0 = max(0, min(ucx - src_w // 2, w - src_w))
    y0 = max(0, min(ucy - src_h // 2, h - src_h))
    x1 = min(x0 + src_w, w)
    y1 = min(y0 + src_h, h)
    x0 = max(0, x1 - src_w)
    y0 = max(0, y1 - src_h)
    main_crop = frame[y0:y1, x0:x1]
    if main_crop.size == 0:
        main_crop = frame
    main_panel = cv2.resize(main_crop, (out_w, sh_main), interpolation=cv2.INTER_LANCZOS4)
    if color_grade and color_grade != "none":
        main_panel = apply_color_grade(main_panel, color_grade)

    # Side panels
    side_panels = []
    for box in side_boxes:
        pad = 1.4
        bw = box[2] - box[0]
        bh_src = box[3] - box[1]
        src_h = int(bh_src * pad)
        src_w = int(src_h * (out_w // 2 / sh_side))
        src_w = max(src_w, bw)
        src_h = max(src_h, bh_src)

        ucx = (box[0] + box[2]) // 2
        ucy = (box[1] + box[3]) // 2
        x0 = max(0, min(ucx - src_w // 2, w - src_w))
        y0 = max(0, min(ucy - src_h // 2, h - src_h))
        x1 = min(x0 + src_w, w)
        y1 = min(y0 + src_h, h)
        x0 = max(0, x1 - src_w)
        y0 = max(0, y1 - src_h)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            crop = frame
        panel = cv2.resize(crop, (out_w // 2, sh_side), interpolation=cv2.INTER_LANCZOS4)
        if color_grade and color_grade != "none":
            panel = apply_color_grade(panel, color_grade)
        side_panels.append(panel)

    while len(side_panels) < 2:
        side_panels.append(side_panels[-1] if side_panels else np.zeros((sh_side, out_w//2, 3), dtype=np.uint8))

    bottom = np.concatenate([
        cv2.resize(side_panels[0], (out_w // 2, sh_side), interpolation=cv2.INTER_LINEAR),
        cv2.resize(side_panels[1], (out_w - out_w // 2, sh_side), interpolation=cv2.INTER_LINEAR)
    ], axis=1)

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    y = 0
    canvas[y:y+sh_main, :] = main_panel
    y += sh_main
    canvas[y:y+div, :] = PANEL_DIVIDER_COLOR
    y += div
    canvas[y:y+sh_side, :] = bottom
    y += sh_side
    if y < out_h:
        canvas[y:, :] = PANEL_DIVIDER_COLOR
    return canvas


# ── Subtitle Helpers ───────────────────────────────────────────────────────────
def _seconds_to_srt_time(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def whisper_available():
    try:
        import whisper
        return True
    except ImportError:
        return False

def translation_available():
    try:
        import deep_translator
        return True
    except ImportError:
        return False

def transcribe_to_srt(video_path, srt_path, whisper_model="base", language=None,
                       max_chars_per_line=42, progress_callback=None) -> bool:
    def _p(v, msg=""):
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass
    if not whisper_available():
        return False
    import whisper as _w
    _p(0.0, "Extracting audio...")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        if not _extract_audio_wav(video_path, wav_path):
            return False
        _p(0.2, f"Transcribing ({whisper_model})...")
        model = _w.load_model(whisper_model)
        opts = {"word_timestamps": True, "verbose": False}
        if language:
            opts["language"] = language
        result = model.transcribe(wav_path, **opts)
        _p(0.85, "Writing subtitles...")
        words = [{"word": w_["word"].strip(), "start": w_["start"], "end": w_["end"]}
                 for seg in result.get("segments", []) for w_ in seg.get("words", [])]
        lines: List[str] = []
        idx = 1
        buf: List[dict] = []
        buf_len = 0

        def flush():
            nonlocal idx, buf, buf_len
            if not buf:
                return
            lines.append(f"{idx}\n{_seconds_to_srt_time(buf[0]['start'])} --> "
                         f"{_seconds_to_srt_time(buf[-1]['end'])}\n"
                         f"{' '.join(x['word'] for x in buf)}\n")
            idx += 1
            buf = []
            buf_len = 0

        for w_ in words:
            wl = len(w_["word"]) + 1
            if buf_len + wl > max_chars_per_line and buf:
                flush()
            buf.append(w_)
            buf_len += wl
        flush()
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        _p(1.0, f"{len(lines)} subtitle lines")
        return True
    except Exception as e:
        print(f"Whisper failed: {e}", file=sys.stderr)
        return False
    finally:
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass

def translate_srt(srt_path, target_language, source_language="auto",
                  progress_callback=None) -> bool:
    def _p(v, msg=""):
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass
    if not translation_available() or not target_language:
        return not target_language
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return False
    try:
        import re
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"\n\n+", content.strip())
        out = []
        tr = GoogleTranslator(source=source_language, target=target_language)
        for i, block in enumerate(blocks):
            ls = block.strip().splitlines()
            if len(ls) < 3:
                out.append(block)
                continue
            try:
                translated = tr.translate(" ".join(ls[2:])) or " ".join(ls[2:])
            except Exception:
                translated = " ".join(ls[2:])
            out.append(f"{ls[0]}\n{ls[1]}\n{translated}")
            if i % 10 == 0:
                _p(i / max(len(blocks), 1), f"{i}/{len(blocks)}")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out) + "\n")
        _p(1.0, "Translation done")
        return True
    except Exception as e:
        print(f"Translation failed: {e}", file=sys.stderr)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(
    input_path, output_path,
    target_preset_label="Match source (no upscale)",
    tracking_mode="subject",
    sample_interval=None,
    confidence=0.45,
    output_fps=None,
    crf=23,
    encoder_preset="fast",
    audio_bitrate="128k",
    yolo_weights="yolov8n.pt",
    burn_subtitles=False,
    whisper_model="base",
    whisper_language=None,
    subtitle_style_name="Bold White (TikTok)",
    subtitle_max_chars=42,
    subtitle_translate_to=None,
    vignette_strength=0.0,
    sharpen_strength=0.0,
    color_grade="none",
    ken_burns=False,
    progress_callback=None,
):
    def _p(v, msg=""):
        if progress_callback:
            try:
                progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception:
                pass

    result_meta = {"output_path": output_path, "subtitle_path": None,
                   "clamped": False, "effective_size": (0, 0),
                   "duration": 0.0, "panel_mode": False}

    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path) / 1024**2 > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB.")

    info = get_video_info(input_path)
    fps = info["fps"]
    total_frames = info["total_frames"]
    orig_w, orig_h = info["width"], info["height"]
    duration = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ProcessingError("Corrupt or unreadable video.")

    # Allow processing vertical videos too (re-convert)
    # if not info["is_landscape"]:
    #     raise ProcessingError("Video is already vertical.")

    if not sample_interval:
        sample_interval = max(1, int(fps // 2))

    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    req_w, req_h = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update(clamped=clamped, effective_size=(target_w, target_h), duration=duration)
    _p(0.01, f"Output {target_w}x{target_h}  source {orig_w}x{orig_h}")

    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)

    # Detection scale
    det_scale = min(1.0, 640 / orig_w)
    det_w = max(1, int(orig_w * det_scale))
    det_h = max(1, int(orig_h * det_scale))
    sx = orig_w / det_w
    sy = orig_h / det_h

    hw, hh = crop_w // 2, crop_h // 2

    # ── Subtitles ──────────────────────────────────────────────────────────────
    srt_path = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "Transcribing...")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt")
        os.close(srt_fd)
        ok = transcribe_to_srt(input_path, srt_path, whisper_model=whisper_model,
                               language=whisper_language, max_chars_per_line=subtitle_max_chars,
                               progress_callback=lambda v, m: _p(0.02 + v * 0.08, m))
        if not ok:
            if os.path.exists(srt_path):
                os.unlink(srt_path)
            srt_path = None
        elif subtitle_translate_to:
            translate_srt(srt_path, target_language=subtitle_translate_to,
                          progress_callback=lambda v, m: _p(0.10 + v * 0.05, m))
        if srt_path:
            result_meta["subtitle_path"] = srt_path

    # ── Model loading ──────────────────────────────────────────────────────────
    model_obj = None
    if tracking_mode == "subject":
        _p(0.10, "Loading YOLO...")
        model_obj = _get_model(yolo_weights)
        if model_obj is None:
            _p(0.10, "YOLO unavailable — saliency fallback")

    # ── Initialize state ───────────────────────────────────────────────────────
    tracker = SmoothTracker(time_constant_sec=SMOOTH_TIME_CONSTANT)
    tracker.set_alpha(render_fps)
    tracker.reset(orig_w // 2, orig_h // 2)

    scene_detector = SceneCutDetector(threshold=SCENE_CUT_THRESHOLD, window=SCENE_CUT_WINDOW)
    layout_state = LayoutState(fps=render_fps, lock_seconds=LAYOUT_LOCK_SECONDS)

    # Current state
    current_layout = LAYOUT_SINGLE
    current_boxes: List[Tuple[int, int, int, int]] = []
    current_cx, current_cy = orig_w // 2, orig_h // 2
    detection_active = False

    # ── Open encoder ───────────────────────────────────────────────────────────
    style = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])
    proc = _open_ffmpeg_encoder(
        output_path, target_w, target_h, render_fps, audio_source=input_path,
        crf=crf, preset=encoder_preset, audio_bitrate=audio_bitrate,
        subtitle_path=srt_path, subtitle_style=style)

    if color_grade and color_grade != "none":
        _build_lut(color_grade)

    rpt_n = max(1, total_frames // 40)
    fi = 0
    fallback_count = 0

    _p(0.13, f"Rendering {total_frames} frames")

    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames:
                    break

                is_sample = (fi % sample_interval == 0)
                det_result = None

                if is_sample:
                    # Resize for detection
                    det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                    det_gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

                    # Scene cut detection
                    cut = scene_detector.feed(det_gray)
                    if cut:
                        # Reset tracker smoothly toward center
                        tracker.reset(orig_w // 2, orig_h // 2)
                        current_boxes = []
                        detection_active = False

                    # Detection
                    if model_obj is not None:
                        det_result = detect_subjects(det_frame, model_obj, confidence)

                    if det_result is not None:
                        det_cx, det_cy, det_boxes = det_result
                        # Scale to original coordinates
                        src_cx = det_cx * sx
                        src_cy = det_cy * sy
                        src_boxes = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                     for x1, y1, x2, y2 in det_boxes]

                        # Update tracker
                        tracker.update(src_cx, src_cy)
                        current_cx, current_cy = tracker.position
                        current_boxes = src_boxes
                        detection_active = True

                        # Update layout
                        new_layout = classify_layout(src_boxes, orig_w, orig_h)
                        current_layout = layout_state.propose(new_layout)
                    else:
                        # No detection — predict forward
                        pred_cx, pred_cy = tracker.predict(frames_ahead=sample_interval)
                        tracker.update(pred_cx, pred_cy)
                        current_cx, current_cy = tracker.position
                        detection_active = False
                        fallback_count += 1
                else:
                    # Non-sample frame: just predict forward
                    pred_cx, pred_cy = tracker.predict()
                    # Smooth toward prediction
                    current_cx = current_cx * 0.7 + pred_cx * 0.3
                    current_cy = current_cy * 0.7 + pred_cy * 0.3

                # Clamp crop center
                crop_cx = max(hw, min(current_cx, orig_w - hw))
                crop_cy = max(hh, min(current_cy, orig_h - hh))

                # ── Render ─────────────────────────────────────────────────────
                if current_layout == LAYOUT_SINGLE or not current_boxes or len(current_boxes) < 2:
                    out_frame = render_single(frame, crop_cx, crop_cy, crop_w, crop_h,
                                              target_w, target_h)
                elif current_layout == LAYOUT_DUO and len(current_boxes) >= 2:
                    out_frame = render_duo(frame, current_boxes, target_w, target_h,
                                           color_grade=color_grade)
                elif current_layout == LAYOUT_TRIO and len(current_boxes) >= 3:
                    out_frame = render_trio(frame, current_boxes, target_w, target_h,
                                            color_grade=color_grade)
                else:
                    out_frame = render_single(frame, crop_cx, crop_cy, crop_w, crop_h,
                                              target_w, target_h)

                # Post-processing
                if color_grade and color_grade != "none" and current_layout == LAYOUT_SINGLE:
                    out_frame = apply_color_grade(out_frame, color_grade)
                if vignette_strength > 0:
                    out_frame = apply_vignette(out_frame, vignette_strength)
                if sharpen_strength > 0:
                    out_frame = apply_sharpen(out_frame, sharpen_strength)
                if ken_burns:
                    out_frame = apply_ken_burns(out_frame, fi, render_fps)

                # Write
                try:
                    proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    break

                fi += 1
                if fi % rpt_n == 0:
                    _p(0.13 + 0.75 * (fi / total_frames), f"{fi}/{total_frames}...")

    except BrokenPipeError:
        pass

    _p(0.88, "Encoding...")
    try:
        _close_ffmpeg_encoder(proc, output_path)
    except ProcessingError:
        raise
    except Exception as e:
        raise ProcessingError(f"Encoder shutdown failed: {e}")

    _p(1.0, "Done!")
    print(f"Output: {output_path}  ({os.path.getsize(output_path)/1024**2:.1f} MB)", file=sys.stderr)
    result_meta["fallback_frames"] = fallback_count
    return result_meta


# ── Batch clip pipeline ────────────────────────────────────────────────────────
def process_clips_batch(
    input_path, output_dir, clips,
    target_preset_label="720p   (720x1280  - HD)",
    tracking_mode="subject",
    confidence=0.45,
    output_fps=None,
    crf=23, encoder_preset="fast", audio_bitrate="128k",
    yolo_weights="yolov8n.pt", burn_subtitles=False, whisper_model="base",
    whisper_language=None, subtitle_style_name="Bold White (TikTok)",
    subtitle_max_chars=42, subtitle_translate_to=None,
    vignette_strength=0.0, sharpen_strength=0.0, color_grade="none",
    ken_burns=False, progress_callback=None,
):
    def _p(v, msg=""):
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, clip in enumerate(clips):
        base_pct = i / max(len(clips), 1)
        next_pct = (i + 1) / max(len(clips), 1)
        _p(base_pct, f"Clip {i+1}/{len(clips)}...")
        trimmed_path = out_path = None
        try:
            fd, trimmed_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            if not _trim_video(input_path, trimmed_path, clip.start_sec, clip.end_sec):
                results.append({"clip": clip, "output_path": None, "error": "trim failed"})
                continue
            out_path = os.path.join(output_dir,
                f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s_vertical.mp4")

            def clip_cb(v, msg="", _b=base_pct, _n=next_pct):
                _p(_b + v * (_n - _b), msg)

            meta = process_video(
                trimmed_path, out_path,
                target_preset_label=target_preset_label,
                tracking_mode=tracking_mode,
                confidence=confidence,
                output_fps=output_fps,
                crf=crf, encoder_preset=encoder_preset, audio_bitrate=audio_bitrate,
                yolo_weights=yolo_weights, burn_subtitles=burn_subtitles,
                whisper_model=whisper_model, whisper_language=whisper_language,
                subtitle_style_name=subtitle_style_name,
                subtitle_max_chars=subtitle_max_chars,
                subtitle_translate_to=subtitle_translate_to,
                vignette_strength=vignette_strength,
                sharpen_strength=sharpen_strength,
                color_grade=color_grade,
                ken_burns=ken_burns,
                progress_callback=clip_cb)
            meta["clip"] = clip
            results.append(meta)
        except Exception as exc:
            results.append({"clip": clip, "output_path": out_path, "error": str(exc)})
        finally:
            if trimmed_path and os.path.exists(trimmed_path):
                try:
                    os.unlink(trimmed_path)
                except OSError:
                    pass
    n_ok = sum(1 for r in results if not r.get("error"))
    _p(1.0, f"{n_ok}/{len(results)} clips done")
    return results
