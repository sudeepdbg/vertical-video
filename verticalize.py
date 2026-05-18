"""
verticalize.py — AI Vertical Video Converter v4.4 (Sports Intelligence Engine)
───────────────────────────────────────────────────────────────────────────────
IMPROVEMENTS over v4.3:
1. PLAY PHASE DETECTION: Distinguishes Fast Break vs. Half Court to adjust prediction horizon.
2. BALL TRAJECTORY PHYSICS: Predicts landing spot of high-arcing balls (shots/passses).
3. COURT-AWARE RESET: Resets tracker to court center-of-mass on scene cuts, not last position.
4. ADAPTIVE Q (Lightweight IMM): Inflates process noise during high-jerk events to reduce lag.
5. HIGH-FREQ SAMPLING: Sports mode now samples at ~2Hz (fps/15) instead of 0.2Hz.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ─── Custom exception ──────────────────────────────────────────────────────────

class ProcessingError(Exception):
    pass


# ─── Constants ─────────────────────────────────────────────────────────────────

PERSON_CLASS_ID      = 0
SPORTS_BALL_CLASS_ID = 32
HIGH_PRIO_CLASSES    = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB     = 2_000
MIN_FRAME_DIM        = 240
MAX_FRAMES_GUARD     = 1_080_000
LOWER_THIRD_GUARD    = 0.80

# Panel detection thresholds
PANEL_MIN_PERSONS          = 2
PANEL_PROBE_COUNT          = 30
PANEL_MAJORITY_FRAC        = 0.60
PANEL_STABILITY_FRAC       = 0.75
PANEL_MAX_PERSON_MOTION    = 8.0
PANEL_MIN_PERSON_AREA_FRAC = 0.06
PANEL_MAX_COUNT_VARIANCE   = 1.5
PANEL_MIN_PERSON_ASPECT    = 1.3

# Sports constants
SPORTS_COURT_COLORS_HSV = [
    {"h": [10, 30],  "s": [40, 180], "v": [80, 220]},   # Basketball
    {"h": [35, 85],  "s": [40, 255], "v": [40, 220]},   # Football/Soccer
    {"h": [90, 130], "s": [0,  60],  "v": [150, 255]},  # Hockey
]

# Kalman Constants
KALMAN_PROCESS_NOISE_BASE    = 1e-2
KALMAN_PROCESS_NOISE_HIGH    = 1e-1  # For high acceleration/jerk
KALMAN_MEASUREMENT_NOISE     = 1e-1
KALMAN_OPTICAL_FLOW_NOISE    = 5e-1
KALMAN_SALIENCY_NOISE        = 2e-0
KALMAN_INITIAL_ERROR         = 1.0
KALMAN_GATE_THRESHOLD        = 4.0

# Physics & Prediction
GRAVITY_PIXELS_PER_SEC2      = 980  # Approx gravity in px/s^2 (depends on resolution)
FAST_BREAK_PREDICT_SEC       = 0.8  # Look ahead 0.8s for fast breaks
HALF_COURT_PREDICT_SEC       = 0.3  # Look ahead 0.3s for set plays
BALL_SPEED_THRESHOLD         = 15.0 # px/frame to consider "fast"

SPORTS_SCENE_CUT_THRESHOLD   = 0.22
SPORTS_SCENE_CUT_MIN_FRAMES  = 3
SPORTS_SWITCH_BALL_BONUS     = 200
SPORTS_BALL_CONFIDENCE       = 0.35
SPORTS_BALL_PROXIMITY_PX     = 120
SPORTS_EVENT_EXPAND_FRAMES   = 15
SPORTS_EVENT_EXPAND_FACTOR   = 1.25

# Legacy velocity -> Gaussian window table (non-sports)
VELOCITY_SMOOTH_TABLE: List[Tuple[float, int]] = [
    (0.0, 61), (3.0, 53), (8.0, 43), (15.0, 33),
    (30.0, 23), (60.0, 15), (120.0, 9),
]

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Match source (no upscale)":    (0, 0),
    "1080p  (1080x1920 - Full HD)": (1080, 1920),
    "720p   (720x1280  - HD)":      (720, 1280),
    "540p   (540x960   - SD)":      (540, 960),
    "480p   (480x854   - Low)":     (480, 854),
}

SUBTITLE_STYLES: Dict[str, Dict[str, Any]] = {
    "Bold White (TikTok)": {
        "fontsize": 18, "primary_color": "&H00FFFFFF", "outline_color": "&H00000000",
        "outline": 2, "bold": 1, "shadow": 0, "back_color": "&H00000000", "margin_v": 80,
    },
    "Yellow (Classic)": {
        "fontsize": 16, "primary_color": "&H0000FFFF", "outline_color": "&H00000000",
        "outline": 2, "bold": 1, "shadow": 1, "back_color": "&H00000000", "margin_v": 80,
    },
    "Box (Accessible)": {
        "fontsize": 15, "primary_color": "&H00FFFFFF", "outline_color": "&H00000000",
        "outline": 0, "bold": 0, "shadow": 0, "back_color": "&H80000000", "margin_v": 80,
    },
}

TRANSLATION_LANGUAGES: Dict[str, str] = {
    "None (keep original)": "",   "French": "fr",       "German": "de",
    "Spanish": "es",              "Italian": "it",       "Portuguese": "pt",
    "Dutch": "nl",                "Polish": "pl",        "Russian": "ru",
    "Japanese": "ja",             "Korean": "ko",        "Chinese (Simplified)": "zh-CN",
    "Arabic": "ar",               "Hindi": "hi",         "Turkish": "tr",
    "Indonesian": "id",           "Swedish": "sv",       "Norwegian": "no",
    "Danish": "da",               "Finnish": "fi",       "Greek": "el",
    "Hebrew": "iw",               "Thai": "th",          "Vietnamese": "vi",
    "Malay": "ms",                "Ukrainian": "uk",
}

# Visual constants
VIGNETTE_STRENGTH       = 0.55
VIGNETTE_FALLOFF        = 1.8
COLOR_GRADES            = ("none", "warm", "cool", "vibrant", "matte")
PANEL_SLOT_EMA          = 0.07
PANEL_SLOT_MAX_JUMP     = 0.08
KEN_BURNS_MAX_ZOOM      = 1.04
KEN_BURNS_PERIOD        = 8.0
DISSOLVE_FRAMES         = 3
PANEL_DIVIDER_PX        = 3
PANEL_DIVIDER_COLOR     = (15, 15, 15)
PANEL_CROP_EXPAND       = 1.55
PANEL_TRANSITION_FRAMES = 6


# ─── Panel Mode Configuration ─────────────────────────────────────────────────

@dataclass
class PanelModeConfig:
    split_mode: str = "auto"
    n_splits: int = 2
    split_orientation: str = "horizontal"
    max_person_motion: float   = PANEL_MAX_PERSON_MOTION
    min_person_area_frac: float = PANEL_MIN_PERSON_AREA_FRAC
    max_count_variance: float  = PANEL_MAX_COUNT_VARIANCE
    stability_frac: float      = PANEL_STABILITY_FRAC

    def __post_init__(self) -> None:
        if self.split_mode not in ("auto", "force_on", "force_off"):
            raise ValueError(f"split_mode must be 'auto', 'force_on', or 'force_off', got '{self.split_mode}'")
        if self.split_orientation not in ("horizontal", "vertical"):
            raise ValueError(f"split_orientation must be 'horizontal' or 'vertical', got '{self.split_orientation}'")
        if not (1 <= self.n_splits <= 4):
            raise ValueError(f"n_splits must be between 1 and 4, got {self.n_splits}")
        if self.n_splits > 2:
            print(
                f"[PanelModeConfig] n_splits={self.n_splits} not fully implemented; "
                "falling back to 2 splits.", file=sys.stderr,
            )


# ─── Clip segment ──────────────────────────────────────────────────────────────

class ClipSegment:
    def __init__(
        self, start_sec: float, end_sec: float, score: float,
        soi_region: str = "center", peak_frame: int = 0, title: str = "",
    ) -> None:
        self.start_sec  = start_sec
        self.end_sec    = end_sec
        self.score      = score
        self.soi_region = soi_region
        self.peak_frame = peak_frame
        self.title      = title
        self.duration   = end_sec - start_sec

    def __repr__(self) -> str:
        return f"<Clip {self.start_sec:.1f}s-{self.end_sec:.1f}s score={self.score:.2f}>"


# ─── Feature availability guards ───────────────────────────────────────────────

def whisper_available() -> bool:
    try:
        import whisper
        return True
    except ImportError:
        return False


def translation_available() -> bool:
    try:
        import deep_translator
        return True
    except ImportError:
        return False


def yolo_available() -> bool:
    if not _YOLO_AVAILABLE:
        return False
    try:
        import urllib.request
        urllib.request.urlopen("https://github.com", timeout=3)
        return True
    except Exception:
        return os.path.exists("yolov8n.pt") or os.path.exists("yolov8s.pt")


# ─── Vignette (cached numpy mask) ─────────────────────────────────────────────

_vignette_cache: Dict[Tuple, np.ndarray] = {}


def _build_vignette(
    w: int, h: int,
    strength: float = VIGNETTE_STRENGTH,
    falloff: float  = VIGNETTE_FALLOFF,
) -> np.ndarray:
    key = (w, h, round(strength, 3), round(falloff, 3))
    if key not in _vignette_cache:
        xs = np.linspace(-1, 1, w, dtype=np.float32)
        ys = np.linspace(-1, 1, h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        dist   = np.sqrt(xg**2 + yg**2)
        dist  /= dist.max()
        mask   = np.clip(1.0 - strength * (dist**falloff), 0.0, 1.0)[:, :, np.newaxis]
        _vignette_cache[key] = mask
    return _vignette_cache[key]


def apply_vignette(frame: np.ndarray, strength: float = VIGNETTE_STRENGTH) -> np.ndarray:
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    return (frame.astype(np.float32) * _build_vignette(w, h, strength)).clip(0, 255).astype(np.uint8)


# ─── Unsharp mask ──────────────────────────────────────────────────────────────

def apply_sharpen(frame: np.ndarray, strength: float = 0.6, radius: int = 1) -> np.ndarray:
    if strength <= 0:
        return frame
    ksize   = radius * 2 + 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)


# ─── Color grade LUT ───────────────────────────────────────────────────────────

_lut_cache: Dict[str, np.ndarray] = {}


def _build_lut(grade: str) -> np.ndarray:
    if grade in _lut_cache:
        return _lut_cache[grade]
    x = np.arange(256, dtype=np.float32)
    if grade == "warm":
        r = np.clip(x * 1.06 + 5,  0, 255)
        g = np.clip(x * 1.02 + 2,  0, 255)
        b = np.clip(x * 0.92 - 4,  0, 255)
    elif grade == "cool":
        r = np.clip(x * 0.92 - 4,  0, 255)
        g = np.clip(x * 1.01 + 1,  0, 255)
        b = np.clip(x * 1.07 + 6,  0, 255)
    elif grade == "vibrant":
        def _sc(v: np.ndarray) -> np.ndarray:
            n = v / 255
            s = n * n * (3 - 2 * n)
            return np.clip((n * 0.6 + s * 0.4) * 255, 0, 255)
        r = _sc(x * 1.04); g = _sc(x * 1.02); b = _sc(x)
    elif grade == "matte":
        r = np.clip(x * 0.88 + 18, 0, 255)
        g = np.clip(x * 0.86 + 16, 0, 255)
        b = np.clip(x * 0.84 + 22, 0, 255)
    else:
        r = g = b = x.copy()
    lut = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(256, 1, 3)
    _lut_cache[grade] = lut
    return lut


def apply_color_grade(frame: np.ndarray, grade: str = "none") -> np.ndarray:
    if not grade or grade == "none":
        return frame
    return cv2.LUT(frame, _build_lut(grade))


# ─── Ken Burns micro-zoom ──────────────────────────────────────────────────────

def apply_ken_burns(
    frame: np.ndarray, frame_idx: int, fps: float,
    max_zoom: float = KEN_BURNS_MAX_ZOOM, period: float = KEN_BURNS_PERIOD,
) -> np.ndarray:
    if max_zoom <= 1.0:
        return frame
    t     = (frame_idx / max(fps, 1)) % period
    scale = 1.0 + (max_zoom - 1.0) * 0.5 * (1 - math.cos(2 * math.pi * t / period))
    if abs(scale - 1.0) < 1e-4:
        return frame
    h, w = frame.shape[:2]
    nw   = max(int(w / scale), 2)
    nh   = max(int(h / scale), 2)
    x0   = (w - nw) // 2
    y0   = (h - nh) // 2
    return cv2.resize(frame[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)


# ─── Cross-dissolve on scene cuts ──────────────────────────────────────────────

class DissolveBuffer:
    def __init__(self, n: int = DISSOLVE_FRAMES) -> None:
        self.n    = n
        self._buf: Optional[np.ndarray] = None
        self._rem = 0

    def on_cut(self, last_frame: np.ndarray) -> None:
        self._buf = last_frame.copy()
        self._rem = self.n

    def blend(self, new_frame: np.ndarray) -> np.ndarray:
        if self._rem <= 0 or self._buf is None:
            return new_frame
        alpha     = self._rem / self.n
        self._rem -= 1
        return cv2.addWeighted(self._buf, alpha, new_frame, 1.0 - alpha, 0)

    @property
    def active(self) -> bool:
        return self._rem > 0


# ─── FFmpeg post-filter chain ──────────────────────────────────────────────────

def _build_ffmpeg_vf(color_grade: str = "none", ffmpeg_sharpen: bool = False) -> List[str]:
    filters: List[str] = []
    eq_map = {
        "warm":    "brightness=0.02:saturation=1.12:gamma_r=1.05:gamma_b=0.95",
        "cool":    "brightness=0.01:saturation=1.08:gamma_r=0.95:gamma_b=1.05",
        "vibrant": "brightness=0.0:saturation=1.25:contrast=1.05",
        "matte":   "brightness=0.03:saturation=0.85:contrast=0.92",
    }
    if color_grade in eq_map:
        filters.append(f"eq={eq_map[color_grade]}")
    if ffmpeg_sharpen:
        filters.append("unsharp=5:5:0.8:3:3:0.0")
    return filters


# ─── FFmpegVideoReader ─────────────────────────────────────────────────────────

class FFmpegVideoReader:
    def __init__(
        self, path: str, width: int, height: int,
        seek_sec: float = 0.0, n_frames: Optional[int] = None,
        scale_w: Optional[int] = None, scale_h: Optional[int] = None,
    ) -> None:
        self.path         = path
        self.width        = width
        self.height       = height
        self.seek_sec     = seek_sec
        self.n_frames     = n_frames
        self.out_w        = scale_w or width
        self.out_h        = scale_h or height
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover    = b""

    def _build_cmd(self, extra_decoder_flags: List[str]) -> List[str]:
        cmd = ["ffmpeg"]
        if self.seek_sec > 0:
            cmd += ["-ss", str(self.seek_sec)]
        cmd += extra_decoder_flags
        cmd += ["-i", self.path, "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-vf", f"scale={self.out_w}:{self.out_h}"]
        if self.n_frames is not None:
            cmd += ["-vframes", str(self.n_frames)]
        cmd += ["pipe:1"]
        return cmd

    def _open(self) -> None:
        candidates = [
            self._build_cmd(["-vcodec", "libdav1d"]),
            self._build_cmd(["-hwaccel", "none"]),
        ]
        for cmd in candidates:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=max(self._frame_bytes * 4, 1 << 20),
                )
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc     = proc
                    self._leftover = test
                    return
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                proc.wait()
            except Exception:
                pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            self._proc.wait()
            self._proc = None

    def __enter__(self) -> "FFmpegVideoReader":
        self._open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __iter__(self):
        if not self._proc:
            self._open()
        buf = self._leftover
        self._leftover = b""
        while True:
            needed = self._frame_bytes - len(buf)
            while needed > 0:
                chunk  = self._proc.stdout.read(needed)
                if not chunk:
                    return
                buf   += chunk
                needed -= len(chunk)
            yield np.frombuffer(buf[:self._frame_bytes], dtype=np.uint8).reshape(
                self.out_h, self.out_w, 3
            )
            buf = buf[self._frame_bytes:]


def _read_frame_at(
    path: str, width: int, height: int, t_sec: float,
    scale_w: Optional[int] = None, scale_h: Optional[int] = None,
) -> Optional[np.ndarray]:
    r = FFmpegVideoReader(path, width, height, seek_sec=t_sec, n_frames=1,
                          scale_w=scale_w, scale_h=scale_h)
    r._open()
    frames = list(r)
    r.close()
    return frames[0] if frames else None


# ─── FFmpeg helpers ────────────────────────────────────────────────────────────

def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{tool} not found. Install FFmpeg.")


def _has_audio(path: str) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15,
        )
        return "audio" in r.stdout
    except Exception:
        return False


def _extract_audio_wav(vpath: str, wpath: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", vpath, "-ar", "16000", "-ac", "1", "-f", "wav", wpath],
        capture_output=True,
    )
    return r.returncode == 0 and os.path.exists(wpath)


def _trim_video(inp: str, out: str, start: float, end: float) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-hwaccel", "none",
         "-ss", str(start), "-to", str(end), "-i", inp,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
         "-c:a", "aac", "-b:a", "128k",
         "-avoid_negative_ts", "make_zero", "-reset_timestamps", "1", out],
        capture_output=True,
    )
    return r.returncode == 0 and os.path.exists(out)


# ─── Encoder ──────────────────────────────────────────────────────────────────

def _open_ffmpeg_encoder(
    output_path: str, width: int, height: int, fps: float,
    audio_source: Optional[str], crf: int = 23, preset: str = "fast",
    audio_bitrate: str = "128k", subtitle_path: Optional[str] = None,
    subtitle_style: Optional[Dict[str, Any]] = None,
    extra_vf: Optional[List[str]] = None,
) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
    ]
    has_aud = bool(audio_source and _has_audio(audio_source))
    if has_aud:
        cmd += ["-hwaccel", "none", "-i", audio_source]

    vf: List[str] = []
    if subtitle_path and os.path.exists(subtitle_path):
        s    = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        sesc = subtitle_path.replace("\\", "/").replace(":", r"\:")
        force = (
            f"Fontsize={s.get('fontsize', 18)},"
            f"PrimaryColour={s.get('primary_color', '&H00FFFFFF')},"
            f"OutlineColour={s.get('outline_color', '&H00000000')},"
            f"Outline={s.get('outline', 2)},Bold={s.get('bold', 1)},"
            f"Shadow={s.get('shadow', 0)},BackColour={s.get('back_color', '&H00000000')},"
            f"MarginV={s.get('margin_v', 80)},Alignment=2"
        )
        vf.append(f"subtitles='{sesc}':force_style='{force}'")
    if extra_vf:
        vf.extend(extra_vf)

    cmd += ["-map", "0:v:0"]
    if has_aud:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    else:
        cmd += ["-an"]
    if vf:
        cmd += ["-vf", ", ".join(vf)]
    cmd += [
        "-aspect", f"{width}:{height}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
        "-shortest", "-movflags", "+faststart", output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _close_ffmpeg_encoder(proc: subprocess.Popen, output_path: str) -> None:
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait()
    if proc.returncode != 0:
        try:
            err = proc.stderr.read(2000).decode(errors="replace")
        except Exception:
            err = ""
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")


# ─── Video metadata ────────────────────────────────────────────────────────────

def get_video_info(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1", path,
    ]
    r  = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    kv: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    w = int(kv.get("width",  0) or 0)
    h = int(kv.get("height", 0) or 0)
    try:
        num, den = kv.get("r_frame_rate", "30/1").split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    dur = float(kv.get("duration", 0.0) or 0.0)
    if dur <= 0:
        nb  = int(kv.get("nb_frames", 0) or 0)
        dur = nb / fps if fps > 0 and nb > 0 else 0.0
    if w == 0 or h == 0:
        raise ProcessingError(f"Cannot read dimensions: {path}")
    return {
        "fps":              fps,
        "total_frames":     min(int(dur * fps), MAX_FRAMES_GUARD),
        "width":            w,
        "height":            h,
        "duration_seconds": dur,
        "is_landscape":     w > h,
    }


def extract_thumbnail(path: str, t: float = 1.0) -> Optional[bytes]:
    info  = get_video_info(path)
    frame = _read_frame_at(path, info["width"], info["height"], t, scale_w=320, scale_h=180)
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ─── Resolution helpers ────────────────────────────────────────────────────────

def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        cw = int(orig_h * 9 / 16)
        if cw > orig_w:
            cw = orig_w
        ch = int(cw * 16 / 9)
    else:
        cw, ch = tw, th

    if ch > orig_h:
        scale = orig_h / ch
        cw    = int(cw * scale)
        ch    = int(orig_h)
    if cw > orig_w:
        scale = orig_w / cw
        cw    = int(orig_w)
        ch    = int(ch * scale)

    return max(cw - (cw % 2), 2), max(ch - (ch % 2), 2)


def calculate_crop_dims(orig_w: int, orig_h: int, tw: int, th: int) -> Tuple[int, int]:
    th    = max(th, 2)
    ratio = tw / th
    if (orig_w / orig_h) > ratio:
        ch = orig_h
        cw = int(round(ch * ratio))
    else:
        cw = orig_w
        ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ─── YOLO model cache ──────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}


def _get_model(weights: str = "yolov8n.pt") -> Optional[Any]:
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


# ─── Face detection ────────────────────────────────────────────────────────────

_haar_cascade: Optional[cv2.CascadeClassifier] = None


def _get_haar() -> Optional[cv2.CascadeClassifier]:
    global _haar_cascade
    if _haar_cascade:
        return _haar_cascade
    p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(p):
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            _haar_cascade = c
            return c
    return None


def detect_faces(
    frame: np.ndarray, confidence_thresh: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    haar = _get_haar()
    if not haar:
        return []
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw  = haar.detectMultiScale(gray, 1.1, 5, minSize=(max(30, w // 20), max(30, h // 20)))
    if len(raw) == 0:
        return []
    faces = [(x, y, x + bw, y + bh) for x, y, bw, bh in raw]
    faces.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
    return faces


# ─── Play Phase Detection (NEW v4.4) ──────────────────────────────────────────

class SportsPlayPhaseDetector:
    """
    Determines if the game is in a 'fast_break', 'half_court', or 'rebound' phase.
    Uses player spread and ball velocity.
    """
    def __init__(self, fps: float):
        self.fps = fps
        self.prev_ball_pos: Optional[Tuple[float, float]] = None
        self.ball_vel_history: List[float] = []
        
    def detect_phase(
        self, 
        persons: List[Tuple[int, int, int, int]], 
        ball_box: Optional[Tuple[int, int, int, int]],
        frame_w: int
    ) -> str:
        """
        Returns: 'fast_break', 'half_court', 'rebound', or 'static'
        """
        # 1. Calculate Player Spread (Standard Deviation of X positions)
        if not persons:
            return 'static'
        
        centers_x = [(p[0] + p[2]) / 2 for p in persons]
        mean_x = np.mean(centers_x)
        spread = np.std(centers_x)
        
        # Normalized spread (0.0 to 1.0)
        norm_spread = spread / (frame_w / 2)
        
        # 2. Calculate Ball Velocity
        ball_speed = 0.0
        if ball_box:
            bx = (ball_box[0] + ball_box[2]) / 2
            by = (ball_box[1] + ball_box[3]) / 2
            if self.prev_ball_pos:
                dx = bx - self.prev_ball_pos[0]
                dy = by - self.prev_ball_pos[1]
                ball_speed = math.sqrt(dx*dx + dy*dy)
            self.prev_ball_pos = (bx, by)
            
            self.ball_vel_history.append(ball_speed)
            if len(self.ball_vel_history) > 10:
                self.ball_vel_history.pop(0)
        
        avg_ball_speed = np.mean(self.ball_vel_history) if self.ball_vel_history else 0
        
        # 3. Phase Logic
        # Fast Break: High ball speed, players spreading out (high spread)
        if avg_ball_speed > BALL_SPEED_THRESHOLD and norm_spread > 0.15:
            return 'fast_break'
        
        # Rebound/Lose Ball: High player clustering (low spread) but high motion
        # Or simply low spread often indicates paint action/rebounding
        if norm_spread < 0.08:
            return 'rebound'
            
        # Half Court: Moderate spread, lower ball speed
        return 'half_court'


# ─── SportsKalmanTracker (IMPROVED v4.4) ──────────────────────────────────────

class SportsKalmanTracker:
    """
    2-D constant-acceleration Kalman filter with Play-Phase Aware Prediction.
    
    IMPROVEMENTS v4.4:
    1. Adaptive Prediction Horizon: Looks further ahead during fast breaks.
    2. Ball Trajectory Physics: Predicts landing spot for high-arcing balls.
    3. Adaptive Q: Inflates process noise during high-jerk events (Lightweight IMM).
    """

    def __init__(self, dt: float = 1.0, fps: float = 30.0) -> None:
        self.dt = dt
        self.fps = fps
        # State: [cx, cy, vx, vy, ax, ay]
        self.F  = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,          0.5*dt**2],
            [0, 0, 1,  0,  dt,         0],
            [0, 0, 0,  1,  0,          dt],
            [0, 0, 0,  0,  1,          0],
            [0, 0, 0,  0,  0,          1],
        ], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        
        # Base Process Noise
        self.Q_base  = np.eye(6, dtype=np.float64) * KALMAN_PROCESS_NOISE_BASE
        self.Q_base[4, 4] *= 4.0 # Acceleration noise
        self.Q_base[5, 5] *= 4.0
        
        # Measurement Noise (R) per sensor
        self.R_yolo      = np.eye(2, dtype=np.float64) * KALMAN_MEASUREMENT_NOISE
        self.R_optical   = np.eye(2, dtype=np.float64) * KALMAN_OPTICAL_FLOW_NOISE
        self.R_saliency  = np.eye(2, dtype=np.float64) * KALMAN_SALIENCY_NOISE
        
        # State Covariance
        self.P  = np.eye(6, dtype=np.float64) * KALMAN_INITIAL_ERROR
        self.x  = np.zeros((6, 1), dtype=np.float64)
        
        self.initialized  = False
        self._stale_count = 0
        self._last_sensor = "none"
        
        # For adaptive Q
        self._prev_accel_mag = 0.0

    def init(self, cx: float, cy: float) -> None:
        self.x = np.array([[cx], [cy], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * KALMAN_INITIAL_ERROR
        self.initialized  = True
        self._stale_count = 0
        self._last_sensor = "init"
        self._prev_accel_mag = 0.0

    def predict_adaptive(self, play_phase: str, ball_is_airborne: bool = False, ball_vel: Optional[Tuple[float,float]] = None) -> Tuple[float, float]:
        """
        Predict position based on play phase.
        """
        if not self.initialized:
            return 0.0, 0.0
            
        # Determine look-ahead time
        if play_phase == "fast_break":
            steps = int(self.fps * FAST_BREAK_PREDICT_SEC)
        elif play_phase == "rebound":
            steps = int(self.fps * 0.1) # Very short lookahead for chaotic rebounds
        else: # half_court
            steps = int(self.fps * HALF_COURT_PREDICT_SEC)
            
        steps = max(1, steps)
        
        # Standard Kinematic Prediction
        dt_s = self.dt * steps
        x_pred = self.x.copy()
        
        # If ball is airborne, we might want to predict trajectory differently
        # But for camera centering, following the predicted kinematic path of the 
        # *player* or *ball carrier* is usually best. 
        # If we are tracking the ball directly and it's airborne:
        if ball_is_airborne and ball_vel:
            # Simple projectile motion adjustment could go here
            # For now, we rely on the constant acceleration model in self.x
            pass
            
        x_pred[0] += self.x[2] * dt_s + 0.5 * self.x[4] * dt_s**2
        x_pred[1] += self.x[3] * dt_s + 0.5 * self.x[5] * dt_s**2
        x_pred[2] += self.x[4] * dt_s
        x_pred[3] += self.x[5] * dt_s
        
        return float(x_pred[0, 0]), float(x_pred[1, 0])

    def _predict_step(self) -> None:
        """Internal: advance state by one dt (called every frame)."""
        if self.initialized:
            # Adaptive Q: If previous acceleration was high, increase Q temporarily
            current_accel_mag = math.sqrt(self.x[4,0]**2 + self.x[5,0]**2)
            if current_accel_mag > 50.0: # High jerk threshold
                Q_used = self.Q_base * 10 # Inflate Q
            else:
                Q_used = self.Q_base
                
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + Q_used
            self._stale_count += 1
            
            # Clamp velocity to prevent explosion during long dropouts
            max_vel = 200.0 # pixels per frame
            if abs(self.x[2,0]) > max_vel: self.x[2,0] = np.sign(self.x[2,0]) * max_vel
            if abs(self.x[3,0]) > max_vel: self.x[3,0] = np.sign(self.x[3,0]) * max_vel

    def update(
        self, cx: float, cy: float, sensor: str = "yolo",
    ) -> Tuple[float, float]:
        """
        Process measurement; returns filtered position.
        Implements Mahalanobis gating to reject outliers.
        """
        if not self.initialized:
            self.init(cx, cy)
            self._last_sensor = sensor
            return cx, cy

        # Select R based on sensor trustworthiness
        if sensor == "optical_flow":
            R = self.R_optical
        elif sensor == "saliency":
            R = self.R_saliency
        else:
            R = self.R_yolo

        # Innovation (measurement residual)
        z = np.array([[cx], [cy]], dtype=np.float64)
        y = z - self.H @ self.x
        
        # Innovation Covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Mahalanobis Distance for Gating
        inv_S = np.linalg.inv(S)
        mahalanobis_dist = np.sqrt(float(y.T @ inv_S @ y))
        
        if mahalanobis_dist > KALMAN_GATE_THRESHOLD:
            # Reject measurement as outlier. Keep prediction.
            self._stale_count += 1
            return float(self.x[0,0]), float(self.x[1,0])

        # Kalman Gain
        K = self.P @ self.H.T @ inv_S
        
        # State Update
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ self.H) @ self.P
        
        self._stale_count = 0
        self._last_sensor = sensor
        
        return float(self.x[0, 0]), float(self.x[1, 0])

    def increment_stale(self) -> None:
        """Call when no measurement is available this frame. Advances prediction."""
        self._predict_step()

    @property
    def is_stale(self) -> bool:
        return self._stale_count > 10

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.sqrt(vx * vx + vy * vy)

    @property
    def last_sensor(self) -> str:
        return self._last_sensor


# ─── Court/field boundary detection ──────────────────────────────────────────

def detect_field_of_play(
    frame: np.ndarray, sport_hint: str = "auto",
) -> Optional[np.ndarray]:
    """Return binary mask: 1 = field-of-play, 0 = crowd/scoreboard/other."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    def _make_mask(color_range: Dict) -> np.ndarray:
        lower = np.array([color_range["h"][0], color_range["s"][0], color_range["v"][0]])
        upper = np.array([color_range["h"][1], color_range["s"][1], color_range["v"][1]])
        m      = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        m      = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    if sport_hint == "auto":
        best_mask, best_area = None, 0
        for cr in SPORTS_COURT_COLORS_HSV:
            m    = _make_mask(cr)
            area = cv2.countNonZero(m)
            if area > best_area and area > (h * w * 0.15):
                best_area, best_mask = area, m
        return best_mask

    sport_ranges = {
        "basketball": [SPORTS_COURT_COLORS_HSV[0]],
        "football":   [SPORTS_COURT_COLORS_HSV[1]],
        "soccer":     [SPORTS_COURT_COLORS_HSV[1]],
        "hockey":     [SPORTS_COURT_COLORS_HSV[2]],
    }
    mask = np.zeros((h, w), dtype=np.uint8)
    for cr in sport_ranges.get(sport_hint, SPORTS_COURT_COLORS_HSV):
        mask = cv2.bitwise_or(mask, _make_mask(cr))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask    = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask if cv2.countNonZero(mask) > (h * w * 0.10) else None


def get_court_center_of_mass(field_mask: np.ndarray) -> Tuple[float, float]:
    """Calculate the centroid of the detected court area."""
    if field_mask is None:
        return None
    moments = cv2.moments(field_mask)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


# ─── Sports-specific optical flow ─────────────────────────────────────────────

def sports_optical_flow_center(
    prev: np.ndarray,
    curr: np.ndarray,
    w: int,
    h: int,
    prev_center: Optional[Tuple[int, int]] = None,
    field_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int]]:
    if prev is None or curr is None:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        b = max(1, int(w * 0.04))
        mag[:, :b] = mag[:, w - b:] = mag[:b, :] = mag[h - b:, :] = 0

        if field_mask is not None:
            if field_mask.shape[:2] != (h, w):
                fm = cv2.resize(field_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                fm = field_mask
            mag = mag * (fm.astype(np.float32) / 255.0)

        if prev_center is not None:
            pcx, pcy = prev_center
            ys, xs   = np.mgrid[0:h, 0:w]
            dist     = np.sqrt((xs - pcx)**2 + (ys - pcy)**2)
            mag      = mag * np.exp(-dist / (max(w, h) * 0.3))

        if mag.max() < 0.5:
            return None
        t = mag.sum()
        if t == 0:
            return None
        ys, xs = np.mgrid[0:h, 0:w]
        return int((xs * mag).sum() / t), int((ys * mag).sum() / t)
    except Exception:
        return None


# ─── Temporal saliency ────────────────────────────────────────────────────────

def temporal_saliency_center(
    frame: np.ndarray,
    prev_saliency: Optional[np.ndarray] = None,
    decay: float = 0.7,
) -> Tuple[int, int, np.ndarray]:
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIM or h < MIN_FRAME_DIM:
        return w // 2, h // 2, np.zeros((h, w), dtype=np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31, 31), 0
    )
    sat  = cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (31, 31), 0
    )
    sal  = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)

    if prev_saliency is not None:
        temporal_diff = np.abs(sal - prev_saliency * decay)
        sal           = sal * (1.0 + temporal_diff * 2.0)

    b = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w - b:] = sal[:b, :] = sal[h - b:, :] = 0

    t = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2, sal

    ys, xs = np.mgrid[0:h, 0:w]
    cx     = int((xs * sal).sum() / t)
    cy     = int((ys * sal).sum() / t)

    return cx, cy, sal


# ─── Sports broadcast cut detection ──────────────────────────────────────────

def _ensure_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def is_sports_scene_change(
    prev: Optional[np.ndarray],
    curr: np.ndarray,
    prev_hist: Optional[np.ndarray] = None,
    frame_count: int = 0,
    last_cut_frame: int = -100,
) -> Tuple[bool, Optional[np.ndarray], int]:
    curr_bgr = _ensure_bgr(curr)
    prev_bgr = _ensure_bgr(prev)

    curr_hist = cv2.calcHist([curr_bgr], [0, 1, 2], None, [8, 8, 8],
                              [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()

    if prev_bgr is None:
        return False, curr_hist, last_cut_frame

    pixel_diff = float(cv2.absdiff(prev_bgr, curr_bgr).mean()) / 255.0
    hist_corr  = 0.0
    if prev_hist is not None:
        hist_corr = cv2.compareHist(
            prev_hist.astype(np.float32), curr_hist.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )

    is_cut = (pixel_diff > SPORTS_SCENE_CUT_THRESHOLD) or \
             (prev_hist is not None and hist_corr < 0.5)

    if is_cut and (frame_count - last_cut_frame) < SPORTS_SCENE_CUT_MIN_FRAMES:
        is_cut = False
    if is_cut:
        last_cut_frame = frame_count

    return is_cut, curr_hist, last_cut_frame


def is_scene_change(
    prev: Optional[np.ndarray],
    curr: np.ndarray,
    threshold: float = 0.35,
    prev_hist: Optional[np.ndarray] = None,
    frame_count: int = 0,
    last_cut_frame: int = -100,
    mode: str = "default",
) -> Tuple[bool, Optional[np.ndarray], int]:
    if mode == "sports":
        return is_sports_scene_change(prev, curr, prev_hist, frame_count, last_cut_frame)

    curr_bgr  = _ensure_bgr(curr)
    prev_bgr  = _ensure_bgr(prev)

    curr_hist = cv2.calcHist([curr_bgr], [0, 1, 2], None, [8, 8, 8],
                              [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()

    if prev_bgr is None:
        return False, curr_hist, last_cut_frame

    pixel_diff = float(cv2.absdiff(prev_bgr, curr_bgr).mean()) / 255.0
    is_cut     = pixel_diff > threshold
    if is_cut:
        last_cut_frame = frame_count

    return is_cut, curr_hist, last_cut_frame


# ─── Shot/play event detector ──────────────────────────────────────────────────

class SportsEventDetector:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps                  = fps
        self.recent_ball_heights: List[float] = []
        self.recent_player_heights: List[float] = []
        self.event_active         = False
        self.event_end_frame      = 0
        self._frame_count         = 0
        self._event_flags: Dict[int, bool] = {}

    def update(
        self,
        ball_box: Optional[Tuple[int, int, int, int]],
        primary_person: Optional[Tuple[int, int, int, int]],
        record_frame: Optional[int] = None,
    ) -> bool:
        self._frame_count += 1
        active = False

        if self._frame_count < self.event_end_frame:
            active = True
        elif ball_box is not None and primary_person is not None:
            bx1, by1, bx2, by2 = ball_box
            px1, py1, px2, py2 = primary_person

            ball_height_ratio = (py1 - by1) / max(py2 - py1, 1) if py2 > py1 else 0
            self.recent_ball_heights.append(ball_height_ratio)
            if len(self.recent_ball_heights) > int(self.fps * 0.5):
                self.recent_ball_heights.pop(0)

            if len(self.recent_ball_heights) >= 3:
                if (ball_height_ratio < -0.3 and
                        self.recent_ball_heights[-1] < self.recent_ball_heights[-2]):
                    self.event_end_frame = self._frame_count + SPORTS_EVENT_EXPAND_FRAMES
                    active = True

            if not active:
                ball_dx = abs((bx2 - bx1) - (px2 - px1))
                if ball_dx > (px2 - px1) * 0.5:
                    self.event_end_frame = (
                        self._frame_count + SPORTS_EVENT_EXPAND_FRAMES // 2
                    )
                    active = True

        self.event_active = active
        if record_frame is not None:
            self._event_flags[record_frame] = active
        return active

    def event_active_for(self, fi: int) -> bool:
        return self._event_flags.get(fi, False)


# ─── Subject / person detection ────────────────────────────────────────────────

DetectionResult = namedtuple(
    "DetectionResult", ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"]
)


def detect_subjects(
    frame: np.ndarray,
    model: Any,
    confidence: float = 0.45,
    prev_center: Optional[Tuple[int, int]] = None,
    prev_ball_carrier: Optional[int] = None,
    tracking_mode: str = "subject",
) -> Tuple[Optional[DetectionResult], Optional[Tuple[int, int, int, int]], int]:
    if model is None:
        return None, None, -1
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"det err: {e}", file=sys.stderr)
        return None, None, -1

    if results.boxes is None or len(results.boxes) == 0:
        return None, None, -1

    persons: List[Tuple] = []
    balls:   List[Tuple] = []

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cls == PERSON_CLASS_ID and conf >= confidence:
            persons.append((x1, y1, x2, y2, cx, cy, conf))
        elif cls == SPORTS_BALL_CLASS_ID and conf >= SPORTS_BALL_CONFIDENCE:
            balls.append((x1, y1, x2, y2, cx, cy, conf))

    if not persons:
        return None, None, -1

    ball_box     = None
    ball_carrier = -1

    if balls:
        best_ball = max(balls, key=lambda b: b[6])
        ball_box  = (best_ball[0], best_ball[1], best_ball[2], best_ball[3])
        min_dist  = float('inf')
        for i, p in enumerate(persons):
            dist = math.hypot(p[4] - best_ball[4], p[5] - best_ball[5])
            if dist < min_dist and dist < SPORTS_BALL_PROXIMITY_PX:
                min_dist, ball_carrier = dist, i

    if tracking_mode == "sports_action" and prev_center is not None and len(persons) > 1:
        pcx, pcy   = prev_center
        best_idx   = 0
        best_score = -1e9
        for i, p in enumerate(persons):
            score = -math.hypot(p[4] - pcx, p[5] - pcy)
            if i == ball_carrier:
                score += SPORTS_SWITCH_BALL_BONUS
            if i == prev_ball_carrier:
                score += SPORTS_SWITCH_BALL_BONUS * 0.5
            if score > best_score:
                best_score, best_idx = score, i
        primary = persons[best_idx]
    else:
        primary = persons[ball_carrier] if ball_carrier >= 0 else None
        if primary is None:
            tw = sum(e[6] for e in persons)
            if tw == 0:
                return None, None, -1
            cx = int(sum(e[6] * e[4] for e in persons) / tw)
            cy = int(sum(e[6] * e[5] for e in persons) / tw)
            return DetectionResult(
                cx, cy,
                min(e[0] for e in persons), min(e[1] for e in persons),
                max(e[2] for e in persons), max(e[3] for e in persons),
                len(persons),
            ), ball_box, ball_carrier

    x1, y1, x2, y2, cx, cy, _conf = primary
    cluster = [primary] + [
        p for p in persons if p is not primary and
        math.hypot(p[4] - cx, p[5] - cy) < (x2 - x1) * 1.5
    ]

    ux1 = min(p[0] for p in cluster)
    uy1 = min(p[1] for p in cluster)
    ux2 = max(p[2] for p in cluster)
    uy2 = max(p[3] for p in cluster)

    return (
        DetectionResult(int(cx), int(cy), ux1, uy1, ux2, uy2, len(persons)),
        ball_box,
        ball_carrier,
    )


def detect_persons_all(
    frame: np.ndarray, model: Any, confidence: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    if model is None:
        return []
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception:
        return []
    if results.boxes is None or len(results.boxes) == 0:
        return []
    persons = [
        tuple(map(int, box.xyxy[0].tolist()))
        for box in results.boxes
        if int(box.cls[0]) == PERSON_CLASS_ID
    ]
    return sorted(persons, key=lambda b: b[0])


# ─── Framing helpers ───────────────────────────────────────────────────────────

def _apply_lower_third_guard(cy: int, crop_h: int, subject_cy_src: int, orig_h: int) -> int:
    hh     = crop_h // 2
    max_cy = subject_cy_src - int((1.0 - LOWER_THIRD_GUARD) * crop_h) + hh
    return min(cy, min(max_cy, orig_h - hh))


def _soi_region_label(cx: int, cy: int, w: int, h: int) -> str:
    col = "left" if cx < w // 3 else ("right" if cx > 2 * w // 3 else "center")
    row = "upper" if cy < h // 3 else ("lower" if cy > 2 * h // 3 else "mid")
    if row == "mid" and col == "
