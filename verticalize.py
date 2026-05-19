"""
verticalize.py — AI Vertical Video Converter v6.0 (Sports Smoothing Revolution)
═══════════════════════════════════════════════════════════════════════════════
Fixed v6.0-final:
- Added missing process_video(), process_sports_video(), process_clips_batch()
- Fixed GameState.FAST_BREAK -> PlayPhase.FAST_BREAK in GameStateEngine
- Fixed talking_head_bias default in subject-tracking path
- Fixed clip_results dict to include "clip" key
- Fixed all undefined variable references (use_kalman, use_ball_tracking, etc.)
- Fixed FFmpegVideoReader to handle edge cases
- Unified analytics dict structure
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections import namedtuple, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum, auto

import cv2
import numpy as np

try:
    from scipy.signal import savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN_AVAILABLE = True
except ImportError:
    _HUNGARIAN_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: Custom exception
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Enums & Constants
# ═══════════════════════════════════════════════════════════════════════════════

class PlayPhase(Enum):
    FAST_BREAK = auto()
    HALF_COURT = auto()
    REBOUND = auto()
    STATIC = auto()
    TRANSITION = auto()

class GameState(Enum):
    LIVE_PLAY = auto()
    TIMEOUT = auto()
    REPLAY = auto()
    FREE_THROW = auto()
    UNKNOWN = auto()

class TrackingStatus(Enum):
    ACTIVE = auto()
    OCCLUDED = auto()
    LOST = auto()

PERSON_CLASS_ID      = 0
SPORTS_BALL_CLASS_ID = 32
HIGH_PRIO_CLASSES    = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB     = 2_000
MIN_FRAME_DIM        = 240
MAX_FRAMES_GUARD     = 1_080_000
LOWER_THIRD_GUARD    = 0.80

PANEL_MIN_PERSONS          = 2
PANEL_PROBE_COUNT          = 30
PANEL_MAJORITY_FRAC        = 0.60
PANEL_STABILITY_FRAC       = 0.75
PANEL_MAX_PERSON_MOTION    = 8.0
PANEL_MIN_PERSON_AREA_FRAC = 0.06
PANEL_MAX_COUNT_VARIANCE   = 1.5
PANEL_MIN_PERSON_ASPECT    = 1.3

SPORTS_COURT_COLORS_HSV = [
    {"h": [10, 30],  "s": [40, 180], "v": [80, 220]},
    {"h": [35, 85],  "s": [40, 255], "v": [40, 220]},
    {"h": [90, 130], "s": [0,  60],  "v": [150, 255]},
]

KALMAN_PLAYER_PROCESS_NOISE_BASE  = 3e-2
KALMAN_PLAYER_PROCESS_NOISE_HIGH  = 3e-1
KALMAN_PLAYER_MEASUREMENT_NOISE   = 3e-2

KALMAN_BALL_PROCESS_NOISE_BASE    = 1e-1
KALMAN_BALL_PROCESS_NOISE_HIGH    = 8e-1
KALMAN_BALL_MEASUREMENT_NOISE     = 8e-2

KALMAN_OPTICAL_FLOW_NOISE         = 5e-1
KALMAN_SALIENCY_NOISE             = 2e-0
KALMAN_INITIAL_ERROR              = 1.0
KALMAN_GATE_THRESHOLD             = 8.0

GRAVITY_PIXELS_PER_SEC2_BASE      = 500.0
FAST_BREAK_PREDICT_SEC            = 0.8
HALF_COURT_PREDICT_SEC            = 0.3
BALL_SPEED_THRESHOLD              = 15.0

BALL_AIRBORNE_THRESHOLD_PX        = 20.0
BALL_BOUNCE_VELOCITY_DAMPING      = 0.6
BALL_MAX_PREDICTION_FRAMES        = 10

SPORTS_SCENE_CUT_THRESHOLD        = 0.18
SPORTS_SCENE_CUT_MIN_FRAMES       = 2
SPORTS_SWITCH_BALL_BONUS          = 200
SPORTS_BALL_CONFIDENCE            = 0.25
SPORTS_BALL_PROXIMITY_PX          = 120
SPORTS_EVENT_EXPAND_FRAMES        = 15
SPORTS_EVENT_EXPAND_FACTOR        = 1.25

AVS_BASE_WINDOW_SEC               = 0.40
AVS_MAX_WINDOW_SEC                = 0.80
AVS_MIN_WINDOW_SEC                = 0.15
AVS_POLYORDER                     = 3

AVS_VELOCITY_FAST_THRESHOLD       = 80.0
AVS_VELOCITY_SLOW_THRESHOLD       = 5.0
AVS_ACCEL_SPIKE_THRESHOLD         = 200.0
AVS_CONFIDENCE_LOW_THRESHOLD      = 0.3

SPORTS_POST_SMOOTH_WINDOW_SEC     = 0.50
SPORTS_POST_SMOOTH_EMA_ALPHA      = 0.04

ICS_LOOKAHEAD_SEC                 = 0.30
ICS_FAST_BREAK_MARGIN_FACTOR      = 1.35
ICS_SET_PLAY_MARGIN_FACTOR        = 1.05
ICS_BOUNDARY_ELASTICITY_PX        = 50
ICS_COURT_PRESERVE_RATIO          = 0.25

MOT_MAX_OCCLUSION_FRAMES          = 30
MOT_IOU_MATCH_THRESHOLD           = 0.3
MOT_APPEARANCE_WEIGHT             = 0.3
MOT_MAX_TRACKS                    = 15
MOT_MIN_TRACK_AGE                 = 3

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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

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


@dataclass
class Track:
    id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    status: TrackingStatus = TrackingStatus.ACTIVE
    appearance: Optional[np.ndarray] = None
    class_id: int = PERSON_CLASS_ID
    confidence: float = 0.0
    kalman_state: np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
    kalman_covariance: np.ndarray = field(default_factory=lambda: np.eye(6))


@dataclass
class BallState:
    bbox: Optional[Tuple[int, int, int, int]]
    center: Optional[Tuple[float, float]]
    velocity: Tuple[float, float]
    is_airborne: bool = False
    is_possessed: bool = False
    possessor_track_id: Optional[int] = None
    bounce_count: int = 0
    airborne_frames: int = 0


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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Feature Availability Guards
# ═══════════════════════════════════════════════════════════════════════════════

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
        return (
            os.path.exists("yolov8n.pt") or
            os.path.exists("yolov8s.pt") or
            os.path.exists("yolo11n.pt")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Visual Effects
# ═══════════════════════════════════════════════════════════════════════════════

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


def apply_sharpen(frame: np.ndarray, strength: float = 0.6, radius: int = 1) -> np.ndarray:
    if strength <= 0:
        return frame
    ksize   = radius * 2 + 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)


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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: FFmpeg Utilities
# ═══════════════════════════════════════════════════════════════════════════════

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
            self._build_cmd([]),
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
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            except Exception:
                pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
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
                chunk = self._proc.stdout.read(needed)
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
    try:
        r = FFmpegVideoReader(path, width, height, seek_sec=t_sec, n_frames=1,
                              scale_w=scale_w, scale_h=scale_h)
        r._open()
        frames = list(r)
        r.close()
        return frames[0] if frames else None
    except Exception:
        return None


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
    try:
        proc.wait(timeout=120)
    except Exception:
        proc.kill()
        proc.wait()
    if proc.returncode != 0:
        try:
            err = proc.stderr.read(2000).decode(errors="replace")
        except Exception:
            err = ""
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")


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
        "height":           h,
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: Model Cache & Face Detection
# ═══════════════════════════════════════════════════════════════════════════════

_model_cache: Dict[str, Any] = {}
_yunet_detector: Optional[Any] = None


def _get_model(weights: str = "yolov8n.pt") -> Optional[Any]:
    if not _YOLO_AVAILABLE:
        return None
    if weights not in _model_cache:
        for w in [weights, "yolo11n.pt", "yolov8n.pt", "yolov8s.pt"]:
            try:
                m = _YOLO(w)
                _model_cache[weights] = m
                print(f"[Model] Loaded {w}", file=sys.stderr)
                return m
            except Exception:
                continue
        print("YOLO unavailable", file=sys.stderr)
        return None
    return _model_cache[weights]


def _get_yunet() -> Optional[Any]:
    global _yunet_detector
    if _yunet_detector is not None:
        return _yunet_detector

    model_paths = [
        "face_detection_yunet_2023mar.onnx",
        "yunet.onnx",
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                net = cv2.dnn.readNet(p)
                _yunet_detector = net
                print(f"[Face] Loaded YuNet from {p}", file=sys.stderr)
                return net
            except Exception:
                pass

    print("[Face] YuNet not found, using Haar cascade", file=sys.stderr)
    return None


def detect_faces(
    frame: np.ndarray, confidence_thresh: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]

    yunet = _get_yunet()
    if yunet is not None:
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), [0, 0, 0], True, False)
            yunet.setInput(blob)
            detections = yunet.forward()
            faces = []
            if detections is not None and detections.ndim >= 3:
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_thresh:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2, y2))
            if faces:
                faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
                return faces
        except Exception:
            pass

    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(haar_path):
        cascade = cv2.CascadeClassifier(haar_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = cascade.detectMultiScale(gray, 1.1, 5, minSize=(max(30, w//20), max(30, h//20)))
        if len(raw) > 0:
            faces = [(x, y, x+bw, y+bh) for x, y, bw, bh in raw]
            faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
            return faces
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: Multi-Object Sports Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObjectSportsTracker:
    def __init__(self, fps: float, frame_w: int, frame_h: int) -> None:
        self.fps = fps
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.ball_state = BallState(bbox=None, center=None, velocity=(0.0, 0.0))
        self.gravity_px = GRAVITY_PIXELS_PER_SEC2_BASE * (frame_h / 1080.0)
        self.track_history: Dict[int, deque] = {}
        self.appearance_gallery: Dict[int, np.ndarray] = {}

    def _compute_iou(self, box_a: Tuple, box_b: Tuple) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _compute_appearance_sim(self, track_id: int, det_frame: np.ndarray,
                                 det_box: Tuple[int, int, int, int]) -> float:
        if track_id not in self.appearance_gallery:
            return 0.5
        x1, y1, x2, y2 = det_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(det_frame.shape[1], x2), min(det_frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = det_frame[y1:y2, x1:x2]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        gallery_hist = self.appearance_gallery[track_id]
        return float(cv2.compareHist(gallery_hist.astype(np.float32), hist.astype(np.float32),
                               cv2.HISTCMP_CORREL))

    def _update_appearance(self, track_id: int, det_frame: np.ndarray,
                           det_box: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = det_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(det_frame.shape[1], x2), min(det_frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = det_frame[y1:y2, x1:x2]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if track_id in self.appearance_gallery:
            self.appearance_gallery[track_id] = (
                0.7 * self.appearance_gallery[track_id] + 0.3 * hist
            )
        else:
            self.appearance_gallery[track_id] = hist

    def _hungarian_match(self, tracks: List[Track], detections: List[Tuple],
                         det_frame: np.ndarray) -> Tuple[Dict[int, int], Set[int], Set[int]]:
        if not tracks or not detections:
            return {}, set(range(len(tracks))), set(range(len(detections)))

        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=float)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det[:4])
                pred_cx = track.center[0] + track.velocity[0]
                pred_cy = track.center[1] + track.velocity[1]
                det_cx = (det[0] + det[2]) / 2
                det_cy = (det[1] + det[3]) / 2
                dist = math.hypot(pred_cx - det_cx, pred_cy - det_cy)
                dist_cost = min(dist / 100.0, 1.0)
                app_sim = self._compute_appearance_sim(track.id, det_frame, det[:4])
                cost = (1 - iou) * 0.4 + dist_cost * 0.3 + (1 - app_sim) * 0.3
                cost_matrix[i, j] = cost

        if _HUNGARIAN_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind_list, col_ind_list = [], []
            used_cols: Set[int] = set()
            for i in range(n_tracks):
                best_j, best_cost = -1, float('inf')
                for j in range(n_dets):
                    if j in used_cols:
                        continue
                    if cost_matrix[i, j] < best_cost:
                        best_cost, best_j = cost_matrix[i, j], j
                if best_j >= 0 and best_cost < 0.7:
                    row_ind_list.append(i)
                    col_ind_list.append(best_j)
                    used_cols.add(best_j)
            row_ind = np.array(row_ind_list)
            col_ind = np.array(col_ind_list)

        matched: Dict[int, int] = {}
        unmatched_tracks = set(range(n_tracks))
        unmatched_dets = set(range(n_dets))

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.7:
                matched[i] = j
                unmatched_tracks.discard(i)
                unmatched_dets.discard(j)

        return matched, unmatched_tracks, unmatched_dets

    def update(self, persons: List[Tuple[int, int, int, int]],
               ball_box: Optional[Tuple[int, int, int, int]],
               det_frame: np.ndarray,
               confidences: Optional[List[float]] = None) -> None:
        for track in self.tracks.values():
            track.time_since_update += 1
            track.center = (
                track.center[0] + track.velocity[0],
                track.center[1] + track.velocity[1]
            )
            track.bbox = (
                int(track.bbox[0] + track.velocity[0]),
                int(track.bbox[1] + track.velocity[1]),
                int(track.bbox[2] + track.velocity[0]),
                int(track.bbox[3] + track.velocity[1]),
            )

        active_tracks = [
            t for t in self.tracks.values()
            if t.status == TrackingStatus.ACTIVE or
            (t.status == TrackingStatus.OCCLUDED and
             t.time_since_update < MOT_MAX_OCCLUSION_FRAMES)
        ]

        matched, unmatched_tracks, unmatched_dets = self._hungarian_match(
            active_tracks, persons, det_frame
        )

        for track_idx, det_idx in matched.items():
            track = active_tracks[track_idx]
            det = persons[det_idx]
            new_cx = (det[0] + det[2]) / 2
            new_cy = (det[1] + det[3]) / 2
            track.velocity = (new_cx - track.center[0], new_cy - track.center[1])
            track.center = (new_cx, new_cy)
            track.bbox = det
            track.hits += 1
            track.time_since_update = 0
            track.status = TrackingStatus.ACTIVE
            track.confidence = confidences[det_idx] if confidences else 0.5
            self._update_appearance(track.id, det_frame, det)
            if track.id not in self.track_history:
                self.track_history[track.id] = deque(maxlen=30)
            self.track_history[track.id].append((new_cx, new_cy))

        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            if track.time_since_update >= MOT_MAX_OCCLUSION_FRAMES:
                track.status = TrackingStatus.LOST
            else:
                track.status = TrackingStatus.OCCLUDED

        for det_idx in unmatched_dets:
            det = persons[det_idx]
            new_track = Track(
                id=self.next_track_id,
                bbox=det,
                center=((det[0]+det[2])/2, (det[1]+det[3])/2),
                velocity=(0.0, 0.0),
                age=0,
                hits=1,
                confidence=confidences[det_idx] if confidences else 0.5,
            )
            self.tracks[self.next_track_id] = new_track
            self._update_appearance(self.next_track_id, det_frame, det)
            self.next_track_id += 1

        self._update_ball(ball_box)
        self.tracks = {k: v for k, v in self.tracks.items()
                      if v.status != TrackingStatus.LOST}

    def _update_ball(self, ball_box: Optional[Tuple[int, int, int, int]]) -> None:
        if ball_box is not None:
            bx = (ball_box[0] + ball_box[2]) / 2
            by = (ball_box[1] + ball_box[3]) / 2
            if self.ball_state.center is not None:
                vx = bx - self.ball_state.center[0]
                vy = by - self.ball_state.center[1]
                self.ball_state.velocity = (vx, vy)
                if vy < -BALL_AIRBORNE_THRESHOLD_PX:
                    self.ball_state.is_airborne = True
                    self.ball_state.airborne_frames += 1
                elif abs(vy) < BALL_AIRBORNE_THRESHOLD_PX and self.ball_state.is_airborne:
                    self.ball_state.is_airborne = False
                    self.ball_state.bounce_count += 1
                    self.ball_state.velocity = (
                        self.ball_state.velocity[0],
                        self.ball_state.velocity[1] * BALL_BOUNCE_VELOCITY_DAMPING
                    )
                    self.ball_state.airborne_frames = 0
            self.ball_state.center = (bx, by)
            self.ball_state.bbox = ball_box
        else:
            if self.ball_state.center is not None and self.ball_state.is_airborne:
                dt = 1.0 / self.fps
                new_vy = self.ball_state.velocity[1] + self.gravity_px * dt
                new_cx = self.ball_state.center[0] + self.ball_state.velocity[0]
                new_cy = self.ball_state.center[1] + new_vy * dt
                self.ball_state.center = (new_cx, new_cy)
                self.ball_state.velocity = (self.ball_state.velocity[0], new_vy)

    def get_primary_track(self, prev_ball_carrier: Optional[int] = None) -> Optional[Track]:
        if not self.tracks:
            return None
        if self.ball_state.center is not None:
            min_dist = float('inf')
            closest_track = None
            for track in self.tracks.values():
                if track.status != TrackingStatus.ACTIVE:
                    continue
                dist = math.hypot(
                    track.center[0] - self.ball_state.center[0],
                    track.center[1] - self.ball_state.center[1]
                )
                if dist < SPORTS_BALL_PROXIMITY_PX and dist < min_dist:
                    min_dist = dist
                    closest_track = track
            if closest_track is not None:
                self.ball_state.is_possessed = True
                self.ball_state.possessor_track_id = closest_track.id
                return closest_track

        active_tracks = [t for t in self.tracks.values() if t.status == TrackingStatus.ACTIVE]
        if not active_tracks:
            return None
        frame_cx = self.frame_w / 2
        frame_cy = self.frame_h / 2
        best_track = None
        best_score = -1e9
        for track in active_tracks:
            dist_to_center = math.hypot(track.center[0] - frame_cx, track.center[1] - frame_cy)
            score = -dist_to_center * 0.3 + track.hits * 10 + track.confidence * 100
            if prev_ball_carrier == track.id:
                score += 200
            if score > best_score:
                best_score = score
                best_track = track
        return best_track

    def predict_ball_trajectory(self, n_frames: int = 10) -> List[Tuple[float, float]]:
        if self.ball_state.center is None or not self.ball_state.is_airborne:
            return []
        trajectory = []
        cx, cy = self.ball_state.center
        vx, vy = self.ball_state.velocity
        dt = 1.0 / self.fps
        for _ in range(min(n_frames, BALL_MAX_PREDICTION_FRAMES)):
            vy += self.gravity_px * dt
            cx += vx
            cy += vy * dt
            trajectory.append((cx, cy))
        return trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: Adaptive Velocity-Aware Smoother (AVS)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveVelocityAwareSmoother:
    def __init__(self, fps: float, base_window_sec: float = AVS_BASE_WINDOW_SEC) -> None:
        self.fps = fps
        self.base_window = base_window_sec
        max_buf = int(fps * AVS_MAX_WINDOW_SEC)
        self.buffer_cx: deque = deque(maxlen=max_buf)
        self.buffer_cy: deque = deque(maxlen=max_buf)
        self.buffer_conf: deque = deque(maxlen=max_buf)
        self.buffer_phase: deque = deque(maxlen=max_buf)
        self.prev_smooth_cx: Optional[float] = None
        self.prev_smooth_cy: Optional[float] = None
        self.prev_velocity: Tuple[float, float] = (0.0, 0.0)
        self.frame_count = 0

    def _compute_adaptive_window(self, velocity: float, accel: float,
                                  confidence: float, phase: PlayPhase) -> int:
        window = int(self.fps * self.base_window)
        if velocity > AVS_VELOCITY_FAST_THRESHOLD:
            window = int(window * 0.4)
        elif velocity < AVS_VELOCITY_SLOW_THRESHOLD:
            window = int(window * 1.8)
        if abs(accel) > AVS_ACCEL_SPIKE_THRESHOLD:
            window = max(3, int(window * 0.3))
        if confidence < AVS_CONFIDENCE_LOW_THRESHOLD:
            window = int(window * 1.4)
        if phase == PlayPhase.FAST_BREAK:
            window = max(3, int(window * 0.5))
        elif phase == PlayPhase.STATIC:
            window = int(window * 1.5)
        window = max(5, min(window, int(self.fps * AVS_MAX_WINDOW_SEC)))
        if window % 2 == 0:
            window += 1
        return window

    def smooth(self, cx: float, cy: float, confidence: float = 1.0,
               phase: PlayPhase = PlayPhase.HALF_COURT) -> Tuple[float, float]:
        self.frame_count += 1
        self.buffer_cx.append(cx)
        self.buffer_cy.append(cy)
        self.buffer_conf.append(confidence)
        self.buffer_phase.append(phase)

        n = len(self.buffer_cx)
        if n < 5:
            if self.prev_smooth_cx is None:
                self.prev_smooth_cx = cx
                self.prev_smooth_cy = cy
                return cx, cy
            alpha = 0.3
            sx = alpha * cx + (1 - alpha) * self.prev_smooth_cx
            sy = alpha * cy + (1 - alpha) * self.prev_smooth_cy
            self.prev_smooth_cx = sx
            self.prev_smooth_cy = sy
            return sx, sy

        if n >= 3:
            v_curr_x = self.buffer_cx[-1] - self.buffer_cx[-2]
            v_curr_y = self.buffer_cy[-1] - self.buffer_cy[-2]
            v_prev_x = self.buffer_cx[-2] - self.buffer_cx[-3]
            v_prev_y = self.buffer_cy[-2] - self.buffer_cy[-3]
            velocity = math.hypot(v_curr_x, v_curr_y)
            accel = abs(math.hypot(v_curr_x, v_curr_y) - math.hypot(v_prev_x, v_prev_y))
        else:
            velocity = 0.0
            accel = 0.0

        window = self._compute_adaptive_window(velocity, accel, confidence, phase)
        window = min(window, n)
        if window < 5:
            window = 5 if n >= 5 else n
        if window % 2 == 0:
            window -= 1

        try:
            polyorder = min(AVS_POLYORDER, window - 1)
            if polyorder < 2:
                polyorder = 2
            arr_cx = np.array(list(self.buffer_cx)[-window:])
            arr_cy = np.array(list(self.buffer_cy)[-window:])

            if _SCIPY_AVAILABLE and window >= 5:
                smooth_cx = float(savgol_filter(arr_cx, window, polyorder)[-1])
                smooth_cy = float(savgol_filter(arr_cy, window, polyorder)[-1])
            else:
                weights = np.exp(-0.5 * (np.arange(window) - window // 2)**2 / ((window / 4)**2 + 1e-9))
                weights /= weights.sum()
                smooth_cx = float(np.sum(arr_cx * weights))
                smooth_cy = float(np.sum(arr_cy * weights))

        except Exception:
            alpha = 0.15
            smooth_cx = alpha * cx + (1 - alpha) * (self.prev_smooth_cx or cx)
            smooth_cy = alpha * cy + (1 - alpha) * (self.prev_smooth_cy or cy)

        if self.prev_smooth_cx is not None:
            v_smooth_x = smooth_cx - self.prev_smooth_cx
            v_smooth_y = smooth_cy - self.prev_smooth_cy
            v_prev_mag = math.hypot(*self.prev_velocity)
            v_curr_mag = math.hypot(v_smooth_x, v_smooth_y)
            if v_curr_mag > v_prev_mag * 2.5 and v_prev_mag > 5.0:
                damp = 0.5
                smooth_cx = self.prev_smooth_cx + v_smooth_x * damp
                smooth_cy = self.prev_smooth_cy + v_smooth_y * damp

        self.prev_smooth_cx = smooth_cx
        self.prev_smooth_cy = smooth_cy
        self.prev_velocity = (smooth_cx - cx, smooth_cy - cy)
        return float(smooth_cx), float(smooth_cy)

    def get_metrics(self) -> Dict[str, float]:
        if len(self.buffer_cx) < 2:
            return {"jitter_raw": 0.0, "jitter_smooth": 0.0, "smoothness_pct": 0.0}
        arr = np.array(self.buffer_cx)
        raw_diff = np.diff(arr)
        raw_jitter = float(np.mean(np.abs(raw_diff)))
        if len(self.buffer_cx) > 5 and _SCIPY_AVAILABLE:
            w = min(7, len(arr))
            if w % 2 == 0:
                w -= 1
            smooth_arr = savgol_filter(arr, max(w, 5), min(3, w - 1))
            smooth_diff = np.diff(smooth_arr)
            smooth_jitter = float(np.mean(np.abs(smooth_diff)))
        else:
            smooth_jitter = raw_jitter * 0.5
        pct = (raw_jitter - smooth_jitter) / raw_jitter * 100 if raw_jitter > 0 else 0.0
        return {
            "jitter_raw": round(raw_jitter, 2),
            "jitter_smooth": round(smooth_jitter, 2),
            "smoothness_pct": round(pct, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: Intelligent Crop Strategy (ICS)
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentCropStrategy:
    def __init__(self, orig_w: int, orig_h: int, crop_w: int, crop_h: int, fps: float) -> None:
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.fps = fps
        self.hw = crop_w // 2
        self.hh = crop_h // 2
        self.center_history: deque = deque(maxlen=int(fps * ICS_LOOKAHEAD_SEC * 2))
        self.velocity_history: deque = deque(maxlen=int(fps * 0.5))

    def compute_crop(self, cx: float, cy: float,
                     phase: PlayPhase = PlayPhase.HALF_COURT,
                     ball_pos: Optional[Tuple[float, float]] = None) -> Tuple[int, int, int, int]:
        self.center_history.append((cx, cy))

        if len(self.center_history) >= 3:
            recent = list(self.center_history)[-5:]
            vx = (recent[-1][0] - recent[0][0]) / len(recent)
            vy = (recent[-1][1] - recent[0][1]) / len(recent)
            self.velocity_history.append((vx, vy))
            lookahead_frames = int(self.fps * ICS_LOOKAHEAD_SEC)
            pred_cx = cx + vx * lookahead_frames
            pred_cy = cy + vy * lookahead_frames
        else:
            pred_cx, pred_cy = cx, cy

        if phase == PlayPhase.FAST_BREAK:
            margin_factor = ICS_FAST_BREAK_MARGIN_FACTOR
        elif phase == PlayPhase.STATIC:
            margin_factor = ICS_SET_PLAY_MARGIN_FACTOR
        else:
            margin_factor = 1.15

        eff_crop_w = int(self.crop_w * margin_factor)
        eff_crop_h = int(self.crop_h * margin_factor)
        eff_hw = eff_crop_w // 2
        eff_hh = eff_crop_h // 2

        # Clamp effective crop to source dimensions
        eff_crop_w = min(eff_crop_w, self.orig_w)
        eff_crop_h = min(eff_crop_h, self.orig_h)
        eff_hw = eff_crop_w // 2
        eff_hh = eff_crop_h // 2

        left = int(np.clip(pred_cx - eff_hw, 0, max(0, self.orig_w - eff_crop_w)))
        top  = int(np.clip(pred_cy - eff_hh, 0, max(0, self.orig_h - eff_crop_h)))

        edge_zone = ICS_BOUNDARY_ELASTICITY_PX
        if left < edge_zone:
            left = int(left * 0.3)
        if top < edge_zone:
            top = int(top * 0.3)
        if left + eff_crop_w > self.orig_w - edge_zone:
            overshoot = left + eff_crop_w - (self.orig_w - edge_zone)
            left = max(0, left - int(overshoot * 0.3))
        if top + eff_crop_h > self.orig_h - edge_zone:
            overshoot = top + eff_crop_h - (self.orig_h - edge_zone)
            top = max(0, top - int(overshoot * 0.3))

        left = max(0, min(left, max(0, self.orig_w - self.crop_w)))
        top  = max(0, min(top,  max(0, self.orig_h - self.crop_h)))
        right  = left + self.crop_w
        bottom = top  + self.crop_h

        if ball_pos is not None:
            bx, by = ball_pos
            margin = self.crop_w * 0.15
            if bx < left + margin:
                shift = int(left + margin - bx)
                left  = max(0, left - shift)
                right = left + self.crop_w
            elif bx > right - margin:
                shift = int(bx - (right - margin))
                left  = min(max(0, self.orig_w - self.crop_w), left + shift)
                right = left + self.crop_w
            if by < top + margin:
                shift  = int(top + margin - by)
                top    = max(0, top - shift)
                bottom = top + self.crop_h
            elif by > bottom - margin:
                shift  = int(by - (bottom - margin))
                top    = min(max(0, self.orig_h - self.crop_h), top + shift)
                bottom = top + self.crop_h

        return left, top, right, bottom


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16: Game State Engine
# ═══════════════════════════════════════════════════════════════════════════════

class GameStateEngine:
    def __init__(self, fps: float, frame_w: int, frame_h: int) -> None:
        self.fps = fps
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.current_state = GameState.UNKNOWN
        self.prev_gray: Optional[np.ndarray] = None
        self.freeze_frame_count = 0
        self.motion_history: deque = deque(maxlen=int(fps * 2))
        self.formation_history: deque = deque(maxlen=int(fps * 1))

    def update(self, persons: List[Tuple[int, int, int, int]],
               gray_frame: np.ndarray) -> GameState:
        if self.prev_gray is not None:
            diff = float(cv2.absdiff(self.prev_gray, gray_frame).mean())
            self.motion_history.append(diff)
            if diff < 1.0:
                self.freeze_frame_count += 1
            else:
                self.freeze_frame_count = max(0, self.freeze_frame_count - 2)

        self.prev_gray = gray_frame.copy()

        if self.freeze_frame_count > self.fps:
            self.current_state = GameState.TIMEOUT
            return self.current_state

        if len(self.motion_history) >= int(self.fps * 1.5):
            recent = list(self.motion_history)[-int(self.fps * 1.5):]
            if len(recent) > 20:
                arr = np.array(recent, dtype=float)
                autocorr = np.correlate(arr - arr.mean(), arr - arr.mean(), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 10:
                    peaks = [i for i in range(5, min(len(autocorr), int(self.fps)))
                            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]]
                    if len(peaks) >= 2:
                        self.current_state = GameState.REPLAY
                        return self.current_state

        if len(persons) >= 2:
            centers_y = [(p[1] + p[3]) / 2 for p in persons]
            centers_x = [(p[0] + p[2]) / 2 for p in persons]
            near_line = sum(1 for y in centers_y if y < self.frame_h * 0.4)
            spread = float(np.std(centers_x)) / self.frame_w if len(centers_x) > 1 else 0.0
            if near_line >= 1 and spread > 0.2 and len(persons) <= 5:
                self.current_state = GameState.FREE_THROW
                return self.current_state

        self.current_state = GameState.LIVE_PLAY
        return self.current_state

    def get_zoom_factor(self) -> float:
        # FIX: was incorrectly referencing GameState.FAST_BREAK which doesn't exist
        # Fast break detection belongs to PlayPhase, not GameState
        if self.current_state == GameState.FREE_THROW:
            return 1.1
        elif self.current_state in (GameState.TIMEOUT, GameState.REPLAY):
            return 1.0
        return 1.15


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17: Sports Play Phase Detector
# ═══════════════════════════════════════════════════════════════════════════════

class SportsPlayPhaseDetector:
    def __init__(self, fps: float):
        self.fps = fps
        self.prev_ball_pos: Optional[Tuple[float, float]] = None
        self.ball_vel_history: deque = deque(maxlen=int(fps * 1.0))
        self.player_spread_history: deque = deque(maxlen=int(fps * 0.5))
        self.phase_history: deque = deque(maxlen=5)
        self.transition_counter = 0

    def detect_phase(self, persons: List[Tuple[int, int, int, int]],
                     ball_box: Optional[Tuple[int, int, int, int]],
                     frame_w: int,
                     ball_state: Optional[BallState] = None) -> PlayPhase:
        if not persons:
            return PlayPhase.STATIC

        centers_x = [(p[0] + p[2]) / 2 for p in persons]
        spread = float(np.std(centers_x)) if len(centers_x) > 1 else 0.0
        norm_spread = spread / (frame_w / 2)
        self.player_spread_history.append(norm_spread)

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
        else:
            self.ball_vel_history.append(0.0)

        avg_ball_speed = float(np.mean(self.ball_vel_history)) if self.ball_vel_history else 0.0
        avg_spread = float(np.mean(self.player_spread_history)) if self.player_spread_history else 0.0

        if len(self.phase_history) >= 3:
            recent_phases = list(self.phase_history)[-3:]
            if len(set(recent_phases)) > 1:
                self.transition_counter += 1
                if self.transition_counter > int(self.fps * 0.3):
                    self.phase_history.clear()
                    self.transition_counter = 0
                    return PlayPhase.TRANSITION
            else:
                self.transition_counter = max(0, self.transition_counter - 1)

        if avg_ball_speed > BALL_SPEED_THRESHOLD * 1.5 and avg_spread > 0.2:
            phase = PlayPhase.FAST_BREAK
        elif avg_ball_speed > BALL_SPEED_THRESHOLD and avg_spread > 0.15:
            phase = PlayPhase.FAST_BREAK
        elif avg_spread < 0.06:
            phase = PlayPhase.REBOUND
        else:
            phase = PlayPhase.HALF_COURT

        self.phase_history.append(phase)
        return phase


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18: Legacy SportsKalmanTracker
# ═══════════════════════════════════════════════════════════════════════════════

class SportsKalmanTracker:
    def __init__(self, dt: float = 1.0, fps: float = 30.0) -> None:
        self.dt = dt
        self.fps = fps
        self.F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,          0.5*dt**2],
            [0, 0, 1,  0,  dt,         0],
            [0, 0, 0,  1,  0,          dt],
            [0, 0, 0,  0,  1,          0],
            [0, 0, 0,  0,  0,          1],
        ], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        self.Q_base = np.eye(6, dtype=np.float64) * KALMAN_PLAYER_PROCESS_NOISE_BASE
        self.Q_base[4, 4] *= 4.0
        self.Q_base[5, 5] *= 4.0
        self.R_yolo     = np.eye(2, dtype=np.float64) * KALMAN_PLAYER_MEASUREMENT_NOISE
        self.R_optical  = np.eye(2, dtype=np.float64) * KALMAN_OPTICAL_FLOW_NOISE
        self.R_saliency = np.eye(2, dtype=np.float64) * KALMAN_SALIENCY_NOISE
        self.P = np.eye(6, dtype=np.float64) * KALMAN_INITIAL_ERROR
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.initialized  = False
        self._stale_count = 0
        self._last_sensor = "none"
        self._prev_accel_mag = 0.0

    def init(self, cx: float, cy: float) -> None:
        self.x = np.array([[float(cx)], [float(cy)], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * KALMAN_INITIAL_ERROR
        self.initialized  = True
        self._stale_count = 0
        self._last_sensor = "init"
        self._prev_accel_mag = 0.0

    def predict(self, steps: int = 1) -> Tuple[float, float]:
        if not self.initialized:
            return float(self.x[0, 0]), float(self.x[1, 0])
        if steps == 0:
            return float(self.x[0, 0]), float(self.x[1, 0])
        dt_s = self.dt * steps
        px = float(self.x[0, 0]) + float(self.x[2, 0]) * dt_s + 0.5 * float(self.x[4, 0]) * dt_s**2
        py = float(self.x[1, 0]) + float(self.x[3, 0]) * dt_s + 0.5 * float(self.x[5, 0]) * dt_s**2
        return px, py

    def predict_adaptive(self, play_phase: str, ball_is_airborne: bool = False,
                         ball_vel: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        if play_phase == "fast_break":
            steps = int(self.fps * FAST_BREAK_PREDICT_SEC)
        elif play_phase == "rebound":
            steps = int(self.fps * 0.1)
        else:
            steps = int(self.fps * HALF_COURT_PREDICT_SEC)
        return self.predict(steps=max(1, steps))

    def _predict_step(self) -> None:
        if not self.initialized:
            return
        current_accel_mag = math.sqrt(float(self.x[4, 0])**2 + float(self.x[5, 0])**2)
        if current_accel_mag > 100.0:
            Q_used = self.Q_base * 3
        elif current_accel_mag > 50.0:
            Q_used = self.Q_base * 2
        else:
            Q_used = self.Q_base
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q_used
        self._stale_count += 1
        max_vel = 200.0
        if abs(float(self.x[2, 0])) > max_vel:
            self.x[2, 0] = float(np.sign(self.x[2, 0])) * max_vel
        if abs(float(self.x[3, 0])) > max_vel:
            self.x[3, 0] = float(np.sign(self.x[3, 0])) * max_vel

    def update(self, cx: float, cy: float, sensor: str = "yolo") -> Tuple[float, float]:
        if not self.initialized:
            self.init(cx, cy)
            self._last_sensor = sensor
            return cx, cy
        if sensor == "optical_flow":
            R = self.R_optical
        elif sensor == "saliency":
            R = self.R_saliency
        else:
            R = self.R_yolo
        z = np.array([[cx], [cy]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        inv_S = np.linalg.inv(S)
        mahalanobis_dist = float(np.sqrt((y.T @ inv_S @ y).item()))
        if mahalanobis_dist > KALMAN_GATE_THRESHOLD:
            self._stale_count += 1
            return float(self.x[0, 0]), float(self.x[1, 0])
        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ self.H) @ self.P
        self._stale_count = 0
        self._last_sensor = sensor
        return float(self.x[0, 0]), float(self.x[1, 0])

    def increment_stale(self) -> None:
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
        return math.sqrt(float(vx) * float(vx) + float(vy) * float(vy))

    @property
    def last_sensor(self) -> str:
        return self._last_sensor


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19: Court/field detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_field_of_play(frame: np.ndarray,
                         sport_hint: str = "auto") -> Optional[np.ndarray]:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    def _make_mask(color_range: Dict) -> np.ndarray:
        lower = np.array([color_range["h"][0], color_range["s"][0], color_range["v"][0]])
        upper = np.array([color_range["h"][1], color_range["s"][1], color_range["v"][1]])
        m     = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
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


def get_court_center_of_mass(field_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    if field_mask is None:
        return None
    moments = cv2.moments(field_mask)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 20: Optical flow & Saliency
# ═══════════════════════════════════════════════════════════════════════════════

def sports_optical_flow_center(
    prev: np.ndarray, curr: np.ndarray, w: int, h: int,
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


def temporal_saliency_center(
    frame: np.ndarray, prev_saliency: Optional[np.ndarray] = None,
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
        if prev_saliency.shape == sal.shape:
            temporal_diff = np.abs(sal - prev_saliency * decay)
            sal = sal * (1.0 + temporal_diff * 2.0)
    b = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w - b:] = sal[:b, :] = sal[h - b:, :] = 0
    t = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2, sal
    ys, xs = np.mgrid[0:h, 0:w]
    cx = int((xs * sal).sum() / t)
    cy = int((ys * sal).sum() / t)
    return cx, cy, sal


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 21: Scene change detection
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def is_sports_scene_change(
    prev: Optional[np.ndarray], curr: np.ndarray,
    prev_hist: Optional[np.ndarray] = None,
    frame_count: int = 0, last_cut_frame: int = -100,
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
    prev: Optional[np.ndarray], curr: np.ndarray,
    threshold: float = 0.35, prev_hist: Optional[np.ndarray] = None,
    frame_count: int = 0, last_cut_frame: int = -100, mode: str = "default",
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 22: Sports event detector
# ═══════════════════════════════════════════════════════════════════════════════

class SportsEventDetector:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps                  = fps
        self.recent_ball_heights: List[float] = []
        self.recent_player_heights: List[float] = []
        self.event_active         = False
        self.event_end_frame      = 0
        self._frame_count         = 0
        self._event_flags: Dict[int, bool] = {}

    def update(self, ball_box: Optional[Tuple[int, int, int, int]],
               primary_person: Optional[Tuple[int, int, int, int]],
               record_frame: Optional[int] = None) -> bool:
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 23: Subject detection
# ═══════════════════════════════════════════════════════════════════════════════

DetectionResult = namedtuple(
    "DetectionResult", ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"]
)


def detect_subjects(
    frame: np.ndarray, model: Any, confidence: float = 0.45,
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

    if tracking_mode == "sports_action" and len(persons) > 0:
        best_idx   = 0
        best_score = -1e9
        frame_cx = sum(p[4] for p in persons) / len(persons)
        frame_cy = sum(p[5] for p in persons) / len(persons)
        for i, p in enumerate(persons):
            if prev_center is not None:
                pcx, pcy = prev_center
                score = -math.hypot(p[4] - pcx, p[5] - pcy) * 0.5
            else:
                score = -math.hypot(p[4] - frame_cx, p[5] - frame_cy) * 0.3
            if i == ball_carrier and ball_carrier >= 0:
                score += SPORTS_SWITCH_BALL_BONUS
            if i == prev_ball_carrier and prev_ball_carrier is not None and prev_ball_carrier >= 0:
                score += SPORTS_SWITCH_BALL_BONUS * 0.5
            area = (p[2] - p[0]) * (p[3] - p[1])
            score += area * 0.001
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 24: Framing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_lower_third_guard(cy: int, crop_h: int, subject_cy_src: int, orig_h: int) -> int:
    hh     = crop_h // 2
    max_cy = subject_cy_src - int((1.0 - LOWER_THIRD_GUARD) * crop_h) + hh
    return min(cy, min(max_cy, orig_h - hh))


def _soi_region_label(cx: int, cy: int, w: int, h: int) -> str:
    col = "left" if cx < w // 3 else ("right" if cx > 2 * w // 3 else "center")
    row = "upper" if cy < h // 3 else ("lower" if cy > 2 * h // 3 else "mid")
    if row == "mid" and col == "center":
        return "center"
    if row == "mid":
        return col
    return f"{row}-{col}"


def frame_for_union(
    ux1: int, uy1: int, ux2: int, uy2: int,
    orig_w: int, orig_h: int, crop_w: int, crop_h: int,
) -> Tuple[int, int]:
    ucx = (ux1 + ux2) // 2
    ucy = (uy1 + uy2) // 2
    hw, hh = crop_w // 2, crop_h // 2
    cx  = max(hw, min(ucx, orig_w - hw))
    cy  = max(hh, min(ucy, orig_h - hh))
    cy  = _apply_lower_third_guard(cy, crop_h, ucy, orig_h)
    return cx, max(hh, min(cy, orig_h - hh))


def talking_head_center(
    faces: List[Tuple[int, int, int, int]], orig_w: int, orig_h: int,
    crop_w: int, crop_h: int, bias: float = 0.30,
) -> Optional[Tuple[int, int]]:
    if not faces:
        return None
    ux1 = min(f[0] for f in faces)
    uy1 = min(f[1] for f in faces)
    ux2 = max(f[2] for f in faces)
    uy2 = max(f[3] for f in faces)
    face_cx = (ux1 + ux2) // 2
    face_cy = (uy1 + uy2) // 2
    cy      = int(face_cy * (1 - bias) + (face_cy + crop_h // 6) * bias)
    hw, hh  = crop_w // 2, crop_h // 2
    cx      = max(hw, min(face_cx, orig_w - hw))
    cy      = max(hh, min(cy, orig_h - hh))
    cy      = _apply_lower_third_guard(cy, crop_h, face_cy, orig_h)
    return cx, max(hh, min(cy, orig_h - hh))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 25: Panel detection & rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_panel_mode(
    input_path: str, model: Any, fps: float, total_frames: int,
    orig_w: int, orig_h: int, confidence: float = 0.45,
    n_probe: int = PANEL_PROBE_COUNT,
    max_person_motion: float    = PANEL_MAX_PERSON_MOTION,
    min_person_area_frac: float = PANEL_MIN_PERSON_AREA_FRAC,
    max_count_variance: float   = PANEL_MAX_COUNT_VARIANCE,
    stability_frac: float       = PANEL_STABILITY_FRAC,
    majority_frac: float        = PANEL_MAJORITY_FRAC,
    min_person_aspect: float    = PANEL_MIN_PERSON_ASPECT,
) -> bool:
    if model is None:
        return False

    det_w       = min(orig_w, 640)
    det_h       = max(1, int(det_w * orig_h / orig_w))
    frame_area  = det_w * det_h
    end_t       = max(2.0, total_frames / fps - 1.0)
    probe_ts    = np.linspace(1.0, end_t, n_probe)

    multi_hits        = 0
    stable_split_hits = 0
    motion_vals:  List[float] = []
    area_vals:    List[float] = []
    aspect_vals:  List[float] = []
    count_vals:   List[int]   = []
    prev_centres_xy: Optional[List[Tuple[float, float]]] = None
    prev_split: Optional[Dict[str, List[float]]] = None

    for t in probe_ts:
        frame = _read_frame_at(input_path, orig_w, orig_h, t, scale_w=det_w, scale_h=det_h)
        if frame is None:
            prev_centres_xy = None
            prev_split      = None
            continue

        persons = detect_persons_all(frame, model, confidence)
        count_vals.append(len(persons))
        curr_centres_xy = [
            ((p[0]+p[2])/2/det_w, (p[1]+p[3])/2/det_h) for p in persons
        ]

        if prev_centres_xy is not None and curr_centres_xy:
            matched_dists: List[float] = []
            used: Set[int] = set()
            for px, py in prev_centres_xy:
                best_d, best_j = 1e9, -1
                for j, (cx, cy) in enumerate(curr_centres_xy):
                    if j in used:
                        continue
                    d = math.hypot(px - cx, py - cy)
                    if d < best_d:
                        best_d, best_j = d, j
                if best_j >= 0:
                    matched_dists.append(best_d * det_w)
                    used.add(best_j)
            if matched_dists:
                motion_vals.append(float(np.mean(matched_dists)))
        prev_centres_xy = curr_centres_xy if persons else None

        if len(persons) < PANEL_MIN_PERSONS:
            prev_split = None
            continue
        multi_hits += 1

        areas   = [(p[2]-p[0])*(p[3]-p[1]) for p in persons]
        area_vals.append(float(np.mean(areas)) / frame_area)
        aspects = [(p[3]-p[1]) / max(p[2]-p[0], 1) for p in persons]
        aspect_vals.append(float(np.mean(aspects)))

        centres_x = [(p[0]+p[2])/2/det_w for p in persons]
        left_x    = [c for c in centres_x if c < 0.40]
        right_x   = [c for c in centres_x if c > 0.60]

        if left_x and right_x:
            if prev_split is not None:
                shift_l = abs(np.mean(left_x) - np.mean(prev_split["left"])) \
                          if prev_split["left"] else 0.0
                shift_r = abs(np.mean(right_x) - np.mean(prev_split["right"])) \
                          if prev_split["right"] else 0.0
                if shift_l <= 0.10 and shift_r <= 0.10:
                    stable_split_hits += 1
            prev_split = {"left": left_x, "right": right_x}
        else:
            prev_split = None

    if multi_hits == 0:
        return False

    cond_ab     = (multi_hits > n_probe * majority_frac and
                   stable_split_hits > int(n_probe * stability_frac))
    mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0
    mean_area   = float(np.mean(area_vals))   if area_vals   else 0.0
    count_std   = float(np.std(count_vals))   if len(count_vals) > 1 else 0.0
    mean_aspect = float(np.mean(aspect_vals)) if aspect_vals  else 0.0

    is_panel = (cond_ab and mean_motion < max_person_motion and
                mean_area >= min_person_area_frac and
                count_std <= max_count_variance and
                mean_aspect >= min_person_aspect)

    print(
        f"[panel_detect] multi={multi_hits} stable={stable_split_hits} "
        f"motion={mean_motion:.1f}px area={mean_area:.3f} "
        f"count_std={count_std:.2f} aspect={mean_aspect:.2f} -> panel={is_panel}",
        file=sys.stderr,
    )
    return is_panel


class PanelSlotSmoother:
    def __init__(self, alpha: float = PANEL_SLOT_EMA,
                 max_jump_frac: float = PANEL_SLOT_MAX_JUMP) -> None:
        self.alpha         = alpha
        self.max_jump_frac = max_jump_frac
        self._slots: List[Optional[Tuple[float, ...]]] = [None, None]
        self._slot_cx: List[Optional[float]] = [None, None]

    def _ema_box(self, prev: Optional[Tuple[float, ...]], new_box: Tuple[int, int, int, int],
                 axis_size: float) -> Tuple[float, ...]:
        if prev is None:
            return tuple(float(v) for v in new_box)
        a        = self.alpha
        max_jump = axis_size * self.max_jump_frac
        smoothed = tuple(prev[i] * (1 - a) + new_box[i] * a for i in range(4))
        return tuple(
            float(np.clip(smoothed[i], prev[i] - max_jump, prev[i] + max_jump))
            for i in range(4)
        )

    def _assign_to_slots(self, groups: List[List[Tuple[int, int, int, int]]]) -> List[List]:
        if not any(groups):
            return [[], []]
        group_cx = [
            float(np.mean([(p[0]+p[2])//2 for p in g])) if g else None
            for g in groups
        ]
        if self._slot_cx[0] is None and self._slot_cx[1] is None:
            non_empty = [(i, cx) for i, cx in enumerate(group_cx) if cx is not None]
            non_empty.sort(key=lambda t: t[1])
            slots: List[List] = [[], []]
            for slot_idx, (grp_idx, _) in enumerate(non_empty[:2]):
                slots[slot_idx] = groups[grp_idx]
            return slots
        used_groups: Set[int] = set()
        result: List[List] = [[], []]
        for slot_idx, slot_cx in enumerate(self._slot_cx):
            if slot_cx is None:
                continue
            best_grp, best_dist = -1, float('inf')
            for grp_idx, cx in enumerate(group_cx):
                if grp_idx in used_groups or cx is None:
                    continue
                d = abs(cx - slot_cx)
                if d < best_dist:
                    best_dist, best_grp = d, grp_idx
            if best_grp >= 0:
                result[slot_idx] = groups[best_grp]
                used_groups.add(best_grp)
        return result

    def update(self, group_a: List[Tuple[int, int, int, int]],
               group_b: List[Tuple[int, int, int, int]],
               strip_w: float) -> Tuple[List, List]:
        assigned = self._assign_to_slots([group_a, group_b])
        result: List[List] = [[], []]
        for i in range(2):
            grp = assigned[i]
            if grp:
                union  = _group_union(grp)
                smooth = self._ema_box(self._slots[i], union, strip_w)
                self._slots[i]   = smooth
                self._slot_cx[i] = (smooth[0] + smooth[2]) / 2.0
                result[i] = [tuple(int(v) for v in smooth)]
            else:
                if self._slots[i] is not None:
                    result[i] = [tuple(int(v) for v in self._slots[i])]
        return result[0], result[1]


def _group_union(persons: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return (
        min(p[0] for p in persons), min(p[1] for p in persons),
        max(p[2] for p in persons), max(p[3] for p in persons),
    )


def _crop_group_to_strip(
    frame: np.ndarray, group: List[Tuple[int, int, int, int]],
    strip_w: int, strip_h: int, expand: float = PANEL_CROP_EXPAND,
    vignette_strength: float = 0.0, color_grade: str = "none",
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    if not group:
        crop = frame
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        ucx     = (ux1 + ux2) // 2
        ucy     = (uy1 + uy2) // 2
        union_w = max(ux2 - ux1, 1)
        strip_r = strip_w / max(strip_h, 1)
        crop_w  = int(union_w * expand)
        crop_h  = int(crop_w / strip_r)
        if crop_h > fh:
            crop_h = fh; crop_w = int(crop_h * strip_r)
        if crop_w > fw:
            crop_w = fw; crop_h = int(crop_w / strip_r)
        crop_w  = max(crop_w, 2)
        crop_h  = max(crop_h, 2)
        x1 = max(0, min(ucx - crop_w // 2, fw - crop_w))
        y1 = max(0, min(ucy - crop_h // 2, fh - crop_h))
        x2 = min(x1 + crop_w, fw)
        y2 = min(y1 + crop_h, fh)
        x1 = max(0, x2 - crop_w)
        y1 = max(0, y2 - crop_h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame

    result = cv2.resize(crop, (strip_w, strip_h), interpolation=cv2.INTER_LANCZOS4)
    if color_grade and color_grade != "none":
        result = apply_color_grade(result, color_grade)
    if vignette_strength > 0:
        result = apply_vignette(result, vignette_strength)
    return result


def _render_panel_frame(
    frame: np.ndarray, persons: List[Tuple[int, int, int, int]],
    out_w: int, out_h: int,
    prev_slots: Optional[List[List[Tuple[int, int, int, int]]]],
    vignette_strength: float = VIGNETTE_STRENGTH * 0.7,
    color_grade: str = "none",
    slot_smoother: Optional[PanelSlotSmoother] = None,
    orientation: str = "horizontal",
) -> Tuple[np.ndarray, List[List[Tuple[int, int, int, int]]]]:
    persons = sorted(persons, key=lambda b: (b[0] + b[2]) // 2)
    n = len(persons)

    if n == 0:
        group_a = prev_slots[0] if prev_slots and prev_slots[0] else []
        group_b = prev_slots[1] if prev_slots and len(prev_slots) > 1 else []
    elif n == 1:
        group_a = persons
        group_b = prev_slots[1] if prev_slots and len(prev_slots) > 1 and prev_slots[1] else persons
    else:
        split   = max(1, n // 2)
        group_a = persons[:split]
        group_b = persons[split:]

    if slot_smoother is not None:
        group_a, group_b = slot_smoother.update(group_a, group_b, strip_w=float(out_w))

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)

    if orientation == "vertical":
        strip_w_a = (out_w // 2) & ~1
        strip_w_b = out_w - strip_w_a
        left  = _crop_group_to_strip(frame, group_a, strip_w_a, out_h,
                                     vignette_strength=vignette_strength, color_grade=color_grade)
        right = _crop_group_to_strip(frame, group_b, strip_w_b, out_h,
                                     vignette_strength=vignette_strength, color_grade=color_grade)
        canvas[:, 0:strip_w_a]               = left
        canvas[:, strip_w_a:strip_w_a + strip_w_b] = right
        dx1 = max(0, strip_w_a - PANEL_DIVIDER_PX // 2)
        dx2 = min(out_w, strip_w_a + (PANEL_DIVIDER_PX + 1) // 2)
        canvas[:, dx1:dx2] = PANEL_DIVIDER_COLOR
    else:
        strip_h_a = (out_h // 2) & ~1
        strip_h_b = out_h - strip_h_a
        top = _crop_group_to_strip(frame, group_a, out_w, strip_h_a,
                                   vignette_strength=vignette_strength, color_grade=color_grade)
        bot = _crop_group_to_strip(frame, group_b, out_w, strip_h_b,
                                   vignette_strength=vignette_strength, color_grade=color_grade)
        canvas[0:strip_h_a, :]                   = top
        canvas[strip_h_a:strip_h_a + strip_h_b, :] = bot
        dy1 = max(0, strip_h_a - PANEL_DIVIDER_PX // 2)
        dy2 = min(out_h, strip_h_a + (PANEL_DIVIDER_PX + 1) // 2)
        canvas[dy1:dy2, :] = PANEL_DIVIDER_COLOR

    return canvas, [list(group_a), list(group_b)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 26: Legacy optical flow / saliency
# ═══════════════════════════════════════════════════════════════════════════════

def optical_flow_center(prev: np.ndarray, curr: np.ndarray,
                        w: int, h: int) -> Optional[Tuple[int, int]]:
    if prev is None or curr is None:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        b    = max(1, int(w * 0.04))
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
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31, 31), 0
    )
    sat  = cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (31, 31), 0
    )
    sal = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    b   = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w - b:] = sal[:b, :] = sal[h - b:, :] = 0
    t   = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 27: Camera-path smoothing
# ═══════════════════════════════════════════════════════════════════════════════

def _vel_to_window(speed: float) -> int:
    t = VELOCITY_SMOOTH_TABLE
    if speed <= t[0][0]:
        return t[0][1]
    if speed >= t[-1][0]:
        return t[-1][1]
    for i in range(len(t) - 1):
        v0, w0 = t[i]; v1, w1 = t[i + 1]
        if v0 <= speed <= v1:
            frac = (speed - v0) / (v1 - v0 + 1e-9)
            w    = int(w0 + frac * (w1 - w0))
            return w if w % 2 == 1 else w + 1
    return 33


def _gauss_seg(xs: np.ndarray, ys: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 3:
        return xs.copy(), ys.copy()
    w = min(window, n - 1)
    w = w if w % 2 == 1 else w - 1
    if w < 3:
        return xs.copy(), ys.copy()
    h2    = w // 2
    sigma = h2 / 2.5 + 1e-9
    k     = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma)**2)
    k    /= k.sum()
    sx    = np.convolve(np.pad(xs, h2, "edge"), k, "valid")[:n]
    sy    = np.convolve(np.pad(ys, h2, "edge"), k, "valid")[:n]
    return sx, sy


def _bidir_ema(xs: np.ndarray, ys: np.ndarray, alpha: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 2:
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def _fwd(v: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float); out[0] = v[0]
        for i in range(1, n):
            out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
        return out

    def _bwd(v: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float); out[-1] = v[-1]
        for i in range(n - 2, -1, -1):
            out[i] = alpha * v[i] + (1 - alpha) * out[i + 1]
        return out

    return (_fwd(xs) + _bwd(xs)) / 2, (_fwd(ys) + _bwd(ys)) / 2


def _apply_sports_post_smooth(
    dense_cx: np.ndarray, dense_cy: np.ndarray,
    fps: float, scene_cuts: List[int], total_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    smooth_window = max(5, int(fps * SPORTS_POST_SMOOTH_WINDOW_SEC))
    if smooth_window % 2 == 0:
        smooth_window += 1

    cuts   = sorted({c for c in scene_cuts if 0 < c < total_frames})
    bounds = [0] + list(cuts) + [total_frames]

    damp_cx = dense_cx.copy().astype(float)
    damp_cy = dense_cy.copy().astype(float)
    damping_factor = 0.65
    spike_threshold = 1.8
    min_velocity = 3.0

    for i in range(2, total_frames):
        v_prev_x = damp_cx[i - 1] - damp_cx[i - 2]
        v_curr_x = dense_cx[i] - dense_cx[i - 1]
        v_prev_y = damp_cy[i - 1] - damp_cy[i - 2]
        v_curr_y = dense_cy[i] - dense_cy[i - 1]
        if abs(v_curr_x) > abs(v_prev_x) * spike_threshold and abs(v_prev_x) > min_velocity:
            damp_cx[i] = damp_cx[i - 1] + v_prev_x * damping_factor
        if abs(v_curr_y) > abs(v_prev_y) * spike_threshold and abs(v_prev_y) > min_velocity:
            damp_cy[i] = damp_cy[i - 1] + v_prev_y * damping_factor

    out_cx = damp_cx.copy()
    out_cy = damp_cy.copy()

    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        seg_len = e - s
        if seg_len < 5:
            continue
        seg_cx = damp_cx[s:e].copy()
        seg_cy = damp_cy[s:e].copy()
        w = min(smooth_window, seg_len - 1)
        w = w if w % 2 == 1 else w - 1
        if w >= 5:
            try:
                polyorder = min(3, w - 1)
                if _SCIPY_AVAILABLE:
                    seg_cx = savgol_filter(seg_cx, w, polyorder)
                    seg_cy = savgol_filter(seg_cy, w, polyorder)
                else:
                    h2 = w // 2
                    sigma = h2 / 2.0 + 1e-9
                    k = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
                    k /= k.sum()
                    seg_cx = np.convolve(np.pad(seg_cx, h2, "edge"), k, "valid")[:seg_len]
                    seg_cy = np.convolve(np.pad(seg_cy, h2, "edge"), k, "valid")[:seg_len]
            except Exception:
                h2 = w // 2
                sigma = h2 / 2.0 + 1e-9
                k = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma) ** 2)
                k /= k.sum()
                seg_cx = np.convolve(np.pad(seg_cx, h2, "edge"), k, "valid")[:seg_len]
                seg_cy = np.convolve(np.pad(seg_cy, h2, "edge"), k, "valid")[:seg_len]
        seg_cx, seg_cy = _bidir_ema(seg_cx, seg_cy, alpha=SPORTS_POST_SMOOTH_EMA_ALPHA)
        out_cx[s:e] = seg_cx
        out_cy[s:e] = seg_cy

    return out_cx, out_cy


def smooth_centers(
    centers: List[Tuple[int, int]], speeds: List[float],
    base_window: int = 33, adaptive: bool = True,
    scene_cuts: Optional[List[int]] = None, use_kalman: bool = False,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    empty: Dict[str, float] = {
        "jitter_raw": 0.0, "jitter_smooth": 0.0,
        "smoothness_pct": 0.0, "max_jump_raw": 0.0,
        "kalman_prediction_frames": 0,
    }
    if not centers or len(centers) < 3:
        return list(centers) if centers else [], empty

    n   = len(centers)
    xs  = np.array([c[0] for c in centers], dtype=float)
    ys  = np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n:
        spd = np.pad(spd, (0, n - len(spd)), mode="edge")

    dx_raw     = np.diff(xs); dy_raw = np.diff(ys)
    dist_raw   = np.sqrt(dx_raw**2 + dy_raw**2)
    jitter_raw = float(np.mean(dist_raw)) if len(dist_raw) > 0 else 0.0
    max_jump   = float(np.max(dist_raw))  if len(dist_raw) > 0 else 0.0

    cuts   = sorted({c for c in (scene_cuts or []) if 0 < c < n})
    bounds = [0] + cuts + [n]
    rx, ry = xs.copy(), ys.copy()

    pred_count = 0
    if use_kalman:
        kalman = SportsKalmanTracker(dt=1.0)
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < 2:
                continue
            kalman.init(xs[s], ys[s])
            for j in range(s, e):
                kx, ky = kalman.update(xs[j], ys[j])
                speed  = spd[j] if j < len(spd) else 0.0
                if speed > 60.0 and not kalman.is_stale:
                    raw_alpha = 0.15
                    rx[j] = raw_alpha * xs[j] + (1 - raw_alpha) * kx
                    ry[j] = raw_alpha * ys[j] + (1 - raw_alpha) * ky
                    pred_count += 1
                else:
                    rx[j] = kx
                    ry[j] = ky
        if n > 5:
            h2 = 1; sigma = 0.8
            k  = np.exp(-0.5 * (np.arange(-h2, h2 + 1) / sigma)**2)
            k /= k.sum()
            rx = np.convolve(np.pad(rx, h2, "edge"), k, "valid")[:n]
            ry = np.convolve(np.pad(ry, h2, "edge"), k, "valid")[:n]
    else:
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < 3:
                continue
            w      = max(_vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window, 13)
            gx, gy = _gauss_seg(xs[s:e], ys[s:e], w)
            bx, by = _bidir_ema(gx, gy, alpha=0.08)
            rx[s:e] = bx
            ry[s:e] = by

    smoothed = [(int(x), int(y)) for x, y in zip(rx, ry)]
    dx_s     = np.diff(rx); dy_s = np.diff(ry)
    jitter_s = float(np.mean(np.sqrt(dx_s**2 + dy_s**2)))
    pct      = (jitter_raw - jitter_s) / jitter_raw * 100 if jitter_raw > 0 else 0.0

    return smoothed, {
        "jitter_raw":     round(jitter_raw, 2),
        "jitter_smooth":  round(jitter_s,   2),
        "smoothness_pct": round(pct, 1),
        "max_jump_raw":   round(max_jump, 1),
        "kalman_prediction_frames": pred_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 28: Whisper / translate
# ═══════════════════════════════════════════════════════════════════════════════

def _seconds_to_srt_time(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"


def transcribe_to_srt(
    video_path: str, srt_path: str, whisper_model: str = "base",
    language: Optional[str] = None, max_chars_per_line: int = 42,
    progress_callback=None,
) -> bool:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

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
        model  = _w.load_model(whisper_model)
        opts: Dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if language:
            opts["language"] = language
        result = model.transcribe(wav_path, **opts)

        _p(0.85, "Writing subtitles...")
        lines: List[str] = []
        idx   = 1
        words: List[Dict[str, Any]] = [
            {"word": w_["word"].strip(), "start": w_["start"], "end": w_["end"]}
            for seg in result.get("segments", [])
            for w_ in seg.get("words", [])
        ]

        buf: List[Dict[str, Any]] = []
        buf_len = 0

        def _flush() -> None:
            nonlocal idx, buf, buf_len
            if not buf:
                return
            lines.append(
                f"{idx}\n"
                f"{_seconds_to_srt_time(buf[0]['start'])} --> {_seconds_to_srt_time(buf[-1]['end'])}\n"
                f"{' '.join(x['word'] for x in buf)}\n"
            )
            idx += 1; buf = []; buf_len = 0

        for w_ in words:
            wl = len(w_["word"]) + 1
            if buf_len + wl > max_chars_per_line and buf:
                _flush()
            buf.append(w_)
            buf_len += wl
        _flush()

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        _p(1.0, f"{len(lines)} subtitle lines")
        return True
    except Exception as e:
        print(f"Whisper failed: {e}", file=sys.stderr)
        return False
    finally:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except OSError: pass


def translate_srt(
    srt_path: str, target_language: str,
    source_language: str = "auto", progress_callback=None,
) -> bool:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    if not translation_available() or not target_language:
        return not bool(target_language)
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return False

    import re
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"\n\n+", content.strip())
        out: List[str] = []
        tr  = GoogleTranslator(source=source_language, target=target_language)
        for i, block in enumerate(blocks):
            ls = block.strip().splitlines()
            if len(ls) < 3:
                out.append(block); continue
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
# SECTION 29: Clip detection
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_saliency_score(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_score = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)
    motion    = 0.0
    if prev_frame is not None:
        motion = min(float(cv2.absdiff(gray, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)).mean()) / 30.0, 1.0)
    sat = min(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 128.0, 1.0)
    return 0.4 * motion + 0.4 * lap_score + 0.2 * sat


def _compute_frame_scores(
    input_path: str, fps: float, total_frames: int, orig_w: int, orig_h: int,
    sample_every: int = 15, progress_callback=None,
) -> Tuple[np.ndarray, List[int]]:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    scores: List[float] = []; scene_cuts: List[int] = []
    prev_gray = prev_frame = None
    sw = min(orig_w, 640)
    sh = max(1, int(sw * orig_h / orig_w))
    fi = 0

    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=sw, scale_h=sh) as reader:
            for frame in reader:
                if fi >= total_frames:
                    break
                if fi % sample_every == 0:
                    cg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None and float(cv2.absdiff(prev_gray, cg).mean()) / 255.0 > 0.30:
                        scene_cuts.append(fi)
                    scores.append(_frame_saliency_score(frame, prev_frame))
                    prev_gray  = cg
                    prev_frame = frame.copy()
                if fi % max(1, total_frames // 20) == 0:
                    _p(fi / total_frames, f"Scanning {fi}/{total_frames}...")
                fi += 1
    except Exception as e:
        print(f"[scan] Error: {e}", file=sys.stderr)

    return np.array(scores, dtype=float), scene_cuts


def detect_clips(
    input_path: str,
    min_duration_sec: float = 25.0,
    max_duration_sec: float = 65.0,
    target_n_clips: int = 10,
    model: Optional[Any] = None,
    confidence: float = 0.45,
    progress_callback=None,
) -> List[ClipSegment]:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    info         = get_video_info(input_path)
    fps          = info["fps"]
    total_frames = info["total_frames"]
    duration     = info["duration_seconds"]
    orig_w       = info["width"]
    orig_h       = info["height"]
    sample_every = max(1, int(fps))

    _p(0.0, "Scanning...")
    scores, scene_cut_frames = _compute_frame_scores(
        input_path, fps, total_frames, orig_w, orig_h, sample_every=sample_every,
        progress_callback=lambda v, m: _p(v * 0.45, m),
    )
    if len(scores) == 0:
        return []

    _p(0.45, "Computing arcs...")
    window = max(5, int(30 / (sample_every / fps)))
    ss     = np.convolve(scores, np.ones(window) / window, "same") if len(scores) >= window else scores.copy()
    if ss.max() > 0:
        ss /= ss.max()

    min_gap = max(1, int(min_duration_sec * fps / sample_every))
    peaks: List[int] = []
    for i in range(1, len(ss) - 1):
        wh = min_gap // 2
        lo = max(0, i - wh); hi = min(len(ss), i + wh + 1)
        if ss[i] == ss[lo:hi].max() and ss[i] > 0.3:
            if not peaks or i - peaks[-1] > min_gap // 2:
                peaks.append(i)
    peaks.sort(key=lambda i: ss[i], reverse=True)
    peaks = peaks[:target_n_clips * 2]

    def _arc(pi: int) -> Tuple[float, float]:
        ps = pi * sample_every / fps
        rs = max(0.0, ps - max_duration_sec * 0.4)
        re = min(duration, rs + max_duration_sec)
        for sc in reversed(scene_cut_frames):
            sc_s = sc / fps
            if 0 < ps - sc_s < 15.0:
                rs = max(0.0, sc_s - 1.0); break
        for sc in scene_cut_frames:
            sc_s = sc / fps
            if 0 < sc_s - ps < 15.0:
                re = min(duration, sc_s + 0.5); break
        cd = re - rs
        if cd < min_duration_sec:
            re = min(duration, rs + min_duration_sec)
        elif cd > max_duration_sec:
            c  = (rs + re) / 2
            rs = max(0.0, c - max_duration_sec / 2)
            re = min(duration, rs + max_duration_sec)
        return rs, re

    cands: List[Tuple[float, float, float]] = []
    for pi in peaks:
        s, e = _arc(pi); sc_ = float(ss[pi])
        if not any(min(e, ce) - max(s, cs) > min_duration_sec * 0.5 for cs, ce, _ in cands):
            cands.append((s, e, sc_))
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:target_n_clips]; cands.sort(key=lambda x: x[0])

    _p(0.55, "SOI per clip...")
    segments: List[ClipSegment] = []
    det_w = min(orig_w, 640); det_h = max(1, int(det_w * orig_h / orig_w))

    for ci, (ss2, se, score) in enumerate(cands):
        _p(0.55 + 0.35 * (ci / max(len(cands), 1)), f"Clip {ci+1}/{len(cands)}...")
        soi_xs: List[int] = []; soi_ys: List[int] = []
        n_s = min(8, max(2, int(se - ss2)))
        for t in np.linspace(ss2 + 1, se - 1, n_s):
            frame = _read_frame_at(input_path, orig_w, orig_h, t, scale_w=det_w, scale_h=det_h)
            if frame is None:
                continue
            if model is not None:
                try:
                    res = model(frame, verbose=False, conf=confidence)[0]
                    if res.boxes is not None:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            soi_xs.append((x1 + x2) // 2); soi_ys.append((y1 + y2) // 2)
                except Exception:
                    pass
            else:
                scx, scy = saliency_center(frame)
                soi_xs.append(scx); soi_ys.append(scy)

        sr = "center"
        if soi_xs:
            sr = _soi_region_label(int(np.median(soi_xs)), int(np.median(soi_ys)), orig_w, orig_h)
        ms   = int(ss2 // 60); secs = int(ss2 % 60)
        me   = int(se  // 60); sece = int(se  % 60)
        segments.append(ClipSegment(
            start_sec=ss2, end_sec=se, score=score, soi_region=sr,
            peak_frame=int(np.linspace(ss2+1, se-1, n_s)[n_s//2] * fps),
            title=f"Clip {ci+1} ({ms}:{secs:02d} - {me}:{sece:02d})",
        ))

    _p(1.0, f"Found {len(segments)} clips")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 30: Analytics helper
# ═══════════════════════════════════════════════════════════════════════════════

def _build_analytics(
    input_path: str,
    output_path: str,
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    smooth_metrics: Optional[Dict[str, float]] = None,
    panel_mode: bool = False,
    kalman_predictions: int = 0,
) -> Dict[str, Any]:
    in_size  = os.path.getsize(input_path) / (1024 * 1024) if os.path.exists(input_path) else 0.0
    out_size = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0

    def _bitrate(path: str) -> int:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=bit_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=10,
            )
            return int(r.stdout.strip()) // 1000
        except Exception:
            return 0

    analytics: Dict[str, Any] = {
        "input_size_mb":       round(in_size, 2),
        "output_size_mb":      round(out_size, 2),
        "compression_ratio":   round(in_size / out_size, 2) if out_size > 0 else 0.0,
        "file_size_reduction_pct": round((1 - out_size / in_size) * 100, 1) if in_size > 0 else 0.0,
        "input_resolution":    f"{orig_w}x{orig_h}",
        "output_resolution":   f"{out_w}x{out_h}",
        "input_bitrate_kbps":  _bitrate(input_path),
        "output_bitrate_kbps": _bitrate(output_path),
        "panel_mode":          panel_mode,
        "kalman_predictions":  kalman_predictions,
        "jitter_raw":          0.0,
        "jitter_smooth":       0.0,
        "smoothness_pct":      0.0,
    }
    if smooth_metrics:
        analytics["jitter_raw"]     = smooth_metrics.get("jitter_raw", 0.0)
        analytics["jitter_smooth"]  = smooth_metrics.get("jitter_smooth", 0.0)
        analytics["smoothness_pct"] = smooth_metrics.get("smoothness_pct", 0.0)
        analytics["kalman_predictions"] = smooth_metrics.get("kalman_prediction_frames", kalman_predictions)
    return analytics


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 31: Core single-pass rendering engine
# ═══════════════════════════════════════════════════════════════════════════════

def _render_video(
    input_path: str,
    output_path: str,
    out_w: int,
    out_h: int,
    crop_w: int,
    crop_h: int,
    orig_w: int,
    orig_h: int,
    fps: float,
    total_frames: int,
    smoothed_centers: List[Tuple[int, int]],
    tracking_mode: str = "subject",
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
    burn_subtitles: bool = False,
    whisper_model: str = "base",
    whisper_language: Optional[str] = None,
    subtitle_style_name: str = "Bold White (TikTok)",
    subtitle_max_chars: int = 42,
    subtitle_translate_to: Optional[str] = None,
    output_fps: Optional[float] = None,
    color_grade: str = "none",
    vignette_strength: float = 0.0,
    sharpen_strength: float = 0.0,
    ffmpeg_sharpen: bool = False,
    progress_callback=None,
    scene_cuts: Optional[List[int]] = None,
    use_panel_mode: bool = False,
    panel_config: Optional[PanelModeConfig] = None,
    panel_persons_map: Optional[Dict[int, List[Tuple[int, int, int, int]]]] = None,
) -> Dict[str, Any]:
    """Core rendering: applies smoothed crop path to produce the output video."""

    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    eff_fps   = output_fps or fps
    srt_path  = None

    # Subtitles
    if burn_subtitles and whisper_available():
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt")
        os.close(srt_fd)
        _p(0.02, "Transcribing...")
        ok = transcribe_to_srt(
            input_path, srt_path,
            whisper_model=whisper_model,
            language=whisper_language,
            max_chars_per_line=subtitle_max_chars,
            progress_callback=lambda v, m: _p(0.02 + v * 0.08, m),
        )
        if ok and subtitle_translate_to:
            translate_srt(srt_path, subtitle_translate_to,
                          progress_callback=lambda v, m: _p(0.10 + v * 0.03, m))
        if not ok:
            try: os.unlink(srt_path)
            except OSError: pass
            srt_path = None

    extra_vf = _build_ffmpeg_vf(color_grade, ffmpeg_sharpen)
    subtitle_style = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])

    enc = _open_ffmpeg_encoder(
        output_path, out_w, out_h, eff_fps,
        audio_source=input_path,
        crf=crf, preset=encoder_preset,
        audio_bitrate=audio_bitrate,
        subtitle_path=srt_path,
        subtitle_style=subtitle_style,
        extra_vf=extra_vf,
    )

    dissolve  = DissolveBuffer(DISSOLVE_FRAMES)
    slot_smoother = PanelSlotSmoother() if use_panel_mode else None
    prev_slots: Optional[List[List]] = None
    orientation = (panel_config.split_orientation if panel_config else "horizontal")

    fi = 0
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames:
                    break

                if use_panel_mode:
                    persons = (panel_persons_map or {}).get(fi, [])
                    if not persons:
                        # fallback: use smoothed center as a rough bounding box
                        if fi < len(smoothed_centers):
                            scx, scy = smoothed_centers[fi]
                        else:
                            scx, scy = orig_w // 2, orig_h // 2
                        hw = min(crop_w // 2, orig_w // 4)
                        hh = min(crop_h // 4, orig_h // 4)
                        persons = [(scx - hw, scy - hh, scx + hw, scy + hh)]

                    out_frame, prev_slots = _render_panel_frame(
                        frame, persons, out_w, out_h,
                        prev_slots=prev_slots,
                        slot_smoother=slot_smoother,
                        orientation=orientation,
                    )
                else:
                    if fi < len(smoothed_centers):
                        cx, cy = smoothed_centers[fi]
                    else:
                        cx, cy = orig_w // 2, orig_h // 2

                    hw, hh = crop_w // 2, crop_h // 2
                    x1 = int(np.clip(cx - hw, 0, orig_w - crop_w))
                    y1 = int(np.clip(cy - hh, 0, orig_h - crop_h))
                    x2 = x1 + crop_w
                    y2 = y1 + crop_h

                    # Safety clamp
                    x2 = min(x2, orig_w)
                    y2 = min(y2, orig_h)
                    x1 = max(0, x2 - crop_w)
                    y1 = max(0, y2 - crop_h)

                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] == 0 or crop.shape[1] == 0:
                        crop = frame[:crop_h, :crop_w] if orig_h >= crop_h and orig_w >= crop_w else frame

                    out_frame = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                    if vignette_strength > 0:
                        out_frame = apply_vignette(out_frame, vignette_strength)
                    if sharpen_strength > 0 and not ffmpeg_sharpen:
                        out_frame = apply_sharpen(out_frame, sharpen_strength)

                # Dissolve on scene cut
                if scene_cuts and fi in scene_cuts:
                    dissolve.on_cut(out_frame)
                if dissolve.active:
                    out_frame = dissolve.blend(out_frame)

                try:
                    enc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    break

                fi += 1
                if fi % max(1, total_frames // 50) == 0:
                    _p(0.15 + (fi / total_frames) * 0.80, f"Rendering {fi}/{total_frames}...")

    finally:
        _close_ffmpeg_encoder(enc, output_path)
        if srt_path and os.path.exists(srt_path):
            return {"subtitle_path": srt_path}

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 32: First-pass tracking (collect centers + persons per frame)
# ═══════════════════════════════════════════════════════════════════════════════

def _tracking_pass(
    input_path: str,
    orig_w: int,
    orig_h: int,
    crop_w: int,
    crop_h: int,
    fps: float,
    total_frames: int,
    tracking_mode: str,
    model: Any,
    confidence: float,
    smooth_window: int,
    adaptive_smoothing: bool,
    use_optical_flow: bool,
    rule_of_thirds: bool,
    scene_cut_threshold: float,
    talking_head_bias: float,
    use_kalman: bool = False,
    panel_mode_active: bool = False,
    progress_callback=None,
) -> Tuple[
    List[Tuple[int, int]],    # raw centers
    List[float],              # speeds
    List[int],                # scene cuts
    Dict[int, List[Tuple[int, int, int, int]]],  # persons per frame (for panel mode)
    int,                      # kalman prediction count
]:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    hw, hh      = crop_w // 2, crop_h // 2
    centers:    List[Tuple[int, int]] = []
    speeds:     List[float] = []
    scene_cuts: List[int]   = []
    persons_map: Dict[int, List[Tuple[int, int, int, int]]] = {}

    prev_gray:   Optional[np.ndarray] = None
    prev_frame:  Optional[np.ndarray] = None
    prev_cx:     Optional[int] = None
    prev_cy:     Optional[int] = None
    prev_hist:   Optional[np.ndarray] = None
    last_cut_frame = -100
    prev_ball_carrier: Optional[int] = None
    prev_saliency: Optional[np.ndarray] = None
    kalman_pred_count = 0

    # Scale-down for detection
    det_scale = 1.0
    if orig_w > 960:
        det_scale = 960 / orig_w
    det_w = max(1, int(orig_w * det_scale))
    det_h = max(1, int(orig_h * det_scale))

    fi = 0
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=det_w, scale_h=det_h) as reader:
            for det_frame in reader:
                if fi >= total_frames:
                    break

                # Full-res gray for flow (resize back)
                gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

                # Scene change detection
                is_cut, prev_hist, last_cut_frame = is_scene_change(
                    prev_frame, det_frame, threshold=scene_cut_threshold,
                    prev_hist=prev_hist, frame_count=fi,
                    last_cut_frame=last_cut_frame,
                    mode="sports" if tracking_mode == "sports_action" else "default",
                )
                if is_cut:
                    scene_cuts.append(fi)
                    prev_gray  = None
                    prev_cx    = None
                    prev_cy    = None

                cx: Optional[int] = None
                cy: Optional[int] = None

                if tracking_mode == "talking_head":
                    faces = detect_faces(det_frame)
                    if faces:
                        scaled_faces = [
                            (int(f[0]/det_scale), int(f[1]/det_scale),
                             int(f[2]/det_scale), int(f[3]/det_scale))
                            for f in faces
                        ]
                        result = talking_head_center(
                            scaled_faces, orig_w, orig_h, crop_w, crop_h, bias=talking_head_bias
                        )
                        if result:
                            cx, cy = result

                elif tracking_mode == "sports_action":
                    det_result, ball_box, ball_carrier = detect_subjects(
                        det_frame, model, confidence,
                        prev_center=(
                            (int(prev_cx * det_scale), int(prev_cy * det_scale))
                            if prev_cx is not None else None
                        ),
                        prev_ball_carrier=prev_ball_carrier,
                        tracking_mode="sports_action",
                    )
                    if det_result is not None:
                        cx = int(det_result.cx / det_scale)
                        cy = int(det_result.cy / det_scale)
                        prev_ball_carrier = ball_carrier

                    if panel_mode_active:
                        # Collect all persons for panel rendering
                        all_persons = detect_persons_all(det_frame, model, confidence)
                        persons_map[fi] = [
                            (int(p[0]/det_scale), int(p[1]/det_scale),
                             int(p[2]/det_scale), int(p[3]/det_scale))
                            for p in all_persons
                        ]

                else:  # subject tracking
                    det_result, _, _ = detect_subjects(
                        det_frame, model, confidence,
                        prev_center=(
                            (int(prev_cx * det_scale), int(prev_cy * det_scale))
                            if prev_cx is not None else None
                        ),
                        tracking_mode="subject",
                    )
                    if det_result is not None:
                        cx = int(det_result.cx / det_scale)
                        cy = int(det_result.cy / det_scale)

                    if panel_mode_active:
                        all_persons = detect_persons_all(det_frame, model, confidence)
                        persons_map[fi] = [
                            (int(p[0]/det_scale), int(p[1]/det_scale),
                             int(p[2]/det_scale), int(p[3]/det_scale))
                            for p in all_persons
                        ]

                # Optical flow fallback
                if cx is None and use_optical_flow and prev_gray is not None:
                    of_result = optical_flow_center(prev_gray, gray, det_w, det_h)
                    if of_result is not None:
                        cx = int(of_result[0] / det_scale)
                        cy = int(of_result[1] / det_scale)

                # Saliency fallback
                if cx is None:
                    scx, scy, prev_saliency = temporal_saliency_center(det_frame, prev_saliency)
                    cx = int(scx / det_scale)
                    cy = int(scy / det_scale)

                # Clamp to valid crop region
                cx = max(hw, min(cx, orig_w - hw))
                cy = max(hh, min(cy, orig_h - hh))

                # Rule of thirds horizontal nudge
                if rule_of_thirds and prev_cx is not None:
                    rot_cx = orig_w // 3 if cx < orig_w // 2 else 2 * orig_w // 3
                    cx = int(cx * 0.85 + rot_cx * 0.15)
                    cx = max(hw, min(cx, orig_w - hw))

                # Lower-third guard
                cy = _apply_lower_third_guard(cy, crop_h, cy, orig_h)
                cy = max(hh, min(cy, orig_h - hh))

                speed = 0.0
                if prev_cx is not None:
                    speed = math.hypot(cx - prev_cx, cy - prev_cy)

                centers.append((cx, cy))
                speeds.append(speed)
                prev_cx    = cx
                prev_cy    = cy
                prev_gray  = gray
                prev_frame = det_frame

                if fi % max(1, total_frames // 50) == 0:
                    _p(fi / total_frames, f"Tracking {fi}/{total_frames}...")
                fi += 1

    except Exception as e:
        print(f"[tracking_pass] Error at frame {fi}: {e}", file=sys.stderr)

    # Pad if short
    while len(centers) < total_frames:
        centers.append(centers[-1] if centers else (orig_w // 2, orig_h // 2))
        speeds.append(0.0)

    return centers, speeds, scene_cuts, persons_map, kalman_pred_count


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 33: process_video — main public API
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(
    input_path: str,
    output_path: str,
    target_preset_label: str = "720p   (720x1280  - HD)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    confidence: float = 0.45,
    smooth_window: int = 15,
    adaptive_smoothing: bool = True,
    use_optical_flow: bool = True,
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
    color_grade: str = "none",
    vignette_strength: float = 0.0,
    sharpen_strength: float = 0.0,
    ffmpeg_sharpen: bool = False,
    ken_burns: bool = False,
    use_kalman: bool = False,
    panel_config: Optional[PanelModeConfig] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Main single-clip verticalization pipeline.
    Returns dict with keys: subtitle_path (optional), analytics (dict).
    """
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    _check_ffmpeg()
    info = get_video_info(input_path)
    orig_w, orig_h = info["width"], info["height"]
    fps            = info["fps"]
    total_frames   = info["total_frames"]

    out_w, out_h = resolve_target_size(target_preset_label, orig_w, orig_h)
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, out_w, out_h)

    _p(0.01, "Loading model...")
    model = _get_model(yolo_weights) if tracking_mode != "talking_head" else None

    # Panel mode decision
    panel_cfg      = panel_config or PanelModeConfig()
    use_panel_mode = False
    if panel_cfg.split_mode == "force_on":
        use_panel_mode = True
    elif panel_cfg.split_mode == "force_off":
        use_panel_mode = False
    elif tracking_mode == "subject":
        _p(0.03, "Detecting panel layout...")
        use_panel_mode = _detect_panel_mode(
            input_path, model, fps, total_frames, orig_w, orig_h,
            confidence=confidence,
            max_person_motion=panel_cfg.max_person_motion,
            min_person_area_frac=panel_cfg.min_person_area_frac,
            max_count_variance=panel_cfg.max_count_variance,
            stability_frac=panel_cfg.stability_frac,
        )

    _p(0.05, "Tracking subjects...")
    raw_centers, speeds, scene_cuts, persons_map, kalman_preds = _tracking_pass(
        input_path=input_path,
        orig_w=orig_w, orig_h=orig_h,
        crop_w=crop_w, crop_h=crop_h,
        fps=fps, total_frames=total_frames,
        tracking_mode=tracking_mode,
        model=model,
        confidence=confidence,
        smooth_window=smooth_window,
        adaptive_smoothing=adaptive_smoothing,
        use_optical_flow=use_optical_flow,
        rule_of_thirds=rule_of_thirds,
        scene_cut_threshold=scene_cut_threshold,
        talking_head_bias=talking_head_bias,
        use_kalman=use_kalman,
        panel_mode_active=use_panel_mode,
        progress_callback=lambda v, m: _p(0.05 + v * 0.45, m),
    )

    _p(0.50, "Smoothing camera path...")
    smoothed, smooth_metrics = smooth_centers(
        raw_centers, speeds,
        base_window=smooth_window,
        adaptive=adaptive_smoothing,
        scene_cuts=scene_cuts,
        use_kalman=use_kalman,
    )

    _p(0.55, "Rendering...")
    render_meta = _render_video(
        input_path=input_path,
        output_path=output_path,
        out_w=out_w, out_h=out_h,
        crop_w=crop_w, crop_h=crop_h,
        orig_w=orig_w, orig_h=orig_h,
        fps=fps, total_frames=total_frames,
        smoothed_centers=smoothed,
        tracking_mode=tracking_mode,
        crf=crf,
        encoder_preset=encoder_preset,
        audio_bitrate=audio_bitrate,
        burn_subtitles=burn_subtitles,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        subtitle_style_name=subtitle_style_name,
        subtitle_max_chars=subtitle_max_chars,
        subtitle_translate_to=subtitle_translate_to,
        output_fps=output_fps,
        color_grade=color_grade,
        vignette_strength=vignette_strength,
        sharpen_strength=sharpen_strength,
        ffmpeg_sharpen=ffmpeg_sharpen,
        scene_cuts=scene_cuts,
        use_panel_mode=use_panel_mode,
        panel_config=panel_cfg,
        panel_persons_map=persons_map,
        progress_callback=lambda v, m: _p(0.55 + v * 0.43, m),
    )

    _p(1.0, "Done!")
    analytics = _build_analytics(
        input_path, output_path,
        orig_w=orig_w, orig_h=orig_h,
        out_w=out_w, out_h=out_h,
        smooth_metrics=smooth_metrics,
        panel_mode=use_panel_mode,
        kalman_predictions=smooth_metrics.get("kalman_prediction_frames", 0),
    )
    result = {"analytics": analytics}
    if render_meta.get("subtitle_path"):
        result["subtitle_path"] = render_meta["subtitle_path"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 34: process_sports_video — sports-specific pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_sports_video(
    input_path: str,
    output_path: str,
    sport_type: str = "auto",
    target_preset_label: str = "720p   (720x1280  - HD)",
    confidence: float = 0.45,
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
    use_ball_tracking: bool = True,
    use_kalman: bool = True,
    smooth_window: int = 5,
    adaptive_smoothing: bool = True,
    use_optical_flow: bool = True,
    rule_of_thirds: bool = True,
    scene_cut_threshold: float = 0.22,
    vignette_strength: float = 0.275,
    sharpen_strength: float = 0.3,
    ffmpeg_sharpen: bool = True,
    color_grade: str = "none",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Sports-optimized pipeline using MOT + AVS + ICS.
    Returns dict with keys: subtitle_path (optional), analytics (dict).
    """
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    _check_ffmpeg()
    info = get_video_info(input_path)
    orig_w, orig_h = info["width"], info["height"]
    fps            = info["fps"]
    total_frames   = info["total_frames"]

    out_w, out_h = resolve_target_size(target_preset_label, orig_w, orig_h)
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, out_w, out_h)

    _p(0.01, "Loading sports model...")
    model = _get_model(yolo_weights)

    # Detect field of play on a sample frame
    sample_frame = _read_frame_at(input_path, orig_w, orig_h, 2.0)
    field_mask: Optional[np.ndarray] = None
    if sample_frame is not None:
        field_mask = detect_field_of_play(sample_frame, sport_hint=sport_type)

    # Sports-specific tracking objects
    mot_tracker   = MultiObjectSportsTracker(fps, orig_w, orig_h)
    avs_smoother  = AdaptiveVelocityAwareSmoother(fps)
    ics           = IntelligentCropStrategy(orig_w, orig_h, crop_w, crop_h, fps)
    phase_detector = SportsPlayPhaseDetector(fps)
    event_detector = SportsEventDetector(fps)
    game_engine   = GameStateEngine(fps, orig_w, orig_h)

    # Detection scale
    det_scale = 1.0
    if orig_w > 960:
        det_scale = 960 / orig_w
    det_w = max(1, int(orig_w * det_scale))
    det_h = max(1, int(orig_h * det_scale))

    raw_centers: List[Tuple[int, int]] = []
    speeds:      List[float] = []
    scene_cuts:  List[int]   = []
    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    prev_frame:  Optional[np.ndarray] = None
    prev_hist:   Optional[np.ndarray] = None
    prev_gray:   Optional[np.ndarray] = None
    last_cut_frame = -100
    prev_ball_carrier: Optional[int] = None

    _p(0.05, "Sports tracking pass...")

    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=det_w, scale_h=det_h) as reader:
            for det_frame in reader:
                if len(raw_centers) >= total_frames:
                    break
                fi = len(raw_centers)

                gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

                # Scene change
                is_cut, prev_hist, last_cut_frame = is_sports_scene_change(
                    prev_frame, det_frame, prev_hist, fi, last_cut_frame
                )
                if is_cut:
                    scene_cuts.append(fi)
                    mot_tracker = MultiObjectSportsTracker(fps, orig_w, orig_h)
                    avs_smoother = AdaptiveVelocityAwareSmoother(fps)
                    prev_cx = prev_cy = None

                # Detect persons + ball
                det_result, ball_box, ball_carrier = detect_subjects(
                    det_frame, model, confidence,
                    prev_center=(
                        (int(prev_cx * det_scale), int(prev_cy * det_scale))
                        if prev_cx is not None else None
                    ),
                    prev_ball_carrier=prev_ball_carrier,
                    tracking_mode="sports_action",
                )

                # Convert detections back to original scale
                all_persons_det = detect_persons_all(det_frame, model, confidence)
                all_persons_orig = [
                    (int(p[0]/det_scale), int(p[1]/det_scale),
                     int(p[2]/det_scale), int(p[3]/det_scale))
                    for p in all_persons_det
                ]
                ball_box_orig = (
                    (int(ball_box[0]/det_scale), int(ball_box[1]/det_scale),
                     int(ball_box[2]/det_scale), int(ball_box[3]/det_scale))
                    if ball_box else None
                ) if use_ball_tracking else None

                # Update MOT
                confs = [0.5] * len(all_persons_orig)
                mot_tracker.update(all_persons_orig, ball_box_orig, det_frame, confs)
                primary_track = mot_tracker.get_primary_track(prev_ball_carrier)

                # Determine phase
                phase = phase_detector.detect_phase(
                    all_persons_orig, ball_box_orig, orig_w, mot_tracker.ball_state
                )

                # Get focus point
                if primary_track is not None:
                    raw_cx = float(primary_track.center[0])
                    raw_cy = float(primary_track.center[1])
                elif det_result is not None:
                    raw_cx = float(det_result.cx) / det_scale
                    raw_cy = float(det_result.cy) / det_scale
                else:
                    if prev_gray is not None and use_optical_flow:
                        of = sports_optical_flow_center(prev_gray, gray, det_w, det_h,
                                                        field_mask=field_mask)
                        if of is not None:
                            raw_cx = float(of[0]) / det_scale
                            raw_cy = float(of[1]) / det_scale
                        else:
                            raw_cx = float(prev_cx) if prev_cx is not None else orig_w / 2
                            raw_cy = float(prev_cy) if prev_cy is not None else orig_h / 2
                    else:
                        raw_cx = float(prev_cx) if prev_cx is not None else orig_w / 2
                        raw_cy = float(prev_cy) if prev_cy is not None else orig_h / 2

                # AVS smoothing
                track_conf = primary_track.confidence if primary_track else 0.5
                smooth_cx, smooth_cy = avs_smoother.smooth(raw_cx, raw_cy, track_conf, phase)

                # ICS look-ahead crop
                ball_pos_orig = (
                    float(mot_tracker.ball_state.center[0]),
                    float(mot_tracker.ball_state.center[1])
                ) if mot_tracker.ball_state.center is not None and use_ball_tracking else None

                left, top, right, bottom = ics.compute_crop(
                    smooth_cx, smooth_cy, phase, ball_pos_orig
                )
                cx_out = (left + right) // 2
                cy_out = (top + bottom) // 2

                # Clamp
                cx_out = max(crop_w // 2, min(cx_out, orig_w - crop_w // 2))
                cy_out = max(crop_h // 2, min(cy_out, orig_h - crop_h // 2))
                cy_out = _apply_lower_third_guard(cy_out, crop_h, cy_out, orig_h)

                speed = math.hypot(cx_out - (prev_cx or cx_out), cy_out - (prev_cy or cy_out))
                raw_centers.append((cx_out, cy_out))
                speeds.append(speed)
                prev_cx    = float(cx_out)
                prev_cy    = float(cy_out)
                prev_frame = det_frame
                prev_gray  = gray
                if ball_carrier >= 0:
                    prev_ball_carrier = ball_carrier

                if fi % max(1, total_frames // 50) == 0:
                    _p(0.05 + (fi / total_frames) * 0.45, f"Sports tracking {fi}/{total_frames}...")

    except Exception as e:
        print(f"[sports_tracking] Error: {e}", file=sys.stderr)

    # Pad if needed
    while len(raw_centers) < total_frames:
        raw_centers.append(raw_centers[-1] if raw_centers else (orig_w // 2, orig_h // 2))
        speeds.append(0.0)

    # Post-smooth pass
    _p(0.50, "Sports post-smoothing...")
    dense_cx = np.array([c[0] for c in raw_centers], dtype=float)
    dense_cy = np.array([c[1] for c in raw_centers], dtype=float)
    dense_cx, dense_cy = _apply_sports_post_smooth(dense_cx, dense_cy, fps, scene_cuts, total_frames)
    smoothed = [(int(x), int(y)) for x, y in zip(dense_cx, dense_cy)]

    # Compute metrics
    avs_metrics = avs_smoother.get_metrics()
    smooth_metrics = {
        "jitter_raw":               avs_metrics.get("jitter_raw", 0.0),
        "jitter_smooth":            avs_metrics.get("jitter_smooth", 0.0),
        "smoothness_pct":           avs_metrics.get("smoothness_pct", 0.0),
        "kalman_prediction_frames": 0,
    }

    _p(0.55, "Rendering sports video...")
    render_meta = _render_video(
        input_path=input_path,
        output_path=output_path,
        out_w=out_w, out_h=out_h,
        crop_w=crop_w, crop_h=crop_h,
        orig_w=orig_w, orig_h=orig_h,
        fps=fps, total_frames=total_frames,
        smoothed_centers=smoothed,
        tracking_mode="sports_action",
        crf=crf,
        encoder_preset=encoder_preset,
        audio_bitrate=audio_bitrate,
        burn_subtitles=burn_subtitles,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        subtitle_style_name=subtitle_style_name,
        subtitle_max_chars=subtitle_max_chars,
        subtitle_translate_to=subtitle_translate_to,
        output_fps=output_fps,
        color_grade=color_grade,
        vignette_strength=vignette_strength,
        sharpen_strength=sharpen_strength,
        ffmpeg_sharpen=ffmpeg_sharpen,
        scene_cuts=scene_cuts,
        use_panel_mode=False,
        progress_callback=lambda v, m: _p(0.55 + v * 0.43, m),
    )

    _p(1.0, "Done!")
    analytics = _build_analytics(
        input_path, output_path,
        orig_w=orig_w, orig_h=orig_h,
        out_w=out_w, out_h=out_h,
        smooth_metrics=smooth_metrics,
        panel_mode=False,
    )
    result = {"analytics": analytics}
    if render_meta.get("subtitle_path"):
        result["subtitle_path"] = render_meta["subtitle_path"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 35: process_clips_batch — batch auto-clip pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_clips_batch(
    input_path: str,
    output_dir: str,
    clips: List[ClipSegment],
    target_preset_label: str = "720p   (720x1280  - HD)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    confidence: float = 0.45,
    smooth_window: int = 15,
    adaptive_smoothing: bool = True,
    rule_of_thirds: bool = True,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
    yolo_weights: str = "yolov8n.pt",
    burn_subtitles: bool = False,
    whisper_model: str = "base",
    subtitle_style_name: str = "Bold White (TikTok)",
    subtitle_max_chars: int = 42,
    sport_type: str = "auto",
    output_fps: Optional[float] = None,
    panel_config: Optional[PanelModeConfig] = None,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    Batch-process a list of ClipSegments.
    Returns list of result dicts, each with keys:
      clip, output_path, error (optional), analytics (dict)
    """
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    n = len(clips)

    for i, clip in enumerate(clips):
        clip_start = i / n
        clip_end   = (i + 1) / n

        def _clip_p(v: float, msg: str = "") -> None:
            _p(clip_start + v * (clip_end - clip_start), msg)

        _clip_p(0.0, f"Clip {i+1}/{n}: trimming...")

        # Trim the clip to a temp file
        fd, trim_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        ok = _trim_video(input_path, trim_path, clip.start_sec, clip.end_sec)
        if not ok:
            results.append({
                "clip": clip,
                "output_path": None,
                "error": "Trim failed",
                "analytics": {},
            })
            if os.path.exists(trim_path):
                try: os.unlink(trim_path)
                except OSError: pass
            continue

        out_path = os.path.join(output_dir, f"clip_{i+1:03d}_vertical.mp4")

        try:
            if tracking_mode == "sports_action":
                meta = process_sports_video(
                    trim_path, out_path,
                    sport_type=sport_type,
                    target_preset_label=target_preset_label,
                    confidence=confidence,
                    output_fps=output_fps,
                    crf=crf,
                    encoder_preset=encoder_preset,
                    audio_bitrate=audio_bitrate,
                    yolo_weights=yolo_weights,
                    burn_subtitles=burn_subtitles,
                    whisper_model=whisper_model,
                    subtitle_style_name=subtitle_style_name,
                    subtitle_max_chars=subtitle_max_chars,
                    progress_callback=lambda v, m: _clip_p(0.05 + v * 0.90, m),
                )
            else:
                meta = process_video(
                    trim_path, out_path,
                    target_preset_label=target_preset_label,
                    tracking_mode=tracking_mode,
                    talking_head_bias=talking_head_bias,
                    confidence=confidence,
                    smooth_window=smooth_window,
                    adaptive_smoothing=adaptive_smoothing,
                    use_optical_flow=True,
                    rule_of_thirds=rule_of_thirds,
                    scene_cut_threshold=0.35,
                    output_fps=output_fps,
                    crf=crf,
                    encoder_preset=encoder_preset,
                    audio_bitrate=audio_bitrate,
                    yolo_weights=yolo_weights,
                    burn_subtitles=burn_subtitles,
                    whisper_model=whisper_model,
                    subtitle_style_name=subtitle_style_name,
                    subtitle_max_chars=subtitle_max_chars,
                    panel_config=panel_config,
                    progress_callback=lambda v, m: _clip_p(0.05 + v * 0.90, m),
                )

            results.append({
                "clip":        clip,
                "output_path": out_path,
                "analytics":   meta.get("analytics", {}),
            })

        except Exception as e:
            print(f"[batch] Clip {i+1} error: {e}", file=sys.stderr)
            results.append({
                "clip":        clip,
                "output_path": None,
                "error":       str(e),
                "analytics":   {},
            })
        finally:
            if os.path.exists(trim_path):
                try: os.unlink(trim_path)
                except OSError: pass

    _p(1.0, f"Batch done: {sum(1 for r in results if not r.get('error'))}/{n} clips")
    return results
