"""
verticalize.py — AI Vertical Video Converter v7.5
═══════════════════════════════════════════════════════════════════
v7.5 CHANGES vs v7.4
• FIXED: process_sports_video NameError — captured _render_video return value.
• FIXED: Ball Kalman predict() now unconditional per frame after new_frame(),
  removed redundant manual predict() calls in color/Kalman fallback branches.
• FIXED: smooth_centers Kalman first-frame double-predict — skip _predict_step()
  on j==s (first frame of each segment).
• FIXED: BallKalmanFilter.predict() gravity runaway — clamped vertical velocity
  to heuristic terminal velocity.
• FIXED: Ball trail drift with moving crop window — trail points now stored in
  output-relative coordinates at detection time.
• FIXED: DynamicPanelSlotSmoother.update() 2-slot return inconsistency — now
  always returns List[List]; caller in _render_panel_frame simplified.
• FIXED: LayoutTransitionManager unbounded slot IDs — reuse smallest free
  non-negative integer instead of monotonic increment.
• FIXED: Stale MOT person detections in non-YOLO frames — pass empty lists to
  mot_tracker.update() during skipped frames so tracks predict forward.
• FIXED: Removed unused person_boxes_map parameter from _render_video signature
  and process_sports_video call site.
• FIXED: Removed unused det_scale parameter from _validate_ball_detection.

v7.4 CHANGES vs v7.3
• FIXED: resolve_target_size no longer clamps explicit presets to source dims,
  allowing intentional upscaling (e.g. 480p source → 1080p output).
• FIXED: calculate_crop_dims clarified with docstring — min() clamps are
  correct safety nets; upscaling from crop→output is handled by cv2.resize.
• FIXED: _open_ffmpeg_encoder now accepts source_fps and inserts an fps
  video filter when output_fps differs from source_fps, preventing
  duration/timing mismatches.
• FIXED: _render_video passes source_fps=fps to the encoder so that
  frame-rate conversion is handled correctly by FFmpeg.
• FIXED: _tracking_pass now returns the expected 5-tuple
  (centers, speeds, scene_cuts, persons_map, 0) to match caller unpacking.

v7.3 CHANGES vs v7.2
• FIXED: ResourceMonitor now tracks per-sample instantaneous CPU so cpu_max_pct
         is correctly distinct from cpu_avg_pct.
• FIXED: Scene-cut re-init in _sports_tracking_pass_optimized now calls proper
         .reset() / .init() methods instead of .__init__() to avoid wiping
         constructor arguments.
• FIXED: AdaptiveVelocityAwareSmoother gains a reset() method; scene cuts use
         it instead of re-constructing the object.
• FIXED: ball_trail_buf.clear() on scene cuts now fires BEFORE dissolve.on_cut,
         preventing stale trail points bleeding into the new shot.
• FIXED: detect_subjects coordinate conversion applied correctly when
         _cached_result is used with det_scale != 1.0.
• FIXED: smooth_centers Kalman path now calls _predict_step() before update()
         so the prior is advanced each frame.
• FIXED: PanelModeConfig clamps n_splits to 2 when > 2 so downstream code
         never receives an unsupported value.
• FIXED: _apply_sports_post_smooth spike-damping replaced np.roll with
         explicit shifted arrays, eliminating circular boundary artifacts.
• FIXED: _detect_panel_mode uses a single FFmpegVideoReader pass for probing
         instead of one seek per frame — dramatically faster on large files.
• IMPROVED: Replaced all print(..., file=sys.stderr) with stdlib logging.
• IMPROVED: BallFrameRecord made frozen (immutable) to prevent accidental aliasing.
• IMPROVED: Added OrigCoord / DetCoord type aliases for coordinate-space clarity.
• IMPROVED: _vignette_cache and _lut_cache capped at 32 entries (LRU-style eviction).
• IMPROVED: apply_sharpen uses a single cv2.filter2D kernel instead of two passes.
• IMPROVED: process_video / process_sports_video share a _common_pipeline_setup()
         helper, eliminating ~40 lines of duplicated boilerplate.
• IMPROVED: VELOCITY_SMOOTH_TABLE converted to sorted numpy arrays for O(log n)
         lookup via np.searchsorted.
• IMPROVED: ClipSegment validates that end_sec > start_sec on construction.
• IMPROVED: Added coordinate-type docstrings to key public functions.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import namedtuple, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum, auto

import cv2
import json
import numpy as np

# ── GPU device management ─────────────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

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

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# --- PyAV decode backend (optional) ------------------------------------------------
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

class PyAVVideoReader:
    '''PyAV-based video reader - faster than FFmpeg pipe for some codecs.'''
    def __init__(self, path: str):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            yield frame.to_ndarray(format="bgr24")

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Logging setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("verticalize")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ── Coordinate-space type aliases (documentation only) ────────────────────────
# OrigCoord: pixel in the original full-resolution frame space
# DetCoord:  pixel in the downscaled detection frame space
OrigCoord = Tuple[float, float]
DetCoord  = Tuple[float, float]

# ── Custom exception ──────────────────────────────────────────────────────────
class ProcessingError(Exception):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────
class PlayPhase(Enum):
    FAST_BREAK = auto()
    HALF_COURT = auto()
    REBOUND    = auto()
    STATIC     = auto()
    TRANSITION = auto()

class GameState(Enum):
    LIVE_PLAY  = auto()
    TIMEOUT    = auto()
    REPLAY     = auto()
    FREE_THROW = auto()
    UNKNOWN    = auto()

class TrackingStatus(Enum):
    ACTIVE   = auto()
    OCCLUDED = auto()
    LOST     = auto()


# ── Constants ─────────────────────────────────────────────────────────────────
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
    {"h": [10, 30],  "s": [40, 180], "v": [80,  220]},
    {"h": [35, 85],  "s": [40, 255], "v": [40,  220]},
    {"h": [90, 130], "s": [0,  60],  "v": [150, 255]},
]

KALMAN_PLAYER_PROCESS_NOISE_BASE  = 3e-2
KALMAN_PLAYER_PROCESS_NOISE_HIGH  = 3e-1
KALMAN_PLAYER_MEASUREMENT_NOISE   = 3e-2
KALMAN_BALL_PROCESS_NOISE_BASE    = 8.0
KALMAN_BALL_PROCESS_NOISE_HIGH    = 40.0
KALMAN_BALL_MEASUREMENT_NOISE     = 4.0
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
SPORTS_BALL_CONFIDENCE            = 0.30
SPORTS_BALL_PROXIMITY_PX          = 120
SPORTS_EVENT_EXPAND_FRAMES        = 15
SPORTS_EVENT_EXPAND_FACTOR        = 1.25

AVS_BASE_WINDOW_SEC          = 0.25
AVS_MAX_WINDOW_SEC           = 0.80
AVS_MIN_WINDOW_SEC           = 0.15
AVS_POLYORDER                = 3
AVS_VELOCITY_FAST_THRESHOLD  = 80.0
AVS_VELOCITY_SLOW_THRESHOLD  = 5.0
AVS_ACCEL_SPIKE_THRESHOLD    = 200.0
AVS_CONFIDENCE_LOW_THRESHOLD = 0.3

SPORTS_POST_SMOOTH_WINDOW_SEC = 0.50
SPORTS_POST_SMOOTH_EMA_ALPHA  = 0.12

ICS_LOOKAHEAD_SEC           = 0.30
ICS_FAST_BREAK_MARGIN_FACTOR = 1.35
ICS_SET_PLAY_MARGIN_FACTOR   = 1.05
ICS_BOUNDARY_ELASTICITY_PX   = 50
ICS_COURT_PRESERVE_RATIO     = 0.25

MOT_MAX_OCCLUSION_FRAMES = 30
MOT_IOU_MATCH_THRESHOLD  = 0.3
MOT_APPEARANCE_WEIGHT    = 0.3
MOT_MAX_TRACKS           = 15
MOT_MIN_TRACK_AGE        = 3

SPORTS_YOLO_SKIP_BASE      = 3
SPORTS_YOLO_SKIP_MAX       = 6
SPORTS_YOLO_SKIP_FASTBREAK = 2
BALL_TRACKER_TYPE          = "CSRT"
BALL_ROI_PAD_PX            = 40

BALL_ROI_MAX_AGE_FRAMES     = 25
BALL_ROI_CONF_DECAY         = 0.96
BALL_OVERLAY_CONF_THRESHOLD = 0.30
BALL_MIN_AREA_PX2           = 16
BALL_MAX_AREA_PX2           = 14400
BALL_MIN_ASPECT             = 0.40
BALL_MAX_ASPECT             = 2.50
BALL_MAX_GATE_PX            = 150
BALL_COLOR_MATCH_THRESHOLD  = 0.30
BALL_COLOR_MODEL_BUILD_FRAMES = 8

# Ball-blend weights (previously magic numbers scattered in the code)
BALL_BLEND_BALL_WEIGHT   = 0.55
BALL_BLEND_PLAYER_WEIGHT = 0.45
ROT_BIAS_WEIGHT          = 0.15   # rule-of-thirds blend

# Spike damping (post-smooth)
SPIKE_THRESH = 1.8
SPIKE_MIN_VEL = 3.0
SPIKE_DAMP    = 0.85

# ── Velocity smooth table — stored as parallel sorted numpy arrays ─────────────
_VST_SPEEDS  = np.array([0.0, 3.0, 8.0, 15.0, 30.0, 60.0, 120.0], dtype=np.float32)
_VST_WINDOWS = np.array([61,  53,  43,  33,   23,   15,   9],       dtype=np.int32)

def _vel_to_window(speed: float) -> int:
    """O(log n) lookup via binary search on the velocity table."""
    idx = int(np.searchsorted(_VST_SPEEDS, float(speed), side="right")) - 1
    idx = max(0, min(idx, len(_VST_WINDOWS) - 1))
    if idx < len(_VST_SPEEDS) - 1:
        v0, v1 = float(_VST_SPEEDS[idx]), float(_VST_SPEEDS[idx + 1])
        w0, w1 = int(_VST_WINDOWS[idx]), int(_VST_WINDOWS[idx + 1])
        frac = (speed - v0) / (v1 - v0 + 1e-9)
        w = int(w0 + frac * (w1 - w0))
        return w if w % 2 == 1 else w + 1
    return int(_VST_WINDOWS[idx])


RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Match source (no upscale)":    (0, 0),
    "1080p  (1080x1920 - Full HD)": (1080, 1920),
    "720p   (720x1280  - HD)":      (720,  1280),
    "540p   (540x960   - SD)":      (540,  960),
    "480p   (480x854   - Low)":     (480,  854),
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
    "None (keep original)": "",     "French": "fr",   "German": "de",
    "Spanish": "es",                "Italian": "it",   "Portuguese": "pt",
    "Dutch": "nl",                  "Polish": "pl",    "Russian": "ru",
    "Japanese": "ja",               "Korean": "ko",    "Chinese (Simplified)": "zh-CN",
    "Arabic": "ar",                 "Hindi": "hi",     "Turkish": "tr",
    "Indonesian": "id",             "Swedish": "sv",   "Norwegian": "no",
    "Danish": "da",                 "Finnish": "fi",   "Greek": "el",
    "Hebrew": "iw",                 "Thai": "th",      "Vietnamese": "vi",
    "Malay": "ms",                  "Ukrainian": "uk",
}

VIGNETTE_STRENGTH    = 0.55
VIGNETTE_FALLOFF     = 1.8
COLOR_GRADES         = ("none", "warm", "cool", "vibrant", "matte")
PANEL_SLOT_EMA       = 0.07
PANEL_SLOT_MAX_JUMP  = 0.08
KEN_BURNS_MAX_ZOOM   = 1.04
KEN_BURNS_PERIOD     = 8.0
DISSOLVE_FRAMES      = 3
PANEL_DIVIDER_PX     = 3
PANEL_DIVIDER_COLOR  = (15, 15, 15)
PANEL_CROP_EXPAND    = 1.15
PANEL_TRANSITION_FRAMES = 6
PANEL_MAX_SLOTS              = 4
PANEL_SPEAKER_FOCUS_RATIO    = 0.60
PANEL_HEAD_TARGET_FRAC       = 0.12
PANEL_LOWER_THIRD_HEIGHT_FRAC = 0.15
PANEL_PORTRAIT_HEAD_RATIO    = 2.5
PANEL_VELOCITY_DAMPING       = 0.85
PANEL_LAYOUT_SWITCH_COOLDOWN = 30

# Cache size limits to prevent unbounded memory growth
_CACHE_MAX_ENTRIES = 32


# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class OverlayConfig:
    """Ball-only overlay config. Person boxes are not rendered."""
    show_ball_box:         bool  = True
    ball_box_style:        str   = "corners"
    box_opacity:           float = 0.85
    min_ball_confidence:   float = BALL_OVERLAY_CONF_THRESHOLD
    show_confidence_label: bool  = False
    show_ball_trail:       bool  = True
    trail_length:          int   = 20

    def __post_init__(self) -> None:
        valid_ball = {"corners", "rect", "dot", "none"}
        if self.ball_box_style not in valid_ball:
            raise ValueError(f"ball_box_style must be one of {valid_ball}")
        self.box_opacity  = float(np.clip(self.box_opacity, 0.0, 1.0))
        self.trail_length = max(1, self.trail_length)


@dataclass
class PanelModeConfig:
    split_mode:           str   = "auto"
    n_splits:             int   = 2
    split_orientation:    str   = "horizontal"
    max_person_motion:    float = PANEL_MAX_PERSON_MOTION
    min_person_area_frac: float = PANEL_MIN_PERSON_AREA_FRAC
    max_count_variance:   float = PANEL_MAX_COUNT_VARIANCE
    stability_frac:       float = PANEL_STABILITY_FRAC
    # v5.0: Enhanced panel features
    layout_mode:          str   = "equal"       # equal | speaker_focus | solo_spotlight | auto
    speaker_focus_ratio:  float = PANEL_SPEAKER_FOCUS_RATIO
    head_normalize:       bool  = False
    lower_third_aware:    bool  = False
    portrait_mode:        bool  = False
    max_slots:            int   = PANEL_MAX_SLOTS

    def __post_init__(self) -> None:
        if self.split_mode not in ("auto", "force_on", "force_off"):
            raise ValueError("split_mode must be 'auto', 'force_on', or 'force_off'")
        if self.split_orientation not in ("horizontal", "vertical"):
            raise ValueError("split_orientation must be 'horizontal' or 'vertical'")
        if not (1 <= self.n_splits <= self.max_slots):
            raise ValueError(f"n_splits must be between 1 and {self.max_slots}")
        if self.n_splits > self.max_slots:
            logger.warning("n_splits=%d clamped to max_slots=%d", self.n_splits, self.max_slots)
            self.n_splits = self.max_slots
        if self.layout_mode not in ("equal", "speaker_focus", "solo_spotlight", "auto"):
            raise ValueError("layout_mode must be 'equal', 'speaker_focus', 'solo_spotlight', or 'auto'")


@dataclass
class Track:
    id:                int
    bbox:              Tuple[int, int, int, int]
    center:            Tuple[float, float]
    velocity:          Tuple[float, float]
    age:               int                = 0
    hits:              int                = 0
    time_since_update: int                = 0
    status:            TrackingStatus     = TrackingStatus.ACTIVE
    appearance:        Optional[np.ndarray] = None
    class_id:          int                = PERSON_CLASS_ID
    confidence:        float              = 0.0
    kalman_state:      np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
    kalman_covariance: np.ndarray = field(default_factory=lambda: np.eye(6))


@dataclass
class BallState:
    bbox:               Optional[Tuple[int, int, int, int]]
    center:             Optional[Tuple[float, float]]
    velocity:           Tuple[float, float]
    is_airborne:        bool              = False
    is_possessed:       bool              = False
    possessor_track_id: Optional[int]    = None
    bounce_count:       int              = 0
    airborne_frames:    int              = 0


class ClipSegment:
    def __init__(self, start_sec: float, end_sec: float, score: float,
                 soi_region: str = "center", peak_frame: int = 0, title: str = "") -> None:
        # FIXED: validate temporal order
        if end_sec <= start_sec:
            raise ValueError(
                f"end_sec ({end_sec:.2f}) must be greater than start_sec ({start_sec:.2f})"
            )
        self.start_sec  = start_sec
        self.end_sec    = end_sec
        self.score      = score
        self.soi_region = soi_region
        self.peak_frame = peak_frame
        self.title      = title
        self.duration   = end_sec - start_sec

    def __repr__(self) -> str:
        return f"<Clip {self.start_sec:.1f}s-{self.end_sec:.1f}s score={self.score:.2f}>"


@dataclass(frozen=True)  # FIXED: immutable to prevent accidental aliasing across frames
class BallFrameRecord:
    """Per-frame ball data — immutable; never shared/stale across frames."""
    bbox:       Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    source:     str   = "none"   # yolo | roi | kalman | color | none


# ── Detection result cache ────────────────────────────────────────────────────
class DetectionCache:
    __slots__ = ("persons", "ball_box", "ball_carrier", "det_result",
                 "frame_idx", "confidences", "ball_confidence")

    def __init__(self) -> None:
        self.persons:         List[Tuple[int, int, int, int]] = []
        self.ball_box:        Optional[Tuple[int, int, int, int]] = None
        self.ball_carrier:    int   = -1
        self.det_result:      Optional[Any] = None
        self.frame_idx:       int   = -1
        self.confidences:     List[float] = []
        self.ball_confidence: float = 0.0

    def update(self, persons, ball_box, ball_carrier, det_result, confidences, fi,
               ball_confidence: float = 0.0) -> None:
        self.persons      = persons
        self.ball_box     = ball_box
        self.ball_carrier = ball_carrier
        self.det_result   = det_result
        self.confidences  = confidences
        self.frame_idx    = fi
        self.ball_confidence = ball_confidence

    def reset(self) -> None:
        self.persons      = []
        self.ball_box     = None
        self.ball_carrier = -1
        self.det_result   = None
        self.frame_idx    = -1
        self.confidences  = []
        self.ball_confidence = 0.0


# ── Ball Kalman Filter ────────────────────────────────────────────────────────
class BallKalmanFilter:
    """
    4-state Kalman filter [cx, cy, vx, vy] operating in det_frame pixel space.
    All coordinates are DetCoord (detection-frame pixels).
    """
    def __init__(self, fps: float, frame_h: int) -> None:
        self.fps     = max(fps, 1.0)
        self.frame_h = frame_h
        dt           = 1.0 / self.fps
        self.dt      = dt
        self.gravity = GRAVITY_PIXELS_PER_SEC2_BASE * (frame_h / 1080.0)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        pn      = KALMAN_BALL_PROCESS_NOISE_BASE
        self.Q  = np.diag([pn, pn, pn * 20.0, pn * 20.0]).astype(np.float64)
        self.R  = np.eye(2, dtype=np.float64) * KALMAN_BALL_MEASUREMENT_NOISE
        self.x  = np.zeros((4, 1), dtype=np.float64)
        self.P  = np.eye(4, dtype=np.float64) * 100.0
        self.initialized             = False
        self._stale_frames           = 0
        self._predicted_this_frame   = False

    def _reset_state(self) -> None:
        """Internal: reset matrices without touching fps/frame_h."""
        self.x  = np.zeros((4, 1), dtype=np.float64)
        self.P  = np.eye(4, dtype=np.float64) * 100.0
        self.initialized           = False
        self._stale_frames         = 0
        self._predicted_this_frame = False

    def init(self, cx: float, cy: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self.x = np.array([[cx], [cy], [vx], [vy]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.initialized           = True
        self._stale_frames         = 0
        self._predicted_this_frame = False

    def reset(self) -> None:
        """Reset to uninitialised state, preserving fps and frame_h."""
        self._reset_state()

    def new_frame(self) -> None:
        self._predicted_this_frame = False

    def predict(self) -> Tuple[float, float]:
        if not self.initialized:
            return float(self.x[0, 0]), float(self.x[1, 0])
        if self._predicted_this_frame:
            return float(self.x[0, 0]), float(self.x[1, 0])
        self.x = self.F @ self.x
        self.x[3, 0] += self.gravity * self.dt
        # FIXED: clamp vertical velocity to prevent gravity runaway
        terminal_vy = 1600.0 * (self.frame_h / 1080.0)
        self.x[3, 0] = float(np.clip(self.x[3, 0], -terminal_vy, terminal_vy))
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._stale_frames += 1
        self._predicted_this_frame = True
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        if not self.initialized:
            self.init(cx, cy)
            self._predicted_this_frame = False
            return cx, cy
        if not self._predicted_this_frame:
            self.predict()
        z = np.array([[cx], [cy]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P
        self._stale_frames         = 0
        self._predicted_this_frame = False
        return float(self.x[0, 0]), float(self.x[1, 0])

    def gate_distance(self, cx: float, cy: float) -> float:
        if not self.initialized:
            return 0.0
        return math.hypot(cx - float(self.x[0, 0]), cy - float(self.x[1, 0]))

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def stale_frames(self) -> int:
        return self._stale_frames


# ── Ball Color Model ──────────────────────────────────────────────────────────
class BallColorModel:
    """HSV histogram appearance model for the ball."""

    def __init__(self, n_build: int = BALL_COLOR_MODEL_BUILD_FRAMES) -> None:
        self._n_build  = max(1, min(n_build, 3))
        self._samples: List[np.ndarray] = []
        self._model:   Optional[np.ndarray] = None
        self._bins     = [8, 8, 4]

    def _extract_hist(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return np.zeros(int(np.prod(self._bins)), dtype=np.float32)
        hsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self._bins,
                            [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten().astype(np.float32)

    def add_sample(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        if self._model is not None:
            return
        x1, y1, x2, y2 = bbox
        patch = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if patch.size == 0:
            return
        self._samples.append(self._extract_hist(patch))
        if len(self._samples) >= self._n_build:
            self._model = np.mean(self._samples, axis=0).astype(np.float32)

    def match(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Returns correlation [0-1]; 1.0 = perfect match.
        Returns 1.0 (pass-through) when model not yet built.
        """
        if self._model is None:
            return 1.0
        x1, y1, x2, y2 = bbox
        patch = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if patch.size == 0:
            return 0.0
        h     = self._extract_hist(patch)
        score = float(cv2.compareHist(self._model, h, cv2.HISTCMP_CORREL))
        return max(0.0, score)

    def reset(self) -> None:
        self._samples = []
        self._model   = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None


# ── Ball Color Detector ───────────────────────────────────────────────────────
class BallColorDetector:
    """HSV contour fallback when YOLO and ROI tracker both fail."""

    _BALL_HSV_RANGES = [
        (np.array([5,  100, 100], np.uint8), np.array([20, 255, 255], np.uint8)),
        (np.array([0,  0,   200], np.uint8), np.array([180, 30, 255], np.uint8)),
        (np.array([25, 100, 100], np.uint8), np.array([45, 255, 255], np.uint8)),
        (np.array([0,  80,  50],  np.uint8), np.array([15, 200, 180], np.uint8)),
    ]

    def __init__(self, color_model: BallColorModel) -> None:
        self._model = color_model

    def detect(self, frame: np.ndarray,
               search_center: Optional[Tuple[float, float]],
               search_radius: int = 120,
               expected_area: float = 400.0) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        if search_center is not None:
            scx, scy = int(search_center[0]), int(search_center[1])
            rx1 = max(0, scx - search_radius); ry1 = max(0, scy - search_radius)
            rx2 = min(w, scx + search_radius); ry2 = min(h, scy + search_radius)
            roi = frame[ry1:ry2, rx1:rx2]
            ox, oy = rx1, ry1
        else:
            roi = frame; ox = oy = 0

        if roi.size == 0:
            return None

        hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self._BALL_HSV_RANGES:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

        k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_bbox, best_score = None, -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < BALL_MIN_AREA_PX2 or area > BALL_MAX_AREA_PX2:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(bw, 1)
            if aspect < BALL_MIN_ASPECT or aspect > BALL_MAX_ASPECT:
                continue
            size_score  = 1.0 - abs(area - expected_area) / max(expected_area, 1.0)
            abs_x1 = ox + x; abs_y1 = oy + y
            abs_x2 = abs_x1 + bw; abs_y2 = abs_y1 + bh
            color_score = self._model.match(frame, (abs_x1, abs_y1, abs_x2, abs_y2))
            score       = size_score * 0.4 + color_score * 0.6
            if score > best_score:
                best_score = score; best_bbox = (abs_x1, abs_y1, abs_x2, abs_y2)

        if best_bbox is not None and best_score >= BALL_COLOR_MATCH_THRESHOLD:
            return best_bbox
        return None


# ── Ball ROI Tracker ──────────────────────────────────────────────────────────
class BallROITracker:
    """Ball ROI tracker with graceful fallback when cv2.Tracker is unavailable."""

    def __init__(self, tracker_type: str = BALL_TRACKER_TYPE,
                 max_age: int = BALL_ROI_MAX_AGE_FRAMES) -> None:
        self._type            = tracker_type
        self._tracker:        Optional[Any] = None
        self._bbox_det:       Optional[Tuple[int, int, int, int]] = None
        self._lost            = True
        self._age             = 0
        self._max_age         = max_age
        self._last_velocity:  Tuple[float, float] = (0.0, 0.0)
        self._last_conf:      float = 0.0
        self._has_cv_tracker  = (
            hasattr(cv2, "TrackerCSRT_create") or
            (hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"))
        )

    def _make_tracker(self) -> Optional[Any]:
        if not self._has_cv_tracker:
            return None
        try:
            # Try modern API first, then legacy namespace (OpenCV 4.5+)
            if self._type == "CSRT":
                if hasattr(cv2, "TrackerCSRT_create"):
                    return cv2.TrackerCSRT_create()
                elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                    return cv2.legacy.TrackerCSRT_create()
            if self._type == "KCF":
                if hasattr(cv2, "TrackerKCF_create"):
                    return cv2.TrackerKCF_create()
                elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                    return cv2.legacy.TrackerKCF_create()
            # Default fallback
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
            return None
        except (cv2.error, AttributeError, ValueError):
            self._has_cv_tracker = False
            return None

    def init(self, det_frame: np.ndarray, bbox_det: Tuple[int, int, int, int],
             velocity: Tuple[float, float] = (0.0, 0.0), confidence: float = 1.0) -> None:
        x1, y1, x2, y2 = bbox_det
        bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
        self._tracker = self._make_tracker()
        if self._tracker is not None:
            try:
                self._tracker.init(det_frame, (x1, y1, bw, bh))
            except (cv2.error, ValueError):
                self._tracker = None; self._has_cv_tracker = False
        self._bbox_det      = bbox_det
        self._lost          = False
        self._age           = 0
        self._last_velocity = velocity
        self._last_conf     = confidence

    def update(self, det_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self._lost or self._age >= self._max_age:
            self._lost = True; return None

        if self._tracker is not None:
            try:
                ok, (x, y, bw, bh) = self._tracker.update(det_frame)
                if not ok or bw < 2 or bh < 2:
                    self._lost = True; return None
                self._age          += 1
                self._last_conf    *= BALL_ROI_CONF_DECAY
                self._bbox_det      = (int(x), int(y), int(x + bw), int(y + bh))
                return self._bbox_det
            except (cv2.error, ValueError):
                self._tracker = None; self._has_cv_tracker = False

        # Fallback: shift by last known velocity
        if self._bbox_det is not None:
            vx, vy = self._last_velocity
            x1, y1, x2, y2 = self._bbox_det
            nx1, ny1 = int(x1 + vx), int(y1 + vy)
            nx2, ny2 = int(x2 + vx), int(y2 + vy)
            fh, fw   = det_frame.shape[:2]
            if nx1 < 0 or ny1 < 0 or nx2 > fw or ny2 > fh:
                self._lost = True; return None
            self._bbox_det  = (nx1, ny1, nx2, ny2)
            self._age      += 1
            self._last_conf *= BALL_ROI_CONF_DECAY
            return self._bbox_det

        self._lost = True
        return None

    def reset(self) -> None:
        self._tracker   = None
        self._bbox_det  = None
        self._lost      = True
        self._age       = 0
        self._last_conf = 0.0

    @property
    def is_active(self) -> bool:
        return not self._lost and self._age < self._max_age

    @property
    def age(self) -> int:
        return self._age

    @property
    def confidence(self) -> float:
        return self._last_conf


# ── Resource Monitor ──────────────────────────────────────────────────────────
class ResourceMonitor:
    """
    CPU/RAM tracker using cpu_times() per sample.
    FIXED: cpu_max_pct now tracks the maximum per-interval instantaneous value,
    not a duplicate of cpu_avg_pct.
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self._stop_event  = threading.Event()
        self._thread:     Optional[threading.Thread] = None
        self._lock        = threading.Lock()
        # Each sample: (wall_time, cumulative_cpu_sec, ram_mb)
        self._samples:    List[Tuple[float, float, float]] = []
        self._cpu_cores   = max(1, psutil.cpu_count()) if _PSUTIL_AVAILABLE else 1
        self._parent_proc = psutil.Process() if _PSUTIL_AVAILABLE else None
        self._active      = False
        self._last_report: Dict[str, float] = {}

    def start(self) -> None:
        if self._active or not _PSUTIL_AVAILABLE or self._parent_proc is None:
            return
        self._stop_event.clear()
        self._samples     = []
        self._last_report = {}
        self._active      = True
        self._thread      = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._active:
            self._active = False
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=self.interval_sec + 0.5)
            self._thread = None
        self._last_report = self._build_report()

    def get_stats(self) -> Dict[str, float]:
        return self._last_report

    def _collect_loop(self) -> None:
        while not self._stop_event.is_set():
            snap = self._sample()
            with self._lock:
                self._samples.append(snap)
            remaining = self.interval_sec
            while remaining > 0 and not self._stop_event.is_set():
                chunk = min(0.05, remaining)
                time.sleep(chunk)
                remaining -= chunk

    def _sample(self) -> Tuple[float, float, float]:
        wall    = time.monotonic()
        cpu_sec = 0.0
        ram_mb  = 0.0
        if self._parent_proc is None:
            return (wall, cpu_sec, ram_mb)
        try:
            seen_pids = {self._parent_proc.pid}
            with self._parent_proc.oneshot():
                pt      = self._parent_proc.cpu_times()
                cpu_sec += pt.user + pt.system
                ram_mb  += self._parent_proc.memory_info().rss / (1024 * 1024)
            for child in self._parent_proc.children(recursive=True):
                if child.pid in seen_pids:
                    continue
                seen_pids.add(child.pid)
                try:
                    with child.oneshot():
                        ct      = child.cpu_times()
                        cpu_sec += ct.user + ct.system
                        ram_mb  += child.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return (wall, cpu_sec, ram_mb)

    def _build_report(self) -> Dict[str, float]:
        with self._lock:
            samples = list(self._samples)
        n = len(samples)
        if n < 2 or not _PSUTIL_AVAILABLE:
            return {"cpu_avg_pct": 0.0, "cpu_max_pct": 0.0,
                    "ram_avg_mb": 0.0, "ram_max_mb": 0.0,
                    "processing_time_sec": 0.0}

        wall_sec = samples[-1][0] - samples[0][0]
        cpu_total_delta = samples[-1][1] - samples[0][1]
        cpu_avg = (
            max(0.0, min(100.0, (cpu_total_delta / wall_sec / self._cpu_cores) * 100.0))
            if wall_sec > 0 else 0.0
        )

        # FIXED: compute per-interval instantaneous CPU for max tracking
        cpu_max = 0.0
        for i in range(1, n):
            dwall = samples[i][0] - samples[i-1][0]
            dcpu  = samples[i][1] - samples[i-1][1]
            if dwall > 0:
                inst = max(0.0, min(100.0, (dcpu / dwall / self._cpu_cores) * 100.0))
                if inst > cpu_max:
                    cpu_max = inst

        ram_values = [s[2] for s in samples]
        return {
            "cpu_avg_pct":         round(cpu_avg, 1),
            "cpu_max_pct":         round(cpu_max, 1),
            "ram_avg_mb":          round(sum(ram_values) / n, 1),
            "ram_max_mb":          round(max(ram_values), 1),
            "processing_time_sec": round(wall_sec, 2),
        }


# ── Availability helpers ──────────────────────────────────────────────────────
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

def yolo_available() -> bool:
    if not _YOLO_AVAILABLE:
        return False
    try:
        import urllib.request
        urllib.request.urlopen("https://github.com", timeout=3)
        return True
    except Exception:
        return (os.path.exists("yolov8n.pt") or
                os.path.exists("yolov8s.pt") or
                os.path.exists("yolo11n.pt"))


# ── Visual Effects ────────────────────────────────────────────────────────────
# FIXED: bounded LRU-style caches to prevent unbounded memory growth
_vignette_cache: Dict[Tuple, np.ndarray] = {}
_vignette_insert_order: List[Tuple]      = []

def _build_vignette(w: int, h: int, strength: float = VIGNETTE_STRENGTH,
                    falloff: float = VIGNETTE_FALLOFF) -> np.ndarray:
    key = (w, h, round(strength, 3), round(falloff, 3))
    if key in _vignette_cache:
        return _vignette_cache[key]
    # Evict oldest if at capacity
    if len(_vignette_cache) >= _CACHE_MAX_ENTRIES:
        old = _vignette_insert_order.pop(0)
        _vignette_cache.pop(old, None)
    xs   = np.linspace(-1, 1, w, dtype=np.float32)
    ys   = np.linspace(-1, 1, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.sqrt(xg**2 + yg**2)
    dist /= dist.max()
    mask = np.clip(1.0 - strength * (dist**falloff), 0.0, 1.0)[:, :, np.newaxis]
    _vignette_cache[key] = mask
    _vignette_insert_order.append(key)
    return mask

def apply_vignette(frame: np.ndarray, strength: float = VIGNETTE_STRENGTH) -> np.ndarray:
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    mask = _build_vignette(w, h, strength)
    # FIXED: proper stale-cache eviction when dimensions don't match.
    # The cache key includes (w, h) so a true key hit should always match,
    # but if the cache was corrupted we force a rebuild.
    if mask.shape[:2] != (h, w) or mask.shape[2] != 1:
        stale_keys = [k for k in list(_vignette_cache.keys()) if k[0] == w and k[1] == h]
        for sk in stale_keys:
            _vignette_cache.pop(sk, None)
            if sk in _vignette_insert_order:
                _vignette_insert_order.remove(sk)
        mask = _build_vignette(w, h, strength)
    return (frame.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

# FIXED: single-pass sharpening kernel instead of two-pass GaussianBlur + addWeighted
_SHARPEN_KERNEL_CACHE: Dict[Tuple[float, int], np.ndarray] = {}

def _make_sharpen_kernel(strength: float, radius: int) -> np.ndarray:
    key = (round(strength, 3), radius)
    if key in _SHARPEN_KERNEL_CACHE:
        return _SHARPEN_KERNEL_CACHE[key]
    size = radius * 2 + 1
    k    = -strength * np.ones((size, size), dtype=np.float32) / (size * size - 1)
    k[radius, radius] = 1.0 + strength
    _SHARPEN_KERNEL_CACHE[key] = k
    return k

def apply_sharpen(frame: np.ndarray, strength: float = 0.6, radius: int = 1) -> np.ndarray:
    if strength <= 0:
        return frame
    return cv2.filter2D(frame, -1, _make_sharpen_kernel(strength, radius))


_lut_cache: Dict[str, np.ndarray] = {}

def _build_lut(grade: str) -> np.ndarray:
    if grade in _lut_cache:
        return _lut_cache[grade]
    x = np.arange(256, dtype=np.float32)
    if grade == "warm":
        r = np.clip(x * 1.06 + 5, 0, 255); g = np.clip(x * 1.02 + 2, 0, 255)
        b = np.clip(x * 0.92 - 4, 0, 255)
    elif grade == "cool":
        r = np.clip(x * 0.92 - 4, 0, 255); g = np.clip(x * 1.01 + 1, 0, 255)
        b = np.clip(x * 1.07 + 6, 0, 255)
    elif grade == "vibrant":
        def _sc(v: np.ndarray) -> np.ndarray:
            n = v / 255; s = n * n * (3 - 2 * n)
            return np.clip((n * 0.6 + s * 0.4) * 255, 0, 255)
        r = _sc(x * 1.04); g = _sc(x * 1.02); b = _sc(x)
    elif grade == "matte":
        r = np.clip(x * 0.88 + 18, 0, 255); g = np.clip(x * 0.86 + 16, 0, 255)
        b = np.clip(x * 0.84 + 22, 0, 255)
    else:
        r = g = b = x.copy()
    lut = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(256, 1, 3)
    # LRU eviction for lut cache
    if len(_lut_cache) >= _CACHE_MAX_ENTRIES:
        _lut_cache.pop(next(iter(_lut_cache)))
    _lut_cache[grade] = lut
    return lut

def apply_color_grade(frame: np.ndarray, grade: str = "none") -> np.ndarray:
    if not grade or grade == "none":
        return frame
    return cv2.LUT(frame, _build_lut(grade))


# ── Overlay rendering ─────────────────────────────────────────────────────────
def _draw_tracking_overlays(
    frame: np.ndarray,
    ball_record: Optional[BallFrameRecord],
    overlay_cfg: Optional[OverlayConfig] = None,
    ball_trail: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Draw ball box and optional trail. Person boxes are NOT rendered."""
    cfg = overlay_cfg or OverlayConfig()
    out = frame.copy()
    h, w = out.shape[:2]

    if cfg.show_ball_trail and ball_trail and len(ball_trail) >= 2:
        n_trail = len(ball_trail)
        for i in range(1, n_trail):
            trail_alpha = int(255 * (i / n_trail) * 0.7)
            radius      = max(2, 4 - (n_trail - i) // 4)
            cv2.circle(out, ball_trail[i], radius,
                       (0, trail_alpha, 255 - trail_alpha), -1, cv2.LINE_AA)

    if (cfg.show_ball_box and ball_record is not None
            and ball_record.bbox is not None
            and ball_record.confidence >= cfg.min_ball_confidence):
        bx1, by1, bx2, by2 = ball_record.bbox
        bx1c = max(0, bx1); by1c = max(0, by1)
        bx2c = min(w-1, bx2); by2c = min(h-1, by2)
        if bx2c > bx1c and by2c > by1c:
            src = ball_record.source
            color, alpha = {
                "yolo":   ((0, 230, 255), 1.0),
                "roi":    ((0, 180, 220), 0.85),
                "color":  ((0, 210, 100), 0.75),
                "kalman": ((80, 80, 220), 0.55),
            }.get(src, ((80, 140, 180), 0.50))

            style = cfg.ball_box_style
            if style == "rect":
                if alpha < 1.0:
                    layer = out.copy()
                    cv2.rectangle(layer, (bx1c, by1c), (bx2c, by2c), color, 2, cv2.LINE_AA)
                    # FIXED: avoid dst aliasing in addWeighted — write to temp then copy
                    blended = cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0)
                    out[:] = blended
                else:
                    cv2.rectangle(out, (bx1c, by1c), (bx2c, by2c), color, 2, cv2.LINE_AA)
            elif style == "dot":
                bcx = (bx1c+bx2c)//2; bcy = (by1c+by2c)//2
                r   = max(4, (bx2c-bx1c)//4)
                if alpha < 1.0:
                    layer = out.copy()
                    cv2.circle(layer, (bcx, bcy), r, color, -1, cv2.LINE_AA)
                    blended = cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0)
                    out[:] = blended
                else:
                    cv2.circle(out, (bcx, bcy), r, color, -1, cv2.LINE_AA)
            elif style in ("corners", "none"):
                if style == "corners":
                    clen  = max(6, min((bx2c-bx1c)//4, (by2c-by1c)//4, 18))
                    if alpha < 1.0:
                        layer = out.copy()
                        for (px, py, dx, dy) in [
                            (bx1c, by1c,  clen,  clen), (bx2c, by1c, -clen,  clen),
                            (bx1c, by2c,  clen, -clen), (bx2c, by2c, -clen, -clen),
                        ]:
                            cv2.line(layer, (px, py), (px+dx, py), color, 2, cv2.LINE_AA)
                            cv2.line(layer, (px, py), (px, py+dy), color, 2, cv2.LINE_AA)
                        bcx = (bx1c+bx2c)//2; bcy = (by1c+by2c)//2
                        cv2.circle(layer, (bcx, bcy), max(2, (bx2c-bx1c)//6),
                                   color, -1, cv2.LINE_AA)
                        blended = cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0)
                        out[:] = blended
                    else:
                        for (px, py, dx, dy) in [
                            (bx1c, by1c,  clen,  clen), (bx2c, by1c, -clen,  clen),
                            (bx1c, by2c,  clen, -clen), (bx2c, by2c, -clen, -clen),
                        ]:
                            cv2.line(out, (px, py), (px+dx, py), color, 2, cv2.LINE_AA)
                            cv2.line(out, (px, py), (px, py+dy), color, 2, cv2.LINE_AA)
                        bcx = (bx1c+bx2c)//2; bcy = (by1c+by2c)//2
                        cv2.circle(out, (bcx, bcy), max(2, (bx2c-bx1c)//6),
                                   color, -1, cv2.LINE_AA)

            if cfg.show_confidence_label and ball_record.confidence > 0:
                label = f"{ball_record.source[0].upper()}{ball_record.confidence:.2f}"
                cv2.putText(out, label, (bx1c, max(0, by1c - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def apply_ken_burns(frame: np.ndarray, frame_idx: int, fps: float,
                    max_zoom: float = KEN_BURNS_MAX_ZOOM,
                    period: float = KEN_BURNS_PERIOD) -> np.ndarray:
    if max_zoom <= 1.0:
        return frame
    t     = (frame_idx / max(fps, 1)) % period
    scale = 1.0 + (max_zoom - 1.0) * 0.5 * (1 - math.cos(2 * math.pi * t / period))
    if abs(scale - 1.0) < 1e-4:
        return frame
    h, w = frame.shape[:2]
    nw = max(int(w / scale), 2); nh = max(int(h / scale), 2)
    x0 = (w - nw) // 2; y0 = (h - nh) // 2
    return cv2.resize(frame[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LINEAR)


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
        # Guard: if stored frame shape differs (e.g. first frame after resize change),
        # resize the buffer to match rather than crashing.
        if self._buf.shape != new_frame.shape:
            self._buf = cv2.resize(self._buf,
                                   (new_frame.shape[1], new_frame.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        # FIXED: ensure addWeighted result is captured (OpenCV 4.13 may have issues
        # with in-place dst; we return the new array directly)
        result = cv2.addWeighted(self._buf, alpha, new_frame, 1.0 - alpha, 0)
        return result

    @property
    def active(self) -> bool:
        return self._rem > 0


# ── FFmpeg Utilities ──────────────────────────────────────────────────────────
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
    def __init__(self, path: str, width: int, height: int,
                 seek_sec: float = 0.0, n_frames: Optional[int] = None,
                 scale_w: Optional[int] = None, scale_h: Optional[int] = None) -> None:
        self.path         = path
        self.width        = width
        self.height       = height
        self.seek_sec     = seek_sec
        self.n_frames     = n_frames
        self.out_w        = scale_w or width
        self.out_h        = scale_h or height
        self._proc:       Optional[subprocess.Popen] = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover    = b""

    def _build_cmd(self, extra: List[str]) -> List[str]:
        cmd = ["ffmpeg"]
        if self.seek_sec > 0:
            cmd += ["-ss", str(self.seek_sec)]
        cmd += extra + ["-i", self.path, "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-vf", f"scale={self.out_w}:{self.out_h}"]
        if self.n_frames is not None:
            cmd += ["-vframes", str(self.n_frames)]
        cmd += ["pipe:1"]
        return cmd

    def _open(self) -> None:
        buf_size  = max(self._frame_bytes * 16, 1 << 22)
        last_err: Optional[Exception] = None
        for extra in ([], ["-hwaccel", "none"]):
            try:
                proc = subprocess.Popen(
                    self._build_cmd(extra),
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=buf_size,
                )
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc     = proc
                    self._leftover = test
                    return
                try: proc.stdout.close()
                except (OSError, subprocess.SubprocessError): pass
                try: proc.wait(timeout=2)
                except (OSError, subprocess.SubprocessError): proc.kill()
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                last_err = exc  # FIXED: preserve last error instead of silently swallowing

        raise ProcessingError(
            f"FFmpeg could not decode: {self.path}"
            + (f" — {last_err}" if last_err else "")
        )

    def close(self) -> None:
        if self._proc:
            try: self._proc.stdout.close()
            except (OSError, subprocess.SubprocessError): pass
            try: self._proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError): self._proc.kill()
            self._proc = None

    def __enter__(self) -> "FFmpegVideoReader":
        self._open(); return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __iter__(self):
        if not self._proc:
            self._open()
        buf = self._leftover; self._leftover = b""
        while True:
            needed = self._frame_bytes - len(buf)
            while needed > 0:
                chunk = self._proc.stdout.read(needed)
                if not chunk:
                    return
                buf   += chunk
                needed -= len(chunk)
            yield np.frombuffer(buf[:self._frame_bytes], dtype=np.uint8).reshape(
                self.out_h, self.out_w, 3)
            buf = buf[self._frame_bytes:]


def _read_frame_at(path: str, width: int, height: int, t_sec: float,
                   scale_w: Optional[int] = None,
                   scale_h: Optional[int] = None) -> Optional[np.ndarray]:
    try:
        r = FFmpegVideoReader(path, width, height, seek_sec=t_sec, n_frames=1,
                              scale_w=scale_w, scale_h=scale_h)
        r._open()
        frames = list(r); r.close()
        return frames[0] if frames else None
    except Exception:
        return None

def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{tool} not found. Install FFmpeg.")

_AUDIO_CACHE: Dict[str, bool] = {}

def _has_audio(path: str) -> bool:
    if path in _AUDIO_CACHE:
        return _AUDIO_CACHE[path]
    try:
        info = get_video_info(path)
        result = bool(info.get("has_audio", False))
    except Exception:
        result = False
    _AUDIO_CACHE[path] = result
    return result

def _extract_audio_wav(vpath: str, wpath: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", vpath, "-ar", "16000", "-ac", "1", "-f", "wav", wpath],
        capture_output=True, timeout=300)
    return r.returncode == 0 and os.path.exists(wpath)

def _trim_video(inp: str, out: str, start: float, end: float,
                crf: int = 18, preset: str = "ultrafast") -> bool:
    """FIXED: crf/preset forwarded from caller instead of hardcoded."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-hwaccel", "none",
         "-ss", str(start), "-to", str(end), "-i", inp,
         "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
         "-c:a", "aac", "-b:a", "128k",
         "-avoid_negative_ts", "make_zero", "-reset_timestamps", "1", out],
        capture_output=True)
    return r.returncode == 0 and os.path.exists(out)

def _open_ffmpeg_encoder(output_path: str, width: int, height: int, fps: float,
                         audio_source: Optional[str], crf: int = 23, preset: str = "fast",
                         audio_bitrate: str = "128k",
                         subtitle_path: Optional[str] = None,
                         subtitle_style: Optional[Dict[str, Any]] = None,
                         extra_vf: Optional[List[str]] = None,
                         source_fps: Optional[float] = None) -> subprocess.Popen:
    # FIXED: use source_fps for the raw input stream rate, fps for output
    input_fps = source_fps if source_fps is not None else fps
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(input_fps), "-i", "pipe:0",
    ]
    has_aud = bool(audio_source and _has_audio(audio_source))
    if has_aud:
        cmd += ["-hwaccel", "none", "-i", audio_source]
    vf: List[str] = []
    # FIXED: insert fps filter when source and output frame rates differ
    if source_fps is not None and abs(source_fps - fps) > 0.01:
        vf.append(f"fps={fps}")
    if subtitle_path and os.path.exists(subtitle_path):
        s    = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        sesc = subtitle_path.replace("\\", "/").replace(":", r"\:")
        force = (
            f"Fontsize={s.get('fontsize',18)}, "
            f"PrimaryColour={s.get('primary_color','&H00FFFFFF')}, "
            f"OutlineColour={s.get('outline_color','&H00000000')}, "
            f"Outline={s.get('outline',2)},Bold={s.get('bold',1)}, "
            f"Shadow={s.get('shadow',0)},BackColour={s.get('back_color','&H00000000')}, "
            f"MarginV={s.get('margin_v',80)},Alignment=2 "
        )
        vf.append(f"subtitles='{sesc}':force_style='{force}'")
    if extra_vf:
        vf.extend(extra_vf)
    cmd += ["-map", "0:v:0"]
    if has_aud:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2",
                "-shortest"]
    else:
        cmd += ["-an"]
    if vf:
        cmd += ["-vf", ", ".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)

def _close_ffmpeg_encoder(proc: subprocess.Popen, output_path: str) -> None:
    try: proc.stdin.close()
    except (OSError, subprocess.SubprocessError): pass
    stderr_buf: List[bytes] = []
    def _drain():
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk: break
                stderr_buf.append(chunk)
        except (OSError, ValueError): pass
    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    try: proc.wait(timeout=180)
    except (OSError, subprocess.SubprocessError): proc.kill(); proc.wait()
    t.join(timeout=5)
    if proc.returncode != 0:
        err = b"".join(stderr_buf).decode(errors="replace")
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")

def get_video_info(path: str) -> Dict[str, Any]:
    """JSON-based ffprobe parsing — robust against locale/format changes."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        path,
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    data = json.loads(r.stdout)
    video_stream = next(
        s for s in data["streams"]
        if s.get("codec_type") == "video"
    )
    w = int(video_stream["width"])
    h = int(video_stream["height"])
    fps_raw = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_raw.split("/")
    fps = float(num) / max(float(den), 1.0)
    duration = float(
        video_stream.get("duration", 0)
        or data["format"].get("duration", 0)
    )
    has_audio = any(
        s.get("codec_type") == "audio"
        for s in data["streams"]
    )
    if w == 0 or h == 0:
        raise ProcessingError(f"Cannot read dimensions: {path}")
    return {
        "fps":              fps,
        "total_frames":     min(int(duration * fps), MAX_FRAMES_GUARD),
        "width":            w,
        "height":           h,
        "duration_seconds": duration,
        "is_landscape":     w >= h,
        "has_audio":        has_audio,
    }

def extract_thumbnail(path: str, t: float = 1.0) -> Optional[bytes]:
    info  = get_video_info(path)
    frame = _read_frame_at(path, info["width"], info["height"], t, scale_w=320, scale_h=180)
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None

def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    """
    Resolve the target output size from a preset label.

    'Match source' mode caps to source dims (no upscale).
    Explicit presets (e.g. '1080p') honor the requested size even if
    it exceeds source dims — upscaling is handled by cv2.resize at
    the render step.
    """
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        # "Match source" — derive 9:16 crop from source, never upscale
        cw = int(orig_h * 9 / 16)
        if cw > orig_w: cw = orig_w
        ch = int(cw * 16 / 9)
        # Clamp to source dims for match-source only
        if ch > orig_h:
            scale = orig_h / ch; cw = int(cw * scale); ch = int(orig_h)
        if cw > orig_w:
            scale = orig_w / cw; cw = int(orig_w); ch = int(ch * scale)
    else:
        # Explicit preset — honor the requested output size (upscale OK)
        cw, ch = tw, th
    return max(cw - (cw % 2), 2), max(ch - (ch % 2), 2)

def calculate_crop_dims(orig_w: int, orig_h: int, tw: int, th: int) -> Tuple[int, int]:
    """
    Compute the crop rectangle dimensions from the source frame.

    The min(..., orig_w/orig_h) clamps are intentional safety nets:
    a crop region cannot exceed the source frame.  Upscaling from the
    cropped region to the target output size is handled downstream by
    cv2.resize in the render step.
    """
    th    = max(th, 2)
    ratio = tw / th
    if (orig_w / orig_h) > ratio:
        ch = orig_h; cw = int(round(ch * ratio))
    else:
        cw = orig_w; ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ── Model Cache & Face Detection ──────────────────────────────────────────────
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
                logger.info("Loaded model %s", w)
                return m
            except (OSError, ValueError, RuntimeError):
                continue
        logger.warning("YOLO unavailable")
        return None
    return _model_cache[weights]

def _get_yunet() -> Optional[Any]:
    global _yunet_detector
    if _yunet_detector is not None:
        return _yunet_detector
    for p in ["face_detection_yunet_2023mar.onnx", "yunet.onnx"]:
        if os.path.exists(p):
            try:
                net = cv2.dnn.readNet(p)
                _yunet_detector = net
                return net
            except (cv2.error, OSError, ValueError):
                pass
    return None

def detect_faces(frame: np.ndarray,
                 confidence_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    yunet = _get_yunet()
    if yunet is not None:
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), [0,0,0], True, False)
            yunet.setInput(blob)
            detections = yunet.forward()
            faces = []
            if detections is not None and detections.ndim >= 3:
                for i in range(detections.shape[2]):
                    c = detections[0, 0, i, 2]
                    if c > confidence_thresh:
                        faces.append((
                            int(detections[0, 0, i, 3] * w), int(detections[0, 0, i, 4] * h),
                            int(detections[0, 0, i, 5] * w), int(detections[0, 0, i, 6] * h),
                        ))
            if faces:
                faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
                return faces
        except (cv2.error, ValueError):
            pass
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(haar_path):
        cascade = cv2.CascadeClassifier(haar_path)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw     = cascade.detectMultiScale(gray, 1.1, 5,
                                           minSize=(max(30, w//20), max(30, h//20)))
        if len(raw) > 0:
            faces = [(x, y, x+bw, y+bh) for x, y, bw, bh in raw]
            faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
            return faces
    return []


# ── Multi-Object Sports Tracker ───────────────────────────────────────────────
class MultiObjectSportsTracker:
    def __init__(self, fps: float, frame_w: int, frame_h: int) -> None:
        self.fps     = fps
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.ball_state    = BallState(bbox=None, center=None, velocity=(0.0, 0.0))
        self.gravity_px    = GRAVITY_PIXELS_PER_SEC2_BASE * (frame_h / 1080.0)
        self.appearance_gallery: Dict[int, np.ndarray] = {}
        # track_history removed — it was populated but never read

    def reset(self) -> None:
        """Reset all tracking state, preserving fps/frame dimensions."""
        self.tracks            = {}
        self.next_track_id     = 0
        self.ball_state        = BallState(bbox=None, center=None, velocity=(0.0, 0.0))
        self.appearance_gallery = {}

    def _compute_iou(self, a: Tuple, b: Tuple) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter  = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _compute_appearance_sim(self, track_id: int, det_frame: np.ndarray,
                                det_box: Tuple) -> float:
        if track_id not in self.appearance_gallery:
            return 0.5
        x1, y1, x2, y2 = det_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(det_frame.shape[1], x2), min(det_frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi  = det_frame[y1:y2, x1:x2]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return float(cv2.compareHist(
            self.appearance_gallery[track_id].astype(np.float32),
            hist.astype(np.float32), cv2.HISTCMP_CORREL))

    def _update_appearance(self, track_id: int, det_frame: np.ndarray,
                           det_box: Tuple) -> None:
        x1, y1, x2, y2 = det_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(det_frame.shape[1], x2), min(det_frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi  = det_frame[y1:y2, x1:x2]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        if track_id in self.appearance_gallery:
            self.appearance_gallery[track_id] = (
                0.7 * self.appearance_gallery[track_id] + 0.3 * hist)
        else:
            self.appearance_gallery[track_id] = hist

    def _hungarian_match(self, tracks: List[Track], detections: List[Tuple],
                         det_frame: np.ndarray
                         ) -> Tuple[Dict[int, int], Set[int], Set[int]]:
        if not tracks or not detections:
            return {}, set(range(len(tracks))), set(range(len(detections)))
        n_t, n_d = len(tracks), len(detections)
        cost = np.zeros((n_t, n_d), dtype=float)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou     = self._compute_iou(track.bbox, det[:4])
                pred_cx = track.center[0] + track.velocity[0]
                pred_cy = track.center[1] + track.velocity[1]
                det_cx  = (det[0]+det[2])/2; det_cy = (det[1]+det[3])/2
                dist_c  = min(math.hypot(pred_cx-det_cx, pred_cy-det_cy)/100.0, 1.0)
                app_sim = self._compute_appearance_sim(track.id, det_frame, det[:4])
                cost[i, j] = (1-iou)*0.4 + dist_c*0.3 + (1-app_sim)*0.3
        if _HUNGARIAN_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind_list: List[int] = []; col_ind_list: List[int] = []; used: Set[int] = set()
            for i in range(n_t):
                best_j, best_c = -1, float("inf")
                for j in range(n_d):
                    if j not in used and cost[i, j] < best_c:
                        best_c, best_j = cost[i, j], j
                if best_j >= 0 and best_c < 0.7:
                    row_ind_list.append(i); col_ind_list.append(best_j); used.add(best_j)
            row_ind = np.array(row_ind_list); col_ind = np.array(col_ind_list)
        matched: Dict[int, int] = {}
        unmatched_t = set(range(n_t)); unmatched_d = set(range(n_d))
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 0.7:
                matched[i] = j; unmatched_t.discard(i); unmatched_d.discard(j)
        return matched, unmatched_t, unmatched_d

    def update(self, persons: List[Tuple[int, int, int, int]],
               ball_box: Optional[Tuple[int, int, int, int]],
               det_frame: np.ndarray,
               confidences: Optional[List[float]] = None) -> None:
        for track in self.tracks.values():
            track.time_since_update += 1
            track.center = (track.center[0]+track.velocity[0],
                            track.center[1]+track.velocity[1])
            track.bbox   = (int(track.bbox[0]+track.velocity[0]),
                            int(track.bbox[1]+track.velocity[1]),
                            int(track.bbox[2]+track.velocity[0]),
                            int(track.bbox[3]+track.velocity[1]))
        active = [t for t in self.tracks.values()
                  if t.status == TrackingStatus.ACTIVE or
                  (t.status == TrackingStatus.OCCLUDED and
                   t.time_since_update < MOT_MAX_OCCLUSION_FRAMES)]
        matched, unm_t, unm_d = self._hungarian_match(active, persons, det_frame)
        for ti, di in matched.items():
            t = active[ti]; det = persons[di]
            ncx = (det[0]+det[2])/2; ncy = (det[1]+det[3])/2
            t.velocity = (ncx-t.center[0], ncy-t.center[1])
            t.center   = (ncx, ncy); t.bbox = det
            t.hits += 1; t.time_since_update = 0; t.status = TrackingStatus.ACTIVE
            t.confidence = confidences[di] if confidences else 0.5
            self._update_appearance(t.id, det_frame, det)
        for ti in unm_t:
            t = active[ti]
            t.status = (TrackingStatus.LOST if t.time_since_update >= MOT_MAX_OCCLUSION_FRAMES
                        else TrackingStatus.OCCLUDED)
        for di in unm_d:
            det = persons[di]
            nt  = Track(id=self.next_track_id,
                        bbox=det, center=((det[0]+det[2])/2, (det[1]+det[3])/2),
                        velocity=(0.0, 0.0), hits=1,
                        confidence=confidences[di] if confidences else 0.5)
            self.tracks[self.next_track_id] = nt
            self._update_appearance(self.next_track_id, det_frame, det)
            self.next_track_id += 1
        self._update_ball(ball_box)
        self.tracks = {k: v for k, v in self.tracks.items()
                       if v.status != TrackingStatus.LOST}

    def _update_ball(self, ball_box: Optional[Tuple[int, int, int, int]]) -> None:
        if ball_box is not None:
            bx = (ball_box[0]+ball_box[2])/2; by = (ball_box[1]+ball_box[3])/2
            if self.ball_state.center is not None:
                vx = bx - self.ball_state.center[0]
                vy = by - self.ball_state.center[1]
                self.ball_state = BallState(
                    bbox=ball_box, center=(bx, by), velocity=(vx, vy),
                    is_airborne=(vy < -BALL_AIRBORNE_THRESHOLD_PX),
                    bounce_count=self.ball_state.bounce_count,
                    airborne_frames=(self.ball_state.airborne_frames + 1
                                     if vy < -BALL_AIRBORNE_THRESHOLD_PX else 0),
                )
            else:
                self.ball_state = BallState(bbox=ball_box, center=(bx, by),
                                            velocity=(0.0, 0.0))
        else:
            if self.ball_state.center is not None and self.ball_state.is_airborne:
                dt  = 1.0 / self.fps
                nvy = self.ball_state.velocity[1] + self.gravity_px * dt
                ncx = self.ball_state.center[0] + self.ball_state.velocity[0]
                ncy = self.ball_state.center[1] + nvy * dt
                self.ball_state = BallState(
                    bbox=None, center=(ncx, ncy), velocity=(self.ball_state.velocity[0], nvy),
                    is_airborne=True,
                    airborne_frames=self.ball_state.airborne_frames,
                    bounce_count=self.ball_state.bounce_count,
                )

    def get_primary_track(self, prev_ball_carrier: Optional[int] = None) -> Optional[Track]:
        if not self.tracks:
            return None
        if self.ball_state.center is not None:
            min_dist, closest = float("inf"), None
            for t in self.tracks.values():
                if t.status != TrackingStatus.ACTIVE:
                    continue
                d = math.hypot(t.center[0]-self.ball_state.center[0],
                               t.center[1]-self.ball_state.center[1])
                if d < SPORTS_BALL_PROXIMITY_PX and d < min_dist:
                    min_dist, closest = d, t
            if closest is not None:
                self.ball_state.is_possessed          = True
                self.ball_state.possessor_track_id    = closest.id
                return closest
            else:
                self.ball_state.is_possessed       = False
                self.ball_state.possessor_track_id = None
                return None
        active = [t for t in self.tracks.values() if t.status == TrackingStatus.ACTIVE]
        if not active:
            return None
        fcx, fcy = self.frame_w / 2, self.frame_h / 2
        best, best_score = None, -1e9
        for t in active:
            d = math.hypot(t.center[0]-fcx, t.center[1]-fcy)
            s = -d * 0.3 + t.hits * 10 + t.confidence * 100
            if prev_ball_carrier == t.id:
                s += SPORTS_SWITCH_BALL_BONUS
            if s > best_score:
                best_score, best = s, t
        return best


# ── Adaptive Velocity-Aware Smoother ─────────────────────────────────────────
class AdaptiveVelocityAwareSmoother:
    def __init__(self, fps: float, base_window_sec: float = AVS_BASE_WINDOW_SEC) -> None:
        self.fps         = fps
        self.base_window = base_window_sec
        self._init_buffers()

    def _init_buffers(self) -> None:
        cap = int(self.fps * AVS_MAX_WINDOW_SEC) + 4
        self._cx   = np.zeros(cap, dtype=np.float32)
        self._cy   = np.zeros(cap, dtype=np.float32)
        self._conf = np.ones(cap, dtype=np.float32)
        self._head  = 0
        self._count = 0
        self._cap   = cap
        self.prev_smooth_cx: Optional[float] = None
        self.prev_smooth_cy: Optional[float] = None
        self.prev_velocity:  Tuple[float, float] = (0.0, 0.0)
        self._raw_diffs:    List[float] = []
        self._smooth_diffs: List[float] = []

    def reset(self) -> None:
        """Reset buffer state without touching fps/base_window."""
        self._init_buffers()

    def _push(self, cx: float, cy: float, conf: float) -> None:
        self._cx[self._head]   = cx
        self._cy[self._head]   = cy
        self._conf[self._head] = conf
        self._head  = (self._head + 1) % self._cap
        if self._count < self._cap:
            self._count += 1

    def _window_arr(self, w: int) -> Tuple[np.ndarray, np.ndarray]:
        n   = min(w, self._count)
        idx = [(self._head - 1 - i) % self._cap for i in range(n)]
        idx.reverse()
        return self._cx[idx], self._cy[idx]

    def _compute_adaptive_window(self, velocity: float, accel: float,
                                 confidence: float, phase: PlayPhase) -> int:
        w = int(self.fps * self.base_window)
        if velocity > AVS_VELOCITY_FAST_THRESHOLD:
            w = int(w * 0.4)
        elif velocity < AVS_VELOCITY_SLOW_THRESHOLD:
            w = int(w * 1.8)
        if abs(accel) > AVS_ACCEL_SPIKE_THRESHOLD:
            w = max(3, int(w * 0.3))
        if confidence < AVS_CONFIDENCE_LOW_THRESHOLD:
            w = int(w * 1.4)
        if phase == PlayPhase.FAST_BREAK:
            w = max(3, int(w * 0.5))
        elif phase == PlayPhase.STATIC:
            w = int(w * 1.5)
        w = max(5, min(w, int(self.fps * AVS_MAX_WINDOW_SEC)))
        return w | 1

    def smooth(self, cx: float, cy: float, confidence: float = 1.0,
               phase: PlayPhase = PlayPhase.HALF_COURT) -> Tuple[float, float]:
        if self.prev_smooth_cx is not None:
            self._raw_diffs.append(math.hypot(cx - self.prev_smooth_cx,
                                              cy - self.prev_smooth_cy))
        # Temporal confidence smoothing
        if self._count > 0:
            prev_conf = float(self._conf[(self._head - 1) % self._cap])
            confidence = 0.8 * prev_conf + 0.2 * confidence
        self._push(cx, cy, confidence)
        n = self._count

        if n < 5:
            if self.prev_smooth_cx is None:
                self.prev_smooth_cx = cx; self.prev_smooth_cy = cy
            return cx, cy

        arr_cx, arr_cy = self._window_arr(min(n, int(self.fps * AVS_MAX_WINDOW_SEC)))
        nn = len(arr_cx)
        if nn >= 3:
            velocity = math.hypot(arr_cx[-1]-arr_cx[-2], arr_cy[-1]-arr_cy[-2])
            accel    = abs(velocity - math.hypot(arr_cx[-2]-arr_cx[-3], arr_cy[-2]-arr_cy[-3]))
        else:
            velocity = accel = 0.0

        w = self._compute_adaptive_window(velocity, accel, confidence, phase)
        w = min(w, nn); w = max(w, 5); w = w | 1

        try:
            polyorder = min(AVS_POLYORDER, w - 1)
            if polyorder < 2: polyorder = 2
            wx, wy = arr_cx[-w:], arr_cy[-w:]
            if _SCIPY_AVAILABLE and w >= 5:
                smooth_cx = float(savgol_filter(wx, w, polyorder)[-1])
                smooth_cy = float(savgol_filter(wy, w, polyorder)[-1])
            else:
                sigma     = (w / 4)**2 + 1e-9
                wts       = np.exp(-0.5 * (np.arange(w) - w//2)**2 / sigma)
                wts      /= wts.sum()
                smooth_cx = float(np.sum(wx * wts))
                smooth_cy = float(np.sum(wy * wts))
        except (ValueError, ZeroDivisionError, RuntimeError):
            alpha     = 0.15
            smooth_cx = alpha*cx + (1-alpha)*(self.prev_smooth_cx or cx)
            smooth_cy = alpha*cy + (1-alpha)*(self.prev_smooth_cy or cy)

        if self.prev_smooth_cx is not None:
            dvx = smooth_cx - self.prev_smooth_cx
            dvy = smooth_cy - self.prev_smooth_cy
            pvm = math.hypot(*self.prev_velocity)
            cvm = math.hypot(dvx, dvy)
            if cvm > pvm * 2.5 and pvm > 5.0:
                smooth_cx = self.prev_smooth_cx + dvx * 0.5
                smooth_cy = self.prev_smooth_cy + dvy * 0.5

        # Update prev_velocity with actual smooth frame-to-frame delta (post-damping)
        self.prev_velocity = (
            smooth_cx - self.prev_smooth_cx if self.prev_smooth_cx is not None else 0.0,
            smooth_cy - self.prev_smooth_cy if self.prev_smooth_cy is not None else 0.0,
        )

        if self.prev_smooth_cx is not None:
            self._smooth_diffs.append(
                math.hypot(smooth_cx - self.prev_smooth_cx,
                           smooth_cy - self.prev_smooth_cy))
        self.prev_smooth_cx = smooth_cx
        self.prev_smooth_cy = smooth_cy
        return float(smooth_cx), float(smooth_cy)

    def get_metrics(self) -> Dict[str, float]:
        raw_j = float(np.mean(self._raw_diffs))    if self._raw_diffs    else 0.0
        smo_j = float(np.mean(self._smooth_diffs)) if self._smooth_diffs else 0.0
        pct   = (raw_j - smo_j) / raw_j * 100 if raw_j > 0 else 0.0
        return {"jitter_raw": round(raw_j, 2), "jitter_smooth": round(smo_j, 2),
                "smoothness_pct": round(max(0.0, min(100.0, pct)), 1)}


# ── Intelligent Crop Strategy ─────────────────────────────────────────────────
class IntelligentCropStrategy:
    """All coordinates in OrigCoord (full-resolution pixel space)."""

    def __init__(self, orig_w: int, orig_h: int, crop_w: int, crop_h: int,
                 fps: float) -> None:
        self.orig_w = orig_w; self.orig_h = orig_h
        self.crop_w = crop_w; self.crop_h = crop_h
        self.fps    = fps
        self.hw     = crop_w // 2; self.hh = crop_h // 2
        self._hist:      deque = deque(maxlen=int(fps * ICS_LOOKAHEAD_SEC * 2))
        self._vel_hist:  deque = deque(maxlen=int(fps * 0.5))
        self._cached_vx: float = 0.0
        self._cached_vy: float = 0.0
        self._prev_cx:   Optional[float] = None
        self._prev_cy:   Optional[float] = None

    def compute_crop(self, cx: float, cy: float,
                     phase: PlayPhase = PlayPhase.HALF_COURT,
                     ball_pos: Optional[Tuple[float, float]] = None,
                     ) -> Tuple[int, int, int, int]:
        self._hist.append((cx, cy))
        moved = (self._prev_cx is None or
                 abs(cx - self._prev_cx) > 1 or abs(cy - self._prev_cy) > 1)
        if moved and len(self._hist) >= 3:
            recent = list(self._hist)[-5:]
            self._cached_vx = (recent[-1][0] - recent[0][0]) / len(recent)
            self._cached_vy = (recent[-1][1] - recent[0][1]) / len(recent)
            self._vel_hist.append((self._cached_vx, self._cached_vy))
            self._prev_cx, self._prev_cy = cx, cy
        lk      = int(self.fps * ICS_LOOKAHEAD_SEC)
        pred_cx = cx + self._cached_vx * lk
        pred_cy = cy + self._cached_vy * lk
        mf      = (ICS_FAST_BREAK_MARGIN_FACTOR if phase == PlayPhase.FAST_BREAK
                   else ICS_SET_PLAY_MARGIN_FACTOR if phase == PlayPhase.STATIC else 1.15)
        ecw = min(int(self.crop_w * mf), self.orig_w)
        ech = min(int(self.crop_h * mf), self.orig_h)
        ehw, ehh = ecw//2, ech//2
        left = int(np.clip(pred_cx - ehw, 0, max(0, self.orig_w - ecw)))
        top  = int(np.clip(pred_cy - ehh, 0, max(0, self.orig_h - ech)))
        ez   = ICS_BOUNDARY_ELASTICITY_PX
        if left < ez:           left = int(left * 0.3)
        if top  < ez:           top  = int(top  * 0.3)
        if left + ecw > self.orig_w - ez:
            left = max(0, left - int((left + ecw - (self.orig_w - ez)) * 0.3))
        if top + ech > self.orig_h - ez:
            top  = max(0, top  - int((top  + ech - (self.orig_h - ez)) * 0.3))
        left = max(0, min(left, max(0, self.orig_w - self.crop_w)))
        top  = max(0, min(top,  max(0, self.orig_h - self.crop_h)))
        right, bottom = left + self.crop_w, top + self.crop_h
        if ball_pos is not None:
            bx, by = ball_pos; m = self.crop_w * 0.15
            if bx < left + m:
                left  = max(0, left - int(left + m - bx)); right = left + self.crop_w
            elif bx > right - m:
                left  = min(max(0, self.orig_w - self.crop_w),
                            left + int(bx - (right - m))); right = left + self.crop_w
            if by < top + m:
                top    = max(0, top - int(top + m - by)); bottom = top + self.crop_h
            elif by > bottom - m:
                top    = min(max(0, self.orig_h - self.crop_h),
                             top + int(by - (bottom - m))); bottom = top + self.crop_h
        return left, top, right, bottom


# ── Game State Engine ─────────────────────────────────────────────────────────
class GameStateEngine:
    def __init__(self, fps: float, frame_w: int, frame_h: int) -> None:
        self.fps = fps; self.frame_w = frame_w; self.frame_h = frame_h
        self.current_state   = GameState.UNKNOWN
        self.prev_gray:      Optional[np.ndarray] = None
        self.freeze_frame_count = 0
        self.motion_history: deque = deque(maxlen=int(fps * 2))

    def update(self, persons: List[Tuple], gray_frame: np.ndarray) -> GameState:
        if self.prev_gray is not None:
            # FIXED: guard against shape mismatch before absdiff
            if self.prev_gray.shape == gray_frame.shape:
                diff = float(cv2.absdiff(self.prev_gray, gray_frame).mean())
                self.motion_history.append(diff)
                if diff < 1.0: self.freeze_frame_count += 1
                else:          self.freeze_frame_count = max(0, self.freeze_frame_count - 2)
            else:
                self.motion_history.append(0.0)
                self.freeze_frame_count = max(0, self.freeze_frame_count - 1)
        self.prev_gray = gray_frame.copy()
        if self.freeze_frame_count > self.fps:
            self.current_state = GameState.TIMEOUT
            return self.current_state
        if len(persons) >= 2:
            cy_list  = [(p[1]+p[3])/2 for p in persons]
            cx_list  = [(p[0]+p[2])/2 for p in persons]
            near_line = sum(1 for y in cy_list if y < self.frame_h * 0.4)
            spread    = float(np.std(cx_list)) / self.frame_w if len(cx_list) > 1 else 0.0
            if near_line >= 1 and spread > 0.2 and len(persons) <= 5:
                self.current_state = GameState.FREE_THROW
                return self.current_state
        self.current_state = GameState.LIVE_PLAY
        return self.current_state

    def get_zoom_factor(self) -> float:
        if self.current_state == GameState.FREE_THROW: return 1.1
        if self.current_state in (GameState.TIMEOUT, GameState.REPLAY): return 1.0
        return 1.15


# ── Sports Play Phase Detector ────────────────────────────────────────────────
class SportsPlayPhaseDetector:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.prev_ball_pos:     Optional[Tuple[float, float]] = None
        self.ball_vel_history:  deque = deque(maxlen=int(fps * 1.0))
        self.player_spread_hist: deque = deque(maxlen=int(fps * 0.5))
        self.phase_history:     deque = deque(maxlen=5)
        self.transition_counter = 0

    def reset(self) -> None:
        self.prev_ball_pos      = None
        self.ball_vel_history.clear()
        self.player_spread_hist.clear()
        self.phase_history.clear()
        self.transition_counter = 0

    def detect_phase(self, persons: List[Tuple], ball_box: Optional[Tuple],
                     frame_w: int, ball_state: Optional[BallState] = None) -> PlayPhase:
        if not persons:
            return PlayPhase.STATIC
        cx_list = [(p[0]+p[2])/2 for p in persons]
        spread  = float(np.std(cx_list)) if len(cx_list) > 1 else 0.0
        self.player_spread_hist.append(spread / (frame_w / 2))
        ball_speed = 0.0
        if ball_box:
            bx = (ball_box[0]+ball_box[2])/2; by = (ball_box[1]+ball_box[3])/2
            if self.prev_ball_pos:
                ball_speed = math.hypot(bx-self.prev_ball_pos[0], by-self.prev_ball_pos[1])
            self.prev_ball_pos = (bx, by)
            self.ball_vel_history.append(ball_speed)
        avg_speed  = float(np.mean(self.ball_vel_history))   if self.ball_vel_history  else 0.0
        avg_spread = float(np.mean(self.player_spread_hist)) if self.player_spread_hist else 0.0
        if len(self.phase_history) >= 3:
            rp = list(self.phase_history)[-3:]
            if len(set(rp)) > 1:
                self.transition_counter += 1
                if self.transition_counter > int(self.fps * 0.3):
                    self.phase_history.clear(); self.transition_counter = 0
                    return PlayPhase.TRANSITION
            else:
                self.transition_counter = max(0, self.transition_counter - 1)
        if avg_speed > BALL_SPEED_THRESHOLD * 1.5 and avg_spread > 0.2:
            phase = PlayPhase.FAST_BREAK
        elif avg_speed > BALL_SPEED_THRESHOLD and avg_spread > 0.15:
            phase = PlayPhase.FAST_BREAK
        elif avg_spread < 0.06:
            phase = PlayPhase.REBOUND
        else:
            phase = PlayPhase.HALF_COURT
        self.phase_history.append(phase)
        return phase


# ── Legacy Kalman Tracker ─────────────────────────────────────────────────────
class SportsKalmanTracker:
    def __init__(self, dt: float = 1.0, fps: float = 30.0) -> None:
        self.dt = dt; self.fps = fps
        self.F = np.array([
            [1,0,dt,0,0.5*dt**2,0],[0,1,0,dt,0,0.5*dt**2],
            [0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1],
        ], dtype=np.float64)
        self.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]], dtype=np.float64)
        self.Q_base = np.eye(6, dtype=np.float64) * KALMAN_PLAYER_PROCESS_NOISE_BASE
        self.Q_base[4,4] *= 4.0; self.Q_base[5,5] *= 4.0
        self.R_yolo     = np.eye(2, dtype=np.float64) * KALMAN_PLAYER_MEASUREMENT_NOISE
        self.R_optical  = np.eye(2, dtype=np.float64) * KALMAN_OPTICAL_FLOW_NOISE
        self.R_saliency = np.eye(2, dtype=np.float64) * KALMAN_SALIENCY_NOISE
        self.P          = np.eye(6, dtype=np.float64) * KALMAN_INITIAL_ERROR
        self.x          = np.zeros((6, 1), dtype=np.float64)
        self.initialized = False; self._stale_count = 0

    def init(self, cx: float, cy: float) -> None:
        self.x = np.array([[cx],[cy],[0.],[0.],[0.],[0.]], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * KALMAN_INITIAL_ERROR
        self.initialized = True; self._stale_count = 0

    def predict(self, steps: int = 1) -> Tuple[float, float]:
        if not self.initialized: return float(self.x[0,0]), float(self.x[1,0])
        if steps == 0: return float(self.x[0,0]), float(self.x[1,0])
        dt_s = self.dt * steps
        px = float(self.x[0,0])+float(self.x[2,0])*dt_s+0.5*float(self.x[4,0])*dt_s**2
        py = float(self.x[1,0])+float(self.x[3,0])*dt_s+0.5*float(self.x[5,0])*dt_s**2
        return px, py

    def _predict_step(self) -> None:
        if not self.initialized: return
        am = math.sqrt(float(self.x[4,0])**2 + float(self.x[5,0])**2)
        Q  = self.Q_base * (3 if am > 100 else 2 if am > 50 else 1)
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + Q
        self._stale_count += 1
        for i in (2, 3):
            if abs(float(self.x[i,0])) > 200:
                self.x[i,0] = float(np.sign(self.x[i,0])) * 200

    def update(self, cx: float, cy: float, sensor: str = "yolo") -> Tuple[float, float]:
        if not self.initialized: self.init(cx, cy); return cx, cy
        R = (self.R_optical  if sensor == "optical_flow" else
             self.R_saliency if sensor == "saliency"     else self.R_yolo)
        z = np.array([[cx],[cy]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        inv_S = np.linalg.inv(S)
        if float(np.sqrt((y.T @ inv_S @ y).item())) > KALMAN_GATE_THRESHOLD:
            self._stale_count += 1
            return float(self.x[0,0]), float(self.x[1,0])
        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ self.H) @ self.P
        self._stale_count = 0
        return float(self.x[0,0]), float(self.x[1,0])

    # FIXED: predict_step now actually called before update in smooth_centers
    def increment_stale(self) -> None:
        self._predict_step()

    @property
    def is_stale(self) -> bool: return self._stale_count > 10

    @property
    def velocity(self) -> Tuple[float, float]: return float(self.x[2,0]), float(self.x[3,0])

    @property
    def speed(self) -> float: return math.hypot(*self.velocity)


# ── Court / field detection ───────────────────────────────────────────────────
def detect_field_of_play(frame: np.ndarray,
                         sport_hint: str = "auto") -> Optional[np.ndarray]:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    def _make_mask(cr: Dict) -> np.ndarray:
        m = cv2.inRange(hsv,
                        np.array([cr["h"][0], cr["s"][0], cr["v"][0]]),
                        np.array([cr["h"][1], cr["s"][1], cr["v"][1]]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

    if sport_hint == "auto":
        best_mask, best_area = None, 0
        for cr in SPORTS_COURT_COLORS_HSV:
            m = _make_mask(cr); a = cv2.countNonZero(m)
            if a > best_area and a > h * w * 0.15:
                best_area, best_mask = a, m
        return best_mask

    ranges = {"basketball": [SPORTS_COURT_COLORS_HSV[0]],
              "football":   [SPORTS_COURT_COLORS_HSV[1]],
              "soccer":     [SPORTS_COURT_COLORS_HSV[1]],
              "hockey":     [SPORTS_COURT_COLORS_HSV[2]]}
    mask = np.zeros((h, w), dtype=np.uint8)
    for cr in ranges.get(sport_hint, SPORTS_COURT_COLORS_HSV):
        mask = cv2.bitwise_or(mask, _make_mask(cr))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask    = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask if cv2.countNonZero(mask) > h * w * 0.10 else None
    return None


def get_court_center_of_mass(field_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    if field_mask is None: return None
    m = cv2.moments(field_mask)
    if m["m00"] == 0: return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


# ── Optical flow & Saliency ───────────────────────────────────────────────────
def sports_optical_flow_center(prev: np.ndarray, curr: np.ndarray, w: int, h: int,
                               prev_center: Optional[Tuple] = None,
                               field_mask: Optional[np.ndarray] = None,
                               ) -> Optional[Tuple[int, int]]:
    if prev is None or curr is None: return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag  = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        b    = max(1, int(w * 0.04))
        mag[:, :b] = mag[:, w-b:] = mag[:b, :] = mag[h-b:, :] = 0
        if field_mask is not None:
            fm  = cv2.resize(field_mask, (w, h), cv2.INTER_NEAREST) if field_mask.shape[:2] != (h, w) else field_mask
            mag = mag * (fm.astype(np.float32) / 255.0)
        if prev_center is not None:
            pcx, pcy = prev_center
            ys, xs   = np.mgrid[0:h, 0:w]
            mag      = mag * np.exp(-np.sqrt((xs-pcx)**2 + (ys-pcy)**2) / (max(w, h) * 0.3))
        if mag.max() < 0.5: return None
        t = mag.sum()
        if t == 0: return None
        ys, xs = np.mgrid[0:h, 0:w]
        return int((xs * mag).sum() / t), int((ys * mag).sum() / t)
    except (cv2.error, ValueError, ZeroDivisionError):
        return None

def temporal_saliency_center(frame: np.ndarray,
                             prev_saliency: Optional[np.ndarray] = None,
                             decay: float = 0.7) -> Tuple[int, int, np.ndarray]:
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIM or h < MIN_FRAME_DIM:
        return w//2, h//2, np.zeros((h, w), dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31, 31), 0)
    sat  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32), (31, 31), 0)
    sal  = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    if prev_saliency is not None and prev_saliency.shape == sal.shape:
        sal = sal * (1.0 + np.abs(sal - prev_saliency * decay)**2.0)
    b = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w-b:] = sal[:b, :] = sal[h-b:, :] = 0
    t = sal.sum()
    if t < 1e-6: return w//2, h//2, sal
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t), sal


# ── Scene change detection ────────────────────────────────────────────────────
def _ensure_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None: return None
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def is_sports_scene_change(prev: Optional[np.ndarray], curr: np.ndarray,
                           prev_hist: Optional[np.ndarray] = None,
                           frame_count: int = 0,
                           last_cut_frame: int = -100,
                           ) -> Tuple[bool, Optional[np.ndarray], int]:
    cb = _ensure_bgr(curr); pb = _ensure_bgr(prev)
    ch = cv2.calcHist([cb], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    ch = cv2.normalize(ch, ch).flatten()
    if pb is None: return False, ch, last_cut_frame
    # FIXED: guard against shape mismatch before absdiff
    if pb.shape != cb.shape:
        logger.debug("Shape mismatch in is_sports_scene_change: %s vs %s", pb.shape, cb.shape)
        return False, ch, last_cut_frame
    pixel_diff = float(cv2.absdiff(pb, cb).mean()) / 255.0
    hist_corr  = (cv2.compareHist(prev_hist.astype(np.float32), ch.astype(np.float32),
                                  cv2.HISTCMP_CORREL) if prev_hist is not None else 0.0)
    is_cut     = pixel_diff > SPORTS_SCENE_CUT_THRESHOLD or (prev_hist is not None and hist_corr < 0.5)
    if is_cut and (frame_count - last_cut_frame) < SPORTS_SCENE_CUT_MIN_FRAMES:
        is_cut = False
    if is_cut: last_cut_frame = frame_count
    return is_cut, ch, last_cut_frame

def is_scene_change(prev: Optional[np.ndarray], curr: np.ndarray,
                    threshold: float = 0.35, prev_hist: Optional[np.ndarray] = None,
                    frame_count: int = 0, last_cut_frame: int = -100,
                    mode: str = "default") -> Tuple[bool, Optional[np.ndarray], int]:
    if mode == "sports":
        return is_sports_scene_change(prev, curr, prev_hist, frame_count, last_cut_frame)
    cb = _ensure_bgr(curr); pb = _ensure_bgr(prev)
    ch = cv2.calcHist([cb], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    ch = cv2.normalize(ch, ch).flatten()
    if pb is None: return False, ch, last_cut_frame
    # FIXED: guard against shape mismatch before absdiff
    if pb.shape != cb.shape:
        logger.debug("Shape mismatch in is_scene_change: %s vs %s", pb.shape, cb.shape)
        return False, ch, last_cut_frame
    is_cut = float(cv2.absdiff(pb, cb).mean()) / 255.0 > threshold
    if is_cut: last_cut_frame = frame_count
    return is_cut, ch, last_cut_frame


# ── Sports Event Detector ─────────────────────────────────────────────────────
class SportsEventDetector:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self.recent_ball_heights: List[float] = []
        self.event_active    = False
        self.event_end_frame = 0
        self._frame_count    = 0
        self._event_flags:   Dict[int, bool] = {}

    def update(self, ball_box: Optional[Tuple], primary_person: Optional[Tuple],
               record_frame: Optional[int] = None) -> bool:
        self._frame_count += 1
        active = False
        if self._frame_count < self.event_end_frame:
            active = True
        elif ball_box is not None and primary_person is not None:
            bx1,by1,bx2,by2 = ball_box; px1,py1,px2,py2 = primary_person
            r = (py1-by1) / max(py2-py1, 1) if py2 > py1 else 0
            self.recent_ball_heights.append(r)
            if len(self.recent_ball_heights) > int(self.fps * 0.5):
                self.recent_ball_heights.pop(0)
            if (len(self.recent_ball_heights) >= 3 and r < -0.3 and
                    self.recent_ball_heights[-1] < self.recent_ball_heights[-2]):
                self.event_end_frame = self._frame_count + SPORTS_EVENT_EXPAND_FRAMES
                active = True
        self.event_active = active
        if record_frame is not None:
            self._event_flags[record_frame] = active
        return active

    def event_active_for(self, fi: int) -> bool:
        return self._event_flags.get(fi, False)


# ── Subject detection ─────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult",
                             ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"])

def _parse_yolo_results(results_boxes, scale: float, confidence: float,
                        ) -> Tuple[List[Tuple], List[Tuple], List[float]]:
    persons: List[Tuple] = []; balls: List[Tuple] = []; person_confs: List[float] = []
    if results_boxes is None or len(results_boxes) == 0:
        return persons, balls, person_confs
    for box in results_boxes:
        cls  = int(box.cls[0]); conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx_b, cy_b = (x1+x2)//2, (y1+y2)//2
        if cls == PERSON_CLASS_ID and conf >= confidence:
            persons.append((x1, y1, x2, y2, cx_b, cy_b, conf))
            person_confs.append(conf)
        elif cls == SPORTS_BALL_CLASS_ID and conf >= SPORTS_BALL_CONFIDENCE:
            balls.append((x1, y1, x2, y2, cx_b, cy_b, conf))
    balls.sort(key=lambda b: b[6], reverse=True)
    persons.sort(key=lambda p: p[6], reverse=True)
    return persons, balls, person_confs


def detect_subjects(frame: np.ndarray, model: Any, confidence: float = 0.45,
                    prev_center: Optional[Tuple] = None,
                    prev_ball_carrier: Optional[int] = None,
                    tracking_mode: str = "subject",
                    _cached_result: Optional[DetectionCache] = None,
                    det_scale: float = 1.0,   # FIXED: added for correct coord conversion
                    ) -> Tuple[Optional[DetectionResult], Optional[Tuple], int]:
    """
    Returns DetectionResult with coordinates in OrigCoord space.
    det_scale: ratio of det_frame / orig_frame (used when _cached_result is supplied).
    """
    if _cached_result is not None:
        # FIXED: convert from det-space to orig-space using the caller-supplied scale
        inv = 1.0 / max(det_scale, 1e-6)
        persons_raw = [
            (int(p[0]*inv), int(p[1]*inv), int(p[2]*inv), int(p[3]*inv),
             int((p[0]+p[2])//2*inv), int((p[1]+p[3])//2*inv), 0.5)
            for p in _cached_result.persons
        ]
        balls: List[Tuple] = []
        if _cached_result.ball_box is not None:
            bb = _cached_result.ball_box
            balls = [(int(bb[0]*inv), int(bb[1]*inv), int(bb[2]*inv), int(bb[3]*inv),
                      int((bb[0]+bb[2])//2*inv), int((bb[1]+bb[3])//2*inv), 0.6)]
    else:
        if model is None: return None, None, -1
        try:
            results = model(frame, verbose=False, conf=confidence)[0]
        except Exception as e:
            logger.warning("Detection error: %s", e); return None, None, -1
        persons_raw, balls, _ = _parse_yolo_results(
            results.boxes if results.boxes is not None else [], 1.0, confidence)

    if not persons_raw: return None, None, -1

    ball_box:    Optional[Tuple] = None
    ball_carrier = -1
    if balls:
        best_ball = max(balls, key=lambda b: b[6])
        ball_box  = (best_ball[0], best_ball[1], best_ball[2], best_ball[3])
        min_dist  = float("inf")
        for i, p in enumerate(persons_raw):
            d = math.hypot(p[4]-best_ball[4], p[5]-best_ball[5])
            if d < min_dist and d < SPORTS_BALL_PROXIMITY_PX:
                min_dist, ball_carrier = d, i

    if tracking_mode == "sports_action":
        best_idx = 0; best_score = -1e9
        fcx = sum(p[4] for p in persons_raw) / len(persons_raw)
        fcy = sum(p[5] for p in persons_raw) / len(persons_raw)
        for i, p in enumerate(persons_raw):
            s = (-math.hypot(p[4]-prev_center[0], p[5]-prev_center[1]) * 0.5
                 if prev_center else -math.hypot(p[4]-fcx, p[5]-fcy) * 0.3)
            if i == ball_carrier and ball_carrier >= 0:
                s += SPORTS_SWITCH_BALL_BONUS
            if i == prev_ball_carrier and prev_ball_carrier is not None and prev_ball_carrier >= 0:
                s += SPORTS_SWITCH_BALL_BONUS * 0.5
            s += (p[2]-p[0]) * (p[3]-p[1]) * 0.001
            if s > best_score: best_score, best_idx = s, i
        primary = persons_raw[best_idx]
    else:
        primary = persons_raw[ball_carrier] if ball_carrier >= 0 else None

    if primary is None:
        tw = sum(e[6] for e in persons_raw)
        if tw == 0: return None, None, -1
        cx_w = int(sum(e[6]*e[4] for e in persons_raw) / tw)
        cy_w = int(sum(e[6]*e[5] for e in persons_raw) / tw)
        return DetectionResult(cx_w, cy_w,
                               min(e[0] for e in persons_raw), min(e[1] for e in persons_raw),
                               max(e[2] for e in persons_raw), max(e[3] for e in persons_raw),
                               len(persons_raw)), ball_box, ball_carrier

    x1, y1, x2, y2, cx, cy, _ = primary
    cluster = [primary] + [p for p in persons_raw if p is not primary
                           and math.hypot(p[4]-cx, p[5]-cy) < (x2-x1)*1.5]
    return (DetectionResult(int(cx), int(cy),
                            min(p[0] for p in cluster), min(p[1] for p in cluster),
                            max(p[2] for p in cluster), max(p[3] for p in cluster),
                            len(persons_raw)),
            ball_box, ball_carrier)


def detect_persons_all(frame: np.ndarray, model: Any,
                       confidence: float = 0.45) -> List[Tuple[int, int, int, int]]:
    if model is None: return []
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except (RuntimeError, ValueError, OSError):
        return []
    if results.boxes is None or len(results.boxes) == 0: return []
    return sorted(
        [tuple(map(int, box.xyxy[0].tolist())) for box in results.boxes
         if int(box.cls[0]) == PERSON_CLASS_ID],
        key=lambda b: b[0])


# ── Ball detection validation ─────────────────────────────────────────────────
def _validate_ball_detection(
    bbox: Tuple[int, int, int, int],
    det_frame: np.ndarray,
    ball_kalman: BallKalmanFilter,
    color_model: BallColorModel,
) -> bool:
    """All coordinates in DetCoord space."""
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
    area = bw * bh
    if area < BALL_MIN_AREA_PX2 or area > BALL_MAX_AREA_PX2:
        return False
    aspect = bh / bw
    if aspect < BALL_MIN_ASPECT or aspect > BALL_MAX_ASPECT:
        return False
    if ball_kalman.initialized:
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        if ball_kalman.gate_distance(cx, cy) > BALL_MAX_GATE_PX:
            return False
    if color_model.is_ready:
        if color_model.match(det_frame, bbox) < BALL_COLOR_MATCH_THRESHOLD:
            return False
    return True


# ── Framing helpers ───────────────────────────────────────────────────────────
def _apply_lower_third_guard(cy: int, crop_h: int, subject_cy_src: int,
                             orig_h: int) -> int:
    hh     = crop_h // 2
    max_cy = subject_cy_src - int((1.0 - LOWER_THIRD_GUARD) * crop_h) + hh
    return min(cy, min(max_cy, orig_h - hh))

def _soi_region_label(cx: int, cy: int, w: int, h: int) -> str:
    col = "left" if cx < w//3 else ("right" if cx > 2*w//3 else "center")
    row = "upper" if cy < h//3 else ("lower" if cy > 2*h//3 else "mid")
    return "center" if (row == "mid" and col == "center") else (col if row == "mid" else f"{row}-{col}")

def frame_for_union(ux1: int, uy1: int, ux2: int, uy2: int,
                    orig_w: int, orig_h: int, crop_w: int, crop_h: int) -> Tuple[int, int]:
    ucx = (ux1+ux2)//2; ucy = (uy1+uy2)//2
    hw, hh = crop_w//2, crop_h//2
    cx = max(hw, min(ucx, orig_w-hw)); cy = max(hh, min(ucy, orig_h-hh))
    cy = _apply_lower_third_guard(cy, crop_h, ucy, orig_h)
    return cx, max(hh, min(cy, orig_h-hh))

def talking_head_center(faces: List[Tuple], orig_w: int, orig_h: int,
                        crop_w: int, crop_h: int,
                        bias: float = 0.30) -> Optional[Tuple[int, int]]:
    if not faces: return None
    ux1 = min(f[0] for f in faces); uy1 = min(f[1] for f in faces)
    ux2 = max(f[2] for f in faces); uy2 = max(f[3] for f in faces)
    fcx = (ux1+ux2)//2; fcy = (uy1+uy2)//2
    cy  = int(fcy * (1-bias) + (fcy + crop_h//6) * bias)
    hw, hh = crop_w//2, crop_h//2
    cx = max(hw, min(fcx, orig_w-hw)); cy = max(hh, min(cy, orig_h-hh))
    cy = _apply_lower_third_guard(cy, crop_h, fcy, orig_h)
    return cx, max(hh, min(cy, orig_h-hh))


# ── Panel detection & rendering ───────────────────────────────────────────────
def _detect_panel_mode(input_path: str, model: Any, fps: float, total_frames: int,
                       orig_w: int, orig_h: int, confidence: float = 0.45,
                       n_probe: int = PANEL_PROBE_COUNT,
                       max_person_motion: float = PANEL_MAX_PERSON_MOTION,
                       min_person_area_frac: float = PANEL_MIN_PERSON_AREA_FRAC,
                       max_count_variance: float = PANEL_MAX_COUNT_VARIANCE,
                       stability_frac: float = PANEL_STABILITY_FRAC,
                       majority_frac: float = PANEL_MAJORITY_FRAC,
                       min_person_aspect: float = PANEL_MIN_PERSON_ASPECT) -> bool:
    """
    FIXED: Uses a single FFmpegVideoReader pass to collect probe frames
    instead of n_probe separate seeks — dramatically faster on large files.
    """
    if model is None: return False
    det_w  = min(orig_w, 640); det_h = max(1, int(det_w * orig_h / orig_w))
    duration  = total_frames / fps
    probe_ts  = np.linspace(1.0, max(2.0, duration - 1.0), n_probe)
    frame_area = det_w * det_h

    # Collect all probe frames in one reader pass
    probe_frames: Dict[int, np.ndarray] = {}
    target_frame_ids = set(int(t * fps) for t in probe_ts)
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=det_w, scale_h=det_h) as rdr:
            for fi, frame in enumerate(rdr):
                nearest = min(target_frame_ids, key=lambda x: abs(x - fi)) if target_frame_ids else None
                if nearest is not None and abs(nearest - fi) <= max(1, int(fps * 0.25)):
                    probe_frames[fi] = frame.copy()
                    target_frame_ids.discard(nearest)
                if fi >= int(probe_ts[-1] * fps) + int(fps):
                    break
    except Exception:
        return False

    if not probe_frames:
        return False

    multi_hits = 0; stable_split_hits = 0
    motion_vals: List[float] = []; area_vals: List[float] = []
    aspect_vals: List[float] = []; count_vals: List[int] = []
    prev_centres: Optional[List[Tuple]] = None; prev_split: Optional[Dict] = None

    for frame in probe_frames.values():
        persons = detect_persons_all(frame, model, confidence)
        count_vals.append(len(persons))
        curr_cx = [((p[0]+p[2])/2/det_w, (p[1]+p[3])/2/det_h) for p in persons]
        if prev_centres and curr_cx:
            used: Set[int] = set(); dists: List[float] = []
            for px, py in prev_centres:
                best_d, best_j = 1e9, -1
                for j, (cx2, cy2) in enumerate(curr_cx):
                    if j in used: continue
                    d = math.hypot(px-cx2, py-cy2)
                    if d < best_d: best_d, best_j = d, j
                if best_j >= 0: dists.append(best_d * det_w); used.add(best_j)
            if dists: motion_vals.append(float(np.mean(dists)))
        prev_centres = curr_cx if persons else None
        if len(persons) < PANEL_MIN_PERSONS:
            prev_split = None; continue
        multi_hits += 1
        area_vals.append(float(np.mean([(p[2]-p[0])*(p[3]-p[1]) for p in persons])) / frame_area)
        aspect_vals.append(float(np.mean([(p[3]-p[1])/max(p[2]-p[0],1) for p in persons])))
        cx_list = [(p[0]+p[2])/2/det_w for p in persons]
        left_x  = [c for c in cx_list if c < 0.40]
        right_x = [c for c in cx_list if c > 0.60]
        if left_x and right_x:
            if (prev_split and
                    abs(np.mean(left_x)  - np.mean(prev_split["left"]))  <= 0.10 and
                    abs(np.mean(right_x) - np.mean(prev_split["right"])) <= 0.10):
                stable_split_hits += 1
            prev_split = {"left": left_x, "right": right_x}
        else:
            prev_split = None

    if multi_hits == 0: return False
    is_panel = (
        multi_hits > n_probe * majority_frac and
        stable_split_hits > int(n_probe * stability_frac) and
        float(np.mean(motion_vals)  if motion_vals  else 0) < max_person_motion and
        float(np.mean(area_vals)    if area_vals    else 0) >= min_person_area_frac and
        float(np.std(count_vals)    if len(count_vals) > 1 else 0) <= max_count_variance and
        float(np.mean(aspect_vals)  if aspect_vals  else 0) >= min_person_aspect
    )
    logger.info("panel_detect: multi=%d stable=%d -> panel=%s",
                multi_hits, stable_split_hits, is_panel)
    return is_panel


class DynamicPanelSlotSmoother:
    """N-slot panel smoother with Hungarian assignment and velocity damping."""

    def __init__(self, max_slots: int = PANEL_MAX_SLOTS,
                 alpha: float = PANEL_SLOT_EMA,
                 max_jump_frac: float = PANEL_SLOT_MAX_JUMP,
                 velocity_damping: float = PANEL_VELOCITY_DAMPING) -> None:
        self.max_slots = max_slots
        self.alpha = alpha
        self.max_jump_frac = max_jump_frac
        self.velocity_damping = velocity_damping
        self._slots:    List[Optional[Tuple]] = [None] * max_slots
        self._slot_cx:  List[Optional[float]] = [None] * max_slots
        self._slot_vel: List[Tuple[float,float,float,float]] = [(0.,0.,0.,0.)] * max_slots
        self._active_count = 0
        self._speaker_weights: List[float] = [1.0] * max_slots

    def set_speaker_weights(self, weights: List[float]) -> None:
        for i in range(min(len(weights), self.max_slots)):
            self._speaker_weights[i] = max(0.0, min(1.0, weights[i]))

    def _ema_box(self, prev: Optional[Tuple], new_box: Tuple,
                 axis_size: float, slot_idx: int = 0) -> Tuple:
        if prev is None:
            return tuple(float(v) for v in new_box)
        a = self.alpha; mj = axis_size * self.max_jump_frac
        raw = list(prev[i]*(1-a) + new_box[i]*a for i in range(4))
        vel = [raw[i] - prev[i] for i in range(4)]
        pv = self._slot_vel[slot_idx]
        for d in range(4):
            if abs(vel[d]) > abs(pv[d]) * 2.5 and abs(pv[d]) > 2.0:
                raw[d] = prev[d] + vel[d] * self.velocity_damping
        self._slot_vel[slot_idx] = tuple(vel)
        return tuple(float(np.clip(raw[i], prev[i]-mj, prev[i]+mj)) for i in range(4))

    def _assign_slots_hungarian(self, groups: List[List]) -> List[List]:
        n_slots = self.max_slots
        result = [[] for _ in range(n_slots)]
        active = [(gi, groups[gi]) for gi in range(len(groups)) if groups[gi]]
        if not active:
            return result
        gcx = [float(np.mean([(p[0]+p[2])//2 for p in g])) for _, g in active]
        if all(sc is None for sc in self._slot_cx[:len(active)]):
            for si, (_, g) in enumerate(sorted(zip(gcx, [g for _, g in active]))[:n_slots]):
                result[si] = g
            self._active_count = min(len(active), n_slots)
            return result
        if _HUNGARIAN_AVAILABLE and len(active) >= 2:
            n = max(len(active), sum(1 for sc in self._slot_cx if sc is not None))
            cost = np.full((n, n), 1e6)
            for ai, (gi, _) in enumerate(active):
                for si in range(n_slots):
                    if self._slot_cx[si] is not None:
                        cost[ai, si] = abs(gcx[ai] - self._slot_cx[si])
                    else:
                        cost[ai, si] = 500.0
            row_ind, col_ind = linear_sum_assignment(cost[:len(active), :n_slots])
            for ai, si in zip(row_ind, col_ind):
                if cost[ai, si] < 1e5:
                    result[si] = active[ai][1]
        else:
            used: Set[int] = set()
            for si in range(n_slots):
                if self._slot_cx[si] is None: continue
                best_g, best_d = -1, float("inf")
                for ai, (gi, _) in enumerate(active):
                    if ai in used: continue
                    d = abs(gcx[ai] - self._slot_cx[si])
                    if d < best_d: best_d, best_g = d, ai
                if best_g >= 0: result[si] = active[best_g][1]; used.add(best_g)
            for ai, (gi, g) in enumerate(active):
                if ai not in used:
                    for si in range(n_slots):
                        if not result[si]: result[si] = g; used.add(ai); break
        self._active_count = sum(1 for r in result if r)
        return result

    def update(self, *groups, strip_w: float) -> List[List]:
        group_list = list(groups)
        assigned = self._assign_slots_hungarian(group_list)
        result: List[List] = [[] for _ in range(self.max_slots)]
        for i in range(self.max_slots):
            grp = assigned[i]
            if grp:
                union = _group_union(grp)
                smooth = self._ema_box(self._slots[i], union, strip_w, slot_idx=i)
                self._slots[i] = smooth
                self._slot_cx[i] = (smooth[0] + smooth[2]) / 2.0
                result[i] = [tuple(int(v) for v in smooth)]
            elif self._slots[i] is not None:
                result[i] = [tuple(int(v) for v in self._slots[i])]
        # FIXED: Always return List[List] regardless of input count
        return result

    @property
    def active_count(self) -> int:
        return self._active_count


# Backward compatibility alias
PanelSlotSmoother = DynamicPanelSlotSmoother


def _group_union(persons: List[Tuple]) -> Tuple[int, int, int, int]:
    return (min(p[0] for p in persons), min(p[1] for p in persons),
            max(p[2] for p in persons), max(p[3] for p in persons))


def _detect_faces_for_panel(frame: np.ndarray,
                            persons: List[Tuple[int,int,int,int]]
                            ) -> List[Optional[Tuple[int,int,int,int]]]:
    """Detect one face per person box using Haar cascade."""
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(haar_path):
        return [None] * len(persons)
    cascade = cv2.CascadeClassifier(haar_path)
    results: List[Optional[Tuple[int,int,int,int]]] = []
    for (px1, py1, px2, py2) in persons:
        px1c, py1c = max(0, px1), max(0, py1)
        px2c, py2c = min(frame.shape[1], px2), min(frame.shape[0], py2)
        if px2c <= px1c or py2c <= py1c:
            results.append(None); continue
        roi_gray = cv2.cvtColor(frame[py1c:py2c, px1c:px2c], cv2.COLOR_BGR2GRAY)
        rw, rh = px2c - px1c, py2c - py1c
        faces = cascade.detectMultiScale(roi_gray, 1.15, 4,
                                         minSize=(max(20, rw//8), max(20, rh//8)))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            results.append((px1c+fx, py1c+fy, px1c+fx+fw, py1c+fy+fh))
        else:
            results.append(None)
    return results


def _detect_lower_third_region(frame: np.ndarray,
                                height_frac: float = PANEL_LOWER_THIRD_HEIGHT_FRAC
                                ) -> Optional[int]:
    """Detect text banner in bottom portion using edge density."""
    h, w = frame.shape[:2]
    y_start = int(h * (1.0 - height_frac))
    roi = frame[y_start:, :]
    if roi.size == 0: return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    if float(edges.mean()) / 255.0 > 0.08:
        return y_start
    return None


def _portrait_crop_from_face(person_box, face_box, frame_h, frame_w,
                              head_ratio=PANEL_PORTRAIT_HEAD_RATIO):
    """Head-and-shoulders crop centered on face."""
    px1, py1, px2, py2 = person_box
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        face_h = max(fy2 - fy1, 1)
        face_cx, face_cy = (fx1+fx2)//2, (fy1+fy2)//2
        crop_h = int(face_h * head_ratio)
        crop_w = int(crop_h * 9 / 16)
        cx1 = max(0, face_cx - crop_w//2)
        cy1 = max(0, face_cy - int(face_h * 0.3))
        return (cx1, cy1, min(frame_w, cx1+crop_w), min(frame_h, cy1+crop_h))
    ph = py2 - py1
    return (px1, py1, px2, min(py1 + int(ph*0.6), frame_h))


class LayoutTransitionManager:
    """
    Manages smooth transitions when panel person count changes.
    - Holdover: keeps layout stable when person briefly disappears
    - Stability: requires N frames of consistent count before switching
    - Cross-dissolve: blends old→new layout during transitions
    """

    def __init__(self, stability_frames: int = 15,
                 transition_frames: int = 6,
                 holdover_frames: int = 8) -> None:
        self.stability_frames = stability_frames
        self.transition_frames = transition_frames
        self.holdover_frames = holdover_frames
        self._current_count = 0
        self._pending_count = 0
        self._pending_frames = 0
        self._transitioning = False
        self._transition_progress = 0
        self._old_frame: Optional[np.ndarray] = None
        self._last_persons: List[Tuple] = []
        self._frames_since_last_seen: Dict[int, int] = {}
        self._person_slots: Dict[int, Tuple] = {}

    def update(self, persons: List[Tuple], frame_shape: Tuple) -> Tuple[List[Tuple], int, bool, float]:
        """
        Returns: (stable_persons, layout_count, is_transitioning, blend_alpha)
        - stable_persons: person list with holdover (missing persons kept)
        - layout_count: which grid to use (1/2/3/4)
        - is_transitioning: True during cross-dissolve
        - blend_alpha: 0.0 = fully old layout, 1.0 = fully new layout
        """
        n = len(persons)

        # Track person persistence by approximate position
        self._update_person_tracking(persons)

        # Add held-over persons (recently disappeared)
        stable_persons = list(persons)
        for slot_id, bbox in list(self._person_slots.items()):
            age = self._frames_since_last_seen.get(slot_id, 999)
            if age > 0 and age <= self.holdover_frames:
                # Person not in current frame but recently seen - keep their slot
                already_covered = any(
                    abs((p[0]+p[2])//2 - (bbox[0]+bbox[2])//2) < 50
                    for p in stable_persons
                )
                if not already_covered:
                    stable_persons.append(bbox)

        stable_n = len(stable_persons)

        # Layout stability check
        if stable_n != self._current_count:
            if stable_n == self._pending_count:
                self._pending_frames += 1
            else:
                self._pending_count = stable_n
                self._pending_frames = 1

            if self._pending_frames >= self.stability_frames:
                # Commit layout change
                self._transitioning = True
                self._transition_progress = 0
                self._current_count = stable_n
                self._pending_count = stable_n
                self._pending_frames = 0
        else:
            self._pending_count = stable_n
            self._pending_frames = 0

        # Progress transition
        blend_alpha = 1.0
        if self._transitioning:
            self._transition_progress += 1
            blend_alpha = min(1.0, self._transition_progress / max(self.transition_frames, 1))
            if self._transition_progress >= self.transition_frames:
                self._transitioning = False

        layout_count = max(1, self._current_count)
        return stable_persons[:layout_count], layout_count, self._transitioning, blend_alpha

    def _update_person_tracking(self, persons: List[Tuple]) -> None:
        """Track person positions across frames for holdover."""
        # Match current persons to known slots by proximity
        matched: set = set()
        for p in persons:
            pcx = (p[0] + p[2]) // 2
            best_id, best_dist = -1, 999999
            for sid, sbbox in self._person_slots.items():
                scx = (sbbox[0] + sbbox[2]) // 2
                dist = abs(pcx - scx)
                if dist < best_dist and sid not in matched:
                    best_dist, best_id = dist, sid
            if best_id >= 0 and best_dist < 200:
                self._person_slots[best_id] = p
                self._frames_since_last_seen[best_id] = 0
                matched.add(best_id)
            else:
                # FIXED: Reuse smallest free non-negative integer instead of monotonic increment
                existing_ids = set(self._person_slots.keys())
                new_id = 0
                while new_id in existing_ids:
                    new_id += 1
                self._person_slots[new_id] = p
                self._frames_since_last_seen[new_id] = 0

        # Age out unmatched slots
        for sid in list(self._frames_since_last_seen.keys()):
            if sid not in matched:
                self._frames_since_last_seen[sid] = self._frames_since_last_seen.get(sid, 0) + 1
                if self._frames_since_last_seen[sid] > self.holdover_frames * 2:
                    del self._frames_since_last_seen[sid]
                    self._person_slots.pop(sid, None)

    def store_old_frame(self, frame: np.ndarray) -> None:
        """Store current rendered frame for cross-dissolve."""
        self._old_frame = frame.copy()

    def blend_transition(self, new_frame: np.ndarray, alpha: float) -> np.ndarray:
        """Blend old and new frames during transition."""
        if self._old_frame is None or alpha >= 1.0:
            return new_frame
        if self._old_frame.shape != new_frame.shape:
            self._old_frame = cv2.resize(self._old_frame,
                                          (new_frame.shape[1], new_frame.shape[0]))
        return cv2.addWeighted(new_frame, alpha, self._old_frame, 1.0 - alpha, 0)


def _crop_group_to_strip(frame: np.ndarray, group: List[Tuple], strip_w: int, strip_h: int,
                         expand: float = PANEL_CROP_EXPAND, vignette_strength: float = 0.0,
                         color_grade: str = "none",
                         face_boxes: Optional[List[Optional[Tuple]]] = None,
                         portrait_mode: bool = False,
                         lower_third_y: Optional[int] = None,
                         upper_body_bias: float = 0.15) -> np.ndarray:
    """
    Crop a person/group to fill a strip. Biases upward for head-and-shoulders framing.
    """
    fh, fw = frame.shape[:2]
    if not group:
        crop = frame
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        if portrait_mode and face_boxes:
            valid_faces = [fb for fb in face_boxes if fb is not None]
            if valid_faces:
                ux1, uy1, ux2, uy2 = _portrait_crop_from_face(
                    (ux1, uy1, ux2, uy2), valid_faces[0], fh, fw)
        ucx = (ux1 + ux2) // 2
        person_h = max(uy2 - uy1, 1)
        ucy = (uy1 + uy2) // 2 - int(person_h * upper_body_bias)
        ucy = max(0, ucy)
        uw = max(ux2 - ux1, 1)
        sr = strip_w / max(strip_h, 1)
        cw = int(max(uw, person_h * sr) * expand)
        ch = int(cw / sr)
        if ch > fh: ch = fh; cw = int(ch * sr)
        if cw > fw: cw = fw; ch = int(cw / sr)
        cw, ch = max(cw, 2), max(ch, 2)
        x1 = max(0, min(ucx - cw // 2, fw - cw))
        y1 = max(0, min(ucy - ch // 2, fh - ch))
        if lower_third_y is not None and (y1 + ch) > lower_third_y:
            y1 = max(0, lower_third_y - ch)
        crop = frame[y1:y1+ch, x1:x1+cw]
        if crop.size == 0: crop = frame
    result = cv2.resize(crop, (strip_w, strip_h), interpolation=cv2.INTER_LANCZOS4)
    if color_grade and color_grade != "none":
        result = apply_color_grade(result, color_grade)
    if vignette_strength > 0:
        result = apply_vignette(result, vignette_strength)
    return result


def _render_panel_frame(frame: np.ndarray, persons: List[Tuple], out_w: int, out_h: int,
                        prev_slots: Optional[List], vignette_strength: float = VIGNETTE_STRENGTH*0.7,
                        color_grade: str = "none",
                        slot_smoother: Optional["DynamicPanelSlotSmoother"] = None,
                        orientation: str = "horizontal",
                        panel_config: Optional[PanelModeConfig] = None,
                        audio_rms_per_person: Optional[List[float]] = None,
                        layout_manager: Optional[LayoutTransitionManager] = None,
                        ) -> Tuple[np.ndarray, List]:
    """
    Render panel frame with intelligent grid layouts and smooth transitions.
    - 1 person:  full frame solo
    - 2 people:  top/bottom 50-50 split
    - 3 people:  top 40% + bottom 60% split (1 top, 2 bottom)
    - 4 people:  2x2 grid
    Transitions smoothly between layouts using LayoutTransitionManager.
    """
    cfg = panel_config or PanelModeConfig()
    persons = sorted(persons, key=lambda b: (b[0]+b[2])//2)
    n = len(persons)

    # Detect faces if needed
    face_boxes = None
    if cfg.head_normalize or cfg.portrait_mode:
        face_boxes = _detect_faces_for_panel(frame, persons[:cfg.max_slots])

    # Detect lower-third
    lower_third_y = None
    if cfg.lower_third_aware:
        lower_third_y = _detect_lower_third_region(frame)

    # Layout transition management
    if layout_manager is not None:
        stable_persons, layout_count, is_transitioning, blend_alpha = \
            layout_manager.update(persons, frame.shape)
        # Use stable person list (includes holdover)
        persons = stable_persons[:cfg.max_slots]
        n = len(persons)
    else:
        layout_count = min(max(n, 1), cfg.max_slots)
        is_transitioning = False
        blend_alpha = 1.0

    # Build groups: each person = 1 group
    groups = []
    if n == 0:
        if prev_slots:
            groups = list(prev_slots)
        else:
            groups = [[(out_w//4, out_h//4, out_w*3//4, out_h*3//4)]]
    else:
        groups = [[p] for p in persons[:cfg.max_slots]]

    # Smooth slot positions
    if slot_smoother is not None and len(groups) >= 2:
        # FIXED: update() now always returns List[List]
        sm = slot_smoother.update(*groups, strip_w=float(out_w))
        groups = [sm[i] for i in range(min(len(sm), len(groups)))]

    n_active = len([g for g in groups if g])
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    div = PANEL_DIVIDER_PX
    div_color = PANEL_DIVIDER_COLOR

    def _crop_person(group, w, h, face_idx=None):
        fb = None
        if face_boxes and face_idx is not None and face_idx < len(face_boxes):
            fb = [face_boxes[face_idx]]
        return _crop_group_to_strip(frame, group, w, h,
                                     vignette_strength=vignette_strength,
                                     color_grade=color_grade,
                                     face_boxes=fb,
                                     portrait_mode=cfg.portrait_mode,
                                     lower_third_y=lower_third_y)

    # Use layout_count (from transition manager) not n_active for grid selection
    effective_count = layout_count if layout_manager else n_active

    if effective_count <= 1:
        # Solo: full frame
        g = groups[0] if groups and groups[0] else []
        canvas[:, :] = _crop_person(g, out_w, out_h, face_idx=0)

    elif effective_count == 2:
        # 2 people: top/bottom 50-50 split
        h_top = out_h // 2
        h_bot = out_h - h_top
        g0 = groups[0] if len(groups) > 0 and groups[0] else []
        g1 = groups[1] if len(groups) > 1 and groups[1] else g0
        canvas[0:h_top, :] = _crop_person(g0, out_w, h_top, face_idx=0)
        canvas[h_top:out_h, :] = _crop_person(g1, out_w, h_bot, face_idx=1)
        d1 = max(0, h_top - div // 2)
        d2 = min(out_h, h_top + (div + 1) // 2)
        canvas[d1:d2, :] = div_color

    elif effective_count == 3:
        # 3 people: top = 1 person (40%), bottom = 2 side-by-side (60%)
        h_top = int(out_h * 0.40)
        h_bot = out_h - h_top
        w_half = out_w // 2
        g0 = groups[0] if len(groups) > 0 and groups[0] else []
        g1 = groups[1] if len(groups) > 1 and groups[1] else g0
        g2 = groups[2] if len(groups) > 2 and groups[2] else g1
        canvas[0:h_top, :] = _crop_person(g0, out_w, h_top, face_idx=0)
        canvas[h_top:out_h, 0:w_half] = _crop_person(g1, w_half, h_bot, face_idx=1)
        canvas[h_top:out_h, w_half:out_w] = _crop_person(g2, out_w - w_half, h_bot, face_idx=2)
        # Horizontal divider
        d1 = max(0, h_top - div // 2)
        d2 = min(out_h, h_top + (div + 1) // 2)
        canvas[d1:d2, :] = div_color
        # Vertical divider in bottom half
        vd1 = max(0, w_half - div // 2)
        vd2 = min(out_w, w_half + (div + 1) // 2)
        canvas[h_top:out_h, vd1:vd2] = div_color

    else:
        # 4+ people: 2x2 grid
        h_half = out_h // 2
        w_half = out_w // 2
        active_groups = [g for g in groups if g][:4]
        while len(active_groups) < 4:
            active_groups.append(active_groups[-1] if active_groups else [])
        canvas[0:h_half, 0:w_half] = _crop_person(active_groups[0], w_half, h_half, face_idx=0)
        canvas[0:h_half, w_half:out_w] = _crop_person(active_groups[1], out_w - w_half, h_half, face_idx=1)
        canvas[h_half:out_h, 0:w_half] = _crop_person(active_groups[2], w_half, out_h - h_half, face_idx=2)
        canvas[h_half:out_h, w_half:out_w] = _crop_person(active_groups[3], out_w - w_half, out_h - h_half, face_idx=3)
        # Horizontal divider
        d1 = max(0, h_half - div // 2)
        d2 = min(out_h, h_half + (div + 1) // 2)
        canvas[d1:d2, :] = div_color
        # Vertical divider
        vd1 = max(0, w_half - div // 2)
        vd2 = min(out_w, w_half + (div + 1) // 2)
        canvas[:, vd1:vd2] = div_color

    # Cross-dissolve during transitions
    if layout_manager is not None:
        if is_transitioning and blend_alpha < 1.0:
            canvas = layout_manager.blend_transition(canvas, blend_alpha)
        layout_manager.store_old_frame(canvas)

    return canvas, [list(g) for g in groups[:max(n_active, 1)]]


# ── Legacy optical flow / saliency ────────────────────────────────────────────
def optical_flow_center(prev: np.ndarray, curr: np.ndarray,
                        w: int, h: int) -> Optional[Tuple[int, int]]:
    if prev is None or curr is None: return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag  = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        b    = max(1, int(w * 0.04))
        mag[:, :b] = mag[:, w-b:] = mag[:b, :] = mag[h-b:, :] = 0
        if mag.max() < 0.8: return None
        t = mag.sum()
        if t == 0: return None
        ys, xs = np.mgrid[0:h, 0:w]
        return int((xs * mag).sum() / t), int((ys * mag).sum() / t)
    except Exception:
        return None

def saliency_center(frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIM or h < MIN_FRAME_DIM: return w//2, h//2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31,31), 0)
    sat  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32), (31,31), 0)
    sal  = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    b    = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w-b:] = sal[:b, :] = sal[h-b:, :] = 0
    t = sal.sum()
    if t < 1e-6: return w//2, h//2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t)


# ── Camera-path smoothing ─────────────────────────────────────────────────────
def _gauss_seg(xs: np.ndarray, ys: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 3: return xs.copy(), ys.copy()
    w = min(window, n-1); w = w if w%2==1 else w-1
    if w < 3: return xs.copy(), ys.copy()
    h2 = w//2; sigma = h2/2.5 + 1e-9
    k  = np.exp(-0.5 * (np.arange(-h2, h2+1) / sigma)**2); k /= k.sum()
    sx = np.convolve(np.pad(xs, h2, "edge"), k, "valid")[:n]
    sy = np.convolve(np.pad(ys, h2, "edge"), k, "valid")[:n]
    return sx, sy

def _bidir_ema(xs: np.ndarray, ys: np.ndarray,
               alpha: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 2: return np.array(xs, dtype=float), np.array(ys, dtype=float)
    def _fwd(v):
        out = np.empty(n, dtype=float); out[0] = v[0]
        for i in range(1, n): out[i] = alpha*v[i] + (1-alpha)*out[i-1]
        return out
    def _bwd(v):
        out = np.empty(n, dtype=float); out[-1] = v[-1]
        for i in range(n-2, -1, -1): out[i] = alpha*v[i] + (1-alpha)*out[i+1]
        return out
    return (_fwd(xs)+_bwd(xs))/2, (_fwd(ys)+_bwd(ys))/2

def _apply_sports_post_smooth(dense_cx: np.ndarray, dense_cy: np.ndarray,
                              fps: float, scene_cuts: List[int],
                              total_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    sw = max(5, int(fps * SPORTS_POST_SMOOTH_WINDOW_SEC))
    if sw % 2 == 0: sw += 1
    damp_cx = dense_cx.copy().astype(float)
    damp_cy = dense_cy.copy().astype(float)

    if total_frames > 2:
        vx = np.diff(damp_cx, prepend=damp_cx[0])
        vy = np.diff(damp_cy, prepend=damp_cy[0])
        # FIXED: use explicit shifted arrays instead of np.roll to avoid
        # circular boundary artifact on frame 0
        prev_vx = np.empty_like(vx); prev_vx[0]  = vx[0];  prev_vx[1:]  = vx[:-1]
        prev_vy = np.empty_like(vy); prev_vy[0]  = vy[0];  prev_vy[1:]  = vy[:-1]
        mask_x  = (np.abs(vx) > np.abs(prev_vx)*SPIKE_THRESH) & (np.abs(prev_vx) > SPIKE_MIN_VEL)
        mask_y  = (np.abs(vy) > np.abs(prev_vy)*SPIKE_THRESH) & (np.abs(prev_vy) > SPIKE_MIN_VEL)
        prev_cx_shifted = np.empty_like(damp_cx)
        prev_cx_shifted[0] = damp_cx[0]; prev_cx_shifted[1:] = damp_cx[:-1]
        prev_cy_shifted = np.empty_like(damp_cy)
        prev_cy_shifted[0] = damp_cy[0]; prev_cy_shifted[1:] = damp_cy[:-1]
        damp_cx[mask_x] = (prev_cx_shifted + prev_vx * SPIKE_DAMP)[mask_x]
        damp_cy[mask_y] = (prev_cy_shifted + prev_vy * SPIKE_DAMP)[mask_y]

    cuts   = sorted({c for c in scene_cuts if 0 < c < total_frames})
    bounds = [0] + list(cuts) + [total_frames]
    out_cx = damp_cx.copy(); out_cy = damp_cy.copy()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i+1]; sl = e - s
        if sl < 5: continue
        w = min(sw, sl-1); w = w if w%2==1 else w-1
        if w < 5: continue
        try:
            po = min(3, w-1)
            if _SCIPY_AVAILABLE:
                seg_cx = savgol_filter(damp_cx[s:e], w, po)
                seg_cy = savgol_filter(damp_cy[s:e], w, po)
            else:
                h2 = w//2; sigma = h2/2.0 + 1e-9
                k  = np.exp(-0.5*(np.arange(-h2, h2+1)/sigma)**2); k /= k.sum()
                seg_cx = np.convolve(np.pad(damp_cx[s:e], h2, "edge"), k, "valid")[:sl]
                seg_cy = np.convolve(np.pad(damp_cy[s:e], h2, "edge"), k, "valid")[:sl]
        except (ValueError, ImportError, RuntimeError):
            seg_cx = damp_cx[s:e]; seg_cy = damp_cy[s:e]
        local_vel      = (np.mean(np.sqrt(np.diff(seg_cx)**2 + np.diff(seg_cy)**2))
                         if len(seg_cx) > 1 else 0)
        adaptive_alpha = min(0.25, max(0.08, SPORTS_POST_SMOOTH_EMA_ALPHA * (1 + local_vel/50)))
        seg_cx, seg_cy = _bidir_ema(seg_cx, seg_cy, alpha=adaptive_alpha)
        out_cx[s:e]    = seg_cx; out_cy[s:e] = seg_cy
    return out_cx, out_cy


def smooth_centers(centers: List[Tuple[int, int]], speeds: List[float],
                   base_window: int = 33, adaptive: bool = True,
                   scene_cuts: Optional[List[int]] = None,
                   use_kalman: bool = False) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    empty = {"jitter_raw":0.0, "jitter_smooth":0.0, "smoothness_pct":0.0,
             "max_jump_raw":0.0, "kalman_prediction_frames":0}
    if not centers or len(centers) < 3:
        return list(centers) if centers else [], empty
    n   = len(centers)
    xs  = np.array([c[0] for c in centers], dtype=float)
    ys  = np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n: spd = np.pad(spd, (0, n-len(spd)), mode="edge")
    dx_raw = np.diff(xs); dy_raw = np.diff(ys)
    dist_r = np.sqrt(dx_raw**2 + dy_raw**2)
    jitter_raw = float(np.mean(dist_r)) if len(dist_r) > 0 else 0.0
    max_jump   = float(np.max(dist_r))  if len(dist_r) > 0 else 0.0
    cuts   = sorted({c for c in (scene_cuts or []) if 0 < c < n})
    bounds = [0] + cuts + [n]
    rx, ry = xs.copy(), ys.copy()
    pred_count = 0
    if use_kalman:
        kalman = SportsKalmanTracker(dt=1.0)
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i+1]
            if e - s < 2: continue
            kalman.init(xs[s], ys[s])
            for j in range(s, e):
                # FIXED: skip predict on first frame to avoid double-predict
                if j > s:
                    kalman._predict_step()
                kx, ky = kalman.update(xs[j], ys[j])
                speed  = spd[j] if j < len(spd) else 0.0
                if speed > 60.0 and not kalman.is_stale:
                    rx[j] = 0.15*xs[j] + 0.85*kx; ry[j] = 0.15*ys[j] + 0.85*ky
                    pred_count += 1
                else:
                    rx[j] = kx; ry[j] = ky
        if n > 5:
            k  = np.exp(-0.5*(np.arange(-1,2)/0.8)**2); k /= k.sum()
            rx = np.convolve(np.pad(rx, 1, "edge"), k, "valid")[:n]
            ry = np.convolve(np.pad(ry, 1, "edge"), k, "valid")[:n]
    else:
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i+1]
            if e - s < 3: continue
            w  = max(_vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window, 13)
            gx, gy = _gauss_seg(xs[s:e], ys[s:e], w)
            bx, by = _bidir_ema(gx, gy, alpha=0.08)
            rx[s:e] = bx; ry[s:e] = by
    smoothed = [(int(x), int(y)) for x, y in zip(rx, ry)]
    dx_s = np.diff(rx); dy_s = np.diff(ry)
    jitter_s = float(np.mean(np.sqrt(dx_s**2 + dy_s**2)))
    pct = max(0.0, min(100.0, (jitter_raw - jitter_s) / jitter_raw * 100 if jitter_raw > 0 else 0.0))
    return smoothed, {"jitter_raw": round(jitter_raw, 2), "jitter_smooth": round(jitter_s, 2),
                      "smoothness_pct": round(pct, 1), "max_jump_raw": round(max_jump, 1),
                      "kalman_prediction_frames": pred_count}

def _compute_final_smoothness(raw_centers: List[Tuple],
                              smoothed_centers: List[Tuple]) -> Dict[str, float]:
    n = min(len(raw_centers), len(smoothed_centers))
    if n < 2:
        return {"jitter_raw": 0.0, "jitter_smooth": 0.0, "smoothness_pct": 0.0}
    rx = np.array([c[0] for c in raw_centers[:n]], dtype=float)
    ry = np.array([c[1] for c in raw_centers[:n]], dtype=float)
    sx = np.array([c[0] for c in smoothed_centers[:n]], dtype=float)
    sy = np.array([c[1] for c in smoothed_centers[:n]], dtype=float)
    d_raw = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2)
    d_smo = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
    jitter_raw    = float(np.mean(d_raw)) if len(d_raw) > 0 else 0.0
    jitter_smooth = float(np.mean(d_smo)) if len(d_smo) > 0 else 0.0
    pct = max(0.0, min(100.0,
              (jitter_raw - jitter_smooth) / jitter_raw * 100 if jitter_raw > 0 else 0.0))
    return {"jitter_raw": round(jitter_raw, 2), "jitter_smooth": round(jitter_smooth, 2),
            "smoothness_pct": round(pct, 1)}


# ── Whisper / translate ───────────────────────────────────────────────────────
def _seconds_to_srt_time(s: float) -> str:
    h  = int(s // 3600); m = int((s % 3600) // 60)
    sc = int(s % 60);    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def transcribe_to_srt(video_path: str, srt_path: str, whisper_model: str = "base",
                      language: Optional[str] = None, max_chars_per_line: int = 42,
                      progress_callback=None) -> bool:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    if not whisper_available(): return False
    import whisper as _w
    _p(0.0, "Extracting audio...")
    fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        if not _extract_audio_wav(video_path, wav_path): return False
        _p(0.2, f"Transcribing ({whisper_model})...")
        mdl    = _w.load_model(whisper_model)
        opts: Dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if language: opts["language"] = language
        result = mdl.transcribe(wav_path, **opts)
        _p(0.85, "Writing subtitles...")
        lines: List[str] = []; idx = 1
        words = [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                 for seg in result.get("segments", []) for w in seg.get("words", [])]
        buf: List[Dict] = []; buf_len = 0
        def _flush():
            nonlocal idx, buf, buf_len
            if not buf: return
            lines.append(f"{idx}\n"
                         f"{_seconds_to_srt_time(buf[0]['start'])} --> "
                         f"{_seconds_to_srt_time(buf[-1]['end'])}\n"
                         f"{' '.join(x['word'] for x in buf)}\n")
            idx += 1; buf = []; buf_len = 0
        for w in words:
            wl = len(w["word"]) + 1
            if buf_len + wl > max_chars_per_line and buf: _flush()
            buf.append(w); buf_len += wl
        _flush()
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n".join(lines))
        _p(1.0, f"{len(lines)} subtitle lines"); return True
    except Exception as e:
        logger.error("Whisper failed: %s", e); return False
    finally:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except OSError: pass

def translate_srt(srt_path: str, target_language: str, source_language: str = "auto",
                  progress_callback=None) -> bool:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    if not translation_available() or not target_language:
        return not bool(target_language)
    try: from deep_translator import GoogleTranslator
    except ImportError: return False
    # FIXED: guard against missing file
    if not os.path.exists(srt_path):
        logger.warning("translate_srt: file not found: %s", srt_path)
        return False
    import re
    try:
        with open(srt_path, "r", encoding="utf-8") as f: content = f.read()
        blocks = re.split(r"\n\n+", content.strip())
        out: List[str] = []; tr = GoogleTranslator(source=source_language, target=target_language)
        for i, block in enumerate(blocks):
            ls = block.strip().splitlines()
            if len(ls) < 3: out.append(block); continue
            try:    translated = tr.translate(" ".join(ls[2:])) or " ".join(ls[2:])
            except Exception: translated = " ".join(ls[2:])
            out.append(f"{ls[0]}\n{ls[1]}\n{translated}")
            if i % 10 == 0: _p(i / max(len(blocks), 1), f"{i}/{len(blocks)}")
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n".join(out) + "\n")
        _p(1.0, "Translation done"); return True
    except Exception as e:
        logger.error("Translation failed: %s", e); return False


# ── Clip detection ────────────────────────────────────────────────────────────
def _compute_audio_energy(input_path: str, duration: float,
                          sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract audio -> compute per-second RMS energy, normalized to [0, 1]."""
    if not _has_audio(input_path):
        return None
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if not _extract_audio_wav(input_path, wav_path):
            return None
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-f", "s16le", "-acodec", "pcm_s16le",
               "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or len(r.stdout) < sample_rate * 2:
            return None
        pcm = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        n_seconds = max(1, int(duration))
        energy = np.zeros(n_seconds, dtype=np.float32)
        for i in range(n_seconds):
            s = i * sample_rate; e = min(s + sample_rate, len(pcm))
            if s >= len(pcm): break
            energy[i] = float(np.sqrt(np.mean(pcm[s:e] ** 2)))
        mx = energy.max()
        if mx > 0: energy /= mx
        return energy
    except Exception as exc:
        logger.debug("Audio energy extraction failed: %s", exc)
        return None
    finally:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except OSError: pass


def _frame_saliency_score(frame: np.ndarray,
                          prev_frame: Optional[np.ndarray]) -> float:
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap    = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)
    motion = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # FIXED: guard against shape mismatch before absdiff
        if prev_gray.shape == gray.shape:
            motion = min(float(cv2.absdiff(gray, prev_gray).mean()) / 30.0, 1.0)
        else:
            logger.debug("Shape mismatch in _frame_saliency_score: %s vs %s",
                         prev_gray.shape, gray.shape)
    sat = min(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].mean()) / 128.0, 1.0)
    return 0.4*motion + 0.4*lap + 0.2*sat

def _compute_frame_scores(input_path: str, fps: float, total_frames: int,
                          orig_w: int, orig_h: int, sample_every: int = 15,
                          progress_callback=None) -> Tuple[np.ndarray, List[int]]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    scores: List[float] = []; scene_cuts: List[int] = []
    prev_gray = prev_frame = None
    sw = min(orig_w, 640); sh = max(1, int(sw * orig_h / orig_w)); fi = 0
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=sw, scale_h=sh) as reader:
            for frame in reader:
                if fi >= total_frames: break
                if fi % sample_every == 0:
                    cg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        # FIXED: guard against shape mismatch before absdiff
                        if prev_gray.shape == cg.shape:
                            if float(cv2.absdiff(prev_gray, cg).mean())/255.0 > 0.30:
                                scene_cuts.append(fi)
                        else:
                            logger.debug("Shape mismatch in _compute_frame_scores: %s vs %s",
                                         prev_gray.shape, cg.shape)
                    scores.append(_frame_saliency_score(frame, prev_frame))
                    prev_gray = cg; prev_frame = frame.copy()
                if fi % max(1, total_frames//20) == 0:
                    _p(fi/total_frames, f"Scanning {fi}/{total_frames}...")
                fi += 1
    except Exception as e:
        logger.error("[scan] Error: %s", e)
    return np.array(scores, dtype=float), scene_cuts

def detect_clips(input_path: str, min_duration_sec: float = 25.0,
                 max_duration_sec: float = 65.0, target_n_clips: int = 10,
                 model: Optional[Any] = None, confidence: float = 0.45,
                 progress_callback=None) -> List[ClipSegment]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    info         = get_video_info(input_path)
    fps          = info["fps"]; total_frames = info["total_frames"]
    duration     = info["duration_seconds"]
    orig_w, orig_h = info["width"], info["height"]
    sample_every = max(1, int(fps))
    _p(0.0, "Scanning...")
    scores, scene_cut_frames = _compute_frame_scores(
        input_path, fps, total_frames, orig_w, orig_h,
        sample_every=sample_every,
        progress_callback=lambda v, m: _p(v*0.45, m))
    if len(scores) == 0: return []
    _p(0.45, "Computing arcs...")
    window = max(5, int(30 / (sample_every / fps)))
    ss     = (np.convolve(scores, np.ones(window)/window, "same")
              if len(scores) >= window else scores.copy())
    if ss.max() > 0: ss /= ss.max()
    min_gap = max(1, int(min_duration_sec * fps / sample_every))
    peaks: List[int] = []
    for i in range(1, len(ss)-1):
        wh = min_gap//2; lo = max(0, i-wh); hi = min(len(ss), i+wh+1)
        if ss[i] == ss[lo:hi].max() and ss[i] > 0.3:
            if not peaks or i - peaks[-1] > min_gap//2:
                peaks.append(i)
    peaks.sort(key=lambda i: ss[i], reverse=True); peaks = peaks[:target_n_clips*2]
    def arc(pi):
        ps = pi * sample_every / fps; rs = max(0.0, ps - max_duration_sec*0.4)
        re = min(duration, rs + max_duration_sec)
        for sc in reversed(scene_cut_frames):
            sc_s = sc / fps
            if 0 < ps - sc_s < 15.0: rs = max(0.0, sc_s - 1.0); break
        for sc in scene_cut_frames:
            sc_s = sc / fps
            if 0 < sc_s - ps < 15.0: re = min(duration, sc_s + 0.5); break
        cd = re - rs
        if cd < min_duration_sec: re = min(duration, rs + min_duration_sec)
        elif cd > max_duration_sec:
            c = (rs+re)/2; rs = max(0.0, c-max_duration_sec/2)
            re = min(duration, rs+max_duration_sec)
        return rs, re
    cands: List[Tuple[float, float, float]] = []
    for pi in peaks:
        s, e = arc(pi); sc = float(ss[pi])
        if not any(min(e,ce)-max(s,cs) > min_duration_sec*0.5 for cs,ce,_ in cands):
            cands.append((s, e, sc))
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:target_n_clips]; cands.sort(key=lambda x: x[0])
    _p(0.55, "SOI per clip...")
    segments: List[ClipSegment] = []
    det_w = min(orig_w, 640); det_h = max(1, int(det_w * orig_h / orig_w))
    for ci, (ss2, se, score) in enumerate(cands):
        _p(0.55 + 0.35*(ci/max(len(cands), 1)), f"Clip {ci+1}/{len(cands)}...")
        soi_xs: List[int] = []; soi_ys: List[int] = []
        n_s = min(8, max(2, int(se - ss2)))
        for t in np.linspace(ss2+1, se-1, n_s):
            frame = _read_frame_at(input_path, orig_w, orig_h, t, scale_w=det_w, scale_h=det_h)
            if frame is None: continue
            if model is not None:
                try:
                    res = model(frame, verbose=False, conf=confidence)[0]
                    if res.boxes is not None:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            soi_xs.append((x1+x2)//2); soi_ys.append((y1+y2)//2)
                except (RuntimeError, ValueError, OSError): pass
            else:
                scx, scy = saliency_center(frame); soi_xs.append(scx); soi_ys.append(scy)
        sr = "center"
        if soi_xs: sr = _soi_region_label(int(np.median(soi_xs)), int(np.median(soi_ys)), orig_w, orig_h)
        ms = int(ss2//60); secs = int(ss2%60); me = int(se//60); sece = int(se%60)
        # ClipSegment validates end > start, so safe
        segments.append(ClipSegment(start_sec=ss2, end_sec=se, score=score,
                                    soi_region=sr,
                                    peak_frame=int(np.linspace(ss2+1, se-1, n_s)[n_s//2]*fps),
                                    title=f"Clip {ci+1} ({ms}:{secs:02d} - {me}:{sece:02d})"))
    _p(1.0, f"Found {len(segments)} clips")
    return segments


# ── Analytics ─────────────────────────────────────────────────────────────────
def _build_analytics(input_path: str, output_path: str, orig_w: int, orig_h: int,
                     out_w: int, out_h: int, smooth_metrics: Optional[Dict] = None,
                     panel_mode: bool = False, kalman_predictions: int = 0,
                     resource_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    def _sz(p): return os.path.getsize(p) / (1024*1024) if os.path.exists(p) else 0.0
    def _br(p):
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=bit_rate", "-of",
                                "default=noprint_wrappers=1:nokey=1", p],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.strip()) // 1000
        except (OSError, subprocess.SubprocessError, ValueError): return 0
    in_sz  = _sz(input_path); out_sz = _sz(output_path)
    jitter_raw = jitter_smooth = smoothness_pct = 0.0
    if smooth_metrics:
        jitter_raw    = smooth_metrics.get("jitter_raw",    0.0)
        jitter_smooth = smooth_metrics.get("jitter_smooth", 0.0)
        smoothness_pct = max(0.0, min(100.0,
            (jitter_raw - jitter_smooth) / jitter_raw * 100
            if jitter_raw > 0 else smooth_metrics.get("smoothness_pct", 0.0)))
    a: Dict[str, Any] = {
        "input_size_mb":           round(in_sz, 2),
        "output_size_mb":          round(out_sz, 2),
        "compression_ratio":       round(in_sz/out_sz, 2) if out_sz > 0 else 0.0,
        "file_size_reduction_pct": round((1-out_sz/in_sz)*100, 1) if in_sz > 0 else 0.0,
        "input_resolution":        f"{orig_w}x{orig_h}",
        "output_resolution":       f"{out_w}x{out_h}",
        "input_bitrate_kbps":      _br(input_path),
        "output_bitrate_kbps":     _br(output_path),
        "panel_mode":              panel_mode,
        "kalman_predictions":      kalman_predictions,
        "jitter_raw":              round(jitter_raw, 2),
        "jitter_smooth":           round(jitter_smooth, 2),
        "smoothness_pct":          round(smoothness_pct, 1),
    }
    if resource_stats:
        a.update(resource_stats)
    return a


# ── Shared pipeline setup helper (DRY) ───────────────────────────────────────
def _get_device(device="auto"):
    '''Return compute device string for torch operations.'''
    if device == "auto":
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def seconds_to_frame(ts: float, fps: float) -> int:
    '''Convert timestamp (seconds) to frame index with rounding.'''
    return int(round(ts * fps))


def _common_pipeline_setup(input_path: str, target_preset_label: str,
                            yolo_weights: str,
                            load_model: bool = True,
                            ) -> Tuple[Dict[str, Any], int, int, int, int, float, int,
                                       Optional[Any], "ResourceMonitor"]:
    """
    FIXED: Eliminates ~40 lines of duplicated boilerplate shared between
    process_video and process_sports_video.
    Returns: (info, orig_w, orig_h, out_w, out_h, fps, total_frames, model, res_mon)
    """
    res_mon = ResourceMonitor()
    res_mon.start()
    _check_ffmpeg()
    info     = get_video_info(input_path)
    orig_w, orig_h = info["width"], info["height"]
    fps            = info["fps"]
    total_frames   = info["total_frames"]
    out_w, out_h   = resolve_target_size(target_preset_label, orig_w, orig_h)
    model = _get_model(yolo_weights) if load_model else None
    return info, orig_w, orig_h, out_w, out_h, fps, total_frames, model, res_mon


# ── Core rendering engine ─────────────────────────────────────────────────────
def _render_video(input_path: str, output_path: str,
                  out_w: int, out_h: int, crop_w: int, crop_h: int,
                  orig_w: int, orig_h: int, fps: float, total_frames: int,
                  smoothed_centers: List[Tuple[int, int]],
                  tracking_mode: str = "subject", crf: int = 23,
                  encoder_preset: str = "fast", audio_bitrate: str = "128k",
                  burn_subtitles: bool = False, whisper_model: str = "base",
                  whisper_language: Optional[str] = None,
                  subtitle_style_name: str = "Bold White (TikTok)",
                  subtitle_max_chars: int = 42,
                  subtitle_translate_to: Optional[str] = None,
                  output_fps: Optional[float] = None,
                  color_grade: str = "none", vignette_strength: float = 0.0,
                  sharpen_strength: float = 0.0, ffmpeg_sharpen: bool = False,
                  progress_callback=None, scene_cuts: Optional[List[int]] = None,
                  use_panel_mode: bool = False,
                  panel_config: Optional[PanelModeConfig] = None,
                  panel_persons_map: Optional[Dict[int, List]] = None,
                  ball_records: Optional[Dict[int, BallFrameRecord]] = None,
                  draw_tracking_boxes: bool = False,
                  overlay_config: Optional[OverlayConfig] = None,
                  ) -> Dict[str, Any]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    eff_overlay: Optional[OverlayConfig] = (overlay_config if overlay_config is not None
                                            else OverlayConfig() if draw_tracking_boxes
                                            else None)
    eff_fps = output_fps or fps
    srt_path: Optional[str] = None
    if burn_subtitles and whisper_available():
        fd, srt_path = tempfile.mkstemp(suffix=".srt"); os.close(fd)
        _p(0.02, "Transcribing...")
        ok = transcribe_to_srt(input_path, srt_path, whisper_model=whisper_model,
                               language=whisper_language, max_chars_per_line=subtitle_max_chars,
                               progress_callback=lambda v, m: _p(0.02+v*0.08, m))
        if ok and subtitle_translate_to:
            translate_srt(srt_path, subtitle_translate_to,
                          progress_callback=lambda v, m: _p(0.10+v*0.03, m))
        if not ok:
            try: os.unlink(srt_path)
            except OSError: pass
            srt_path = None

    extra_vf       = _build_ffmpeg_vf(color_grade, ffmpeg_sharpen)
    subtitle_style = SUBTITLE_STYLES.get(subtitle_style_name,
                                         SUBTITLE_STYLES["Bold White (TikTok)"])
    enc = _open_ffmpeg_encoder(
        output_path, out_w, out_h, eff_fps, audio_source=input_path,
        crf=crf, preset=encoder_preset, audio_bitrate=audio_bitrate,
        subtitle_path=srt_path, subtitle_style=subtitle_style, extra_vf=extra_vf,
        source_fps=fps)  # FIXED: pass actual source frame rate

    dissolve       = DissolveBuffer(DISSOLVE_FRAMES)
    slot_smoother  = PanelSlotSmoother() if use_panel_mode else None
    layout_mgr     = LayoutTransitionManager() if use_panel_mode else None
    prev_slots     = None
    orientation    = panel_config.split_orientation if panel_config else "horizontal"
    scene_cut_set  = set(scene_cuts or [])

    # Ball trail stores original-space coordinates (v7.2 fix retained)
    ball_trail_buf: deque = deque(maxlen=(eff_overlay.trail_length if eff_overlay else 20))

    fi = 0
    _prev_out_frame: Optional[np.ndarray] = None   # last rendered output frame (output-res)
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames: break

                # Scene cut: clear trail and prime dissolve with the PREVIOUS
                # output-resolution frame — not the raw orig-res input frame.
                if fi in scene_cut_set:
                    ball_trail_buf.clear()
                    if _prev_out_frame is not None:
                        # FIXED: ensure _prev_out_frame matches current output dimensions
                        # before passing to dissolve.on_cut (panel mode can change sizes)
                        if _prev_out_frame.shape[:2] == (out_h, out_w):
                            dissolve.on_cut(_prev_out_frame)
                        else:
                            resized = cv2.resize(_prev_out_frame, (out_w, out_h),
                                                 interpolation=cv2.INTER_LINEAR)
                            dissolve.on_cut(resized)

                if use_panel_mode:
                    persons = (panel_persons_map or {}).get(fi, [])
                    if not persons:
                        scx, scy = (smoothed_centers[fi] if fi < len(smoothed_centers)
                                    else (orig_w//2, orig_h//2))
                        hw = min(crop_w//2, orig_w//4); hh = min(crop_h//4, orig_h//4)
                        persons = [(scx-hw, scy-hh, scx+hw, scy+hh)]
                    try:
                        out_frame, prev_slots = _render_panel_frame(
                            frame, persons, out_w, out_h, prev_slots=prev_slots,
                            slot_smoother=slot_smoother, orientation=orientation,
                            panel_config=panel_config, layout_manager=layout_mgr)
                    except (IndexError, ValueError):
                        cx, cy = (smoothed_centers[fi] if fi < len(smoothed_centers)
                                  else (orig_w//2, orig_h//2))
                        hw, hh = crop_w//2, crop_h//2
                        x1 = int(np.clip(cx-hw, 0, orig_w-crop_w))
                        y1 = int(np.clip(cy-hh, 0, orig_h-crop_h))
                        x2 = min(x1+crop_w, orig_w)
                        y2 = min(y1+crop_h, orig_h)
                        out_frame = cv2.resize(
                            frame[y1:y2, x1:x2], (out_w, out_h),
                            interpolation=cv2.INTER_LANCZOS4)
                else:
                    cx, cy = (smoothed_centers[fi] if fi < len(smoothed_centers)
                              else (orig_w//2, orig_h//2))
                    hw, hh = crop_w//2, crop_h//2
                    x1 = int(np.clip(cx-hw, 0, orig_w-crop_w))
                    y1 = int(np.clip(cy-hh, 0, orig_h-crop_h))
                    x2 = min(x1+crop_w, orig_w); y2 = min(y1+crop_h, orig_h)
                    x1 = max(0, x2-crop_w);      y1 = max(0, y2-crop_h)
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] == 0 or crop.shape[1] == 0:
                        crop = (frame[:crop_h, :crop_w]
                                if orig_h >= crop_h and orig_w >= crop_w else frame)
                    out_frame = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                    if eff_overlay is not None:
                        sx = out_w / crop_w; sy = out_h / crop_h
                        ball_rec_out: Optional[BallFrameRecord] = None
                        if ball_records and fi in ball_records:
                            br = ball_records[fi]
                            if (br.bbox is not None and
                                    br.confidence >= eff_overlay.min_ball_confidence):
                                bx1, by1, bx2, by2 = br.bbox
                                cbx1 = bx1-x1; cby1 = by1-y1
                                cbx2 = bx2-x1; cby2 = by2-y1
                                if cbx2 > 0 and cby2 > 0 and cbx1 < crop_w and cby1 < crop_h:
                                    ball_rec_out = BallFrameRecord(
                                        bbox=(int(cbx1*sx), int(cby1*sy),
                                              int(cbx2*sx), int(cby2*sy)),
                                        confidence=br.confidence,
                                        source=br.source,
                                    )
                                    # FIXED: Store trail in output-relative coordinates at detection time
                        # to avoid drift when the crop window moves
                        obx = (bx1 + bx2) // 2
                        oby = (by1 + by2) // 2
                        out_tx = int((obx - x1) * sx)
                        out_ty = int((oby - y1) * sy)
                        ball_trail_buf.append((out_tx, out_ty))

                        # Trail already in output coordinates, just clip to bounds
                        output_trail: List[Tuple[int, int]] = [
                            (tx, ty) for tx, ty in ball_trail_buf
                            if 0 <= tx < out_w and 0 <= ty < out_h
                        ]

                        if ball_rec_out is not None:
                            out_frame = _draw_tracking_overlays(
                                out_frame, ball_rec_out,
                                overlay_cfg=eff_overlay, ball_trail=output_trail)

                if vignette_strength > 0:
                    out_frame = apply_vignette(out_frame, vignette_strength)
                if sharpen_strength > 0 and not ffmpeg_sharpen:
                    out_frame = apply_sharpen(out_frame, sharpen_strength)

                if dissolve.active:
                    out_frame = dissolve.blend(out_frame)

                # Save output-res frame for next scene-cut dissolve
                _prev_out_frame = out_frame

                # Frame validation before encoder writes
                if (
                    out_frame is None
                    or out_frame.size == 0
                    or len(out_frame.shape) != 3
                ):
                    raise RuntimeError(
                        f"Invalid frame at {fi}"
                    )
                h, w = out_frame.shape[:2]
                if h != out_h or w != out_w:
                    raise RuntimeError(
                        f"Frame mismatch at {fi}: {w}x{h}"
                    )

                try: enc.stdin.write(out_frame.tobytes())
                except BrokenPipeError as e:
                    stderr_output = ""
                    try:
                        if enc.stderr:
                            stderr_output = enc.stderr.read().decode(errors="ignore")[-4000:]
                    except Exception:
                        pass
                    logger.error(
                        "FFmpeg encoder failed at frame %d\n%s",
                        fi,
                        stderr_output
                    )
                    raise RuntimeError(
                        f"FFmpeg pipe broke at frame {fi}"
                    )
                fi += 1
                if fi % max(1, total_frames//50) == 0:
                    _p(0.15 + (fi/total_frames)*0.80, f"Rendering {fi}/{total_frames}...")
    finally:
        # FIXED: subtitle tempfile cleaned up even if encoder raises
        _enc_error: Optional[Exception] = None
        try:
            _close_ffmpeg_encoder(enc, output_path)
        except ProcessingError as _e:
            _enc_error = _e
        if srt_path and os.path.exists(srt_path):
            try: os.unlink(srt_path)
            except OSError: pass
            srt_path = None   # don't return stale path
        if _enc_error:
            raise _enc_error

    return {}


# ── First-pass tracking ───────────────────────────────────────────────────────
def _tracking_pass(input_path: str, orig_w: int, orig_h: int, crop_w: int, crop_h: int,
                   fps: float, total_frames: int, tracking_mode: str, model: Any,
                   confidence: float, smooth_window: int, adaptive_smoothing: bool,
                   use_optical_flow: bool, rule_of_thirds: bool,
                   scene_cut_threshold: float, talking_head_bias: float = 0.30,
                   use_kalman: bool = False, panel_mode_active: bool = False,
                   progress_callback=None) -> Tuple[List, List, List, Dict, int]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    hw, hh = crop_w//2, crop_h//2
    centers: List[Tuple[int,int]] = []; speeds: List[float] = []
    scene_cuts: List[int] = []; persons_map: Dict[int, List] = {}
    prev_gray = prev_frame = prev_cx = prev_cy = prev_hist = None
    last_cut_frame = -100; prev_ball_carrier = None; prev_saliency = None
    det_scale = min(1.0, 960/orig_w)
    det_w = max(1, int(orig_w*det_scale)); det_h = max(1, int(orig_h*det_scale))
    fi = 0
    yolo_failures = 0
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=det_w, scale_h=det_h) as reader:
            for det_frame in reader:
                if fi >= total_frames: break
                gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
                is_cut, prev_hist, last_cut_frame = is_scene_change(
                    prev_frame, det_frame, threshold=scene_cut_threshold,
                    prev_hist=prev_hist, frame_count=fi, last_cut_frame=last_cut_frame,
                    mode="sports" if tracking_mode == "sports_action" else "default")
                if is_cut: scene_cuts.append(fi); prev_gray = prev_cx = prev_cy = None
                cx = cy = None
                if tracking_mode == "talking_head":
                    faces = detect_faces(det_frame)
                    if faces:
                        sf = [(int(f[0]/det_scale), int(f[1]/det_scale),
                               int(f[2]/det_scale), int(f[3]/det_scale)) for f in faces]
                        r  = talking_head_center(sf, orig_w, orig_h, crop_w, crop_h,
                                                 bias=talking_head_bias)
                        if r: cx, cy = r
                elif tracking_mode == "sports_action":
                    dr, bb, bc = detect_subjects(det_frame, model, confidence,
                                                 prev_center=((int(prev_cx*det_scale),
                                                               int(prev_cy*det_scale))
                                                              if prev_cx is not None else None),
                                                 prev_ball_carrier=prev_ball_carrier,
                                                 tracking_mode="sports_action")
                    if dr: cx = int(dr.cx/det_scale); cy = int(dr.cy/det_scale); prev_ball_carrier = bc
                    if panel_mode_active:
                        ap = detect_persons_all(det_frame, model, confidence)
                        persons_map[fi] = [(int(p[0]/det_scale), int(p[1]/det_scale),
                                            int(p[2]/det_scale), int(p[3]/det_scale)) for p in ap]
                else:
                    dr, _, _ = detect_subjects(det_frame, model, confidence,
                                               prev_center=((int(prev_cx*det_scale),
                                                             int(prev_cy*det_scale))
                                                            if prev_cx is not None else None),
                                               tracking_mode="subject")
                    if dr: cx = int(dr.cx/det_scale); cy = int(dr.cy/det_scale)
                    if panel_mode_active:
                        ap = detect_persons_all(det_frame, model, confidence)
                        persons_map[fi] = [(int(p[0]/det_scale), int(p[1]/det_scale),
                                            int(p[2]/det_scale), int(p[3]/det_scale)) for p in ap]
                if cx is None and use_optical_flow and prev_gray is not None:
                    of = optical_flow_center(prev_gray, gray, det_w, det_h)
                    if of: cx = int(of[0]/det_scale); cy = int(of[1]/det_scale)
                if cx is None:
                    scx, scy, prev_saliency = temporal_saliency_center(det_frame, prev_saliency)
                    cx = int(scx/det_scale); cy = int(scy/det_scale)
                cx = max(hw, min(cx, orig_w-hw)); cy = max(hh, min(cy, orig_h-hh))
                if rule_of_thirds and prev_cx is not None:
                    rot_cx = orig_w//3 if cx < orig_w//2 else 2*orig_w//3
                    cx = max(hw, min(int(cx*(1-ROT_BIAS_WEIGHT) + rot_cx*ROT_BIAS_WEIGHT), orig_w-hw))
                cy = _apply_lower_third_guard(cy, crop_h, cy, orig_h)
                cy = max(hh, min(cy, orig_h-hh))
                speed = math.hypot(cx-prev_cx, cy-prev_cy) if prev_cx is not None else 0.0
                centers.append((cx, cy)); speeds.append(speed)
                prev_cx = cx; prev_cy = cy; prev_gray = gray.copy()
                # FIXED: store a copy to prevent aliasing with FFmpegVideoReader buffer
                prev_frame = det_frame.copy()
                if fi % max(1, total_frames//50) == 0:
                    _p(fi/total_frames, f"Tracking {fi}/{total_frames}...")
                fi += 1
    except Exception as e:
        logger.error("[tracking_pass] Error at frame %d: %s", fi, e)
        while len(centers) < total_frames:
            centers.append(centers[-1] if centers else (orig_w//2, orig_h//2))
            speeds.append(0.0)
    return centers, speeds, scene_cuts, persons_map, 0  # FIXED: 5-tuple to match caller


# ── Optimized sports tracking pass ───────────────────────────────────────────
def _sports_tracking_pass_optimized(
    input_path: str, orig_w: int, orig_h: int, crop_w: int, crop_h: int,
    fps: float, total_frames: int, model: Any, confidence: float,
    use_ball_tracking: bool, use_optical_flow: bool,
    field_mask: Optional[np.ndarray],
    mot_tracker: MultiObjectSportsTracker,
    avs_smoother: AdaptiveVelocityAwareSmoother,
    ics: IntelligentCropStrategy,
    phase_detector: SportsPlayPhaseDetector,
    min_ball_confidence: float = SPORTS_BALL_CONFIDENCE,
    progress_callback=None,
) -> Tuple[
    List[Tuple[int,int]],
    List[float],
    List[int],
    Dict[int, BallFrameRecord],
    Dict[int, List[Tuple[int,int,int,int]]],
]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    # Ensure minimum 640px width for vertical inputs
    det_scale = min(1.0, 960 / orig_w)
    if orig_w < 640:
        det_scale = 640 / orig_w
    det_w = max(1, int(orig_w * det_scale))
    det_h = max(1, int(orig_h * det_scale))
    hw, hh = crop_w//2, crop_h//2

    raw_centers:     List[Tuple[int,int]]                          = []
    speeds:          List[float]                                   = []
    scene_cuts:      List[int]                                     = []
    ball_records:    Dict[int, BallFrameRecord]                    = {}
    person_boxes_map: Dict[int, List[Tuple[int,int,int,int]]]      = {}

    det_cache    = DetectionCache()
    ball_tracker = BallROITracker(BALL_TRACKER_TYPE, max_age=BALL_ROI_MAX_AGE_FRAMES)
    ball_kalman  = BallKalmanFilter(fps, det_h)
    ball_color_model = BallColorModel(n_build=BALL_COLOR_MODEL_BUILD_FRAMES)
    ball_color_det   = BallColorDetector(ball_color_model)

    prev_frame:       Optional[np.ndarray] = None
    prev_hist:        Optional[np.ndarray] = None
    prev_gray:        Optional[np.ndarray] = None
    last_cut_frame    = -100
    prev_cx:          Optional[float] = None
    prev_cy:          Optional[float] = None
    prev_ball_carrier: Optional[int]  = None
    current_phase     = PlayPhase.HALF_COURT

    ball_found_ever   = False
    tracker_lost_ball = False
    yolo_skip         = SPORTS_YOLO_SKIP_BASE
    frames_since_yolo = 0

    ball_size_history:  deque = deque(maxlen=10)
    expected_ball_area: float = 400.0

    fi = 0
    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h,
                               scale_w=det_w, scale_h=det_h) as reader:
            for det_frame in reader:
                if len(raw_centers) >= total_frames:
                    break
                fi = len(raw_centers)
                gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

                is_cut, prev_hist, last_cut_frame = is_sports_scene_change(
                    prev_frame, det_frame, prev_hist, fi, last_cut_frame)
                if is_cut:
                    scene_cuts.append(fi)
                    # FIXED: use .reset() methods, not .__init__(), to preserve params
                    mot_tracker.reset()
                    avs_smoother.reset()
                    ball_tracker.reset()
                    ball_kalman.reset()
                    ball_color_model.reset()
                    det_cache.reset()
                    prev_cx = prev_cy = None
                    frames_since_yolo = yolo_skip
                    tracker_lost_ball = False
                    ball_found_ever   = False
                    ball_size_history.clear()
                    phase_detector.reset()

                ball_kalman.new_frame()

                run_yolo = (
                    fi == 0
                    or not ball_found_ever
                    or tracker_lost_ball
                    or frames_since_yolo >= yolo_skip
                    or det_cache.frame_idx < 0
                    or is_cut
                    or (ball_found_ever and
                        det_cache.ball_confidence < min_ball_confidence * 0.8)
                )

                frames_since_yolo = 0 if run_yolo else frames_since_yolo + 1
                tracker_lost_ball = False

                this_ball_rec = BallFrameRecord(bbox=None, confidence=0.0, source="none")

                if run_yolo and model is not None:
                    try:
                        results = model(det_frame, verbose=False, conf=confidence)[0]
                        persons_raw, balls_raw, p_confs = _parse_yolo_results(
                            results.boxes, 1.0, confidence)
                        persons_orig = [
                            (int(p[0]/det_scale), int(p[1]/det_scale),
                             int(p[2]/det_scale), int(p[3]/det_scale))
                            for p in persons_raw
                        ]
                        ball_box_orig:  Optional[Tuple[int,int,int,int]] = None
                        ball_conf_yolo = 0.0
                        if balls_raw and use_ball_tracking:
                            for bb in balls_raw:
                                if bb[6] < min_ball_confidence:
                                    continue
                                cand_bbox_det = (bb[0], bb[1], bb[2], bb[3])
                                if not _validate_ball_detection(
                                    cand_bbox_det, det_frame, ball_kalman,
                                    ball_color_model):
                                    continue
                                ball_conf_yolo = bb[6]
                                ball_box_orig  = (int(bb[0]/det_scale), int(bb[1]/det_scale),
                                                  int(bb[2]/det_scale), int(bb[3]/det_scale))
                                ball_found_ever = True
                                bw = bb[2]-bb[0]; bh = bb[3]-bb[1]
                                ball_size_history.append((bw, bh))
                                if ball_size_history:
                                    expected_ball_area = float(
                                        np.mean([s[0]*s[1] for s in ball_size_history]))
                                bcx_det = (bb[0]+bb[2])/2.0; bcy_det = (bb[1]+bb[3])/2.0
                                if not ball_kalman.initialized:
                                    ball_kalman.init(bcx_det, bcy_det)
                                else:
                                    ball_kalman.update(bcx_det, bcy_det)
                                ball_color_model.add_sample(det_frame, cand_bbox_det)
                                vkx, vky = ball_kalman.velocity
                                ball_tracker.init(det_frame, cand_bbox_det,
                                                  velocity=(vkx, vky),
                                                  confidence=ball_conf_yolo)
                                break

                        this_ball_rec = BallFrameRecord(
                            bbox=ball_box_orig,
                            confidence=ball_conf_yolo,
                            source="yolo" if ball_box_orig is not None else "none",
                        )
                        ball_carrier = -1
                        if ball_box_orig is not None and persons_orig:
                            bcx = (ball_box_orig[0]+ball_box_orig[2])/2
                            bcy = (ball_box_orig[1]+ball_box_orig[3])/2
                            min_d = float("inf")
                            for i, p in enumerate(persons_orig):
                                pcx = (p[0]+p[2])/2; pcy = (p[1]+p[3])/2
                                d = math.hypot(pcx-bcx, pcy-bcy)
                                if d < min_d and d < SPORTS_BALL_PROXIMITY_PX:
                                    min_d, ball_carrier = d, i
                        det_cache.update(persons_orig, ball_box_orig, ball_carrier,
                                         None, p_confs, fi, ball_conf_yolo)
                        mot_tracker.update(persons_orig, ball_box_orig, det_frame, p_confs)
                    except Exception as e:
                        logger.warning("[yolo] frame %d: %s", fi, e)

                else:
                    # FIXED: Unconditional predict once per frame (after new_frame)
                    ball_kalman.predict()

                    resolved = False
                    if use_ball_tracking and ball_tracker.is_active:
                        tracked_bb_det = ball_tracker.update(det_frame)
                        if tracked_bb_det is not None:
                            roi_cx   = (tracked_bb_det[0]+tracked_bb_det[2])/2.0
                            roi_cy   = (tracked_bb_det[1]+tracked_bb_det[3])/2.0
                            roi_area = ((tracked_bb_det[2]-tracked_bb_det[0]) *
                                        (tracked_bb_det[3]-tracked_bb_det[1]))
                            gate_ok  = (
                                (not ball_kalman.initialized or
                                 ball_kalman.gate_distance(roi_cx, roi_cy) <= BALL_MAX_GATE_PX)
                                and BALL_MIN_AREA_PX2 <= roi_area <= BALL_MAX_AREA_PX2
                            )
                            if gate_ok:
                                ball_kalman.update(roi_cx, roi_cy)
                                tracked_ball_orig = (int(tracked_bb_det[0]/det_scale),
                                                     int(tracked_bb_det[1]/det_scale),
                                                     int(tracked_bb_det[2]/det_scale),
                                                     int(tracked_bb_det[3]/det_scale))
                                this_ball_rec = BallFrameRecord(
                                    bbox=tracked_ball_orig,
                                    confidence=ball_tracker.confidence,
                                    source="roi",
                                )
                                det_cache.ball_box = tracked_ball_orig
                                resolved = True
                            else:
                                ball_tracker.reset(); tracker_lost_ball = True
                        else:
                            tracker_lost_ball = True; det_cache.ball_box = None

                    if not resolved and use_ball_tracking and ball_found_ever:
                        pred_center = ball_kalman.position if ball_kalman.initialized else None
                        color_bbox  = ball_color_det.detect(
                            det_frame, pred_center,
                            search_radius=int(BALL_MAX_GATE_PX),
                            expected_area=expected_ball_area)
                        if color_bbox is not None:
                            ccx = (color_bbox[0]+color_bbox[2])/2.0
                            ccy = (color_bbox[1]+color_bbox[3])/2.0
                            ball_kalman.update(ccx, ccy)
                            color_orig = (int(color_bbox[0]/det_scale), int(color_bbox[1]/det_scale),
                                          int(color_bbox[2]/det_scale), int(color_bbox[3]/det_scale))
                            this_ball_rec = BallFrameRecord(
                                bbox=color_orig, confidence=0.45, source="color")
                            det_cache.ball_box = color_orig
                            resolved = True

                    if not resolved:
                        if ball_kalman.initialized and ball_found_ever:
                            kpx, kpy = ball_kalman.position
                            avg_w = avg_h = 40
                            if ball_size_history:
                                avg_w = int(np.mean([s[0] for s in ball_size_history]))
                                avg_h = int(np.mean([s[1] for s in ball_size_history]))
                            rx = avg_w // 2; ry = avg_h // 2
                            kal_orig = (int((kpx-rx)/det_scale), int((kpy-ry)/det_scale),
                                        int((kpx+rx)/det_scale), int((kpy+ry)/det_scale))
                            this_ball_rec = BallFrameRecord(
                                bbox=kal_orig, confidence=0.20, source="kalman")
                            det_cache.ball_box = kal_orig
                        else:
                            this_ball_rec = BallFrameRecord(
                                bbox=None, confidence=0.0, source="none")
                            det_cache.ball_box = None

                    # FIXED: Pass empty lists on non-YOLO frames so tracker predicts
                    # tracks forward via internal velocity instead of snapping to stale positions
                    if run_yolo:
                        mot_tracker.update(det_cache.persons, det_cache.ball_box,
                                           det_frame, det_cache.confidences)
                    else:
                        mot_tracker.update([], det_cache.ball_box,
                                           det_frame, [])

                ball_records[fi]     = this_ball_rec
                person_boxes_map[fi] = list(det_cache.persons)

                current_phase = phase_detector.detect_phase(
                    det_cache.persons, det_cache.ball_box, orig_w, mot_tracker.ball_state)

                yolo_skip = (SPORTS_YOLO_SKIP_FASTBREAK if current_phase == PlayPhase.FAST_BREAK
                             else SPORTS_YOLO_SKIP_MAX   if current_phase == PlayPhase.STATIC
                             else SPORTS_YOLO_SKIP_BASE)

                ball_center   = mot_tracker.ball_state.center
                primary_track = mot_tracker.get_primary_track(prev_ball_carrier)

                if primary_track is not None:
                    raw_cx = float(primary_track.center[0])
                    raw_cy = float(primary_track.center[1])
                    if primary_track.id == mot_tracker.ball_state.possessor_track_id:
                        prev_ball_carrier = primary_track.id
                elif ball_center is not None and use_ball_tracking:
                    active_tracks = [t for t in mot_tracker.tracks.values()
                                     if t.status == TrackingStatus.ACTIVE]
                    if active_tracks:
                        nearest = min(active_tracks, key=lambda t: math.hypot(
                            t.center[0]-ball_center[0], t.center[1]-ball_center[1]))
                        raw_cx = (ball_center[0]*BALL_BLEND_BALL_WEIGHT +
                                  nearest.center[0]*BALL_BLEND_PLAYER_WEIGHT)
                        raw_cy = (ball_center[1]*BALL_BLEND_BALL_WEIGHT +
                                  nearest.center[1]*BALL_BLEND_PLAYER_WEIGHT)
                    else:
                        raw_cx = float(ball_center[0]); raw_cy = float(ball_center[1])
                elif det_cache.persons:
                    raw_cx = float(sum((p[0]+p[2])/2 for p in det_cache.persons) / len(det_cache.persons))
                    raw_cy = float(sum((p[1]+p[3])/2 for p in det_cache.persons) / len(det_cache.persons))
                else:
                    if prev_gray is not None and use_optical_flow:
                        of = sports_optical_flow_center(prev_gray, gray, det_w, det_h,
                                                        field_mask=field_mask)
                        if of:
                            raw_cx = float(of[0]) / det_scale
                            raw_cy = float(of[1]) / det_scale
                        else:
                            raw_cx = float(prev_cx) if prev_cx is not None else orig_w/2
                            raw_cy = float(prev_cy) if prev_cy is not None else orig_h/2
                    else:
                        raw_cx = float(prev_cx) if prev_cx is not None else orig_w/2
                        raw_cy = float(prev_cy) if prev_cy is not None else orig_h/2

                track_conf = (primary_track.confidence if primary_track is not None
                              else 0.7 if ball_center is not None else 0.4)
                smooth_cx, smooth_cy = avs_smoother.smooth(raw_cx, raw_cy, track_conf,
                                                           current_phase)

                if use_ball_tracking:
                    if mot_tracker.ball_state.center is not None:
                        ball_pos_orig: Optional[Tuple[float,float]] = (
                            float(mot_tracker.ball_state.center[0]),
                            float(mot_tracker.ball_state.center[1]),
                        )
                    elif ball_kalman.initialized and ball_found_ever:
                        kpx, kpy     = ball_kalman.position
                        ball_pos_orig = (float(kpx/det_scale), float(kpy/det_scale))
                    else:
                        ball_pos_orig = None
                else:
                    ball_pos_orig = None

                left, top, right, bottom = ics.compute_crop(
                    smooth_cx, smooth_cy, current_phase, ball_pos_orig)
                cx_out = (left+right)//2;  cy_out = (top+bottom)//2
                cx_out = max(hw, min(cx_out, orig_w-hw))
                cy_out = max(hh, min(cy_out, orig_h-hh))
                cy_out = _apply_lower_third_guard(cy_out, crop_h, cy_out, orig_h)

                speed = math.hypot(cx_out-(prev_cx or cx_out), cy_out-(prev_cy or cy_out))
                raw_centers.append((cx_out, cy_out)); speeds.append(speed)
                prev_cx = float(cx_out); prev_cy = float(cy_out)
                prev_gray = gray.copy()
                # FIXED: store a copy to prevent aliasing with FFmpegVideoReader buffer
                prev_frame = det_frame.copy()

                if fi % max(1, total_frames//50) == 0:
                    ball_src = this_ball_rec.source
                    focus    = ("carrier" if primary_track and mot_tracker.ball_state.is_possessed
                                else "ball" if ball_center is not None else "person/flow")
                    _p(fi/total_frames,
                       f"Sports tracking {fi}/{total_frames} "
                       f"(skip={yolo_skip}, phase={current_phase.name}, "
                       f"ball={ball_src}, focus={focus})...")

    except Exception as e:
        logger.error("[sports_tracking_opt] Error: %s", e)
        while len(raw_centers) < total_frames:
            raw_centers.append(raw_centers[-1] if raw_centers else (orig_w//2, orig_h//2))
            speeds.append(0.0)
            missing_fi = len(raw_centers) - 1
            ball_records.setdefault(missing_fi, BallFrameRecord())
            person_boxes_map.setdefault(missing_fi, [])

    # Pad / truncate to exact length
    actual = len(raw_centers)
    if actual < total_frames:
        last = raw_centers[-1] if raw_centers else (orig_w//2, orig_h//2)
        for i in range(actual, total_frames):
            raw_centers.append(last); speeds.append(0.0)
            ball_records.setdefault(i, BallFrameRecord())
            person_boxes_map.setdefault(i, [])
    elif actual > total_frames:
        raw_centers      = raw_centers[:total_frames]
        speeds           = speeds[:total_frames]
        ball_records     = {k: v for k, v in ball_records.items()     if k < total_frames}
        person_boxes_map = {k: v for k, v in person_boxes_map.items() if k < total_frames}

    return raw_centers, speeds, scene_cuts, ball_records, person_boxes_map


# ── process_video — main public API ──────────────────────────────────────────
def process_video(input_path: str, output_path: str,
                  target_preset_label: str = "720p   (720x1280  - HD)",
                  tracking_mode: str = "subject", talking_head_bias: float = 0.30,
                  confidence: float = 0.45, smooth_window: int = 15,
                  adaptive_smoothing: bool = True, use_optical_flow: bool = True,
                  rule_of_thirds: bool = True, scene_cut_threshold: float = 0.35,
                  output_fps: Optional[float] = None, crf: int = 23,
                  encoder_preset: str = "fast", audio_bitrate: str = "128k",
                  yolo_weights: str = "yolov8n.pt", burn_subtitles: bool = False,
                  whisper_model: str = "base", whisper_language: Optional[str] = None,
                  subtitle_style_name: str = "Bold White (TikTok)",
                  subtitle_max_chars: int = 42, subtitle_translate_to: Optional[str] = None,
                  color_grade: str = "none", vignette_strength: float = 0.0,
                  sharpen_strength: float = 0.0, ffmpeg_sharpen: bool = False,
                  ken_burns: bool = False, use_kalman: bool = False,
                  panel_config: Optional[PanelModeConfig] = None,
                  progress_callback=None) -> Dict[str, Any]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)

    (info, orig_w, orig_h, out_w, out_h, fps, total_frames,
     model, res_mon) = _common_pipeline_setup(
        input_path, target_preset_label,
        yolo_weights if tracking_mode != "talking_head" else "",
        load_model=(tracking_mode != "talking_head"),
    )
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, out_w, out_h)
    panel_config      = panel_config or PanelModeConfig()
    use_panel_mode = False
    if panel_config.split_mode == "force_on":
        use_panel_mode = True
    elif panel_config.split_mode == "auto" and tracking_mode == "subject":
        _p(0.03, "Detecting panel layout...")
        use_panel_mode = _detect_panel_mode(
            input_path, model, fps, total_frames, orig_w, orig_h,
            confidence=confidence,
            max_person_motion=panel_config.max_person_motion,
            min_person_area_frac=panel_config.min_person_area_frac,
            max_count_variance=panel_config.max_count_variance,
            stability_frac=panel_config.stability_frac)
    _p(0.05, "Tracking subjects...")
    raw_centers, speeds, scene_cuts, persons_map, kalman_preds = _tracking_pass(
        input_path=input_path, orig_w=orig_w, orig_h=orig_h, crop_w=crop_w, crop_h=crop_h,
        fps=fps, total_frames=total_frames, tracking_mode=tracking_mode, model=model,
        confidence=confidence, smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
        use_optical_flow=use_optical_flow, rule_of_thirds=rule_of_thirds,
        scene_cut_threshold=scene_cut_threshold, talking_head_bias=talking_head_bias,
        use_kalman=use_kalman, panel_mode_active=use_panel_mode,
        progress_callback=lambda v, m: _p(0.05+v*0.45, m))
    _p(0.50, "Smoothing camera path...")
    smoothed, smooth_metrics = smooth_centers(
        raw_centers, speeds, base_window=smooth_window, adaptive=adaptive_smoothing,
        scene_cuts=scene_cuts, use_kalman=use_kalman)
    _p(0.55, "Rendering...")
    render_meta = _render_video(
        input_path=input_path, output_path=output_path,
        out_w=out_w, out_h=out_h, crop_w=crop_w, crop_h=crop_h,
        orig_w=orig_w, orig_h=orig_h, fps=fps, total_frames=total_frames,
        smoothed_centers=smoothed, tracking_mode=tracking_mode,
        crf=crf, encoder_preset=encoder_preset, audio_bitrate=audio_bitrate,
        burn_subtitles=burn_subtitles, whisper_model=whisper_model,
        whisper_language=whisper_language, subtitle_style_name=subtitle_style_name,
        subtitle_max_chars=subtitle_max_chars, subtitle_translate_to=subtitle_translate_to,
        output_fps=output_fps, color_grade=color_grade,
        vignette_strength=vignette_strength, sharpen_strength=sharpen_strength,
        ffmpeg_sharpen=ffmpeg_sharpen, scene_cuts=scene_cuts,
        use_panel_mode=use_panel_mode, panel_config=panel_config,
        panel_persons_map=persons_map,
        progress_callback=lambda v, m: _p(0.55+v*0.43, m))
    res_mon.stop()
    _p(1.0, "Done!")
    analytics = _build_analytics(
        input_path, output_path, orig_w=orig_w, orig_h=orig_h, out_w=out_w, out_h=out_h,
        smooth_metrics=smooth_metrics, panel_mode=use_panel_mode,
        kalman_predictions=smooth_metrics.get("kalman_prediction_frames", 0),
        resource_stats=res_mon.get_stats())
    return {"analytics": analytics, "subtitle_path": render_meta.get("subtitle_path")}


# ── process_sports_video — optimized sports pipeline ─────────────────────────
def process_sports_video(input_path: str, output_path: str,
                         sport_type: str = "auto",
                         target_preset_label: str = "720p   (720x1280  - HD)",
                         confidence: float = 0.45, output_fps: Optional[float] = None,
                         crf: int = 23, encoder_preset: str = "fast",
                         audio_bitrate: str = "128k", yolo_weights: str = "yolov8n.pt",
                         burn_subtitles: bool = False, whisper_model: str = "base",
                         whisper_language: Optional[str] = None,
                         subtitle_style_name: str = "Bold White (TikTok)",
                         subtitle_max_chars: int = 42,
                         subtitle_translate_to: Optional[str] = None,
                         use_ball_tracking: bool = True, use_kalman: bool = True,
                         smooth_window: int = 5, adaptive_smoothing: bool = True,
                         use_optical_flow: bool = True, rule_of_thirds: bool = True,
                         scene_cut_threshold: float = 0.22,
                         vignette_strength: float = 0.275, sharpen_strength: float = 0.3,
                         ffmpeg_sharpen: bool = True, color_grade: str = "none",
                         draw_tracking_boxes: bool = True,
                         overlay_config: Optional[OverlayConfig] = None,
                         min_ball_confidence: float = SPORTS_BALL_CONFIDENCE,
                         progress_callback=None) -> Dict[str, Any]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)

    (info, orig_w, orig_h, out_w, out_h, fps, total_frames,
     model, res_mon) = _common_pipeline_setup(
        input_path, target_preset_label, yolo_weights)
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, out_w, out_h)

    sample_frame = _read_frame_at(input_path, orig_w, orig_h, 2.0)
    field_mask:  Optional[np.ndarray] = None
    if sample_frame is not None:
        field_mask = detect_field_of_play(sample_frame, sport_hint=sport_type)

    mot_tracker    = MultiObjectSportsTracker(fps, orig_w, orig_h)
    avs_smoother   = AdaptiveVelocityAwareSmoother(fps)
    ics            = IntelligentCropStrategy(orig_w, orig_h, crop_w, crop_h, fps)
    phase_detector = SportsPlayPhaseDetector(fps)

    _p(0.05, "Sports tracking (v7.3 — Kalman + colour model + 4-gate validation)...")
    raw_centers, speeds, scene_cuts, ball_records, person_boxes_map = \
        _sports_tracking_pass_optimized(
            input_path=input_path, orig_w=orig_w, orig_h=orig_h,
            crop_w=crop_w, crop_h=crop_h, fps=fps, total_frames=total_frames,
            model=model, confidence=confidence,
            use_ball_tracking=use_ball_tracking, use_optical_flow=use_optical_flow,
            field_mask=field_mask,
            mot_tracker=mot_tracker, avs_smoother=avs_smoother,
            ics=ics, phase_detector=phase_detector,
            min_ball_confidence=min_ball_confidence,
            progress_callback=lambda v, m: _p(0.05+v*0.45, m),
        )

    total_frames = len(raw_centers)

    _p(0.50, "Sports post-smoothing...")
    dense_cx = np.array([c[0] for c in raw_centers], dtype=float)
    dense_cy = np.array([c[1] for c in raw_centers], dtype=float)
    dense_cx, dense_cy = _apply_sports_post_smooth(dense_cx, dense_cy, fps,
                                                    scene_cuts, total_frames)
    smoothed = [(int(x), int(y)) for x, y in zip(dense_cx, dense_cy)]
    final_smooth_metrics = _compute_final_smoothness(raw_centers, smoothed)

    eff_overlay: Optional[OverlayConfig] = (overlay_config if overlay_config is not None
                                            else OverlayConfig() if draw_tracking_boxes
                                            else None)
    _p(0.55, "Rendering sports video...")
    render_meta = _render_video(
        input_path=input_path, output_path=output_path,
        out_w=out_w, out_h=out_h, crop_w=crop_w, crop_h=crop_h,
        orig_w=orig_w, orig_h=orig_h, fps=fps, total_frames=total_frames,
        smoothed_centers=smoothed, tracking_mode="sports_action",
        crf=crf, encoder_preset=encoder_preset, audio_bitrate=audio_bitrate,
        burn_subtitles=burn_subtitles, whisper_model=whisper_model,
        whisper_language=whisper_language, subtitle_style_name=subtitle_style_name,
        subtitle_max_chars=subtitle_max_chars, subtitle_translate_to=subtitle_translate_to,
        output_fps=output_fps, color_grade=color_grade,
        vignette_strength=vignette_strength, sharpen_strength=sharpen_strength,
        ffmpeg_sharpen=ffmpeg_sharpen, scene_cuts=scene_cuts,
        use_panel_mode=False, ball_records=ball_records,
        overlay_config=eff_overlay,
        progress_callback=lambda v, m: _p(0.55+v*0.43, m),
    )
    res_mon.stop()
    _p(1.0, "Done!")
    analytics = _build_analytics(
        input_path, output_path, orig_w=orig_w, orig_h=orig_h,
        out_w=out_w, out_h=out_h, smooth_metrics=final_smooth_metrics,
        panel_mode=False, resource_stats=res_mon.get_stats())
    return {"analytics": analytics, "subtitle_path": render_meta.get("subtitle_path")}


# ── process_clips_batch ───────────────────────────────────────────────────────
# -------------
def process_clips_batch(input_path: str, output_dir: str, clips: List[ClipSegment],
                        target_preset_label: str = "720p   (720x1280  - HD)",
                        tracking_mode: str = "subject", talking_head_bias: float = 0.30,
                        confidence: float = 0.45, smooth_window: int = 15,
                        adaptive_smoothing: bool = True, rule_of_thirds: bool = True,
                        crf: int = 23, encoder_preset: str = "fast",
                        audio_bitrate: str = "128k", yolo_weights: str = "yolov8n.pt",
                        burn_subtitles: bool = False, whisper_model: str = "base",
                        subtitle_style_name: str = "Bold White (TikTok)",
                        subtitle_max_chars: int = 42,
                        subtitle_translate_to: Optional[str] = None,
                        sport_type: str = "auto",
                        use_optical_flow: bool = True,
                        scene_cut_threshold: float = 0.35,
                        whisper_language: Optional[str] = None,
                        output_fps: Optional[float] = None,
                        panel_config: Optional[PanelModeConfig] = None,
                        draw_tracking_boxes: bool = True,
                        overlay_config: Optional[OverlayConfig] = None,
                        min_ball_confidence: float = SPORTS_BALL_CONFIDENCE,
                        progress_callback=None) -> List[Dict[str, Any]]:
    def _p(v, msg=""): progress_callback and progress_callback(v, msg)
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []; n = len(clips)
    for i, clip in enumerate(clips):
        cs = i/n; ce = (i+1)/n
        def _cp(v, msg="", _cs=cs, _ce=ce): _p(_cs + v*(_ce-_cs), msg)
        _cp(0.0, f"Clip {i+1}/{n}: trimming...")
        fd, trim_path = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
        ok = _trim_video(input_path, trim_path, clip.start_sec, clip.end_sec,
                         crf=crf, preset=encoder_preset)
        if not ok:
            results.append({"clip": clip, "output_path": None,
                            "error": "Trim failed", "analytics": {}})
            if os.path.exists(trim_path):
                try: os.unlink(trim_path)
                except OSError: pass
            continue
        out_path = os.path.join(output_dir, f"clip_{i+1:03d}_vertical.mp4")
        try:
            if tracking_mode == "sports_action":
                meta = process_sports_video(
                    trim_path, out_path, sport_type=sport_type,
                    target_preset_label=target_preset_label, confidence=confidence,
                    output_fps=output_fps, crf=crf, encoder_preset=encoder_preset,
                    audio_bitrate=audio_bitrate, yolo_weights=yolo_weights,
                    burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                    subtitle_style_name=subtitle_style_name,
                    subtitle_max_chars=subtitle_max_chars,
                    draw_tracking_boxes=draw_tracking_boxes, overlay_config=overlay_config,
                    min_ball_confidence=min_ball_confidence,
                    progress_callback=lambda v, m: _cp(0.05+v*0.90, m))
            else:
                meta = process_video(
                    trim_path, out_path, target_preset_label=target_preset_label,
                    tracking_mode=tracking_mode, talking_head_bias=talking_head_bias,
                    confidence=confidence, smooth_window=smooth_window,
                    adaptive_smoothing=adaptive_smoothing, use_optical_flow=use_optical_flow,
                    rule_of_thirds=rule_of_thirds, scene_cut_threshold=scene_cut_threshold,
                    output_fps=output_fps, crf=crf, encoder_preset=encoder_preset,
                    audio_bitrate=audio_bitrate, yolo_weights=yolo_weights,
                    burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                    subtitle_style_name=subtitle_style_name,
                    subtitle_max_chars=subtitle_max_chars, panel_config=panel_config,
                    progress_callback=lambda v, m: _cp(0.05+v*0.90, m))
            results.append({"clip": clip, "output_path": out_path,
                            "analytics": meta.get("analytics", {})})
        except Exception as e:
            logger.error("[batch] Clip %d error: %s", i+1, e)
            results.append({"clip": clip, "output_path": None,
                            "error": str(e), "analytics": {}})
        finally:
            if os.path.exists(trim_path):
                try: os.unlink(trim_path)
                except OSError: pass
    _p(1.0, f"Batch done: {sum(1 for r in results if not r.get('error'))}/{n} clips")
    return results

# --- Processor architecture ---------------------------------------------------------
class VerticalProcessor:
    '''High-level API wrapper for batch/queued processing.'''
    def __init__(self):
        self.audio_cache = {}
        self.model_cache = {}

    def process_video(self, *args, **kwargs):
        return process_video(*args, **kwargs)

    def process_sports_video(self, *args, **kwargs):
        return process_sports_video(*args, **kwargs)
