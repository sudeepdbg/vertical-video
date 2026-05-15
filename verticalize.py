"""
verticalize.py  —  AI Vertical Video converter  v3.1
──────────────────────────────────────────────────────
v3.1 CHANGES (anti-shake + player analytics):
  SHAKE-1  Spring-damped camera physics — floating-point crop center with
           velocity + acceleration constraints. Kills micro-jitter while
           keeping natural ease-in/ease-out on large moves.
  SHAKE-2  Sub-pixel deadzone — camera only moves when target displacement
           exceeds 1.5 px, eliminating breathing/noise-induced drift.
  SHAKE-3  Floating-point crop & Lanczos4 — all coordinates kept as float
           until the final slice, preventing integer-rounding stutter.
  ANAL-1   Player tracker — SORT-style IOU tracker on top of YOLO person
           detections. Persistent IDs across frames.
  ANAL-2   Per-player analytics — distance covered (m), peak/average speed,
           time-in-zone (thirds), acceleration intensity, heatmap grid.
  ANAL-3   Ball possession heuristic — when sports ball (cls 32) overlaps a
           player bbox, possession is credited to that player.
  ANAL-4   Optional analytics overlay — speed labels, motion trails, ball
           possession badge, zone coloration burned into output frames.
  ANAL-5   JSON export — full per-frame trajectories + aggregate stats
           written alongside the video file.
"""

from __future__ import annotations
import bisect, subprocess, sys, os, tempfile, math, json
from collections import namedtuple, deque
from typing import Any, Dict, List, Optional, Tuple
import cv2, numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

class ProcessingError(Exception):
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
PERSON_CLASS_ID    = 0
SPORTS_BALL_CLASS  = 32
HIGH_PRIO_CLASSES  = {0, 2, 3, 5, 7, 15, 16, 32}
MAX_FILE_SIZE_MB   = 2000
MIN_FRAME_DIM      = 240
MAX_FRAMES_GUARD   = 1_080_000
LOWER_THIRD_GUARD  = 0.80
PANEL_MIN_PERSONS  = 2

VELOCITY_SMOOTH_TABLE = [
    (0.0, 51), (3.0, 45), (8.0, 37), (15.0, 27), (30.0, 19),
    (60.0, 13), (120.0, 7), (300.0, 5),
]

RESOLUTION_PRESETS = {
    "Match source (no upscale)":    (0, 0),
    "1080p  (1080x1920 - Full HD)": (1080, 1920),
    "720p   (720x1280  - HD)":      (720, 1280),
    "540p   (540x960   - SD)":      (540, 960),
    "480p   (480x854   - Low)":     (480, 854),
}

SUBTITLE_STYLES = {
    "Bold White (TikTok)": {"fontsize":18,"primary_color":"&H00FFFFFF","outline_color":"&H00000000","outline":2,"bold":1,"shadow":0,"back_color":"&H00000000","margin_v":80},
    "Yellow (Classic)":    {"fontsize":16,"primary_color":"&H0000FFFF","outline_color":"&H00000000","outline":2,"bold":1,"shadow":1,"back_color":"&H00000000","margin_v":80},
    "Box (Accessible)":    {"fontsize":15,"primary_color":"&H00FFFFFF","outline_color":"&H00000000","outline":0,"bold":0,"shadow":0,"back_color":"&H80000000","margin_v":80},
}

TRANSLATION_LANGUAGES = {
    "None (keep original)":"","French":"fr","German":"de","Spanish":"es",
    "Italian":"it","Portuguese":"pt","Dutch":"nl","Polish":"pl","Russian":"ru",
    "Japanese":"ja","Korean":"ko","Chinese (Simplified)":"zh-CN","Arabic":"ar",
    "Hindi":"hi","Turkish":"tr","Indonesian":"id","Swedish":"sv","Norwegian":"no",
    "Danish":"da","Finnish":"fi","Greek":"el","Hebrew":"iw","Thai":"th",
    "Vietnamese":"vi","Malay":"ms","Ukrainian":"uk",
}

# VFX constants
VIGNETTE_STRENGTH  = 0.55
VIGNETTE_FALLOFF   = 1.8
COLOR_GRADES       = ("none","warm","cool","vibrant","matte")
PANEL_SLOT_EMA     = 0.25
KEN_BURNS_MAX_ZOOM = 1.04
KEN_BURNS_PERIOD   = 8.0
DISSOLVE_FRAMES    = 3
PANEL_DIVIDER_PX   = 3
PANEL_DIVIDER_COLOR= (15, 15, 15)
PANEL_CROP_EXPAND  = 1.55

# Sports / motion constants
SPORT_FAST_FLOW_THRESH = 8.0
SPORT_MARGIN_EXPAND    = 1.15
SPORT_SETTLE_SEC       = 0.5
SPORT_PREDICT_FRAMES   = 3
TRANSITION_HOLD_FRAMES = 12

# SHAKE constants
CAM_STIFFNESS   = 0.08   # spring strength toward target
CAM_DAMPING     = 0.35   # velocity decay per frame (0-1)
CAM_DEADZONE    = 1.5    # px: ignore micro-jitter below this
CAM_MAX_VEL     = 48.0   # px/frame cap (prevents impossible jumps)

# ANALYTICS constants
TRACK_MAX_DISAPPEARED = 5
TRACK_MAX_DISTANCE    = 120  # px in det space for matching
ANALYTICS_HISTORY_SEC = 3.0  # trail length
POSSESSION_IOU_THRESH = 0.15
HEATMAP_GRID        = (6, 10)  # rows, cols overlay grid


# ── Segment class ─────────────────────────────────────────────────────────────
class ClipSegment:
    def __init__(self, start_sec, end_sec, score, soi_region="center", peak_frame=0, title=""):
        self.start_sec=start_sec; self.end_sec=end_sec; self.score=score
        self.soi_region=soi_region; self.peak_frame=peak_frame; self.title=title
        self.duration=end_sec-start_sec
    def __repr__(self): return f"<Clip {self.start_sec:.1f}s-{self.end_sec:.1f}s score={self.score:.2f}>"

def whisper_available():
    try: import whisper; return True
    except ImportError: return False

def translation_available():
    try: import deep_translator; return True
    except ImportError: return False

def yolo_available():
    if not _YOLO_AVAILABLE: return False
    try:
        import urllib.request; urllib.request.urlopen("https://github.com",timeout=3); return True
    except Exception:
        return os.path.exists("yolov8n.pt") or os.path.exists("yolov8s.pt")


# ── SHAKE-1: Spring-damped camera physics ─────────────────────────────────────
class CameraSpring:
    """
    Critically-damped spring for crop-center smoothing.
    Eliminates micro-jitter via deadzone while preserving natural ease-in/out.
    All coordinates are float until final slice.
    """
    __slots__ = ("x","y","vx","vy","stiffness","damping","deadzone","max_vel")
    def __init__(self, x: float, y: float, stiffness: float = CAM_STIFFNESS,
                 damping: float = CAM_DAMPING, deadzone: float = CAM_DEADZONE,
                 max_vel: float = CAM_MAX_VEL):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.stiffness = stiffness
        self.damping = damping
        self.deadzone = deadzone
        self.max_vel = max_vel

    def update(self, tx: float, ty: float) -> Tuple[float, float]:
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < self.deadzone:
            # Kill residual velocity when nearly stationary
            self.vx *= 0.3
            self.vy *= 0.3
        # Spring force
        ax = dx * self.stiffness
        ay = dy * self.stiffness
        self.vx += ax
        self.vy += ay
        # Damping
        self.vx *= (1.0 - self.damping)
        self.vy *= (1.0 - self.damping)
        # Velocity cap
        v = math.hypot(self.vx, self.vy)
        if v > self.max_vel and v > 0:
            self.vx = (self.vx / v) * self.max_vel
            self.vy = (self.vy / v) * self.max_vel
        self.x += self.vx
        self.y += self.vy
        return self.x, self.y


# ── ANAL-1: Player / Object Tracker ───────────────────────────────────────────
class TrackedObject:
    __slots__ = ("oid","cls","history","disappeared","total_distance",
                 "max_speed","sum_speed","speed_samples","possession_frames",
                 "zone_time")
    def __init__(self, oid: int, cls: int, x1: float, y1: float, x2: float, y2: float,
                 frame_idx: int, fps: float):
        self.oid = oid
        self.cls = cls
        self.history: deque[Tuple[int,float,float,float,float]] = deque(maxlen=int(fps*ANALYTICS_HISTORY_SEC*2))
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        self.history.append((frame_idx, cx, cy, x2-x1, y2-y1))
        self.disappeared = 0
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.sum_speed = 0.0
        self.speed_samples = 0
        self.possession_frames = 0
        self.zone_time: Dict[str,int] = {"left":0,"center":0,"right":0,"upper":0,"mid":0,"lower":0}

    def update(self, frame_idx: int, x1: float, y1: float, x2: float, y2: float,
               orig_w: int, orig_h: int):
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        if self.history:
            pfi, px, py, _, _ = self.history[-1]
            d = math.hypot(cx-px, cy-py)
            self.total_distance += d
            dt = max(frame_idx - pfi, 1)
            speed = d / dt
            self.max_speed = max(self.max_speed, speed)
            self.sum_speed += speed
            self.speed_samples += 1
        self.history.append((frame_idx, cx, cy, x2-x1, y2-y1))
        self.disappeared = 0
        # Zone logging
        col = "left" if cx < orig_w/3 else ("right" if cx > 2*orig_w/3 else "center")
        row = "upper" if cy < orig_h/3 else ("lower" if cy > 2*orig_h/3 else "mid")
        self.zone_time[col] += 1
        self.zone_time[row] += 1

    def predict(self) -> Tuple[float,float,float,float]:
        if len(self.history) < 2:
            h = self.history[-1]
            return (h[1]-h[3]/2, h[2]-h[4]/2, h[3], h[4])
        f0, x0, y0, w0, h0 = self.history[-2]
        f1, x1, y1, w1, h1 = self.history[-1]
        dt = max(f1-f0, 1)
        vx, vy = (x1-x0)/dt, (y1-y0)/dt
        return (x1+w1/2 - w1/2 + vx, y1+h1/2 - h1/2 + vy, w1, h1)

    @property
    def last_center(self) -> Tuple[float,float]:
        _, cx, cy, _, _ = self.history[-1]
        return cx, cy

    def to_dict(self, fps: float, sx: float = 1.0, sy: float = 1.0) -> Dict[str,Any]:
        d = {
            "id": self.oid,
            "class": "person" if self.cls == PERSON_CLASS_ID else "ball" if self.cls == SPORTS_BALL_CLASS else str(self.cls),
            "distance_px": round(self.total_distance, 1),
            "max_speed_px_per_frame": round(self.max_speed, 2),
            "avg_speed_px_per_frame": round(self.sum_speed/max(self.speed_samples,1), 2),
            "possession_frames": self.possession_frames,
            "zone_time": dict(self.zone_time),
            "trajectory": [
                {"frame":int(fi),"x":round(cx*sx,1),"y":round(cy*sy,1)}
                for fi,cx,cy,_,_ in list(self.history)[::max(1,len(self.history)//60)]
            ],
        }
        return d


class PlayerTracker:
    def __init__(self, max_disappeared: int = TRACK_MAX_DISAPPEARED,
                 max_distance: float = TRACK_MAX_DISTANCE):
        self.next_id = 1
        self.trackers: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    @staticmethod
    def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
        iy = max(0, min(ay+ah, by+bh) - max(ay, by))
        inter = ix * iy
        union = aw*ah + bw*bh - inter
        return inter / (union + 1e-9)

    def update(self, detections: List[Tuple[int, int, int, int, int]],
               frame_idx: int, orig_w: int, orig_h: int, fps: float) -> List[TrackedObject]:
        # detections: list of (cls, x1, y1, x2, y2)
        dets = [(cls, float(x1), float(y1), float(x2), float(y2)) for cls, x1, y1, x2, y2 in detections]
        tids = list(self.trackers.keys())
        matched = set()

        if tids and dets:
            # Build cost matrix (1 - IOU)
            cost = np.zeros((len(tids), len(dets)), dtype=float)
            for i, tid in enumerate(tids):
                tr = self.trackers[tid]
                pred = tr.predict()
                pw, ph = pred[2], pred[3]
                for j, (cls, x1, y1, x2, y2) in enumerate(dets):
                    dw, dh = x2-x1, y2-y1
                    cdist = math.hypot((pred[0]+pw/2)-(x1+dw/2), (pred[1]+ph/2)-(y1+dh/2))
                    if cdist > self.max_distance or cls != tr.cls:
                        cost[i,j] = 1e6
                    else:
                        iou = self._iou(pred, (x1, y1, dw, dh))
                        cost[i,j] = 1.0 - iou
            # Greedy matching
            while cost.min() < 1e5:
                i, j = np.unravel_index(cost.argmin(), cost.shape)
                tid = tids[i]
                cls, x1, y1, x2, y2 = dets[j]
                self.trackers[tid].update(frame_idx, x1, y1, x2, y2, orig_w, orig_h)
                matched.add(j)
                cost[i,:] = 1e9
                cost[:,j] = 1e9

        # Unmatched detections -> new trackers
        for j, (cls, x1, y1, x2, y2) in enumerate(dets):
            if j in matched:
                continue
            t = TrackedObject(self.next_id, cls, x1, y1, x2, y2, frame_idx, fps)
            self.trackers[self.next_id] = t
            self.next_id += 1

        # Unmatched trackers -> mark disappeared
        used_tids = set()
        for i, tid in enumerate(tids):
            if cost[i].min() < 1e5:  # still matched check via cost row
                continue
            tr = self.trackers[tid]
            if j not in matched:
                tr.disappeared += 1
                if tr.disappeared > self.max_disappeared:
                    used_tids.add(tid)
        for tid in used_tids:
            del self.trackers[tid]

        return list(self.trackers.values())

    def compute_possession(self):
        # Simple: ball inside player bbox => possession
        ball = [t for t in self.trackers.values() if t.cls == SPORTS_BALL_CLASS]
        persons = [t for t in self.trackers.values() if t.cls == PERSON_CLASS_ID]
        if not ball or not persons:
            return
        for b in ball:
            bx, by = b.last_center
            best = None
            best_iou = 0
            for p in persons:
                if not p.history:
                    continue
                _, px, py, pw, ph = p.history[-1]
                x1, y1, x2, y2 = px-pw/2, py-ph/2, px+pw/2, py+ph/2
                iou = self._iou((bx-5, by-5, 10, 10), (x1, y1, x2-x1, y2-y1))
                if iou > best_iou:
                    best_iou = iou
                    best = p
            if best and best_iou > POSSESSION_IOU_THRESH:
                best.possession_frames += 1


class AnalyticsRenderer:
    """Optional overlay of trails, speed text, and possession."""
    def __init__(self, fps: float):
        self.fps = fps

    def draw(self, frame: np.ndarray, trackers: List[TrackedObject],
             show_trails: bool = True, show_speed: bool = True,
             show_possession: bool = True) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for t in trackers:
            if t.cls != PERSON_CLASS_ID or len(t.history) < 2:
                continue
            # Trail
            if show_trails:
                pts = [(int(cx), int(cy)) for _,cx,cy,_,_ in t.history]
                for i in range(1, len(pts)):
                    alpha = int(255 * i / len(pts))
                    cv2.line(out, pts[i-1], pts[i], (0, 255, 255), 2, cv2.LINE_AA)
            # Speed label
            if show_speed and t.speed_samples:
                avg = t.sum_speed / t.speed_samples
                label = f"ID:{t.oid} {avg*self.fps:.1f} px/s"
                cx, cy = int(t.last_center[0]), int(t.last_center[1])
                cv2.putText(out, label, (cx-40, cy-20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(out, label, (cx-40, cy-20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 1, cv2.LINE_AA)
            # Possession badge
            if show_possession and t.possession_frames > 10:
                cx, cy = int(t.last_center[0]), int(t.last_center[1])
                cv2.circle(out, (cx, cy+25), 8, (0,165,255), -1)
                cv2.putText(out, "BALL", (cx-12, cy+29), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255,255,255), 1, cv2.LINE_AA)
        return out


# ── Caches ────────────────────────────────────────────────────────────────────
_vignette_cache: Dict[Tuple, np.ndarray] = {}
_lut_cache: Dict[str, np.ndarray] = {}

def _build_vignette(w: int, h: int, strength: float = VIGNETTE_STRENGTH, falloff: float = VIGNETTE_FALLOFF) -> np.ndarray:
    key = (w, h, round(strength, 3), round(falloff, 3))
    if key in _vignette_cache:
        return _vignette_cache[key]
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.sqrt(xg * xg + yg * yg)
    dist /= dist.max() + 1e-9
    mask = np.clip(1.0 - strength * (dist ** falloff), 0.0, 1.0)[:, :, np.newaxis]
    _vignette_cache[key] = mask
    return mask

def apply_vignette(frame: np.ndarray, strength: float = VIGNETTE_STRENGTH) -> np.ndarray:
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    mask = _build_vignette(w, h, strength)
    return (frame.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

def apply_sharpen(frame: np.ndarray, strength: float = 0.6, radius: int = 1) -> np.ndarray:
    if strength <= 0:
        return frame
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)

def _build_lut(grade: str) -> np.ndarray:
    if grade in _lut_cache:
        return _lut_cache[grade]
    x = np.arange(256, dtype=np.float32)
    if grade == "warm":
        r = np.clip(x * 1.06 + 5, 0, 255); g = np.clip(x * 1.02 + 2, 0, 255); b = np.clip(x * 0.92 - 4, 0, 255)
    elif grade == "cool":
        r = np.clip(x * 0.92 - 4, 0, 255); g = np.clip(x * 1.01 + 1, 0, 255); b = np.clip(x * 1.07 + 6, 0, 255)
    elif grade == "vibrant":
        def sc(v):
            n = v / 255.0; s = n * n * (3 - 2 * n)
            return np.clip((n * 0.6 + s * 0.4) * 255, 0, 255)
        r = sc(x * 1.04); g = sc(x * 1.02); b = sc(x)
    elif grade == "matte":
        r = np.clip(x * 0.88 + 18, 0, 255); g = np.clip(x * 0.86 + 16, 0, 255); b = np.clip(x * 0.84 + 22, 0, 255)
    else:
        r = g = b = x.copy()
    lut = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(256, 1, 3)
    _lut_cache[grade] = lut
    return lut

def apply_color_grade(frame: np.ndarray, grade: str = "none") -> np.ndarray:
    if not grade or grade == "none":
        return frame
    return cv2.LUT(frame, _build_lut(grade))

def apply_ken_burns(frame: np.ndarray, frame_idx: int, fps: float,
                    max_zoom: float = KEN_BURNS_MAX_ZOOM, period: float = KEN_BURNS_PERIOD) -> np.ndarray:
    if max_zoom <= 1.0:
        return frame
    t = (frame_idx / max(fps, 1)) % period
    scale = 1.0 + (max_zoom - 1.0) * 0.5 * (1 - math.cos(2 * math.pi * t / period))
    if abs(scale - 1.0) < 1e-4:
        return frame
    h, w = frame.shape[:2]
    nw = max(int(w / scale), 2); nh = max(int(h / scale), 2)
    x0 = (w - nw) // 2; y0 = (h - nh) // 2
    return cv2.resize(frame[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LINEAR)


class DissolveBuffer:
    def __init__(self, n: int = DISSOLVE_FRAMES):
        self.n = n; self._buf: Optional[np.ndarray] = None; self._rem = 0
    def on_cut(self, last_frame: np.ndarray):
        self._buf = last_frame.copy(); self._rem = self.n
    def blend(self, new_frame: np.ndarray) -> np.ndarray:
        if self._rem <= 0 or self._buf is None:
            return new_frame
        a = self._rem / self.n; self._rem -= 1
        return cv2.addWeighted(self._buf, a, new_frame, 1.0 - a, 0)
    @property
    def active(self) -> bool:
        return self._rem > 0


class VelocityTracker:
    def __init__(self, window: int = 5, predict_frames: int = SPORT_PREDICT_FRAMES):
        self.window = window
        self.predict_frames = predict_frames
        self._history: deque[Tuple[int, float, float]] = deque(maxlen=window)
        self._last_pred: Optional[Tuple[float, float]] = None
    def update(self, fi: int, cx: float, cy: float) -> Tuple[float, float]:
        self._history.append((fi, cx, cy))
        if len(self._history) < 2:
            self._last_pred = (cx, cy)
            return self._last_pred
        f0, x0, y0 = self._history[-2]
        f1, x1, y1 = self._history[-1]
        dt = max(f1 - f0, 1)
        vx = (x1 - x0) / dt; vy = (y1 - y0) / dt
        self._last_pred = (x1 + vx * self.predict_frames, y1 + vy * self.predict_frames)
        return self._last_pred
    @property
    def speed(self) -> float:
        if len(self._history) < 2:
            return 0.0
        f0, x0, y0 = self._history[0]
        f1, x1, y1 = self._history[-1]
        dt = max(f1 - f0, 1)
        return math.hypot(x1 - x0, y1 - y0) / dt


class TransitionHandler:
    def __init__(self, fps: float, settle_sec: float = SPORT_SETTLE_SEC,
                 hold_frames: int = TRANSITION_HOLD_FRAMES):
        self.fps = fps
        self.settle_frames = int(settle_sec * fps)
        self.hold_frames = hold_frames
        self._prev_count = 0
        self._hold_rem = 0
        self._settle_rem = 0
        self._margin = 1.0
    def update(self, subject_count: int, fast_motion: bool) -> float:
        count_dropped = (self._prev_count >= 2 and subject_count == 1 and self._prev_count > subject_count)
        self._prev_count = subject_count
        if count_dropped:
            self._hold_rem = self.hold_frames
            self._settle_rem = self.settle_frames
            self._margin = SPORT_MARGIN_EXPAND
        if fast_motion and self._margin < SPORT_MARGIN_EXPAND:
            self._margin = min(self._margin + 0.05, SPORT_MARGIN_EXPAND)
        if self._hold_rem > 0:
            self._hold_rem -= 1
            return self._margin
        if self._settle_rem > 0:
            self._settle_rem -= 1
            t = 1.0 - (self._settle_rem / self.settle_frames)
            self._margin = SPORT_MARGIN_EXPAND + (1.0 - SPORT_MARGIN_EXPAND) * t
            return self._margin
        if self._margin > 1.0:
            self._margin = max(self._margin - 0.02, 1.0)
        return self._margin


# ── FFmpeg helpers ────────────────────────────────────────────────────────────
class FFmpegVideoReader:
    def __init__(self, path: str, width: int, height: int, seek_sec: float = 0.0,
                 n_frames: Optional[int] = None, scale_w: Optional[int] = None,
                 scale_h: Optional[int] = None):
        self.path = path; self.width = width; self.height = height
        self.seek_sec = seek_sec; self.n_frames = n_frames
        self.out_w = scale_w or width; self.out_h = scale_h or height
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover = b""
    def _candidate_cmds(self) -> List[List[str]]:
        head = ["ffmpeg"]
        if self.seek_sec > 0:
            head += ["-ss", str(self.seek_sec)]
        tail = ["-i", self.path, "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-vf", f"scale={self.out_w}:{self.out_h}"]
        if self.n_frames is not None:
            tail += ["-vframes", str(self.n_frames)]
        tail += ["pipe:1"]
        return [head + ["-vcodec", "libdav1d"] + tail,
                head + ["-hwaccel", "none"] + tail]
    def _open(self):
        for cmd in self._candidate_cmds():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        bufsize=max(self._frame_bytes * 4, 1 << 20))
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc = proc; self._leftover = test; return
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                proc.wait()
            except Exception:
                pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")
    def close(self):
        if self._proc:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            self._proc.wait(); self._proc = None
    def __enter__(self):
        self._open(); return self
    def __exit__(self, *_):
        self.close()
    def __iter__(self):
        if not self._proc:
            self._open()
        buf = self._leftover; self._leftover = b""
        fb = self._frame_bytes
        while True:
            needed = fb - len(buf)
            while needed > 0:
                chunk = self._proc.stdout.read(needed)
                if not chunk:
                    return
                buf += chunk; needed -= len(chunk)
            yield np.frombuffer(buf[:fb], dtype=np.uint8).reshape(self.out_h, self.out_w, 3)
            buf = buf[fb:]


def _read_frame_at(path: str, width: int, height: int, t_sec: float,
                   scale_w: Optional[int] = None, scale_h: Optional[int] = None) -> Optional[np.ndarray]:
    r = FFmpegVideoReader(path, width, height, seek_sec=t_sec, n_frames=1,
                          scale_w=scale_w, scale_h=scale_h)
    r._open(); frames = list(r); r.close()
    return frames[0] if frames else None


def _check_ffmpeg():
    for t in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([t, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{t} not found. Install FFmpeg.")

def _has_audio(path: str) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=15)
        return "audio" in r.stdout
    except Exception:
        return False

def _extract_audio_wav(vpath: str, wpath: str) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-i", vpath, "-ar", "16000", "-ac", "1", "-f", "wav", wpath],
                       capture_output=True)
    return r.returncode == 0 and os.path.exists(wpath)

def _trim_video(inp: str, out: str, start: float, end: float) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-hwaccel", "none", "-ss", str(start), "-to", str(end),
                        "-i", inp, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                        "-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero",
                        "-reset_timestamps", "1", out],
                       capture_output=True)
    return r.returncode == 0 and os.path.exists(out)


def _open_ffmpeg_encoder(output_path: str, width: int, height: int, fps: float,
                         audio_source: Optional[str], crf: int = 23, preset: str = "fast",
                         audio_bitrate: str = "128k", subtitle_path: Optional[str] = None,
                         subtitle_style: Optional[Dict] = None, extra_vf: Optional[List[str]] = None) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    has_aud = audio_source and _has_audio(audio_source)
    if has_aud:
        cmd += ["-hwaccel", "none", "-i", audio_source]
    vf = []
    if subtitle_path and os.path.exists(subtitle_path):
        s = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        sesc = subtitle_path.replace("\\", "/").replace(":", "\\:")
        force = (f"Fontsize={s.get('fontsize', 18)},PrimaryColour={s.get('primary_color', '&H00FFFFFF')},"
                 f"OutlineColour={s.get('outline_color', '&H00000000')},Outline={s.get('outline', 2)},"
                 f"Bold={s.get('bold', 1)},Shadow={s.get('shadow', 0)},BackColour={s.get('back_color', '&H00000000')},"
                 f"MarginV={s.get('margin_v', 80)},Alignment=2")
        vf.append(f"subtitles='{sesc}':force_style='{force}'")
    if extra_vf:
        vf.extend(extra_vf)
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
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _close_ffmpeg_encoder(proc: subprocess.Popen, output_path: str):
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


# ── Metadata ──────────────────────────────────────────────────────────────────
def get_video_info(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
           "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    kv = {}
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

def extract_thumbnail(path: str, t: float = 1.0) -> Optional[bytes]:
    info = get_video_info(path)
    frame = _read_frame_at(path, info["width"], info["height"], t, scale_w=320, scale_h=180)
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ── Resolution ────────────────────────────────────────────────────────────────
def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
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

def calculate_crop_dims(orig_w: int, orig_h: int, tw: int, th: int) -> Tuple[int, int]:
    ratio = tw / th
    if (orig_w / orig_h) > ratio:
        ch = orig_h
        cw = int(round(ch * ratio))
    else:
        cw = orig_w
        ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ── YOLO / Detection ──────────────────────────────────────────────────────────
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


# ── Face detection ────────────────────────────────────────────────────────────
_face_net = None
_haar_cascade = None
_FACE_PROTO = "deploy.prototxt"
_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

def _load_face_net() -> Optional[Any]:
    global _face_net
    if _face_net:
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
    if _haar_cascade:
        return _haar_cascade
    p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(p):
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            _haar_cascade = c
            return c
    return None

def detect_faces(frame: np.ndarray, confidence_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    net = _load_face_net()
    if net:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        dets = net.forward()
        faces = []
        for i in range(dets.shape[2]):
            if float(dets[0, 0, i, 2]) < confidence_thresh:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
        faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        return faces
    haar = _get_haar()
    if not haar:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw = haar.detectMultiScale(gray, 1.1, 5, minSize=(max(30, w//20), max(30, h//20)))
    if len(raw) == 0:
        return []
    faces2 = [(x, y, x+bw, y+bh) for x, y, bw, bh in raw]
    faces2.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
    return faces2


# ── Subject detection ─────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult", ["cx","cy","ux1","uy1","ux2","uy2","count","has_ball"])

def detect_subjects(frame: np.ndarray, model: Any, confidence: float = 0.45) -> Optional[DetectionResult]:
    if model is None:
        return None
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"det err: {e}", file=sys.stderr)
        return None
    if results.boxes is None or len(results.boxes) == 0:
        return None
    pp = []; hp = []; ap = []
    has_ball = False
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w_ = (x2 - x1) * conf
        if cls == SPORTS_BALL_CLASS:
            w_ *= 3.0
            has_ball = True
        e = (w_, x1, y1, x2, y2)
        if cls == PERSON_CLASS_ID:
            pp.append(e)
        elif cls in HIGH_PRIO_CLASSES:
            hp.append(e)
        ap.append(e)
    pool = pp or hp or ap
    if not pool:
        return None
    tw = sum(e[0] for e in pool)
    if tw == 0:
        return None
    cx = int(sum(e[0] * (e[1] + e[3]) / 2 for e in pool) / tw)
    cy = int(sum(e[0] * (e[2] + e[4]) / 2 for e in pool) / tw)
    return DetectionResult(cx, cy, min(e[1] for e in pool), min(e[2] for e in pool),
                           max(e[3] for e in pool), max(e[4] for e in pool), len(pool), has_ball)

def detect_persons_all(frame: np.ndarray, model: Any, confidence: float = 0.45) -> List[Tuple[int, int, int, int]]:
    if model is None:
        return []
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception:
        return []
    if results.boxes is None or len(results.boxes) == 0:
        return []
    p = []
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            p.append((x1, y1, x2, y2))
    p.sort(key=lambda b: b[0])
    return p


# ── Framing helpers ───────────────────────────────────────────────────────────
def _apply_lower_third_guard(cy: float, crop_h: int, subject_cy_src: float, orig_h: int) -> float:
    hh = crop_h // 2
    max_cy = subject_cy_src - (1.0 - LOWER_THIRD_GUARD) * crop_h + hh
    return min(cy, min(max_cy, orig_h - hh))

def _soi_region_label(cx: int, cy: int, w: int, h: int) -> str:
    col = "left" if cx < w // 3 else ("right" if cx > 2 * w // 3 else "center")
    row = "upper" if cy < h // 3 else ("lower" if cy > 2 * h // 3 else "mid")
    if row == "mid" and col == "center":
        return "center"
    if row == "mid":
        return col
    return f"{row}-{col}"

def frame_for_union(ux1: int, uy1: int, ux2: int, uy2: int,
                    orig_w: int, orig_h: int, crop_w: int, crop_h: int) -> Tuple[int, int]:
    ucx = (ux1 + ux2) // 2; ucy = (uy1 + uy2) // 2
    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(ucx, orig_w - hw))
    cy = max(hh, min(ucy, orig_h - hh))
    cy = _apply_lower_third_guard(cy, crop_h, ucy, orig_h)
    return cx, max(hh, min(cy, orig_h - hh))

def talking_head_center(faces: List[Tuple[int, int, int, int]], orig_w: int, orig_h: int,
                        crop_w: int, crop_h: int, bias: float = 0.30) -> Optional[Tuple[int, int]]:
    if not faces:
        return None
    ux1 = min(f[0] for f in faces); uy1 = min(f[1] for f in faces)
    ux2 = max(f[2] for f in faces); uy2 = max(f[3] for f in faces)
    face_cx = (ux1 + ux2) // 2; face_cy = (uy1 + uy2) // 2
    cy = int(face_cy * (1 - bias) + (face_cy + crop_h // 6) * bias)
    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(face_cx, orig_w - hw))
    cy = max(hh, min(cy, orig_h - hh))
    cy = _apply_lower_third_guard(cy, crop_h, face_cy, orig_h)
    return cx, max(hh, min(cy, orig_h - hh))


# ── Panel helpers ─────────────────────────────────────────────────────────────
def _group_union(persons: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return (min(p[0] for p in persons), min(p[1] for p in persons),
            max(p[2] for p in persons), max(p[3] for p in persons))

def _crop_group_to_strip(frame: np.ndarray, group: List[Tuple[int, int, int, int]],
                         strip_w: int, strip_h: int, expand: float = PANEL_CROP_EXPAND,
                         vignette_strength: float = 0.0, color_grade: str = "none") -> np.ndarray:
    fh, fw = frame.shape[:2]
    if not group:
        crop = frame
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        ucx = (ux1 + ux2) // 2; ucy = (uy1 + uy2) // 2
        union_w = max(ux2 - ux1, 1); strip_ratio = strip_w / strip_h
        crop_w = int(union_w * expand); crop_h = int(crop_w / strip_ratio)
        if crop_h > fh:
            crop_h = fh; crop_w = int(crop_h * strip_ratio)
        if crop_w > fw:
            crop_w = fw; crop_h = int(crop_w / strip_ratio)
        crop_w = max(crop_w, 2); crop_h = max(crop_h, 2)
        x1 = max(0, min(ucx - crop_w // 2, fw - crop_w))
        y1 = max(0, min(ucy - crop_h // 2, fh - crop_h))
        x2 = min(x1 + crop_w, fw); y2 = min(y1 + crop_h, fh)
        x1 = max(0, x2 - crop_w); y1 = max(0, y2 - crop_h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
    result = cv2.resize(crop, (strip_w, strip_h), interpolation=cv2.INTER_LANCZOS4)
    if color_grade and color_grade != "none":
        result = apply_color_grade(result, color_grade)
    if vignette_strength > 0:
        result = apply_vignette(result, vignette_strength)
    return result

def _detect_panel_mode(input_path: str, model: Any, fps: float, total_frames: int,
                       orig_w: int, orig_h: int, confidence: float = 0.45,
                       n_probe: int = 16) -> Tuple[bool, int]:
    if model is None:
        return False, 0
    probe_ts = np.linspace(1.0, max(1.5, total_frames / fps - 1.0), n_probe)
    hits = 0; max_persons = 0
    for t in probe_ts:
        frame = _read_frame_at(input_path, orig_w, orig_h, t,
                               scale_w=640, scale_h=max(1, int(640 * orig_h / orig_w)))
        if frame is None:
            continue
        persons = detect_persons_all(frame, model, confidence)
        n = len(persons)
        max_persons = max(max_persons, n)
        if n >= PANEL_MIN_PERSONS:
            hits += 1
    return hits > n_probe * 0.5, max_persons

class PanelSlotSmoother:
    def __init__(self, alpha: float = PANEL_SLOT_EMA):
        self.alpha = alpha
        self._slots: List[Optional[Tuple[float, float, float, float]]] = [None, None, None, None]
    def update(self, groups: List[List[Tuple[int, int, int, int]]]) -> List[List[Tuple[int, int, int, int]]]:
        def _ema_box(prev: Optional[Tuple], new_box: Tuple) -> Tuple[float, float, float, float]:
            if prev is None:
                return new_box
            a = self.alpha
            return (prev[0]*(1-a) + new_box[0]*a, prev[1]*(1-a) + new_box[1]*a,
                    prev[2]*(1-a) + new_box[2]*a, prev[3]*(1-a) + new_box[3]*a)
        out = []
        for i, group in enumerate(groups):
            if not group:
                out.append(group)
                continue
            u = _group_union(group)
            s = _ema_box(self._slots[i], u)
            self._slots[i] = s
            out.append([tuple(int(v) for v in s)])
        return out

def _render_panel_frame(frame: np.ndarray, persons: List[Tuple[int, int, int, int]],
                        out_w: int, out_h: int, prev_slots: Optional[List] = None,
                        vignette_strength: float = VIGNETTE_STRENGTH * 0.7,
                        color_grade: str = "none", slot_smoother: Optional[PanelSlotSmoother] = None,
                        n_rows: int = 2) -> Tuple[np.ndarray, List]:
    persons = sorted(persons, key=lambda b: (b[0] + b[2]) // 2)
    n = len(persons)
    if n == 0:
        groups = [prev_slots[i] if prev_slots and i < len(prev_slots) and prev_slots[i] else []
                  for i in range(n_rows)]
    elif n <= n_rows:
        groups = [[p] for p in persons] + [[] for _ in range(n_rows - n)]
    else:
        per_row = max(1, n // n_rows)
        groups = []
        for i in range(n_rows):
            start = i * per_row
            end = n if i == n_rows - 1 else (i + 1) * per_row
            groups.append(persons[start:end])
    if slot_smoother:
        groups = slot_smoother.update(groups)
    strip_hs = []
    rem = out_h - (n_rows - 1) * PANEL_DIVIDER_PX
    base = rem // n_rows
    for i in range(n_rows):
        if i == n_rows - 1:
            strip_hs.append(rem - sum(strip_hs))
        else:
            strip_hs.append(base & ~1)
    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    y_off = 0
    for i, (group, sh) in enumerate(zip(groups, strip_hs)):
        strip = _crop_group_to_strip(frame, group or [], out_w, sh,
                                     vignette_strength=vignette_strength, color_grade=color_grade)
        canvas[y_off:y_off+sh, :] = strip
        y_off += sh
        if i < n_rows - 1:
            dy1 = max(0, y_off - PANEL_DIVIDER_PX // 2)
            dy2 = min(out_h, y_off + (PANEL_DIVIDER_PX + 1) // 2)
            canvas[dy1:dy2, :] = PANEL_DIVIDER_COLOR
            y_off = dy2
    return canvas, groups


# ── Optical flow / saliency ─────────────────────────────────────────────────
def optical_flow_center(prev: Optional[np.ndarray], curr: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int]]:
    if prev is None or curr is None:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        b = max(1, int(w * 0.04))
        mag[:, :b] = mag[:, w-b:] = mag[:b, :] = mag[h-b:, :] = 0
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
    lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31, 31), 0)
    sat = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (31, 31), 0)
    sal = lap / (lap.max() + 1e-6) + sat / (sat.max() + 1e-6)
    b = max(1, int(w * 0.05))
    sal[:, :b] = sal[:, w-b:] = sal[:b, :] = sal[h-b:, :] = 0
    t = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t)

def is_scene_change(prev: Optional[np.ndarray], curr: np.ndarray, threshold: float = 0.35) -> bool:
    if prev is None:
        return False
    try:
        return float(cv2.absdiff(prev, curr).mean()) / 255.0 > threshold
    except Exception:
        return False


# ── Smoothing helpers (kept for reference; spring replaces live path) ──────────
def smooth_centers(centers: List[Tuple[int, int]], speeds: List[float],
                   base_window: int = 27, adaptive: bool = True,
                   scene_cuts: Optional[List[int]] = None,
                   fast_motion_mask: Optional[List[bool]] = None) -> List[Tuple[int, int]]:
    if not centers or len(centers) < 3:
        return list(centers) if centers else []
    n = len(centers)
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n:
        spd = np.pad(spd, (0, n - len(spd)), mode="edge")
    bounds = [0] + sorted(set(scene_cuts or [])) + [n]
    rx, ry = xs.copy(), ys.copy()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i+1]
        if e - s < 3:
            continue
        base_w = max(_vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window, 13)
        if fast_motion_mask:
            fast_ratio = sum(fast_motion_mask[s:e]) / max(e - s, 1)
            if fast_ratio > 0.3:
                base_w = max(5, int(base_w * 0.5))
        xs_s, ys_s = _gauss_seg(xs[s:e], ys[s:e], base_w)
        rx[s:e] = xs_s
        ry[s:e] = ys_s
    return [(int(x), int(y)) for x, y in zip(rx, ry)]

def _vel_to_window(speed: float) -> int:
    t = VELOCITY_SMOOTH_TABLE
    if speed <= t[0][0]:
        return t[0][1]
    if speed >= t[-1][0]:
        return t[-1][1]
    for i in range(len(t) - 1):
        v0, w0 = t[i]
        v1, w1 = t[i+1]
        if v0 <= speed <= v1:
            tt = (speed - v0) / (v1 - v0 + 1e-9)
            w = int(w0 + tt * (w1 - w0))
            return w if w % 2 == 1 else w + 1
    return 27

def _gauss_seg(xs: np.ndarray, ys: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n < 3:
        return xs.copy(), ys.copy()
    w = min(window, n - 1)
    w = w if w % 2 == 1 else w - 1
    if w < 3:
        return xs.copy(), ys.copy()
    h2 = w // 2
    sigma = h2 / 2.5 + 1e-9
    k = np.exp(-0.5 * (np.arange(-h2, h2+1) / sigma) ** 2)
    k /= k.sum()
    sx = np.convolve(np.pad(xs, h2, "edge"), k, "valid")[:n]
    sy = np.convolve(np.pad(ys, h2, "edge"), k, "valid")[:n]
    return sx, sy


# ── Whisper / translate ───────────────────────────────────────────────────────
def _seconds_to_srt_time(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sc = int(s % 60); ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def transcribe_to_srt(video_path: str, srt_path: str, whisper_model: str = "base",
                      language: Optional[str] = None, max_chars_per_line: int = 42,
                      progress_callback: Optional[Any] = None) -> bool:
    def _p(v: float, msg: str = ""):
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
        lines = []
        idx = 1
        words = []
        for seg in result.get("segments", []):
            for w_ in seg.get("words", []):
                words.append({"word": w_["word"].strip(), "start": w_["start"], "end": w_["end"]})
        buf = []
        buf_len = 0
        def flush():
            nonlocal idx, buf, buf_len
            if not buf:
                return
            lines.append(f"{idx}\n{_seconds_to_srt_time(buf[0]['start'])} --> {_seconds_to_srt_time(buf[-1]['end'])}\n{' '.join(x['word'] for x in buf)}\n")
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

def translate_srt(srt_path: str, target_language: str, source_language: str = "auto",
                  progress_callback: Optional[Any] = None) -> bool:
    def _p(v: float, msg: str = ""):
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass
    if not translation_available() or not target_language:
        return bool(not target_language)
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


# ── Clip detection ────────────────────────────────────────────────────────────
def _frame_saliency_score(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_score = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)
    motion_score = 0.0
    if prev_frame is not None:
        motion_score = min(float(cv2.absdiff(gray, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)).mean()) / 30.0, 1.0)
    sat_score = min(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 128.0, 1.0)
    return 0.4 * motion_score + 0.4 * lap_score + 0.2 * sat_score

def _compute_frame_scores(input_path: str, fps: float, total_frames: int,
                          orig_w: int, orig_h: int, sample_every: int = 15,
                          progress_callback: Optional[Any] = None) -> Tuple[np.ndarray, List[int]]:
    def _p(v: float, msg: str = ""):
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass
    scores = []
    scene_cuts = []
    prev_gray = None
    prev_frame = None
    sw = min(orig_w, 640)
    sh = max(1, int(sw * orig_h / orig_w))
    report_n = max(1, total_frames // 20)
    fi = 0
    with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=sw, scale_h=sh) as reader:
        for frame in reader:
            if fi >= total_frames:
                break
            if fi % sample_every == 0:
                cg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and float(cv2.absdiff(prev_gray, cg).mean()) / 255.0 > 0.30:
                    scene_cuts.append(fi)
                scores.append(_frame_saliency_score(frame, prev_frame))
                prev_gray = cg
                prev_frame = frame.copy()
            if fi % report_n == 0:
                _p(fi / total_frames, f"Scanning {fi}/{total_frames}...")
            fi += 1
    return np.array(scores, dtype=float), scene_cuts

def detect_clips(input_path: str, min_duration_sec: float = 25.0,
                 max_duration_sec: float = 65.0, target_n_clips: int = 10,
                 model: Optional[Any] = None, confidence: float = 0.45,
                 progress_callback: Optional[Any] = None) -> List[ClipSegment]:
    def _p(v: float, msg: str = ""):
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass
    info = get_video_info(input_path)
    fps, total_frames = info["fps"], info["total_frames"]
    duration = info["duration_seconds"]
    orig_w, orig_h = info["width"], info["height"]
    sample_every = max(1, int(fps))
    _p(0.0, "Scanning...")
    scores, scene_cuts_frames = _compute_frame_scores(input_path, fps, total_frames, orig_w, orig_h,
        sample_every=sample_every, progress_callback=lambda v, m: _p(v * 0.45, m))
    if len(scores) == 0:
        return []
    _p(0.45, "Computing arcs...")
    window = max(5, int(30 / (sample_every / fps)))
    ss = (np.convolve(scores, np.ones(window) / window, mode="same") if len(scores) >= window else scores.copy())
    if ss.max() > 0:
        ss = ss / ss.max()
    min_gap = max(1, int(min_duration_sec * fps / sample_every))
    peaks = []
    for i in range(1, len(ss) - 1):
        wh = min_gap // 2
        lo = max(0, i - wh)
        hi = min(len(ss), i + wh + 1)
        if ss[i] == ss[lo:hi].max() and ss[i] > 0.3:
            if not peaks or i - peaks[-1] > min_gap // 2:
                peaks.append(i)
    peaks.sort(key=lambda i: ss[i], reverse=True)
    peaks = peaks[:target_n_clips * 2]

    def _arc(pi: int) -> Tuple[float, float]:
        ps = pi * sample_every / fps
        rs = max(0.0, ps - max_duration_sec * 0.4)
        re = min(duration, rs + max_duration_sec)
        for sc in reversed(scene_cuts_frames):
            sc_s = sc / fps
            if 0 < ps - sc_s < 15.0:
                rs = max(0.0, sc_s - 1.0)
                break
        for sc in scene_cuts_frames:
            sc_s = sc / fps
            if 0 < sc_s - ps < 15.0:
                re = min(duration, sc_s + 0.5)
                break
        cd = re - rs
        if cd < min_duration_sec:
            re = min(duration, rs + min_duration_sec)
        elif cd > max_duration_sec:
            c = (rs + re) / 2
            rs = max(0.0, c - max_duration_sec / 2)
            re = min(duration, rs + max_duration_sec)
        return rs, re

    cands = []
    for pi in peaks:
        s, e = _arc(pi)
        sc = float(ss[pi])
        if not any(min(e, ce) - max(s, cs) > min_duration_sec * 0.5 for cs, ce, _ in cands):
            cands.append((s, e, sc))
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:target_n_clips]
    cands.sort(key=lambda x: x[0])
    _p(0.55, "SOI per clip...")
    segments = []
    for ci, (ss2, se, score) in enumerate(cands):
        _p(0.55 + 0.35 * (ci / max(len(cands), 1)), f"Clip {ci+1}/{len(cands)}...")
        soi_xs = []
        soi_ys = []
        n_s = min(8, max(2, int(se - ss2)))
        for t in np.linspace(ss2 + 1, se - 1, n_s):
            frame = _read_frame_at(input_path, orig_w, orig_h, t,
                                   scale_w=640, scale_h=max(1, int(640 * orig_h / orig_w)))
            if frame is None:
                continue
            if model is not None:
                try:
                    res = model(frame, verbose=False, conf=confidence)[0]
                    if res.boxes is not None:
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
        sr = "center"
        if soi_xs:
            sr = _soi_region_label(int(np.median(soi_xs)), int(np.median(soi_ys)), orig_w, orig_h)
        ms = int(ss2 // 60)
        secs = int(ss2 % 60)
        me = int(se // 60)
        sece = int(se % 60)
        segments.append(ClipSegment(start_sec=ss2, end_sec=se, score=score, soi_region=sr,
            peak_frame=int(np.linspace(ss2 + 1, se - 1, n_s)[n_s // 2] * fps),
            title=f"Clip {ci+1}  ({ms}:{secs:02d} - {me}:{sece:02d})"))
    _p(1.0, f"Found {len(segments)} clips")
    return segments


# ── process_video — main entry point ─────────────────────────────────────────
def process_video(
    input_path: str, output_path: str,
    target_preset_label: str = "Match source (no upscale)",
    tracking_mode: str = "subject", talking_head_bias: float = 0.30,
    sample_interval: Optional[int] = None, confidence: float = 0.45,
    use_optical_flow: bool = True,
    smooth_window: int = 27, adaptive_smoothing: bool = True,
    rule_of_thirds: bool = True,
    scene_cut_threshold: float = 0.35, output_fps: Optional[float] = None,
    crf: int = 23, encoder_preset: str = "fast",
    audio_bitrate: str = "128k", yolo_weights: str = "yolov8n.pt",
    burn_subtitles: bool = False, whisper_model: str = "base",
    whisper_language: Optional[str] = None,
    subtitle_style_name: str = "Bold White (TikTok)", subtitle_max_chars: int = 42,
    subtitle_translate_to: Optional[str] = None,
    vignette_strength: float = VIGNETTE_STRENGTH,
    sharpen_strength: float = 0.0,
    color_grade: str = "none",
    ken_burns: bool = False,
    dissolve_cuts: bool = True,
    ffmpeg_sharpen: bool = False,
    sport_mode: bool = False,
    enable_analytics: bool = False,
    analytics_overlay: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    enable_analytics: Track players/ball and export JSON stats.
    analytics_overlay: Burn speed labels, trails, and possession into output.
    """
    def _p(v: float, msg: str = ""):
        if progress_callback:
            try:
                progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception:
                pass

    result_meta = {"output_path": output_path, "subtitle_path": None, "clamped": False,
                   "effective_size": (0, 0), "duration": 0.0, "panel_mode": False,
                   "sport_mode": sport_mode, "analytics": None}

    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path) / 1024**2 > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB.")

    info = get_video_info(input_path)
    fps, total_frames = info["fps"], info["total_frames"]
    orig_w, orig_h = info["width"], info["height"]
    duration = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ProcessingError("Corrupt or unreadable video.")
    if not info["is_landscape"]:
        raise ProcessingError("Video is already vertical.")

    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    req_w, req_h = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update(clamped=clamped, effective_size=(target_w, target_h), duration=duration)
    _p(0.01, f"Output {target_w}x{target_h}  source {orig_w}x{orig_h}")

    if not sample_interval:
        sample_interval = max(1, int(fps / 5))
    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)

    det_scale = min(1.0, 640 / orig_w)
    det_w = max(1, int(orig_w * det_scale))
    det_h = max(1, int(orig_h * det_scale))
    sx, sy = orig_w / det_w, orig_h / det_h

    srt_path = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "Transcribing...")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt")
        os.close(srt_fd)
        ok = transcribe_to_srt(input_path, srt_path, whisper_model=whisper_model,
                               language=whisper_language,
                               max_chars_per_line=subtitle_max_chars,
                               progress_callback=lambda v, m: _p(0.02 + v * 0.08, m))
        if not ok:
            if os.path.exists(srt_path):
                os.unlink(srt_path)
                srt_path = None
        else:
            if subtitle_translate_to:
                translate_srt(srt_path, target_language=subtitle_translate_to,
                              progress_callback=lambda v, m: _p(0.10 + v * 0.05, m))
            result_meta["subtitle_path"] = srt_path

    start_pct = 0.10
    model_obj = None
    if tracking_mode == "subject":
        _p(start_pct, "Loading YOLO...")
        model_obj = _get_model(yolo_weights)
        if model_obj is None:
            _p(start_pct, "YOLO unavailable - saliency fallback")
    elif tracking_mode == "talking_head":
        _p(start_pct, "Loading face detector...")
        if _get_haar() is None and _load_face_net() is None:
            _p(start_pct, "No face detector - saliency fallback")

    # Panel detection
    is_panel = False
    panel_n_rows = 2
    slot_smoother = None
    if tracking_mode == "subject" and model_obj is not None:
        _p(start_pct + 0.01, "Checking panel/group shot...")
        is_panel, max_persons = _detect_panel_mode(input_path, model_obj, fps, total_frames,
                                                    orig_w, orig_h, confidence, n_probe=8)
        if is_panel:
            panel_n_rows = min(4, max(2, max_persons))
            result_meta["panel_mode"] = True
            result_meta["panel_rows"] = panel_n_rows
            _p(start_pct + 0.02, f"Panel mode - {panel_n_rows}-row vertical split")
            slot_smoother = PanelSlotSmoother()

    extra_vf = []
    eq_map = {"warm":"brightness=0.02:saturation=1.12:gamma_r=1.05:gamma_b=0.95",
              "cool":"brightness=0.01:saturation=1.08:gamma_r=0.95:gamma_b=1.05",
              "vibrant":"brightness=0.0:saturation=1.25:contrast=1.05",
              "matte":"brightness=0.03:saturation=0.85:contrast=0.92"}
    if color_grade in eq_map:
        extra_vf.append(f"eq={eq_map[color_grade]}")
    if ffmpeg_sharpen:
        extra_vf.append("unsharp=5:5:0.8:3:3:0.0")

    style = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])
    proc = _open_ffmpeg_encoder(output_path, target_w, target_h, render_fps,
                                 audio_source=input_path, crf=crf, preset=encoder_preset,
                                 audio_bitrate=audio_bitrate, subtitle_path=srt_path,
                                 subtitle_style=style, extra_vf=extra_vf if extra_vf else None)

    if vignette_strength > 0:
        _build_vignette(target_w, target_h, vignette_strength)
    if color_grade and color_grade != "none":
        _build_lut(color_grade)

    dissolve_buf = DissolveBuffer(DISSOLVE_FRAMES) if dissolve_cuts else None
    velocity_tracker = VelocityTracker() if sport_mode else None
    transition_handler = TransitionHandler(fps) if sport_mode else None

    # SHAKE-1: Spring camera
    cam_spring = CameraSpring(orig_w / 2.0, orig_h / 2.0)

    # ANAL-1: Analytics
    player_tracker = PlayerTracker() if enable_analytics else None
    analytics_renderer = AnalyticsRenderer(render_fps) if enable_analytics and analytics_overlay else None
    all_raw_detections: List[List[Tuple[int,int,int,int,int]]] = []  # per frame

    scene_cuts = []
    prev_gray = None
    prev_flow = None
    last_det = None
    det_dropout = 0
    MAX_DROPOUT = int(fps * 1.5)
    hw, hh = crop_w // 2, crop_h // 2
    prev_slots = None
    last_out_frame = None
    rpt_n = max(1, total_frames // 40)
    fi = 0

    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames:
                    break

                is_sample = (fi % sample_interval == 0)
                fast_motion = False
                anchor_cx: Optional[float] = None
                anchor_cy: Optional[float] = None
                subject_count = 0
                frame_dets: List[Tuple[int,int,int,int,int]] = []

                if is_sample:
                    det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                    cg = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
                    cut = is_scene_change(prev_gray, cg, scene_cut_threshold)
                    if cut:
                        scene_cuts.append(fi)
                        prev_flow = None
                        det_dropout = 0
                        if dissolve_buf and last_out_frame is not None:
                            dissolve_buf.on_cut(last_out_frame)

                    # Flow for sports
                    flow_mag = 0.0
                    if use_optical_flow and sport_mode:
                        sm = cv2.resize(cg, (max(1, det_w // 2), max(1, det_h // 2)))
                        if prev_flow is not None:
                            fc = optical_flow_center(prev_flow, sm, det_w // 2, det_h // 2)
                            if fc:
                                flow_mag = math.hypot(fc[0] - det_w//4, fc[1] - det_h//4)
                        prev_flow = sm

                    fast_motion = flow_mag > SPORT_FAST_FLOW_THRESH * det_scale
                    prev_gray = cg

                    if is_panel:
                        pass
                    elif tracking_mode == "talking_head":
                        faces = detect_faces(det_frame, confidence_thresh=0.5)
                        if faces:
                            faces_orig = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                          for x1, y1, x2, y2 in faces]
                            r = talking_head_center(faces_orig, orig_w, orig_h, crop_w, crop_h, talking_head_bias)
                            if r:
                                anchor_cx, anchor_cy = float(r[0]), float(r[1])
                                det_dropout = 0
                        if anchor_cx is None and use_optical_flow:
                            sm = cv2.resize(cg, (max(1, det_w // 2), max(1, det_h // 2)))
                            if prev_flow is not None:
                                fc = optical_flow_center(prev_flow, sm, det_w // 2, det_h // 2)
                                if fc:
                                    anchor_cx, anchor_cy = float(fc[0] * 2 * sx), float(fc[1] * 2 * sy)
                            prev_flow = sm
                            det_dropout += sample_interval
                    else:
                        if model_obj is not None:
                            det = detect_subjects(det_frame, model_obj, confidence)
                            if det is not None:
                                subject_count = det.count
                                anchor_cx, anchor_cy = float(det.cx * sx), float(det.cy * sy)
                                # Union-based framing is better for groups
                                anchor_cx, anchor_cy = frame_for_union(
                                    int(det.ux1 * sx), int(det.uy1 * sy),
                                    int(det.ux2 * sx), int(det.uy2 * sy),
                                    orig_w, orig_h, crop_w, crop_h)
                                anchor_cx, anchor_cy = float(anchor_cx), float(anchor_cy)
                                last_det = (anchor_cx, anchor_cy)
                                det_dropout = 0
                                # Predict ball
                                if sport_mode and det.has_ball and velocity_tracker:
                                    pred = velocity_tracker.update(fi, anchor_cx, anchor_cy)
                                    if pred:
                                        anchor_cx, anchor_cy = pred

                                # Collect analytics detections
                                if player_tracker is not None:
                                    try:
                                        res = model_obj(det_frame, verbose=False, conf=confidence)[0]
                                        if res.boxes is not None:
                                            for box in res.boxes:
                                                cls = int(box.cls[0])
                                                if cls in (PERSON_CLASS_ID, SPORTS_BALL_CLASS):
                                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                                    frame_dets.append((cls, x1, y1, x2, y2))
                                    except Exception:
                                        pass

                        if anchor_cx is None and use_optical_flow:
                            sm = cv2.resize(cg, (max(1, det_w // 2), max(1, det_h // 2)))
                            if prev_flow is not None:
                                fc = optical_flow_center(prev_flow, sm, det_w // 2, det_h // 2)
                                if fc:
                                    anchor_cx, anchor_cy = float(fc[0] * 2 * sx), float(fc[1] * 2 * sy)
                            prev_flow = sm
                            det_dropout += sample_interval
                        if anchor_cx is None:
                            if last_det and det_dropout < MAX_DROPOUT:
                                anchor_cx, anchor_cy = last_det
                            else:
                                sc_ = saliency_center(det_frame)
                                anchor_cx, anchor_cy = float(sc_[0] * sx), float(sc_[1] * sy)

                    # Transition margin
                    margin = 1.0
                    if sport_mode and transition_handler:
                        margin = transition_handler.update(subject_count, fast_motion)
                    if anchor_cx is not None and not is_panel:
                        if margin > 1.0:
                            anchor_cx = float(anchor_cx + (anchor_cx - orig_w/2.0) * (margin - 1.0) * 0.5)
                            anchor_cy = float(anchor_cy + (anchor_cy - orig_h/2.0) * (margin - 1.0) * 0.5)

                # Spring update every frame (float coords)
                if anchor_cx is not None and not is_panel:
                    cam_spring.update(anchor_cx, anchor_cy)

                # Clamp float camera to bounds
                eff_cw, eff_ch = crop_w, crop_h
                margin = 1.0
                if sport_mode and transition_handler:
                    margin = transition_handler._margin
                eff_cw = min(int(crop_w * margin), orig_w)
                eff_ch = min(int(crop_h * margin), orig_h)
                eff_hw, eff_hh = eff_cw / 2.0, eff_ch / 2.0

                cam_x = max(eff_hw, min(cam_spring.x, orig_w - eff_hw))
                cam_y = max(eff_hh, min(cam_spring.y, orig_h - eff_hh))

                if is_panel:
                    if is_sample:
                        det_frame_p = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                        persons_det = detect_persons_all(det_frame_p, model_obj, confidence)
                        persons_full = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                        for x1, y1, x2, y2 in persons_det]
                    else:
                        persons_full = [b for grp in (prev_slots or []) if grp for b in grp]
                    out_frame, prev_slots = _render_panel_frame(
                        frame, persons_full, target_w, target_h, prev_slots,
                        vignette_strength=vignette_strength * 0.7,
                        color_grade=color_grade, slot_smoother=slot_smoother,
                        n_rows=panel_n_rows)
                else:
                    left = int(max(0, min(cam_x - eff_hw, orig_w - eff_cw)))
                    top = int(max(0, min(cam_y - eff_hh, orig_h - eff_ch)))
                    right = min(left + eff_cw, orig_w)
                    bottom = min(top + eff_ch, orig_h)
                    left = max(0, right - eff_cw)
                    top = max(0, bottom - eff_ch)
                    crop = frame[top:bottom, left:right]

                    if crop.shape[1] != target_w or crop.shape[0] != target_h:
                        out_frame = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        out_frame = crop

                    if ken_burns:
                        out_frame = apply_ken_burns(out_frame, fi, fps)
                    if sharpen_strength > 0:
                        out_frame = apply_sharpen(out_frame, sharpen_strength)
                    if color_grade and color_grade != "none":
                        out_frame = apply_color_grade(out_frame, color_grade)
                    if vignette_strength > 0:
                        out_frame = apply_vignette(out_frame, vignette_strength)

                # Analytics overlay (on full-res output frame)
                if player_tracker is not None:
                    # Update tracker with this frame's detections (in original coords)
                    if frame_dets:
                        dets_full = [(cls, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                     for cls, x1, y1, x2, y2 in frame_dets]
                        player_tracker.update(dets_full, fi, orig_w, orig_h, fps)
                        player_tracker.compute_possession()
                    if analytics_renderer is not None:
                        out_frame = analytics_renderer.draw(out_frame, list(player_tracker.trackers.values()))

                if dissolve_buf and dissolve_buf.active:
                    out_frame = dissolve_buf.blend(out_frame)
                last_out_frame = out_frame

                try:
                    proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    break

                fi += 1
                if fi % rpt_n == 0:
                    _p(0.12 + 0.75 * (fi / total_frames), f"{fi}/{total_frames}...")
    finally:
        pass

    _p(0.88, "Finalizing encode...")
    _close_ffmpeg_encoder(proc, output_path)

    # Export analytics JSON
    if player_tracker is not None:
        analytics_data = {
            "video": {"width": orig_w, "height": orig_h, "fps": fps, "frames": total_frames},
            "subjects": [t.to_dict(fps, sx, sy) for t in player_tracker.trackers.values()],
        }
        json_path = os.path.splitext(output_path)[0] + "_analytics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analytics_data, f, indent=2)
        result_meta["analytics"] = json_path
        print(f"Analytics: {json_path}", file=sys.stderr)

    _p(1.0, "Done!")
    print(f"Output: {output_path}  ({os.path.getsize(output_path)/1024**2:.1f} MB)"
          f"  cuts={len(scene_cuts)}", file=sys.stderr)
    return result_meta


# ── Batch clip pipeline ───────────────────────────────────────────────────────
def process_clips_batch(
    input_path: str, output_dir: str, clips: List[ClipSegment],
    target_preset_label: str = "720p   (720x1280  - HD)",
    tracking_mode: str = "subject", talking_head_bias: float = 0.30,
    confidence: float = 0.45, smooth_window: int = 27,
    adaptive_smoothing: bool = True, use_optical_flow: bool = True,
    rule_of_thirds: bool = True, crf: int = 23, encoder_preset: str = "fast",
    audio_bitrate: str = "128k", yolo_weights: str = "yolov8n.pt",
    burn_subtitles: bool = False, whisper_model: str = "base",
    subtitle_style_name: str = "Bold White (TikTok)", subtitle_max_chars: int = 42,
    vignette_strength: float = VIGNETTE_STRENGTH, sharpen_strength: float = 0.0,
    color_grade: str = "none", ken_burns: bool = False,
    dissolve_cuts: bool = True, ffmpeg_sharpen: bool = False,
    sport_mode: bool = False,
    enable_analytics: bool = False,
    analytics_overlay: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    def _p(v: float, msg: str = ""):
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
        trimmed_path = None
        out_path = None
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

            meta = process_video(trimmed_path, out_path,
                target_preset_label=target_preset_label,
                tracking_mode=tracking_mode, talking_head_bias=talking_head_bias,
                confidence=confidence, smooth_window=smooth_window,
                adaptive_smoothing=adaptive_smoothing,
                use_optical_flow=use_optical_flow, rule_of_thirds=rule_of_thirds,
                crf=crf, encoder_preset=encoder_preset, audio_bitrate=audio_bitrate,
                yolo_weights=yolo_weights, burn_subtitles=burn_subtitles,
                whisper_model=whisper_model,
                subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                vignette_strength=vignette_strength, sharpen_strength=sharpen_strength,
                color_grade=color_grade, ken_burns=ken_burns,
                dissolve_cuts=dissolve_cuts, ffmpeg_sharpen=ffmpeg_sharpen,
                sport_mode=sport_mode,
                enable_analytics=enable_analytics,
                analytics_overlay=analytics_overlay,
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
