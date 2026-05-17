"""
verticalize.py  —  AI Vertical Video Converter  v3.1
──────────────────────────────────────────────────────
CHANGES v3.1 (bug-fixes & improvements over v3.0):

BUG-1   scene_cuts index mismatch in smooth_centers(): scene cut indices were
        raw frame numbers but smooth_centers received detection-sample indices.
        Fixed by converting frame indices → sample indices before the call.

BUG-2   PanelSlotSmoother._smooth() returned group unchanged when empty, so
        the internal EMA state was never updated for absent slots. Now returns
        the EMA-smoothed last known box when group is empty (hold-last logic).

BUG-3   FFmpegVideoReader._candidate_cmds() put seek BEFORE -i for the
        libdav1d candidate but not for the hwaccel candidate, producing
        slightly inconsistent seek behaviour. Unified into one helper.

BUG-4   _open_ffmpeg_encoder() wrote -aspect W:H after -profile:v / -level
        flags, which some FFmpeg builds reject. Moved -aspect before codec
        flags where it is always accepted.

BUG-5   process_video() used undefined variables `hw`/`hh` (from the outer
        scope) inside the panel render branch, risking stale values if crop
        geometry differed. Removed stale usage; each branch now computes its
        own geometry.

BUG-6   _render_panel_frame() iterated prev_slots assuming it was always a
        list of lists, but PanelSlotSmoother could return a tuple of lists
        containing un-boxed tuples. Added explicit normalisation.

PANEL-5 Sports false-positive fix (the basketball problem):
        _detect_panel_mode() now checks five independent signals before
        declaring panel mode. A video must pass ALL of:
          a. ≥ PANEL_MIN_PERSONS persons visible                (was already there)
          b. Stable L/R split across probes                     (was already there)
          c. Low inter-frame person-motion (px/frame)          NEW
          d. Mean person bounding-box area ≥ MIN_PANEL_PERSON_AREA_FRAC
             of the frame area (panel guests are large/close-up) NEW
          e. Person count variance ≤ MAX_PANEL_COUNT_VARIANCE  NEW
        Basketball fails (c) and (d) and (e) simultaneously.

ANALYTICS-2  get_analytics_meta() extended with:
        tracking_mode, panel_mode, scene_cuts_count, smooth_jitter_raw,
        smooth_jitter_smooth, smoothness_pct, max_jump_raw, color_grade,
        subtitle_burned, subtitle_language, vignette_strength, crf, preset.
        All values are JSON-serialisable.

CLEAN   Removed dead variables, redundant comments, stale TODO markers;
        unified progress-callback helper pattern; consistent type hints.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections import namedtuple
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

PERSON_CLASS_ID   = 0
HIGH_PRIO_CLASSES = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB  = 2_000
MIN_FRAME_DIM     = 240
MAX_FRAMES_GUARD  = 1_080_000
LOWER_THIRD_GUARD = 0.80

# Panel detection thresholds
PANEL_MIN_PERSONS            = 2    # minimum persons per probe frame
PANEL_PROBE_COUNT            = 24   # number of frames probed for panel detection
PANEL_MAJORITY_FRAC          = 0.50 # fraction of probes that must show stable L/R split
PANEL_STABILITY_FRAC         = 0.70 # fraction of multi-person frames that must be stable
PANEL_MAX_PERSON_MOTION      = 12.0 # px/frame normalised to det_w — above this = sports
PANEL_MIN_PERSON_AREA_FRAC   = 0.04 # person bbox must be ≥ 4 % of frame area (close-up)
PANEL_MAX_COUNT_VARIANCE     = 2.5  # std-dev of person count across probes

# Velocity → Gaussian smoothing window (frames). Wider = smoother but more lag.
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
VIGNETTE_STRENGTH     = 0.55
VIGNETTE_FALLOFF      = 1.8
COLOR_GRADES          = ("none", "warm", "cool", "vibrant", "matte")
PANEL_SLOT_EMA        = 0.15
PANEL_SLOT_MAX_JUMP   = 0.25
KEN_BURNS_MAX_ZOOM    = 1.04
KEN_BURNS_PERIOD      = 8.0
DISSOLVE_FRAMES       = 3
PANEL_DIVIDER_PX      = 3
PANEL_DIVIDER_COLOR   = (15, 15, 15)
PANEL_CROP_EXPAND     = 1.55
PANEL_TRANSITION_FRAMES = 6


# ─── Clip segment ──────────────────────────────────────────────────────────────

class ClipSegment:
    """Represents one detected highlight segment."""

    def __init__(
        self,
        start_sec: float,
        end_sec: float,
        score: float,
        soi_region: str = "center",
        peak_frame: int = 0,
        title: str = "",
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
        return os.path.exists("yolov8n.pt") or os.path.exists("yolov8s.pt")


# ─── Vignette (cached numpy mask) ─────────────────────────────────────────────

_vignette_cache: Dict[Tuple, np.ndarray] = {}


def _build_vignette(
    w: int,
    h: int,
    strength: float = VIGNETTE_STRENGTH,
    falloff: float = VIGNETTE_FALLOFF,
) -> np.ndarray:
    key = (w, h, round(strength, 3), round(falloff, 3))
    if key not in _vignette_cache:
        xs = np.linspace(-1, 1, w, dtype=np.float32)
        ys = np.linspace(-1, 1, h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        dist = np.sqrt(xg**2 + yg**2)
        dist /= dist.max()
        mask = np.clip(1.0 - strength * (dist**falloff), 0.0, 1.0)[:, :, np.newaxis]
        _vignette_cache[key] = mask
    return _vignette_cache[key]


def apply_vignette(frame: np.ndarray, strength: float = VIGNETTE_STRENGTH) -> np.ndarray:
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    mask = _build_vignette(w, h, strength)
    return (frame.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)


# ─── Unsharp mask ──────────────────────────────────────────────────────────────

def apply_sharpen(frame: np.ndarray, strength: float = 0.6, radius: int = 1) -> np.ndarray:
    if strength <= 0:
        return frame
    ksize = radius * 2 + 1
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
        r = _sc(x * 1.04)
        g = _sc(x * 1.02)
        b = _sc(x)
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
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    max_zoom: float = KEN_BURNS_MAX_ZOOM,
    period: float = KEN_BURNS_PERIOD,
) -> np.ndarray:
    if max_zoom <= 1.0:
        return frame
    t = (frame_idx / max(fps, 1)) % period
    scale = 1.0 + (max_zoom - 1.0) * 0.5 * (1 - math.cos(2 * math.pi * t / period))
    if abs(scale - 1.0) < 1e-4:
        return frame
    h, w = frame.shape[:2]
    nw = max(int(w / scale), 2)
    nh = max(int(h / scale), 2)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    return cv2.resize(frame[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)


# ─── Cross-dissolve on scene cuts ──────────────────────────────────────────────

class DissolveBuffer:
    """Alpha-blends N frames at scene cuts to mask hard transitions."""

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
        alpha = self._rem / self.n
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
    """Software-decode pipe reader; tries libdav1d first, falls back to default."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        seek_sec: float = 0.0,
        n_frames: Optional[int] = None,
        scale_w: Optional[int] = None,
        scale_h: Optional[int] = None,
    ) -> None:
        self.path        = path
        self.width       = width
        self.height      = height
        self.seek_sec    = seek_sec
        self.n_frames    = n_frames
        self.out_w       = scale_w or width
        self.out_h       = scale_h or height
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover    = b""

    def _build_cmd(self, extra_decoder_flags: List[str]) -> List[str]:
        """Build an ffmpeg raw-pipe command with optional decoder flags."""
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
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=max(self._frame_bytes * 4, 1 << 20),
                )
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc    = proc
                    self._leftover = test
                    return
                # Command produced no valid data — clean up and try next
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
    path: str,
    width: int,
    height: int,
    t_sec: float,
    scale_w: Optional[int] = None,
    scale_h: Optional[int] = None,
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
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: Optional[str],
    crf: int = 23,
    preset: str = "fast",
    audio_bitrate: str = "128k",
    subtitle_path: Optional[str] = None,
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
        s = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
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

    # BUG-4 fix: -aspect must come before codec/profile flags
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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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


# ─── Resolution helpers ────────────────────────────────────────────────────────

def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        # Match source: derive 9:16 from original height
        cw = int(orig_h * 9 / 16)
        if cw > orig_w:
            cw = orig_w
        ch = int(cw * 16 / 9)
    else:
        cw, ch = tw, th

    # Never upscale
    if ch > orig_h:
        scale = orig_h / ch
        cw = int(cw * scale)
        ch = int(orig_h)
    if cw > orig_w:
        scale = orig_w / cw
        cw = int(orig_w)
        ch = int(ch * scale)

    return max(cw - (cw % 2), 2), max(ch - (ch % 2), 2)


def calculate_crop_dims(orig_w: int, orig_h: int, tw: int, th: int) -> Tuple[int, int]:
    th = max(th, 2)
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


def detect_faces(frame: np.ndarray, confidence_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
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


# ─── Subject / person detection ────────────────────────────────────────────────

DetectionResult = namedtuple("DetectionResult", ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"])


def detect_subjects(
    frame: np.ndarray,
    model: Any,
    confidence: float = 0.45,
) -> Optional[DetectionResult]:
    if model is None:
        return None
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"det err: {e}", file=sys.stderr)
        return None
    if results.boxes is None or len(results.boxes) == 0:
        return None

    pp, hp, ap = [], [], []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w_  = (x2 - x1) * conf
        entry = (w_, x1, y1, x2, y2)
        if cls == PERSON_CLASS_ID:
            pp.append(entry)
        elif cls in HIGH_PRIO_CLASSES:
            hp.append(entry)
        ap.append(entry)

    pool = pp or hp or ap
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


def detect_persons_all(
    frame: np.ndarray,
    model: Any,
    confidence: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    if model is None:
        return []
    try:
        results = model(frame, verbose=False, conf=confidence)[0]
    except Exception:
        return []
    if results.boxes is None or len(results.boxes) == 0:
        return []
    persons = []
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            persons.append((x1, y1, x2, y2))
    persons.sort(key=lambda b: b[0])
    return persons


# ─── Framing helpers ───────────────────────────────────────────────────────────

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
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
) -> Tuple[int, int]:
    ucx = (ux1 + ux2) // 2
    ucy = (uy1 + uy2) // 2
    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(ucx, orig_w - hw))
    cy = max(hh, min(ucy, orig_h - hh))
    cy = _apply_lower_third_guard(cy, crop_h, ucy, orig_h)
    return cx, max(hh, min(cy, orig_h - hh))


def talking_head_center(
    faces: List[Tuple[int, int, int, int]],
    orig_w: int, orig_h: int,
    crop_w: int, crop_h: int,
    bias: float = 0.30,
) -> Optional[Tuple[int, int]]:
    if not faces:
        return None
    ux1 = min(f[0] for f in faces)
    uy1 = min(f[1] for f in faces)
    ux2 = max(f[2] for f in faces)
    uy2 = max(f[3] for f in faces)
    face_cx = (ux1 + ux2) // 2
    face_cy = (uy1 + uy2) // 2
    cy = int(face_cy * (1 - bias) + (face_cy + crop_h // 6) * bias)
    hw, hh = crop_w // 2, crop_h // 2
    cx = max(hw, min(face_cx, orig_w - hw))
    cy = max(hh, min(cy, orig_h - hh))
    cy = _apply_lower_third_guard(cy, crop_h, face_cy, orig_h)
    return cx, max(hh, min(cy, orig_h - hh))


# ─── PANEL-1+5: Robust panel detection ────────────────────────────────────────

def _detect_panel_mode(
    input_path: str,
    model: Any,
    fps: float,
    total_frames: int,
    orig_w: int,
    orig_h: int,
    confidence: float = 0.45,
    n_probe: int = PANEL_PROBE_COUNT,
) -> bool:
    """
    Return True only for genuine podcast / news-desk / interview layouts.

    Five independent signals must ALL pass:
      (a) ≥ PANEL_MIN_PERSONS persons in frame
      (b) Stable L/R split across probes (persons stay in their column)
      (c) Low inter-frame person motion (sports players run; panel guests sit)
      (d) Mean person bbox area ≥ PANEL_MIN_PERSON_AREA_FRAC of frame area
          (close-up guests fill the frame; sports players are small)
      (e) Person count variance ≤ PANEL_MAX_COUNT_VARIANCE across probes

    Basketball / football fails (b) because players constantly cross sides,
    (c) because players move fast, (d) because players are small relative to
    the frame, and (e) because player counts change every probe.
    """
    if model is None:
        return False

    det_w = min(orig_w, 640)
    det_h = max(1, int(det_w * orig_h / orig_w))
    frame_area = det_w * det_h

    probe_ts = np.linspace(1.0, max(1.5, total_frames / fps - 1.0), n_probe)

    multi_hits       = 0
    stable_split_hits = 0
    motion_vals: List[float] = []
    area_vals:   List[float] = []
    count_vals:  List[int]   = []
    prev_split: Optional[Dict[str, List[float]]] = None

    for t in probe_ts:
        frame = _read_frame_at(input_path, orig_w, orig_h, t, scale_w=det_w, scale_h=det_h)
        if frame is None:
            continue

        persons = detect_persons_all(frame, model, confidence)
        count_vals.append(len(persons))

        if len(persons) < PANEL_MIN_PERSONS:
            prev_split = None
            continue
        multi_hits += 1

        # (d) average bbox area relative to frame
        areas = [(p[2] - p[0]) * (p[3] - p[1]) for p in persons]
        mean_area_frac = float(np.mean(areas)) / frame_area
        area_vals.append(mean_area_frac)

        # Normalised X centres
        centres_x = [(p[0] + p[2]) / 2 / det_w for p in persons]
        left_x  = [c for c in centres_x if c < 0.40]
        right_x = [c for c in centres_x if c > 0.60]

        if left_x and right_x:
            if prev_split is not None:
                # (b) stability: did the column positions shift?
                shift_l = abs(np.mean(left_x)  - np.mean(prev_split["left"]))  if prev_split["left"]  else 0.0
                shift_r = abs(np.mean(right_x) - np.mean(prev_split["right"])) if prev_split["right"] else 0.0

                # (c) motion proxy: normalised movement of column centres
                motion = (shift_l + shift_r) / 2 * det_w  # convert to px
                motion_vals.append(motion)

                if shift_l <= 0.12 and shift_r <= 0.12:
                    stable_split_hits += 1

            prev_split = {"left": left_x, "right": right_x}
        else:
            prev_split = None

    if multi_hits == 0:
        return False

    majority_threshold = n_probe * PANEL_MAJORITY_FRAC

    # (a) + (b): enough frames with stable L/R split
    cond_ab = multi_hits > majority_threshold and stable_split_hits > majority_threshold

    # (c) low motion — mean inter-frame column movement in pixels
    mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0
    cond_c = mean_motion < PANEL_MAX_PERSON_MOTION

    # (d) persons are large in frame (close-up / seated)
    mean_area = float(np.mean(area_vals)) if area_vals else 0.0
    cond_d = mean_area >= PANEL_MIN_PERSON_AREA_FRAC

    # (e) count is stable (not fluctuating like sports)
    count_std = float(np.std(count_vals)) if len(count_vals) > 1 else 0.0
    cond_e = count_std <= PANEL_MAX_COUNT_VARIANCE

    return cond_ab and cond_c and cond_d and cond_e


# ─── Panel slot smoother ───────────────────────────────────────────────────────

class PanelSlotSmoother:
    """PANEL-2: Slower EMA + per-frame displacement clamp for stable strips."""

    def __init__(
        self,
        alpha: float = PANEL_SLOT_EMA,
        max_jump_frac: float = PANEL_SLOT_MAX_JUMP,
    ) -> None:
        self.alpha         = alpha
        self.max_jump_frac = max_jump_frac
        # Internal EMA state: last smoothed bounding box per slot
        self._slots: List[Optional[Tuple[float, ...]]] = [None, None]

    def _ema_box(
        self,
        prev: Optional[Tuple[float, ...]],
        new_box: Tuple[int, int, int, int],
        axis_size: float,
    ) -> Tuple[float, ...]:
        if prev is None:
            return tuple(float(v) for v in new_box)
        a        = self.alpha
        max_jump = axis_size * self.max_jump_frac
        smoothed = tuple(prev[i] * (1 - a) + new_box[i] * a for i in range(4))
        clamped  = tuple(
            float(np.clip(smoothed[i], prev[i] - max_jump, prev[i] + max_jump))
            for i in range(4)
        )
        return clamped

    def _smooth_slot(
        self,
        slot_idx: int,
        group: List[Tuple[int, int, int, int]],
        strip_w: float,
    ) -> List[Tuple[int, int, int, int]]:
        if not group:
            # BUG-2 fix: return held EMA box when group is absent this frame
            held = self._slots[slot_idx]
            if held is not None:
                return [tuple(int(v) for v in held)]
            return []
        union  = _group_union(group)
        smooth = self._ema_box(self._slots[slot_idx], union, strip_w)
        self._slots[slot_idx] = smooth
        return [tuple(int(v) for v in smooth)]

    def update(
        self,
        group_a: List[Tuple[int, int, int, int]],
        group_b: List[Tuple[int, int, int, int]],
        strip_w: float,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        return (
            self._smooth_slot(0, group_a, strip_w),
            self._smooth_slot(1, group_b, strip_w),
        )


def _group_union(
    persons: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    return (
        min(p[0] for p in persons), min(p[1] for p in persons),
        max(p[2] for p in persons), max(p[3] for p in persons),
    )


def _crop_group_to_strip(
    frame: np.ndarray,
    group: List[Tuple[int, int, int, int]],
    strip_w: int,
    strip_h: int,
    expand: float = PANEL_CROP_EXPAND,
    vignette_strength: float = 0.0,
    color_grade: str = "none",
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    if not group:
        crop = frame
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        ucx     = (ux1 + ux2) // 2
        ucy     = (uy1 + uy2) // 2
        union_w = max(ux2 - ux1, 1)
        strip_r = strip_w / strip_h
        crop_w  = int(union_w * expand)
        crop_h  = int(crop_w / strip_r)
        if crop_h > fh:
            crop_h = fh
            crop_w = int(crop_h * strip_r)
        if crop_w > fw:
            crop_w = fw
            crop_h = int(crop_w / strip_r)
        crop_w = max(crop_w, 2)
        crop_h = max(crop_h, 2)
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
    frame: np.ndarray,
    persons: List[Tuple[int, int, int, int]],
    out_w: int,
    out_h: int,
    prev_slots: Optional[List[List[Tuple[int, int, int, int]]]],
    vignette_strength: float = VIGNETTE_STRENGTH * 0.7,
    color_grade: str = "none",
    slot_smoother: Optional[PanelSlotSmoother] = None,
) -> Tuple[np.ndarray, List[List[Tuple[int, int, int, int]]]]:
    """PANEL-3/4: render with EMA-smoothed slots; divider drawn after blend."""
    persons = sorted(persons, key=lambda b: (b[0] + b[2]) // 2)
    n = len(persons)

    if n == 0:
        # Hold previous slot contents
        group_a = (prev_slots[0] if prev_slots and prev_slots[0] else [])
        group_b = (prev_slots[1] if prev_slots and len(prev_slots) > 1 and prev_slots[1] else [])
    elif n == 1:
        group_a = persons
        group_b = (prev_slots[1] if prev_slots and len(prev_slots) > 1 and prev_slots[1] else persons)
    else:
        split   = max(1, n // 2)
        group_a = persons[:split]
        group_b = persons[split:]

    if slot_smoother is not None:
        group_a, group_b = slot_smoother.update(group_a, group_b, strip_w=float(out_w))

    strip_h_a = (out_h // 2) & ~1
    strip_h_b = out_h - strip_h_a

    top = _crop_group_to_strip(
        frame, group_a, out_w, strip_h_a,
        vignette_strength=vignette_strength, color_grade=color_grade,
    )
    bot = _crop_group_to_strip(
        frame, group_b, out_w, strip_h_b,
        vignette_strength=vignette_strength, color_grade=color_grade,
    )

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[0:strip_h_a, :]               = top
    canvas[strip_h_a:strip_h_a + strip_h_b, :] = bot

    # PANEL-4: divider drawn last so it stays sharp
    dy1 = max(0, strip_h_a - PANEL_DIVIDER_PX // 2)
    dy2 = min(out_h, strip_h_a + (PANEL_DIVIDER_PX + 1) // 2)
    canvas[dy1:dy2, :] = PANEL_DIVIDER_COLOR

    return canvas, [list(group_a), list(group_b)]


# ─── Optical flow / saliency ───────────────────────────────────────────────────

def optical_flow_center(
    prev: np.ndarray,
    curr: np.ndarray,
    w: int,
    h: int,
) -> Optional[Tuple[int, int]]:
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
    t = sal.sum()
    if t < 1e-6:
        return w // 2, h // 2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum() / t), int((ys * sal).sum() / t)


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


# ─── Camera-path smoothing ─────────────────────────────────────────────────────

def _vel_to_window(speed: float) -> int:
    t = VELOCITY_SMOOTH_TABLE
    if speed <= t[0][0]:
        return t[0][1]
    if speed >= t[-1][0]:
        return t[-1][1]
    for i in range(len(t) - 1):
        v0, w0 = t[i]
        v1, w1 = t[i + 1]
        if v0 <= speed <= v1:
            frac = (speed - v0) / (v1 - v0 + 1e-9)
            w    = int(w0 + frac * (w1 - w0))
            return w if w % 2 == 1 else w + 1
    return 33


def _gauss_seg(
    xs: np.ndarray, ys: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    """Bidirectional EMA for zero phase-lag smoothing."""
    n = len(xs)
    if n < 2:
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def _fwd(v: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float)
        out[0] = v[0]
        for i in range(1, n):
            out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
        return out

    def _bwd(v: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float)
        out[-1] = v[-1]
        for i in range(n - 2, -1, -1):
            out[i] = alpha * v[i] + (1 - alpha) * out[i + 1]
        return out

    rx = (_fwd(xs) + _bwd(xs)) / 2
    ry = (_fwd(ys) + _bwd(ys)) / 2
    return rx, ry


def smooth_centers(
    centers: List[Tuple[int, int]],
    speeds: List[float],
    base_window: int = 33,
    adaptive: bool = True,
    scene_cuts: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """
    Three-stage smoothing pipeline per scene segment:
      1. Velocity-adaptive Gaussian
      2. Bidirectional EMA

    Parameters
    ----------
    centers     : list of (cx, cy) detection centres (one per sample)
    speeds      : list of camera speeds (px/frame), same length as centers
    base_window : fallback Gaussian window when adaptive=False
    adaptive    : scale window by local camera speed
    scene_cuts  : sample indices (NOT frame indices) where cuts occur

    Returns (smoothed_centers, metrics_dict)
    """
    empty_metrics: Dict[str, float] = {
        "jitter_raw": 0.0, "jitter_smooth": 0.0,
        "smoothness_pct": 0.0, "max_jump_raw": 0.0,
    }
    if not centers or len(centers) < 3:
        return list(centers) if centers else [], empty_metrics

    n   = len(centers)
    xs  = np.array([c[0] for c in centers], dtype=float)
    ys  = np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n:
        spd = np.pad(spd, (0, n - len(spd)), mode="edge")

    # Raw jitter metrics (before smoothing)
    dx_raw     = np.diff(xs)
    dy_raw     = np.diff(ys)
    dist_raw   = np.sqrt(dx_raw**2 + dy_raw**2)
    jitter_raw = float(np.mean(dist_raw)) if len(dist_raw) > 0 else 0.0
    max_jump   = float(np.max(dist_raw))  if len(dist_raw) > 0 else 0.0

    # Clamp scene-cut indices to valid sample range
    cuts = sorted({c for c in (scene_cuts or []) if 0 < c < n})
    bounds = [0] + cuts + [n]

    rx, ry = xs.copy(), ys.copy()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 3:
            continue
        w       = max(_vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window, 13)
        gx, gy  = _gauss_seg(xs[s:e], ys[s:e], w)
        bx, by  = _bidir_ema(gx, gy, alpha=0.08)
        rx[s:e] = bx
        ry[s:e] = by

    smoothed = [(int(x), int(y)) for x, y in zip(rx, ry)]

    dx_s         = np.diff(rx)
    dy_s         = np.diff(ry)
    jitter_smooth = float(np.mean(np.sqrt(dx_s**2 + dy_s**2)))
    smoothness   = (jitter_raw - jitter_smooth) / jitter_raw * 100 if jitter_raw > 0 else 0.0

    metrics: Dict[str, float] = {
        "jitter_raw":     round(jitter_raw, 2),
        "jitter_smooth":  round(jitter_smooth, 2),
        "smoothness_pct": round(smoothness, 1),
        "max_jump_raw":   round(max_jump, 1),
    }
    return smoothed, metrics


# ─── Whisper / translate ───────────────────────────────────────────────────────

def _seconds_to_srt_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sc  = int(s % 60)
    ms  = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"


def transcribe_to_srt(
    video_path: str,
    srt_path: str,
    whisper_model: str = "base",
    language: Optional[str] = None,
    max_chars_per_line: int = 42,
    progress_callback=None,
) -> bool:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass

    if not whisper_available():
        return False
    import whisper as _w

    _p(0.0, "Extracting audio…")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        if not _extract_audio_wav(video_path, wav_path):
            return False
        _p(0.2, f"Transcribing ({whisper_model})…")
        model  = _w.load_model(whisper_model)
        opts: Dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if language:
            opts["language"] = language
        result = model.transcribe(wav_path, **opts)

        _p(0.85, "Writing subtitles…")
        lines: List[str] = []
        idx   = 1
        words: List[Dict[str, Any]] = []
        for seg in result.get("segments", []):
            for w_ in seg.get("words", []):
                words.append({"word": w_["word"].strip(), "start": w_["start"], "end": w_["end"]})

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
            idx    += 1
            buf     = []
            buf_len = 0

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
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def translate_srt(
    srt_path: str,
    target_language: str,
    source_language: str = "auto",
    progress_callback=None,
) -> bool:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass

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


# ─── Clip detection ────────────────────────────────────────────────────────────

def _frame_saliency_score(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_score = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)
    motion    = 0.0
    if prev_frame is not None:
        motion = min(
            float(cv2.absdiff(gray, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)).mean()) / 30.0,
            1.0,
        )
    sat = min(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 128.0, 1.0)
    return 0.4 * motion + 0.4 * lap_score + 0.2 * sat


def _compute_frame_scores(
    input_path: str,
    fps: float,
    total_frames: int,
    orig_w: int,
    orig_h: int,
    sample_every: int = 15,
    progress_callback=None,
) -> Tuple[np.ndarray, List[int]]:
    def _p(v: float, msg: str = "") -> None:
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass

    scores: List[float]  = []
    scene_cuts: List[int] = []
    prev_gray = prev_frame = None
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
                prev_gray  = cg
                prev_frame = frame.copy()
            if fi % report_n == 0:
                _p(fi / total_frames, f"Scanning {fi}/{total_frames}…")
            fi += 1

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
            try:
                progress_callback(v, msg)
            except Exception:
                pass

    info         = get_video_info(input_path)
    fps          = info["fps"]
    total_frames = info["total_frames"]
    duration     = info["duration_seconds"]
    orig_w       = info["width"]
    orig_h       = info["height"]
    sample_every = max(1, int(fps))

    _p(0.0, "Scanning…")
    scores, scene_cut_frames = _compute_frame_scores(
        input_path, fps, total_frames, orig_w, orig_h,
        sample_every=sample_every,
        progress_callback=lambda v, m: _p(v * 0.45, m),
    )
    if len(scores) == 0:
        return []

    _p(0.45, "Computing arcs…")
    window = max(5, int(30 / (sample_every / fps)))
    ss     = (
        np.convolve(scores, np.ones(window) / window, mode="same")
        if len(scores) >= window else scores.copy()
    )
    if ss.max() > 0:
        ss /= ss.max()

    min_gap = max(1, int(min_duration_sec * fps / sample_every))
    peaks: List[int] = []
    for i in range(1, len(ss) - 1):
        wh   = min_gap // 2
        lo   = max(0, i - wh)
        hi   = min(len(ss), i + wh + 1)
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
                rs = max(0.0, sc_s - 1.0)
                break
        for sc in scene_cut_frames:
            sc_s = sc / fps
            if 0 < sc_s - ps < 15.0:
                re = min(duration, sc_s + 0.5)
                break
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
        s, e = _arc(pi)
        sc_  = float(ss[pi])
        if not any(min(e, ce) - max(s, cs) > min_duration_sec * 0.5 for cs, ce, _ in cands):
            cands.append((s, e, sc_))
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:target_n_clips]
    cands.sort(key=lambda x: x[0])

    _p(0.55, "SOI per clip…")
    segments: List[ClipSegment] = []
    det_w = min(orig_w, 640)
    det_h = max(1, int(det_w * orig_h / orig_w))

    for ci, (ss2, se, score) in enumerate(cands):
        _p(0.55 + 0.35 * (ci / max(len(cands), 1)), f"Clip {ci+1}/{len(cands)}…")
        soi_xs: List[int] = []
        soi_ys: List[int] = []
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
                            soi_xs.append((x1 + x2) // 2)
                            soi_ys.append((y1 + y2) // 2)
                except Exception:
                    pass
            else:
                scx, scy = saliency_center(frame)
                soi_xs.append(scx)
                soi_ys.append(scy)

        sr   = "center"
        if soi_xs:
            sr = _soi_region_label(
                int(np.median(soi_xs)), int(np.median(soi_ys)), orig_w, orig_h
            )
        ms   = int(ss2 // 60)
        secs = int(ss2 % 60)
        me   = int(se  // 60)
        sece = int(se  % 60)
        segments.append(ClipSegment(
            start_sec=ss2, end_sec=se, score=score, soi_region=sr,
            peak_frame=int(np.linspace(ss2 + 1, se - 1, n_s)[n_s // 2] * fps),
            title=f"Clip {ci+1} ({ms}:{secs:02d} – {me}:{sece:02d})",
        ))

    _p(1.0, f"Found {len(segments)} clips")
    return segments


# ─── Analytics ────────────────────────────────────────────────────────────────

def get_analytics_meta(
    input_path: str,
    output_path: str,
    *,
    tracking_mode: str = "",
    panel_mode: bool = False,
    scene_cuts_count: int = 0,
    smooth_metrics: Optional[Dict[str, float]] = None,
    color_grade: str = "none",
    subtitle_burned: bool = False,
    subtitle_language: str = "",
    vignette_strength: float = VIGNETTE_STRENGTH,
    crf: int = 23,
    encoder_preset: str = "fast",
) -> Dict[str, Any]:
    """
    Return a dict with comprehensive metrics about the conversion,
    ready to feed the companion analytics player widget.

    All values are JSON-serialisable.
    """
    def _safe_info(p: str) -> Dict[str, Any]:
        try:
            return get_video_info(p)
        except Exception:
            return {}

    def _size_mb(p: str) -> float:
        try:
            return os.path.getsize(p) / 1024**2
        except Exception:
            return 0.0

    def _bitrate_kbps(mb: float, dur: float) -> int:
        if dur and dur > 0:
            return round(mb * 8 * 1024 / dur)
        return 0

    in_info  = _safe_info(input_path)
    out_info = _safe_info(output_path)
    in_mb    = _size_mb(input_path)
    out_mb   = _size_mb(output_path)
    in_dur   = in_info.get("duration_seconds", 0.0)
    out_dur  = out_info.get("duration_seconds", 0.0)

    meta: Dict[str, Any] = {
        # File properties
        "input_path":               input_path,
        "output_path":              output_path,
        "input_size_mb":            round(in_mb, 2),
        "output_size_mb":           round(out_mb, 2),
        "compression_ratio":        round(in_mb / out_mb, 2) if out_mb else 0,
        "file_size_reduction_pct":  round((1 - out_mb / in_mb) * 100, 1) if in_mb else 0.0,
        # Duration
        "input_duration_s":         round(in_dur, 2),
        "output_duration_s":        round(out_dur, 2),
        # Resolution
        "input_resolution":         f"{in_info.get('width', 0)}x{in_info.get('height', 0)}",
        "output_resolution":        f"{out_info.get('width', 0)}x{out_info.get('height', 0)}",
        # Frame rate
        "input_fps":                round(in_info.get("fps", 0.0), 2),
        "output_fps":               round(out_info.get("fps", 0.0), 2),
        # Bitrate
        "input_bitrate_kbps":       _bitrate_kbps(in_mb,  in_dur),
        "output_bitrate_kbps":      _bitrate_kbps(out_mb, out_dur),
        # Audio
        "has_audio":                _has_audio(input_path),
        # Processing parameters
        "tracking_mode":            tracking_mode,
        "panel_mode":               panel_mode,
        "scene_cuts_count":         scene_cuts_count,
        "color_grade":              color_grade,
        "vignette_strength":        round(vignette_strength, 3),
        "crf":                      crf,
        "encoder_preset":           encoder_preset,
        # Subtitle info
        "subtitle_burned":          subtitle_burned,
        "subtitle_language":        subtitle_language,
        # Smoothing metrics (populated from smooth_centers if available)
        "jitter_raw":               0.0,
        "jitter_smooth":            0.0,
        "smoothness_pct":           0.0,
        "max_jump_raw":             0.0,
    }

    if smooth_metrics:
        meta.update({
            "jitter_raw":     smooth_metrics.get("jitter_raw",    0.0),
            "jitter_smooth":  smooth_metrics.get("jitter_smooth", 0.0),
            "smoothness_pct": smooth_metrics.get("smoothness_pct", 0.0),
            "max_jump_raw":   smooth_metrics.get("max_jump_raw",  0.0),
        })

    return meta


# ─── process_video — main entry point ─────────────────────────────────────────

def process_video(
    input_path: str,
    output_path: str,
    target_preset_label: str = "Match source (no upscale)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    sample_interval: Optional[int] = None,
    confidence: float = 0.45,
    use_optical_flow: bool = True,
    smooth_window: int = 33,
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
    vignette_strength: float = VIGNETTE_STRENGTH,
    sharpen_strength: float = 0.0,
    color_grade: str = "none",
    ken_burns: bool = False,
    dissolve_cuts: bool = True,
    ffmpeg_sharpen: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Convert a landscape video to vertical (9:16) with smart subject tracking.

    Returns a dict with keys:
      output_path, subtitle_path, clamped, effective_size, duration,
      panel_mode, analytics (see get_analytics_meta).
    """
    def _p(v: float, msg: str = " ") -> None:
        if progress_callback:
            try:
                progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception:
                pass

    result_meta: Dict[str, Any] = {
        "output_path":    output_path,
        "subtitle_path":  None,
        "clamped":        False,
        "effective_size": (0, 0),
        "duration":       0.0,
        "panel_mode":     False,
    }

    _check_ffmpeg()
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path) / 1024**2 > MAX_FILE_SIZE_MB:
        raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB.")

    info         = get_video_info(input_path)
    fps          = info["fps"]
    total_frames = info["total_frames"]
    orig_w       = info["width"]
    orig_h       = info["height"]
    duration     = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ProcessingError("Corrupt or unreadable video.")
    if not info["is_landscape"]:
        raise ProcessingError("Video is already vertical.")

    lbl      = target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    req_w, req_h       = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped            = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update(clamped=clamped, effective_size=(target_w, target_h), duration=duration)
    _p(0.01, f"Output {target_w}x{target_h} ← source {orig_w}x{orig_h}")

    if not sample_interval:
        sample_interval = max(1, int(fps / 5))
    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)

    det_scale = min(1.0, 640 / orig_w)
    det_w     = max(1, int(orig_w * det_scale))
    det_h     = max(1, int(orig_h * det_scale))
    sx        = orig_w / det_w
    sy        = orig_h / det_h

    # ── Subtitles ──────────────────────────────────────────────────────────────
    srt_path: Optional[str] = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "Transcribing…")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt")
        os.close(srt_fd)
        ok = transcribe_to_srt(
            input_path, srt_path,
            whisper_model=whisper_model, language=whisper_language,
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
                    srt_path, target_language=subtitle_translate_to,
                    progress_callback=lambda v, m: _p(0.10 + v * 0.05, m),
                )
            result_meta["subtitle_path"] = srt_path

    # ── Load model ─────────────────────────────────────────────────────────────
    start_pct  = 0.10
    model_obj  = None
    if tracking_mode == "subject":
        _p(start_pct, "Loading YOLO…")
        model_obj = _get_model(yolo_weights)
        if model_obj is None:
            _p(start_pct, "YOLO unavailable – saliency fallback")
    elif tracking_mode == "talking_head":
        _p(start_pct, "Loading face detector…")
        if _get_haar() is None:
            _p(start_pct, "No face detector – saliency fallback")

    # ── Panel detection ────────────────────────────────────────────────────────
    is_panel     = False
    slot_smoother: Optional[PanelSlotSmoother] = None
    if tracking_mode == "subject" and model_obj is not None:
        _p(start_pct + 0.01, "Checking panel/group-shot…")
        is_panel = _detect_panel_mode(
            input_path, model_obj, fps, total_frames, orig_w, orig_h,
            confidence, n_probe=PANEL_PROBE_COUNT,
        )
        if is_panel:
            _p(start_pct + 0.02, "Panel mode – 2-row vertical split")
            result_meta["panel_mode"] = True
            slot_smoother = PanelSlotSmoother()

    # ── Open encoder ───────────────────────────────────────────────────────────
    extra_vf = _build_ffmpeg_vf(color_grade="none", ffmpeg_sharpen=ffmpeg_sharpen) or None
    style    = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])

    _p(0.12, f"Single-pass detect+render ({total_frames} frames)…")
    proc = _open_ffmpeg_encoder(
        output_path, target_w, target_h, render_fps,
        audio_source=input_path,
        crf=crf, preset=encoder_preset, audio_bitrate=audio_bitrate,
        subtitle_path=srt_path, subtitle_style=style,
        extra_vf=extra_vf,
    )

    # Pre-build cached resources
    if vignette_strength > 0:
        _build_vignette(target_w, target_h, vignette_strength)
    if color_grade and color_grade != "none":
        _build_lut(color_grade)

    dissolve_buf = DissolveBuffer(DISSOLVE_FRAMES) if dissolve_cuts else None
    smooth_metrics: Dict[str, float] = {}
    scene_cuts: List[int] = []       # frame indices of detected scene cuts
    last_out_frame: Optional[np.ndarray] = None
    rpt_n = max(1, total_frames // 40)

    # ══════════════════════════════════════════════════════════════════════════
    # Non-panel path: two-pass (detect → smooth → render)
    # ══════════════════════════════════════════════════════════════════════════
    if not is_panel:
        _p(0.12, "Pass 1/2: detecting subjects…")

        det_centers_raw: List[Tuple[int, int]] = []
        det_frame_indices: List[int]           = []   # frame index for each sample
        frame_speeds: List[float]              = []
        prev_c:     Optional[Tuple[int, int]]  = None
        prev_gray2: Optional[np.ndarray]       = None
        prev_flow2: Optional[np.ndarray]       = None
        last_det2:  Optional[Tuple[int, int]]  = None
        det_dropout2 = 0
        fi2 = 0

        with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=det_w, scale_h=det_h) as reader:
            for det_frame in reader:
                if fi2 >= total_frames:
                    break
                if fi2 % sample_interval == 0:
                    cg  = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
                    cut = is_scene_change(prev_gray2, cg, scene_cut_threshold)
                    if cut:
                        scene_cuts.append(fi2)
                        prev_flow2   = None
                        det_dropout2 = 0
                    prev_gray2   = cg
                    anchor_cx = anchor_cy = None

                    if tracking_mode == "talking_head":
                        faces = detect_faces(det_frame, confidence_thresh=0.5)
                        if faces:
                            faces_orig = [
                                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                                for x1, y1, x2, y2 in faces
                            ]
                            r = talking_head_center(
                                faces_orig, orig_w, orig_h, crop_w, crop_h, talking_head_bias
                            )
                            if r:
                                anchor_cx, anchor_cy = r
                                det_dropout2 = 0
                        if anchor_cx is None and use_optical_flow:
                            sm = cv2.resize(cg, (max(1, det_w // 2), max(1, det_h // 2)))
                            if prev_flow2 is not None:
                                fc = optical_flow_center(prev_flow2, sm, det_w // 2, det_h // 2)
                                if fc:
                                    anchor_cx, anchor_cy = int(fc[0] * 2 * sx), int(fc[1] * 2 * sy)
                            prev_flow2 = sm
                            det_dropout2 += sample_interval
                    else:
                        if model_obj is not None:
                            det = detect_subjects(det_frame, model_obj, confidence)
                            if det is not None:
                                anchor_cx, anchor_cy = frame_for_union(
                                    int(det.ux1 * sx), int(det.uy1 * sy),
                                    int(det.ux2 * sx), int(det.uy2 * sy),
                                    orig_w, orig_h, crop_w, crop_h,
                                )
                                last_det2    = (anchor_cx, anchor_cy)
                                det_dropout2 = 0
                        if anchor_cx is None and use_optical_flow:
                            sm = cv2.resize(cg, (max(1, det_w // 2), max(1, det_h // 2)))
                            if prev_flow2 is not None:
                                fc = optical_flow_center(prev_flow2, sm, det_w // 2, det_h // 2)
                                if fc:
                                    anchor_cx, anchor_cy = int(fc[0] * 2 * sx), int(fc[1] * 2 * sy)
                            prev_flow2 = sm
                            det_dropout2 += sample_interval
                        if anchor_cx is None:
                            max_dropout = int(fps * 1.5)
                            if last_det2 and det_dropout2 < max_dropout:
                                anchor_cx, anchor_cy = last_det2
                            else:
                                sc_ = saliency_center(det_frame)
                                anchor_cx, anchor_cy = int(sc_[0] * sx), int(sc_[1] * sy)

                    if anchor_cx is not None:
                        spd = 0.0
                        if prev_c is not None:
                            dx  = anchor_cx - prev_c[0]
                            dy  = anchor_cy - prev_c[1]
                            spd = math.sqrt(dx * dx + dy * dy) * render_fps / sample_interval
                        det_centers_raw.append((anchor_cx, anchor_cy))
                        det_frame_indices.append(fi2)
                        frame_speeds.append(spd)
                        prev_c = (anchor_cx, anchor_cy)

                if fi2 % rpt_n == 0:
                    _p(0.12 + 0.30 * (fi2 / total_frames), f"Det {fi2}/{total_frames}…")
                fi2 += 1

        # ── Smooth the full detection path ─────────────────────────────────────
        _p(0.42, "Smoothing camera path…")
        dense_cx = np.full(total_frames, orig_w // 2, dtype=float)
        dense_cy = np.full(total_frames, orig_h // 2, dtype=float)

        if det_centers_raw:
            # BUG-1 fix: convert frame-index scene cuts → detection-sample indices
            # det_frame_indices maps sample k → frame index fi2
            # We need cuts expressed as sample indices so smooth_centers segments correctly
            frame_to_sample = {fi_k: k for k, fi_k in enumerate(det_frame_indices)}
            scene_cuts_sample: List[int] = []
            for cut_fi in scene_cuts:
                # Find the nearest sample to this cut frame
                nearest_sample = min(
                    range(len(det_frame_indices)),
                    key=lambda k: abs(det_frame_indices[k] - cut_fi),
                    default=None,
                )
                if nearest_sample is not None and 0 < nearest_sample < len(det_centers_raw):
                    scene_cuts_sample.append(nearest_sample)
            scene_cuts_sample = sorted(set(scene_cuts_sample))

            smoothed_det, smooth_metrics = smooth_centers(
                det_centers_raw, frame_speeds,
                base_window=smooth_window, adaptive=adaptive_smoothing,
                scene_cuts=scene_cuts_sample,
            )
            # Interpolate smoothed samples to a dense per-frame map
            known_frames = np.array(det_frame_indices, dtype=float)
            known_cx     = np.array([smoothed_det[k][0] for k in range(len(known_frames))])
            known_cy     = np.array([smoothed_det[k][1] for k in range(len(known_frames))])
            all_frames   = np.arange(total_frames, dtype=float)
            dense_cx     = np.interp(all_frames, known_frames, known_cx)
            dense_cy     = np.interp(all_frames, known_frames, known_cy)

        # ── Render pass ────────────────────────────────────────────────────────
        _p(0.44, "Pass 2/2: rendering…")
        hw = crop_w // 2
        hh = crop_h // 2
        scene_cut_set = set(scene_cuts)
        fi = 0

        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames:
                    break
                if fi in scene_cut_set and dissolve_buf and last_out_frame is not None:
                    dissolve_buf.on_cut(last_out_frame)

                cur_cx = int(np.clip(dense_cx[fi], hw, orig_w - hw))
                cur_cy = int(np.clip(dense_cy[fi], hh, orig_h - hh))

                left = max(0, min(cur_cx - hw, orig_w - crop_w))
                top  = max(0, min(cur_cy - hh, orig_h - crop_h))
                crop = frame[top:top + crop_h, left:left + crop_w]

                if crop.shape[1] != target_w or crop.shape[0] != target_h:
                    crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                out_frame = crop
                if ken_burns:
                    out_frame = apply_ken_burns(out_frame, fi, fps)
                if sharpen_strength > 0:
                    out_frame = apply_sharpen(out_frame, sharpen_strength)
                if color_grade and color_grade != "none":
                    out_frame = apply_color_grade(out_frame, color_grade)
                if vignette_strength > 0:
                    out_frame = apply_vignette(out_frame, vignette_strength)
                if dissolve_buf and dissolve_buf.active:
                    out_frame = dissolve_buf.blend(out_frame)

                last_out_frame = out_frame
                try:
                    proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    break

                fi += 1
                if fi % rpt_n == 0:
                    _p(0.44 + 0.44 * (fi / total_frames), f"Render {fi}/{total_frames}…")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel path: single-pass detect + render
    # ══════════════════════════════════════════════════════════════════════════
    else:
        _p(0.12, "Panel: single-pass detect+render…")
        prev_slots: Optional[List[List[Tuple[int, int, int, int]]]] = None
        fi = 0

        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames:
                    break
                is_sample = (fi % sample_interval == 0)
                if is_sample:
                    det_frame_p  = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                    persons_det  = detect_persons_all(det_frame_p, model_obj, confidence)
                    persons_full = [
                        (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                        for x1, y1, x2, y2 in persons_det
                    ]
                else:
                    # Hold last known slot boxes
                    persons_full = [b for grp in (prev_slots or []) if grp for b in grp]

                out_frame, prev_slots = _render_panel_frame(
                    frame, persons_full, target_w, target_h, prev_slots,
                    vignette_strength=vignette_strength * 0.7,
                    color_grade=color_grade,
                    slot_smoother=slot_smoother,
                )
                if dissolve_buf and dissolve_buf.active:
                    out_frame = dissolve_buf.blend(out_frame)

                last_out_frame = out_frame
                try:
                    proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    break

                fi += 1
                if fi % rpt_n == 0:
                    _p(0.12 + 0.75 * (fi / total_frames), f"{fi}/{total_frames}…")

    # ── Finalize ───────────────────────────────────────────────────────────────
    _p(0.88, "Encoding…")
    _close_ffmpeg_encoder(proc, output_path)

    analytics = get_analytics_meta(
        input_path, output_path,
        tracking_mode=tracking_mode,
        panel_mode=is_panel,
        scene_cuts_count=len(scene_cuts),
        smooth_metrics=smooth_metrics if smooth_metrics else None,
        color_grade=color_grade,
        subtitle_burned=burn_subtitles and srt_path is not None,
        subtitle_language=subtitle_translate_to or whisper_language or "",
        vignette_strength=vignette_strength,
        crf=crf,
        encoder_preset=encoder_preset,
    )
    result_meta["analytics"] = analytics

    _p(1.0, "Done!")
    print(
        f"Output: {output_path} ({os.path.getsize(output_path) / 1024**2:.1f} MB) "
        f"cuts={len(scene_cuts)} panel={is_panel}",
        file=sys.stderr,
    )
    return result_meta


# ─── Batch clip pipeline ───────────────────────────────────────────────────────

def process_clips_batch(
    input_path: str,
    output_dir: str,
    clips: List[ClipSegment],
    target_preset_label: str = "720p   (720x1280  - HD)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    confidence: float = 0.45,
    smooth_window: int = 33,
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
    vignette_strength: float = VIGNETTE_STRENGTH,
    sharpen_strength: float = 0.0,
    color_grade: str = "none",
    ken_burns: bool = False,
    dissolve_cuts: bool = True,
    ffmpeg_sharpen: bool = False,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Process multiple ClipSegments in sequence; returns one result dict per clip."""

    def _p(v: float, msg: str = " ") -> None:
        if progress_callback:
            try:
                progress_callback(v, msg)
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for i, clip in enumerate(clips):
        base_pct = i / max(len(clips), 1)
        next_pct = (i + 1) / max(len(clips), 1)
        _p(base_pct, f"Clip {i+1}/{len(clips)}…")

        trimmed_path: Optional[str] = None
        out_path: Optional[str]     = None
        try:
            fd, trimmed_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            if not _trim_video(input_path, trimmed_path, clip.start_sec, clip.end_sec):
                results.append({"clip": clip, "output_path": None, "error": "trim failed"})
                continue

            out_path = os.path.join(
                output_dir,
                f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s_vertical.mp4",
            )

            def clip_cb(v: float, msg: str = " ", _b: float = base_pct, _n: float = next_pct) -> None:
                _p(_b + v * (_n - _b), msg)

            meta = process_video(
                trimmed_path, out_path,
                target_preset_label=target_preset_label,
                tracking_mode=tracking_mode, talking_head_bias=talking_head_bias,
                confidence=confidence, smooth_window=smooth_window,
                adaptive_smoothing=adaptive_smoothing, use_optical_flow=use_optical_flow,
                rule_of_thirds=rule_of_thirds, crf=crf, encoder_preset=encoder_preset,
                audio_bitrate=audio_bitrate, yolo_weights=yolo_weights,
                burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                vignette_strength=vignette_strength, sharpen_strength=sharpen_strength,
                color_grade=color_grade, ken_burns=ken_burns, dissolve_cuts=dissolve_cuts,
                ffmpeg_sharpen=ffmpeg_sharpen, progress_callback=clip_cb,
            )
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
