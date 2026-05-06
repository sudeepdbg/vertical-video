"""
verticalize.py
──────────────
Convert landscape video to 9:16 vertical with AI subject tracking.

KEY FIXES:
  FIX-1  FFmpegVideoReader — decodes ALL codecs (AV1, HEVC, VP9 …) via
         FFmpeg software decode → raw BGR24 pipe, bypassing OpenCV's
         broken hardware-accelerated decoder path.
  FIX-2  Camera path: Cubic Hermite spline + Gaussian + EMA smoothing
         → eliminates micro-jitter and frame jumps.
  FIX-3  Panel Discussion mode — detects ≥2 people, renders 2×2 split-
         screen grid instead of a jumpy single-subject crop.

Modes:
  • Subject tracking    — YOLOv8 + optical flow
  • Talking Head Mode   — DNN/Haar face detector, upper-third framing
  • Panel Discussion    — Multi-person split-screen (2×2 grid) [auto]
  • Auto-clip detect    — scan long video, find high-engagement segments

Dependencies: opencv-python, ultralytics, numpy, ffmpeg (system binary)
Optional:     openai-whisper, deep-translator
"""

from __future__ import annotations

import bisect
import subprocess
import sys
import os
import shutil
import tempfile
from collections import namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


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

LOWER_THIRD_GUARD = 0.80    # bottom 20% reserved for platform UI
PANEL_MIN_PERSONS = 2       # ≥ this many persons → panel mode

VELOCITY_SMOOTH_TABLE: List[Tuple[float, int]] = [
    (0.0, 51), (3.0, 45), (8.0, 37),
    (15.0, 27), (30.0, 19), (60.0, 13), (120.0, 7),
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
        "bold": 1, "shadow": 0, "back_color": "&H00000000", "margin_v": 80,
    },
    "Yellow (Classic)": {
        "fontsize": 16, "primary_color": "&H0000FFFF",
        "outline_color": "&H00000000", "outline": 2,
        "bold": 1, "shadow": 1, "back_color": "&H00000000", "margin_v": 80,
    },
    "Box (Accessible)": {
        "fontsize": 15, "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000", "outline": 0,
        "bold": 0, "shadow": 0, "back_color": "&H80000000", "margin_v": 80,
    },
}

TRANSLATION_LANGUAGES: Dict[str, str] = {
    "None (keep original)": "",
    "French 🇫🇷": "fr", "German 🇩🇪": "de", "Spanish 🇪🇸": "es",
    "Italian 🇮🇹": "it", "Portuguese 🇵🇹": "pt", "Dutch 🇳🇱": "nl",
    "Polish 🇵🇱": "pl", "Russian 🇷🇺": "ru", "Japanese 🇯🇵": "ja",
    "Korean 🇰🇷": "ko", "Chinese (Simplified) 🇨🇳": "zh-CN",
    "Arabic 🇸🇦": "ar", "Hindi 🇮🇳": "hi", "Turkish 🇹🇷": "tr",
    "Indonesian 🇮🇩": "id", "Swedish 🇸🇪": "sv", "Norwegian 🇳🇴": "no",
    "Danish 🇩🇰": "da", "Finnish 🇫🇮": "fi", "Greek 🇬🇷": "el",
    "Hebrew 🇮🇱": "iw", "Thai 🇹🇭": "th", "Vietnamese 🇻🇳": "vi",
    "Malay 🇲🇾": "ms", "Ukrainian 🇺🇦": "uk",
}


# ─────────────────────────────────────────────────────────────────────────────
#  ClipSegment
# ─────────────────────────────────────────────────────────────────────────────
class ClipSegment:
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
#  Optional dependency guards
# ─────────────────────────────────────────────────────────────────────────────
def whisper_available() -> bool:
    try:
        import whisper; return True  # noqa: F401
    except ImportError:
        return False


def translation_available() -> bool:
    try:
        import deep_translator; return True  # noqa: F401
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  FIX-1: FFmpegVideoReader
#  ─────────────────────────────────────────────────────────────────────────
#  OpenCV VideoCapture fails silently on AV1 (and sometimes VP9/HEVC) when
#  hardware decode is unavailable.  This reader drives FFmpeg directly with
#  software decoders and yields BGR24 numpy frames from stdout.
# ─────────────────────────────────────────────────────────────────────────────
class FFmpegVideoReader:
    """
    Universal FFmpeg-backed frame reader.  Works with AV1, HEVC, VP9, H.264…

    Usage (streaming all frames):
        with FFmpegVideoReader(path, width, height) as reader:
            for frame in reader:   # ndarray (H, W, 3) BGR
                process(frame)

    Optional scale_w / scale_h resize output on-the-fly (faster than Python resize).
    seek_sec + n_frames for random-access spot-reads.
    """

    def __init__(self, path: str, width: int, height: int,
                 seek_sec: float = 0.0,
                 n_frames: Optional[int] = None,
                 scale_w: Optional[int] = None,
                 scale_h: Optional[int] = None):
        self.path        = path
        self.width       = width
        self.height      = height
        self.seek_sec    = seek_sec
        self.n_frames    = n_frames
        self.out_w       = scale_w or width
        self.out_h       = scale_h or height
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover   = b""

    # Build two candidate command lists — try AV1-specific decoder first,
    # then a generic software-decode fallback.
    def _cmds(self) -> List[List[str]]:
        common_head = ["ffmpeg"]
        if self.seek_sec > 0:
            common_head += ["-ss", str(self.seek_sec)]
        common_tail = [
            "-i", self.path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-vf", f"scale={self.out_w}:{self.out_h}",
        ]
        if self.n_frames is not None:
            common_tail += ["-vframes", str(self.n_frames)]
        common_tail += ["pipe:1"]

        return [
            # Option A: force libdav1d (fast AV1 software decoder)
            common_head + ["-vcodec", "libdav1d"] + common_tail,
            # Option B: disable all hardware acceleration, let FFmpeg pick
            common_head + ["-hwaccel", "none"] + common_tail,
        ]

    def _open(self) -> None:
        for cmd in self._cmds():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=max(self._frame_bytes * 4, 1 << 20),
                )
                # Verify we actually get pixels
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc     = proc
                    self._leftover = test
                    return
                # Decoder produced nothing — kill and try next
                try: proc.stdout.close()
                except Exception: pass
                proc.wait()
            except Exception:
                pass
        raise ProcessingError(
            f"FFmpeg could not decode video: {self.path}\n"
            "Ensure ffmpeg is installed with libdav1d / software decoders.")

    def close(self) -> None:
        if self._proc is not None:
            try: self._proc.stdout.close()
            except Exception: pass
            self._proc.wait()
            self._proc = None

    def __enter__(self) -> "FFmpegVideoReader":
        self._open(); return self

    def __exit__(self, *_) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._proc is None:
            self._open()
        buf = self._leftover
        self._leftover = b""
        while True:
            needed = self._frame_bytes - len(buf)
            while needed > 0:
                chunk = self._proc.stdout.read(needed)
                if not chunk:
                    return
                buf += chunk
                needed -= len(chunk)
            yield np.frombuffer(buf[:self._frame_bytes],
                                dtype=np.uint8).reshape(self.out_h, self.out_w, 3)
            buf = buf[self._frame_bytes:]


def _read_frame_at(path: str, width: int, height: int, t_sec: float,
                   scale_w: Optional[int] = None,
                   scale_h: Optional[int] = None) -> Optional[np.ndarray]:
    """Read one frame at timestamp t_sec."""
    reader = FFmpegVideoReader(path, width, height, seek_sec=t_sec, n_frames=1,
                                scale_w=scale_w, scale_h=scale_h)
    reader._open()
    frames = list(reader)
    reader.close()
    return frames[0] if frames else None


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{tool} not found. Install FFmpeg.")


def _has_audio(path: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return "audio" in r.stdout
    except Exception:
        return False


def _get_video_duration_ffprobe(path: str) -> Optional[float]:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip())
    except Exception:
        return None


def _extract_audio_wav(video_path: str, wav_path: str) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and os.path.exists(wav_path)


def _trim_video(input_path: str, output_path: str,
                start_sec: float, end_sec: float) -> bool:
    """Re-encode trim with software decode to fix keyframe + codec issues."""
    cmd = [
        "ffmpeg", "-y", "-hwaccel", "none",
        "-ss", str(start_sec), "-to", str(end_sec),
        "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero", "-reset_timestamps", "1",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and os.path.exists(output_path)


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg pipe encoder — raw BGR24 stdin → H.264 MP4 output
# ─────────────────────────────────────────────────────────────────────────────
def _open_ffmpeg_encoder(
    output_path: str, width: int, height: int, fps: float,
    audio_source: Optional[str],
    crf: int = 23, preset: str = "fast", audio_bitrate: str = "128k",
    subtitle_path: Optional[str] = None,
    subtitle_style: Optional[Dict[str, Any]] = None,
) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
    ]
    if audio_source and _has_audio(audio_source):
        cmd += ["-hwaccel", "none", "-i", audio_source]

    vf_chain: List[str] = []
    if subtitle_path and os.path.exists(subtitle_path):
        s = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        srt_esc = subtitle_path.replace("\\", "/").replace(":", "\\:")
        force = (f"Fontsize={s.get('fontsize',18)},"
                 f"PrimaryColour={s.get('primary_color','&H00FFFFFF')},"
                 f"OutlineColour={s.get('outline_color','&H00000000')},"
                 f"Outline={s.get('outline',2)},Bold={s.get('bold',1)},"
                 f"Shadow={s.get('shadow',0)},BackColour={s.get('back_color','&H00000000')},"
                 f"MarginV={s.get('margin_v',80)},Alignment=2")
        vf_chain.append(f"subtitles='{srt_esc}':force_style='{force}'")

    cmd += ["-map", "0:v:0"]
    if audio_source and _has_audio(audio_source):
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    else:
        cmd += ["-an"]
    if vf_chain:
        cmd += ["-vf", ",".join(vf_chain)]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
        "-shortest", "-movflags", "+faststart", output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _close_ffmpeg_encoder(proc: subprocess.Popen, output_path: str) -> None:
    try: proc.stdin.close()
    except BrokenPipeError: pass
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise ProcessingError(
            f"FFmpeg encoder failed (rc={proc.returncode}):\n"
            f"{stderr[-2000:].decode(errors='replace')}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")


# ─────────────────────────────────────────────────────────────────────────────
#  Video metadata  (ffprobe-based for accuracy with all codecs)
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
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

    w = int(kv.get("width", 0) or 0)
    h = int(kv.get("height", 0) or 0)

    fps_raw = kv.get("r_frame_rate", "30/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0

    duration = float(kv.get("duration", 0.0) or 0.0)
    if duration <= 0:
        nb = int(kv.get("nb_frames", 0) or 0)
        duration = nb / fps if fps > 0 and nb > 0 else 0.0

    total_frames = min(int(duration * fps), MAX_FRAMES_GUARD)
    if w == 0 or h == 0:
        raise ProcessingError(f"Cannot read video dimensions: {path}")

    return {
        "fps": fps, "total_frames": total_frames,
        "width": w, "height": h,
        "duration_seconds": duration,
        "is_landscape": w > h,
    }


def extract_thumbnail(path: str, t: float = 1.0) -> Optional[bytes]:
    info = get_video_info(path)
    frame = _read_frame_at(path, info["width"], info["height"], t,
                            scale_w=320, scale_h=180)
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ─────────────────────────────────────────────────────────────────────────────
#  Resolution / geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        cw = int(orig_h * 9 / 16)
        if cw > orig_w: cw = orig_w; ch = int(cw * 16 / 9)
        else:           ch = orig_h
        return cw - (cw % 2), ch - (ch % 2)
    if th > orig_h: scale = orig_h / th; tw = int(tw * scale); th = int(orig_h)
    if tw > orig_w: scale = orig_w / tw; tw = int(orig_w); th = int(th * scale)
    return max(tw - (tw % 2), 2), max(th - (th % 2), 2)


def calculate_crop_dims(orig_w: int, orig_h: int, tw: int, th: int) -> Tuple[int, int]:
    ratio = tw / th
    if (orig_w / orig_h) > ratio: ch = orig_h; cw = int(round(ch * ratio))
    else:                          cw = orig_w; ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ─────────────────────────────────────────────────────────────────────────────
#  YOLO model cache
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}

def _get_model(weights: str = "yolov8n.pt") -> Any:
    if weights not in _model_cache:
        try: _model_cache[weights] = YOLO(weights)
        except Exception as e: raise ProcessingError(f"Failed to load '{weights}': {e}")
    return _model_cache[weights]


# ─────────────────────────────────────────────────────────────────────────────
#  Face detection  (DNN → Haar fallback)
# ─────────────────────────────────────────────────────────────────────────────
_face_net = None; _haar_cascade = None
_FACE_PROTO = "deploy.prototxt"
_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

def _load_face_net():
    global _face_net
    if _face_net is not None: return _face_net
    if os.path.exists(_FACE_PROTO) and os.path.exists(_FACE_MODEL):
        try: _face_net = cv2.dnn.readNetFromCaffe(_FACE_PROTO, _FACE_MODEL); return _face_net
        except Exception: pass
    return None

def _get_haar():
    global _haar_cascade
    if _haar_cascade is not None: return _haar_cascade
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(path):
        c = cv2.CascadeClassifier(path)
        if not c.empty(): _haar_cascade = c; return c
    return None

def detect_faces(frame, confidence_thresh=0.6):
    h, w = frame.shape[:2]
    net = _load_face_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,123))
        net.setInput(blob); dets = net.forward(); faces = []
        for i in range(dets.shape[2]):
            if float(dets[0,0,i,2]) < confidence_thresh: continue
            x1,y1 = max(0,int(dets[0,0,i,3]*w)), max(0,int(dets[0,0,i,4]*h))
            x2,y2 = min(w,int(dets[0,0,i,5]*w)), min(h,int(dets[0,0,i,6]*h))
            if x2>x1 and y2>y1: faces.append((x1,y1,x2,y2))
        faces.sort(key=lambda f:(f[2]-f[0])*(f[3]-f[1]),reverse=True); return faces
    haar = _get_haar()
    if haar is None: return []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    raw  = haar.detectMultiScale(gray,1.1,5,minSize=(max(30,w//20),max(30,h//20)))
    if len(raw)==0: return []
    faces2=[(x,y,x+bw,y+bh) for x,y,bw,bh in raw]
    faces2.sort(key=lambda f:(f[2]-f[0])*(f[3]-f[1]),reverse=True); return faces2


# ─────────────────────────────────────────────────────────────────────────────
#  Subject detection helpers
# ─────────────────────────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult",["cx","cy","ux1","uy1","ux2","uy2","count"])

def detect_subjects(frame, model, confidence=0.45):
    try: results = model(frame,verbose=False,conf=confidence)[0]
    except Exception as e: print(f"⚠ Detection: {e}",file=sys.stderr); return None
    if results.boxes is None or len(results.boxes)==0: return None
    pp=[]; hp=[]; ap=[]
    for box in results.boxes:
        cls=int(box.cls[0]); conf=float(box.conf[0])
        x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        w=(x2-x1)*conf; e=(w,x1,y1,x2,y2)
        if cls==PERSON_CLASS_ID: pp.append(e)
        elif cls in HIGH_PRIO_CLASSES: hp.append(e)
        ap.append(e)
    pool=pp or hp or ap
    if not pool: return None
    tw=sum(e[0] for e in pool)
    if tw==0: return None
    cx=int(sum(e[0]*(e[1]+e[3])/2 for e in pool)/tw)
    cy=int(sum(e[0]*(e[2]+e[4])/2 for e in pool)/tw)
    return DetectionResult(cx,cy,
        min(e[1] for e in pool),min(e[2] for e in pool),
        max(e[3] for e in pool),max(e[4] for e in pool),len(pool))

def detect_persons_all(frame, model, confidence=0.45):
    try: results=model(frame,verbose=False,conf=confidence)[0]
    except Exception: return []
    if results.boxes is None or len(results.boxes)==0: return []
    persons=[]
    for box in results.boxes:
        if int(box.cls[0])==PERSON_CLASS_ID:
            x1,y1,x2,y2=map(int,box.xyxy[0].tolist()); persons.append((x1,y1,x2,y2))
    persons.sort(key=lambda b:b[0]); return persons


# ─────────────────────────────────────────────────────────────────────────────
#  Framing / guard helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_lower_third_guard(cy,crop_h,subject_cy_src,orig_h):
    hh=crop_h//2
    max_cy=subject_cy_src-int((1.0-LOWER_THIRD_GUARD)*crop_h)+hh
    return min(cy,min(max_cy,orig_h-hh))

def _soi_region_label(cx,cy,w,h):
    col="left" if cx<w//3 else("right" if cx>2*w//3 else "center")
    row="upper" if cy<h//3 else("lower" if cy>2*h//3 else "mid")
    if row=="mid" and col=="center": return "center"
    if row=="mid": return col
    return f"{row}-{col}"

def frame_for_union(ux1,uy1,ux2,uy2,orig_w,orig_h,crop_w,crop_h):
    ucx=(ux1+ux2)//2; ucy=(uy1+uy2)//2
    hw,hh=crop_w//2,crop_h//2
    cx=max(hw,min(ucx,orig_w-hw)); cy=max(hh,min(ucy,orig_h-hh))
    cy=_apply_lower_third_guard(cy,crop_h,ucy,orig_h)
    return cx,max(hh,min(cy,orig_h-hh))

def talking_head_center(faces,orig_w,orig_h,crop_w,crop_h,upper_third_bias=0.30):
    if not faces: return None
    ux1=min(f[0] for f in faces); uy1=min(f[1] for f in faces)
    ux2=max(f[2] for f in faces); uy2=max(f[3] for f in faces)
    face_cx=(ux1+ux2)//2; face_cy=(uy1+uy2)//2
    target_cy=face_cy+crop_h//6
    cy=int(face_cy*(1-upper_third_bias)+target_cy*upper_third_bias)
    hw,hh=crop_w//2,crop_h//2
    cx=max(hw,min(face_cx,orig_w-hw)); cy=max(hh,min(cy,orig_h-hh))
    cy=_apply_lower_third_guard(cy,crop_h,face_cy,orig_h)
    return cx,max(hh,min(cy,orig_h-hh))

def apply_framing_bias(cx,cy,vx,vy,speed,orig_w,orig_h,crop_w,crop_h,
                        look_room_frac=0.12,rot_bias=0.15):
    hw,hh=crop_w//2,crop_h//2; look=min(speed/60.0,1.0)
    if look>0.05:
        n=speed+1e-9
        lx=int(cx+(vx/n)*look_room_frac*crop_w*look)
        ly=int(cy+(vy/n)*look_room_frac*crop_h*look)
    else: lx,ly=cx,cy
    still=max(0.0,1.0-look*3)
    if still>0.01:
        tx=min([orig_w//3,2*orig_w//3],key=lambda x:abs(x-cx))
        ty=min([orig_h//3,2*orig_h//3],key=lambda y:abs(y-cy))
        rx=int(cx+rot_bias*still*(tx-cx)); ry=int(cy+rot_bias*still*(ty-cy))
    else: rx,ry=cx,cy
    nx=int(lx*look+rx*(1-look)); ny=int(ly*look+ry*(1-look))
    return max(hw,min(nx,orig_w-hw)),max(hh,min(ny,orig_h-hh))


# ─────────────────────────────────────────────────────────────────────────────
#  FIX-3: Panel Discussion — 2×2 split-screen grid
# ─────────────────────────────────────────────────────────────────────────────
def _detect_panel_mode(input_path,model,fps,total_frames,orig_w,orig_h,
                        confidence=0.45,n_probe=16):
    if model is None: return False
    probe_ts = np.linspace(1.0,max(1.5,total_frames/fps-1.0),n_probe)
    hits = 0
    for t in probe_ts:
        frame=_read_frame_at(input_path,orig_w,orig_h,t,scale_w=640,scale_h=int(640*orig_h/orig_w))
        if frame is None: continue
        if len(detect_persons_all(frame,model,confidence))>=PANEL_MIN_PERSONS:
            hits+=1
    return hits>n_probe*0.5

def _render_panel_frame(frame,persons,out_w,out_h,prev_slots=None):
    cell_w,cell_h=out_w//2,out_h//2
    canvas=np.zeros((out_h,out_w,3),dtype=np.uint8)
    orig_h,orig_w=frame.shape[:2]
    slots=[None]*4
    for i,p in enumerate(persons[:4]): slots[i]=p
    if prev_slots:
        for i in range(4):
            if slots[i] is None and prev_slots[i] is not None: slots[i]=prev_slots[i]
    for qi,(qx,qy) in enumerate([(0,0),(cell_w,0),(0,cell_h),(cell_w,cell_h)]):
        bbox=slots[qi]
        if bbox is not None:
            px1,py1,px2,py2=bbox
            cx=(px1+px2)//2; cy=(py1+py2)//2
            bw=int((px2-px1)*1.6); bh=int(bw*cell_h/cell_w)
            x1=max(0,cx-bw//2); y1=max(0,cy-bh//2)
            x2=min(orig_w,x1+bw); y2=min(orig_h,y1+bh)
            x1=max(0,x2-bw);      y1=max(0,y2-bh)
            crop=frame[y1:y2,x1:x2]
        else: crop=frame
        if crop.size==0: crop=frame
        canvas[qy:qy+cell_h,qx:qx+cell_w]=cv2.resize(crop,(cell_w,cell_h),interpolation=cv2.INTER_LANCZOS4)
    cv2.line(canvas,(cell_w,0),(cell_w,out_h),(20,20,20),2)
    cv2.line(canvas,(0,cell_h),(out_w,cell_h),(20,20,20),2)
    return canvas,slots


# ─────────────────────────────────────────────────────────────────────────────
#  Optical flow / saliency
# ─────────────────────────────────────────────────────────────────────────────
def optical_flow_center(prev,curr,w,h):
    if prev is None or curr is None: return None
    try:
        flow=cv2.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)
        mag=np.sqrt(flow[...,0]**2+flow[...,1]**2)
        b=max(1,int(w*0.04))
        mag[:,:b]=mag[:,w-b:]=mag[:b,:]=mag[h-b:,:]=0
        if mag.max()<0.8: return None
        t=mag.sum()
        if t==0: return None
        ys,xs=np.mgrid[0:h,0:w]
        return int((xs*mag).sum()/t),int((ys*mag).sum()/t)
    except Exception: return None

def saliency_center(frame):
    h,w=frame.shape[:2]
    if w<MIN_FRAME_DIM or h<MIN_FRAME_DIM: return w//2,h//2
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    lap=cv2.GaussianBlur(np.abs(cv2.Laplacian(gray,cv2.CV_64F)).astype(np.float32),(31,31),0)
    sat=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32),(31,31),0)
    sal=lap/(lap.max()+1e-6)+sat/(sat.max()+1e-6)
    b=max(1,int(w*0.05)); sal[:,:b]=sal[:,w-b:]=sal[:b,:]=sal[h-b:,:]=0
    t=sal.sum()
    if t<1e-6: return w//2,h//2
    ys,xs=np.mgrid[0:h,0:w]
    return int((xs*sal).sum()/t),int((ys*sal).sum()/t)

def is_scene_change(prev,curr,threshold=0.35):
    if prev is None: return False
    try: return float(cv2.absdiff(prev,curr).mean())/255.0>threshold
    except Exception: return False


# ─────────────────────────────────────────────────────────────────────────────
#  FIX-2: Smooth camera path
#  1) Cubic Hermite spline interpolation between keyframes
#  2) Velocity-adaptive Gaussian smoothing (edge-padded, scene-cut-aware)
#  3) Zero-phase EMA polish
# ─────────────────────────────────────────────────────────────────────────────
def _cubic_hermite(p0,p1,m0,m1,t):
    t2=t*t; t3=t2*t
    return (2*t3-3*t2+1)*p0+(t3-2*t2+t)*m0+(-2*t3+3*t2)*p1+(t3-t2)*m1

def interpolate_centers(centers,indices,total):
    if total<=0: return []
    if not centers: return [(0,0)]*total
    if len(centers)==1: return [centers[0]]*total
    n=len(indices)
    xs=[float(c[0]) for c in centers]; ys=[float(c[1]) for c in centers]
    def _tan(v):
        m=[0.0]*len(v)
        for i in range(len(v)):
            if i==0:         m[i]=v[1]-v[0] if len(v)>1 else 0.0
            elif i==len(v)-1: m[i]=v[-1]-v[-2]
            else:             m[i]=0.5*(v[i+1]-v[i-1])
        return m
    mx=_tan(xs); my=_tan(ys); result=[]
    for fi in range(total):
        if fi<=indices[0]:  result.append(centers[0]); continue
        if fi>=indices[-1]: result.append(centers[-1]); continue
        r=bisect.bisect_right(indices,fi); l=r-1
        if r>=n: result.append(centers[-1]); continue
        span=max(indices[r]-indices[l],1); t=(fi-indices[l])/span
        result.append((int(_cubic_hermite(xs[l],xs[r],mx[l]*span,mx[r]*span,t)),
                       int(_cubic_hermite(ys[l],ys[r],my[l]*span,my[r]*span,t))))
    while len(result)<total: result.append(result[-1] if result else (0,0))
    return result[:total]

def _compute_speeds(centers,smooth=9):
    n=len(centers)
    if n<2: return [0.0]*n
    raw=[0.0]+[float(np.sqrt((centers[i][0]-centers[i-1][0])**2+(centers[i][1]-centers[i-1][1])**2)) for i in range(1,n)]
    w=min(smooth,n); return np.convolve(raw,np.ones(w)/w,mode="same").tolist()

def _compute_vel_vecs(centers,look=6):
    n=len(centers); out=[]
    for i in range(n):
        j=min(i+look,n-1); k=max(i-look,0); span=j-k
        out.append(((centers[j][0]-centers[k][0])/span,(centers[j][1]-centers[k][1])/span) if span>0 else (0.0,0.0))
    return out

def _vel_to_window(speed):
    t=VELOCITY_SMOOTH_TABLE
    if speed<=t[0][0]: return t[0][1]
    if speed>=t[-1][0]: return t[-1][1]
    for i in range(len(t)-1):
        v0,w0=t[i]; v1,w1=t[i+1]
        if v0<=speed<=v1:
            tt=(speed-v0)/(v1-v0+1e-9); w=int(w0+tt*(w1-w0))
            return w if w%2==1 else w+1
    return 27

def _gauss_seg(xs,ys,window):
    n=len(xs)
    if n<3: return xs.copy(),ys.copy()
    w=min(window,n-1); w=w if w%2==1 else w-1
    if w<3: return xs.copy(),ys.copy()
    h2=w//2; sigma=h2/2.5+1e-9
    k=np.exp(-0.5*(np.arange(-h2,h2+1)/sigma)**2); k/=k.sum()
    sx=np.convolve(np.pad(xs,h2,"edge"),k,"valid")[:n]
    sy=np.convolve(np.pad(ys,h2,"edge"),k,"valid")[:n]
    return sx,sy

def smooth_centers(centers,speeds,base_window=27,adaptive=True,scene_cuts=None):
    if not centers or len(centers)<3: return list(centers) if centers else []
    n=len(centers)
    xs=np.array([c[0] for c in centers],dtype=float); ys=np.array([c[1] for c in centers],dtype=float)
    spd=np.array(speeds[:n],dtype=float)
    if len(spd)<n: spd=np.pad(spd,(0,n-len(spd)),mode="edge")
    bounds=[0]+sorted(set(scene_cuts or []))+[n]; rx,ry=xs.copy(),ys.copy()
    for i in range(len(bounds)-1):
        s,e=bounds[i],bounds[i+1]
        if e-s<3: continue
        w=max(_vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window,13)
        xs_s,ys_s=_gauss_seg(xs[s:e],ys[s:e],w); rx[s:e]=xs_s; ry[s:e]=ys_s
    return [(int(x),int(y)) for x,y in zip(rx,ry)]

def _ema_polish(centers,alpha=0.08):
    if len(centers)<3: return centers
    n=len(centers)
    fx=[float(centers[0][0])]; fy=[float(centers[0][1])]
    for i in range(1,n):
        fx.append(alpha*centers[i][0]+(1-alpha)*fx[-1])
        fy.append(alpha*centers[i][1]+(1-alpha)*fy[-1])
    rx=[fx[-1]]; ry=[fy[-1]]
    for i in range(n-2,-1,-1):
        rx.append(alpha*fx[i]+(1-alpha)*rx[-1])
        ry.append(alpha*fy[i]+(1-alpha)*ry[-1])
    rx.reverse(); ry.reverse()
    return [(int(x),int(y)) for x,y in zip(rx,ry)]


# ─────────────────────────────────────────────────────────────────────────────
#  Whisper → SRT + translation
# ─────────────────────────────────────────────────────────────────────────────
def _seconds_to_srt_time(s):
    h=int(s//3600); m=int((s%3600)//60); sc=int(s%60); ms=int((s-int(s))*1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def transcribe_to_srt(video_path,srt_path,whisper_model="base",language=None,
                       max_chars_per_line=42,progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    if not whisper_available(): return False
    import whisper as _w
    _p(0.0,"🎙️ Extracting audio…")
    wav_fd,wav_path=tempfile.mkstemp(suffix=".wav"); os.close(wav_fd)
    try:
        if not _extract_audio_wav(video_path,wav_path): return False
        _p(0.2,f"📝 Transcribing ({whisper_model})…")
        model=_w.load_model(whisper_model)
        opts={"word_timestamps":True,"verbose":False}
        if language: opts["language"]=language
        result=model.transcribe(wav_path,**opts)
        _p(0.85,"✍️ Writing subtitles…")
        lines=[]; idx=1; words=[]
        for seg in result.get("segments",[]):
            for w in seg.get("words",[]): words.append({"word":w["word"].strip(),"start":w["start"],"end":w["end"]})
        buf=[]; buf_len=0
        def flush():
            nonlocal idx,buf,buf_len
            if not buf: return
            lines.append(f"{idx}\n{_seconds_to_srt_time(buf[0]['start'])} --> {_seconds_to_srt_time(buf[-1]['end'])}\n{' '.join(x['word'] for x in buf)}\n")
            idx+=1; buf=[]; buf_len=0
        for w in words:
            wl=len(w["word"])+1
            if buf_len+wl>max_chars_per_line and buf: flush()
            buf.append(w); buf_len+=wl
        flush()
        with open(srt_path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
        _p(1.0,f"✅ {len(lines)} subtitle lines"); return True
    except Exception as e: print(f"Whisper failed: {e}",file=sys.stderr); return False
    finally:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except OSError: pass

def translate_srt(srt_path,target_language,source_language="auto",progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    if not translation_available() or not target_language: return bool(not target_language)
    try: from deep_translator import GoogleTranslator
    except ImportError: return False
    try:
        import re
        with open(srt_path,"r",encoding="utf-8") as f: content=f.read()
        blocks=re.split(r"\n\n+",content.strip()); out=[]; tr=GoogleTranslator(source=source_language,target=target_language)
        for i,block in enumerate(blocks):
            ls=block.strip().splitlines()
            if len(ls)<3: out.append(block); continue
            try: translated=tr.translate(" ".join(ls[2:])) or " ".join(ls[2:])
            except Exception: translated=" ".join(ls[2:])
            out.append(f"{ls[0]}\n{ls[1]}\n{translated}")
            if i%10==0: _p(i/max(len(blocks),1),f"🌐 {i}/{len(blocks)}")
        with open(srt_path,"w",encoding="utf-8") as f: f.write("\n\n".join(out)+"\n")
        _p(1.0,"✅ Translation done"); return True
    except Exception as e: print(f"Translation failed: {e}",file=sys.stderr); return False


# ─────────────────────────────────────────────────────────────────────────────
#  Saliency scoring for clip detection
# ─────────────────────────────────────────────────────────────────────────────
def _frame_saliency_score(frame,prev_frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    lap_score=min(float(cv2.Laplacian(gray,cv2.CV_64F).var())/3000.0,1.0)
    motion_score=0.0
    if prev_frame is not None:
        motion_score=min(float(cv2.absdiff(gray,cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)).mean())/30.0,1.0)
    sat_score=min(float(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1].mean())/128.0,1.0)
    return 0.4*motion_score+0.4*lap_score+0.2*sat_score

def _compute_frame_scores(input_path,fps,total_frames,orig_w,orig_h,
                           sample_every=15,progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    scores=[]; scene_cuts=[]; prev_gray=None; prev_frame=None
    sw,sh=min(orig_w,640),min(orig_h,int(640*orig_h/orig_w))
    report_n=max(1,total_frames//20); fi=0
    with FFmpegVideoReader(input_path,orig_w,orig_h,scale_w=sw,scale_h=sh) as reader:
        for frame in reader:
            if fi>=total_frames: break
            if fi%sample_every==0:
                cg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and float(cv2.absdiff(prev_gray,cg).mean())/255.0>0.30:
                    scene_cuts.append(fi)
                scores.append(_frame_saliency_score(frame,prev_frame))
                prev_gray=cg; prev_frame=frame.copy()
            if fi%report_n==0: _p(fi/total_frames,f"Scanning {fi}/{total_frames}…")
            fi+=1
    return np.array(scores,dtype=float),scene_cuts


# ─────────────────────────────────────────────────────────────────────────────
#  Clip detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_clips(input_path,min_duration_sec=25.0,max_duration_sec=65.0,
                  target_n_clips=10,model=None,confidence=0.45,
                  progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    info=get_video_info(input_path)
    fps,total_frames=info["fps"],info["total_frames"]
    duration=info["duration_seconds"]; orig_w,orig_h=info["width"],info["height"]
    sample_every=max(1,int(fps))
    _p(0.0,"🔍 Scanning…")
    scores,scene_cuts_frames=_compute_frame_scores(
        input_path,fps,total_frames,orig_w,orig_h,sample_every=sample_every,
        progress_callback=lambda v,m:_p(v*0.45,m))
    if len(scores)==0: return []
    _p(0.45,"📊 Computing arcs…")
    window=max(5,int(30/(sample_every/fps)))
    ss=np.convolve(scores,np.ones(window)/window,mode="same") if len(scores)>=window else scores.copy()
    if ss.max()>0: ss=ss/ss.max()
    min_gap=max(1,int(min_duration_sec*fps/sample_every))
    peaks=[]
    for i in range(1,len(ss)-1):
        wh=min_gap//2; lo=max(0,i-wh); hi=min(len(ss),i+wh+1)
        if ss[i]==ss[lo:hi].max() and ss[i]>0.3:
            if not peaks or i-peaks[-1]>min_gap//2: peaks.append(i)
    peaks.sort(key=lambda i:ss[i],reverse=True); peaks=peaks[:target_n_clips*2]
    def _arc(pi):
        ps=pi*sample_every/fps; rs=max(0.0,ps-max_duration_sec*0.4); re=min(duration,rs+max_duration_sec)
        for sc in reversed(scene_cuts_frames):
            sc_s=sc/fps
            if 0<ps-sc_s<15.0: rs=max(0.0,sc_s-1.0); break
        for sc in scene_cuts_frames:
            sc_s=sc/fps
            if 0<sc_s-ps<15.0: re=min(duration,sc_s+0.5); break
        cd=re-rs
        if cd<min_duration_sec: re=min(duration,rs+min_duration_sec)
        elif cd>max_duration_sec:
            c=(rs+re)/2; rs=max(0.0,c-max_duration_sec/2); re=min(duration,rs+max_duration_sec)
        return rs,re
    cands=[]
    for pi in peaks:
        s,e=_arc(pi); sc=float(ss[pi])
        if not any(min(e,ce)-max(s,cs)>min_duration_sec*0.5 for cs,ce,_ in cands):
            cands.append((s,e,sc))
    cands.sort(key=lambda x:x[2],reverse=True); cands=cands[:target_n_clips]; cands.sort(key=lambda x:x[0])
    _p(0.55,"🎯 SOI per clip…")
    segments=[]
    for ci,(ss2,se,score) in enumerate(cands):
        _p(0.55+0.35*(ci/max(len(cands),1)),f"🎯 Clip {ci+1}/{len(cands)}…")
        soi_xs=[]; soi_ys=[]; n_s=min(8,max(2,int(se-ss2)))
        for t in np.linspace(ss2+1,se-1,n_s):
            frame=_read_frame_at(input_path,orig_w,orig_h,t,scale_w=640,scale_h=int(640*orig_h/orig_w))
            if frame is None: continue
            if model is not None:
                try:
                    res=model(frame,verbose=False,conf=confidence)[0]
                    if res.boxes is not None:
                        for box in res.boxes:
                            x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                            soi_xs.append((x1+x2)//2); soi_ys.append((y1+y2)//2)
                except Exception: pass
            else:
                scx,scy=saliency_center(frame); soi_xs.append(scx); soi_ys.append(scy)
        sr="center"
        if soi_xs: sr=_soi_region_label(int(np.median(soi_xs)),int(np.median(soi_ys)),orig_w,orig_h)
        ms=int(ss2//60); secs=int(ss2%60); me=int(se//60); sece=int(se%60)
        segments.append(ClipSegment(start_sec=ss2,end_sec=se,score=score,soi_region=sr,
            peak_frame=int(np.linspace(ss2+1,se-1,n_s)[n_s//2]*fps),
            title=f"Clip {ci+1}  ({ms}:{secs:02d} – {me}:{sece:02d})"))
    _p(1.0,f"✅ Found {len(segments)} clips"); return segments


# ─────────────────────────────────────────────────────────────────────────────
#  process_video — main entry point
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    input_path: str, output_path: str,
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

    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(min(max(v,0.0),1.0),msg)
            except Exception: pass

    result_meta: Dict[str,Any]={
        "output_path":output_path,"subtitle_path":None,
        "clamped":False,"effective_size":(0,0),"duration":0.0,"panel_mode":False,
    }

    _check_ffmpeg()
    if not os.path.exists(input_path): raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path)/1024**2>MAX_FILE_SIZE_MB: raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB.")

    info=get_video_info(input_path)
    fps,total_frames=info["fps"],info["total_frames"]
    orig_w,orig_h=info["width"],info["height"]
    duration=info["duration_seconds"]

    if total_frames<=0 or orig_w<=0 or orig_h<=0: raise ProcessingError("Corrupt or unreadable video.")
    if not info["is_landscape"]: raise ProcessingError("Video is already vertical.")

    lbl=target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w,target_h=resolve_target_size(lbl,orig_w,orig_h)
    req_w,req_h=RESOLUTION_PRESETS.get(lbl,(0,0))
    clamped=req_h>0 and (target_h<req_h or target_w<req_w)
    result_meta.update(clamped=clamped,effective_size=(target_w,target_h),duration=duration)
    _p(0.01,f"📐 Output {target_w}×{target_h}  source {orig_w}×{orig_h}")

    if not sample_interval: sample_interval=max(1,int(fps/3))
    render_fps=float(output_fps) if output_fps and output_fps>0 else fps
    crop_w,crop_h=calculate_crop_dims(orig_w,orig_h,target_w,target_h)

    # Scale factor for detection pass (run at ≤960px wide for speed)
    det_scale=min(1.0,960/orig_w)
    det_w,det_h=int(orig_w*det_scale),int(orig_h*det_scale)
    sx,sy=orig_w/det_w,orig_h/det_h   # scale back to original coords

    # ── Whisper ──────────────────────────────────────────────────────────
    srt_path: Optional[str]=None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02,"🎙️ Transcribing…")
        srt_fd,srt_path=tempfile.mkstemp(suffix=".srt"); os.close(srt_fd)
        ok=transcribe_to_srt(input_path,srt_path,whisper_model=whisper_model,
                              language=whisper_language,max_chars_per_line=subtitle_max_chars,
                              progress_callback=lambda v,m:_p(0.02+v*0.08,m))
        if not ok:
            if os.path.exists(srt_path): os.unlink(srt_path); srt_path=None
        else:
            if subtitle_translate_to:
                translate_srt(srt_path,target_language=subtitle_translate_to,
                              progress_callback=lambda v,m:_p(0.10+v*0.05,m))
            result_meta["subtitle_path"]=srt_path

    # ── Load model ────────────────────────────────────────────────────────
    start_pct=0.10; model_obj=None
    if tracking_mode=="subject":
        _p(start_pct,"🤖 Loading YOLO model…"); model_obj=_get_model(yolo_weights)
    elif tracking_mode=="talking_head":
        _p(start_pct,"👤 Loading face detector…")
        if _get_haar() is None and _load_face_net() is None:
            raise ProcessingError("No face detector available.")

    # ── Panel mode check ──────────────────────────────────────────────────
    is_panel=False
    if tracking_mode=="subject" and model_obj is not None:
        _p(start_pct+0.01,"👥 Checking for panel/group shot…")
        is_panel=_detect_panel_mode(input_path,model_obj,fps,total_frames,orig_w,orig_h,confidence)
        if is_panel:
            _p(start_pct+0.02,"🖥️ Panel Discussion mode — split-screen grid")
            result_meta["panel_mode"]=True

    # ── Detection pass ────────────────────────────────────────────────────
    all_centers: List[Tuple[int,int]]=[]
    scene_cuts: List[int]=[]

    if not is_panel:
        _p(start_pct+0.02,f"🔎 Analysing {total_frames} frames…")
        det_centers: List[Tuple[int,int]]=[]
        det_indices: List[int]=[]
        sal_centers: List[Tuple[int,int]]=[]
        sal_indices:  List[int]=[]
        prev_gray=None; prev_flow=None
        last_det: Optional[Tuple[int,int]]=None
        det_dropout=0; MAX_DROPOUT=int(fps*1.5)
        report_n=max(1,total_frames//25); det_end=0.42; fi=0

        with FFmpegVideoReader(input_path,orig_w,orig_h,scale_w=det_w,scale_h=det_h) as reader:
            for frame in reader:
                if fi>=total_frames: break
                cg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                if is_scene_change(prev_gray,cg,scene_cut_threshold):
                    scene_cuts.append(fi); prev_flow=None; det_dropout=0
                prev_gray=cg

                if fi%sample_interval==0:
                    center=None
                    if tracking_mode=="talking_head":
                        faces=detect_faces(frame,confidence_thresh=0.5)
                        if faces:
                            faces_orig=[(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in faces]
                            center=talking_head_center(faces_orig,orig_w,orig_h,crop_w,crop_h,talking_head_bias)
                            det_dropout=0
                        elif use_optical_flow:
                            sm=cv2.resize(cg,(det_w//2,det_h//2))
                            if prev_flow is not None:
                                fc=optical_flow_center(prev_flow,sm,det_w//2,det_h//2)
                                if fc is not None: center=(int(fc[0]*2*sx),int(fc[1]*2*sy))
                            prev_flow=sm; det_dropout+=sample_interval
                    else:
                        det=detect_subjects(frame,model_obj,confidence)
                        if det is not None:
                            center=frame_for_union(int(det.ux1*sx),int(det.uy1*sy),
                                                    int(det.ux2*sx),int(det.uy2*sy),orig_w,orig_h,crop_w,crop_h)
                            last_det=center; det_dropout=0
                        elif use_optical_flow:
                            sm=cv2.resize(cg,(det_w//2,det_h//2))
                            if prev_flow is not None:
                                fc=optical_flow_center(prev_flow,sm,det_w//2,det_h//2)
                                if fc is not None: center=(int(fc[0]*2*sx),int(fc[1]*2*sy))
                            prev_flow=sm; det_dropout+=sample_interval

                    if center is not None:
                        det_centers.append(center); det_indices.append(fi)
                    elif last_det is not None and det_dropout<MAX_DROPOUT:
                        det_centers.append(last_det); det_indices.append(fi)
                    else:
                        sal_centers.append(saliency_center(frame)); sal_indices.append(fi)

                fi+=1
                if fi%report_n==0:
                    _p(start_pct+0.02+(det_end-start_pct-0.02)*(fi/total_frames),f"🔎 {fi}/{total_frames}…")

        _p(det_end,f"📍 {len(det_centers)} anchors · {len(scene_cuts)} cuts")

        if not det_centers:
            det_centers=sal_centers or [(orig_w//2,orig_h//2)]; det_indices=sal_indices or [0]
        else:
            gap=sample_interval*6
            for si,sc in zip(sal_indices,sal_centers):
                if not det_indices or min(abs(si-di) for di in det_indices)>gap:
                    det_indices.append(si); det_centers.append(sc)
            pairs=sorted(zip(det_indices,det_centers))
            det_indices=[p[0] for p in pairs]; det_centers=[p[1] for p in pairs]

        _p(0.43,"📈 Computing crop path…")
        all_centers=interpolate_centers(det_centers,det_indices,total_frames)
        speeds=_compute_speeds(all_centers,smooth=11)

        if rule_of_thirds and tracking_mode!="talking_head":
            vel_vecs=_compute_vel_vecs(all_centers,look=6)
            all_centers=[apply_framing_bias(cx,cy,vx,vy,speeds[i],orig_w,orig_h,crop_w,crop_h)
                         for i,((cx,cy),(vx,vy)) in enumerate(zip(all_centers,vel_vecs))]
        elif tracking_mode=="talking_head" and rule_of_thirds:
            hw2,hh2=crop_w//2,crop_h//2; framed=[]
            for cx,cy in all_centers:
                tx=min([orig_w//3,2*orig_w//3],key=lambda x:abs(x-cx))
                nx=int(cx+0.10*(tx-cx)); framed.append((max(hw2,min(nx,orig_w-hw2)),cy))
            all_centers=framed

        speeds=_compute_speeds(all_centers,smooth=11)
        all_centers=smooth_centers(all_centers,speeds,base_window=smooth_window,
                                    adaptive=adaptive_smoothing,scene_cuts=scene_cuts)
        all_centers=_ema_polish(all_centers,alpha=0.08)
        hw,hh=crop_w//2,crop_h//2
        all_centers=[(max(hw,min(cx,orig_w-hw)),max(hh,min(cy,orig_h-hh))) for cx,cy in all_centers]
        all_centers+=[all_centers[-1]]*max(0,total_frames-len(all_centers))
        all_centers=all_centers[:total_frames]

    # ── Render via pipe ───────────────────────────────────────────────────
    _p(0.46,"✂️ Rendering frames…")
    style=SUBTITLE_STYLES.get(subtitle_style_name,SUBTITLE_STYLES["Bold White (TikTok)"])
    proc=_open_ffmpeg_encoder(output_path,target_w,target_h,render_fps,
                               audio_source=input_path,crf=crf,preset=encoder_preset,
                               audio_bitrate=audio_bitrate,subtitle_path=srt_path,subtitle_style=style)
    rpt_n=max(1,total_frames//40); fi=0; prev_slots=None

    try:
        with FFmpegVideoReader(input_path,orig_w,orig_h) as reader:
            for frame in reader:
                if fi>=total_frames: break
                if is_panel:
                    persons=(detect_persons_all(frame,model_obj,confidence)
                             if fi%sample_interval==0
                             else [s for s in (prev_slots or []) if s is not None])
                    out_frame,prev_slots=_render_panel_frame(frame,persons,target_w,target_h,prev_slots)
                else:
                    cx,cy=all_centers[fi]
                    left=max(0,min(cx-crop_w//2,orig_w-crop_w))
                    top =max(0,min(cy-crop_h//2,orig_h-crop_h))
                    crop=frame[top:top+crop_h,left:left+crop_w]
                    if crop.shape[1]!=target_w or crop.shape[0]!=target_h:
                        crop=cv2.resize(crop,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)
                    out_frame=crop
                try: proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError: break
                fi+=1
                if fi%rpt_n==0: _p(0.46+0.40*(fi/total_frames),f"✂️ {fi}/{total_frames}…")
    finally:
        pass

    _p(0.87,"🎵 Encoding…")
    _close_ffmpeg_encoder(proc,output_path)
    _p(1.0,"✅ Done!")
    print(f"✅  {output_path}  ({os.path.getsize(output_path)/1024**2:.1f} MB)",file=sys.stderr)
    return result_meta


# ─────────────────────────────────────────────────────────────────────────────
#  Batch clip pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_clips_batch(
    input_path: str, output_dir: str, clips: List[ClipSegment],
    target_preset_label: str = "720p   (720×1280  — HD)",
    tracking_mode: str = "subject", talking_head_bias: float = 0.30,
    confidence: float = 0.45, smooth_window: int = 27,
    adaptive_smoothing: bool = True, use_optical_flow: bool = True,
    rule_of_thirds: bool = True, crf: int = 23, encoder_preset: str = "fast",
    audio_bitrate: str = "128k", yolo_weights: str = "yolov8n.pt",
    burn_subtitles: bool = False, whisper_model: str = "base",
    subtitle_style_name: str = "Bold White (TikTok)", subtitle_max_chars: int = 42,
    progress_callback: Optional[Callable[[float,str],None]] = None,
) -> List[Dict[str,Any]]:
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    os.makedirs(output_dir,exist_ok=True); results=[]
    for i,clip in enumerate(clips):
        base_pct=i/max(len(clips),1); next_pct=(i+1)/max(len(clips),1)
        _p(base_pct,f"✂️ Clip {i+1}/{len(clips)}…")
        trimmed_path=None; out_path=None
        try:
            fd,trimmed_path=tempfile.mkstemp(suffix=".mp4"); os.close(fd)
            if not _trim_video(input_path,trimmed_path,clip.start_sec,clip.end_sec):
                results.append({"clip":clip,"output_path":None,"error":"trim failed"}); continue
            out_path=os.path.join(output_dir,f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s_vertical.mp4")
            def clip_cb(v,msg="",_b=base_pct,_n=next_pct): _p(_b+v*(_n-_b),msg)
            meta=process_video(trimmed_path,out_path,target_preset_label=target_preset_label,
                tracking_mode=tracking_mode,talking_head_bias=talking_head_bias,
                confidence=confidence,smooth_window=smooth_window,adaptive_smoothing=adaptive_smoothing,
                use_optical_flow=use_optical_flow,rule_of_thirds=rule_of_thirds,
                crf=crf,encoder_preset=encoder_preset,audio_bitrate=audio_bitrate,
                yolo_weights=yolo_weights,burn_subtitles=burn_subtitles,whisper_model=whisper_model,
                subtitle_style_name=subtitle_style_name,subtitle_max_chars=subtitle_max_chars,
                progress_callback=clip_cb)
            meta["clip"]=clip; results.append(meta)
        except Exception as exc:
            results.append({"clip":clip,"output_path":out_path,"error":str(exc)})
        finally:
            if trimmed_path and os.path.exists(trimmed_path):
                try: os.unlink(trimmed_path)
                except OSError: pass
    n_ok=sum(1 for r in results if not r.get("error"))
    _p(1.0,f"✅ {n_ok}/{len(results)} clips done"); return results
