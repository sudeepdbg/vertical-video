"""
verticalize.py  —  AI Vertical Video Converter  v3.0
──────────────────────────────────────────────────────
Rewrite goals vs v2.x:
  • Removed broken DUO/TRIO/WIDE layout system — split-frame mode caused
    flickering, mis-assigned slots, and artifacts on fast-cut content.
  • All scenes use a single clean crop: union of detected persons/faces,
    padded for context. Works correctly for 1..N people.
  • EMA crop-center smoothing is now applied EVERY frame (it was dead code
    before — only collected into arrays, never used for rendering).
  • Adaptive EMA alpha: slow for talking heads (~0.08), fast for action/
    sports (~0.55). Alpha itself is EMA-smoothed to avoid sudden jumps.
  • Scene cuts snap the crop position instantly then dissolve in.
  • Removed ~250 lines of dead code (interpolate_centers, smooth_centers,
    _ema_polish, LayoutState, LayoutSlotSmoother, _classify_layout, etc.)
  • Single render pass — no second FFmpeg vf pass needed.
  • Detection runs at ~8 fps by default; auto-increases on high motion.
"""

from __future__ import annotations
import subprocess, sys, os, tempfile, math
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
PERSON_CLASS_ID   = 0
HIGH_PRIO_CLASSES = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB  = 2000
MAX_FRAMES_GUARD  = 1_080_000
LOWER_THIRD_GUARD = 0.80   # max fraction of crop below subject centre

# EMA alpha range: 0.08 = very smooth (interviews), 0.55 = responsive (sports)
EMA_ALPHA_SLOW    = 0.08
EMA_ALPHA_FAST    = 0.55
# Motion level at which alpha saturates to EMA_ALPHA_FAST (0-1 normalised)
MOTION_SAT        = 0.08

VIGNETTE_STRENGTH = 0.45
VIGNETTE_FALLOFF  = 1.8
KEN_BURNS_MAX_ZOOM = 1.04
KEN_BURNS_PERIOD   = 8.0
DISSOLVE_FRAMES    = 3

CLIP_BOUNDARY_SEARCH_SEC = 3.0
CLIP_PREROLL_PAD  = 0.35
CLIP_POSTROLL_PAD = 0.35

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


# ── ClipSegment ───────────────────────────────────────────────────────────────
class ClipSegment:
    def __init__(self, start_sec, end_sec, score, soi_region="center", peak_frame=0, title=""):
        self.start_sec  = start_sec
        self.end_sec    = end_sec
        self.score      = score
        self.soi_region = soi_region
        self.peak_frame = peak_frame
        self.title      = title
        self.duration   = end_sec - start_sec

    def __repr__(self):
        return f"<Clip {self.start_sec:.1f}s–{self.end_sec:.1f}s score={self.score:.2f}>"


def whisper_available():
    try: import whisper; return True
    except ImportError: return False

def translation_available():
    try: import deep_translator; return True
    except ImportError: return False

def yolo_available():
    if not _YOLO_AVAILABLE: return False
    try:
        import urllib.request; urllib.request.urlopen("https://github.com", timeout=3); return True
    except Exception:
        return os.path.exists("yolov8n.pt") or os.path.exists("yolov8s.pt")


# ── Visual enhancements ───────────────────────────────────────────────────────
_vignette_cache: Dict[Tuple, np.ndarray] = {}

def _build_vignette(w, h, strength=VIGNETTE_STRENGTH, falloff=VIGNETTE_FALLOFF):
    key = (w, h, round(strength, 3), round(falloff, 3))
    if key in _vignette_cache: return _vignette_cache[key]
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.sqrt(xg**2 + yg**2); dist /= dist.max()
    mask = np.clip(1.0 - strength * (dist**falloff), 0.0, 1.0)[:, :, np.newaxis]
    _vignette_cache[key] = mask; return mask

def apply_vignette(frame, strength=VIGNETTE_STRENGTH):
    if strength <= 0: return frame
    h, w = frame.shape[:2]
    return (frame.astype(np.float32) * _build_vignette(w, h, strength)).clip(0, 255).astype(np.uint8)


def apply_sharpen(frame, strength=0.6, radius=1):
    if strength <= 0: return frame
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)


_lut_cache: Dict[str, np.ndarray] = {}

def _build_lut(grade: str) -> np.ndarray:
    if grade in _lut_cache: return _lut_cache[grade]
    x = np.arange(256, dtype=np.float32)
    if grade == "warm":
        r = np.clip(x*1.06+5,  0, 255); g = np.clip(x*1.02+2, 0, 255); b = np.clip(x*0.92-4, 0, 255)
    elif grade == "cool":
        r = np.clip(x*0.92-4,  0, 255); g = np.clip(x*1.01+1, 0, 255); b = np.clip(x*1.07+6, 0, 255)
    elif grade == "vibrant":
        def sc(v):
            n = v / 255; s = n*n*(3-2*n); return np.clip((n*0.6+s*0.4)*255, 0, 255)
        r = sc(x*1.04); g = sc(x*1.02); b = sc(x)
    elif grade == "matte":
        r = np.clip(x*0.88+18, 0, 255); g = np.clip(x*0.86+16, 0, 255); b = np.clip(x*0.84+22, 0, 255)
    else:
        r = g = b = x.copy()
    lut = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(256, 1, 3)
    _lut_cache[grade] = lut; return lut

def apply_color_grade(frame, grade="none"):
    if not grade or grade == "none": return frame
    return cv2.LUT(frame, _build_lut(grade))


def apply_ken_burns(frame, frame_idx, fps,
                     max_zoom=KEN_BURNS_MAX_ZOOM, period=KEN_BURNS_PERIOD):
    if max_zoom <= 1.0: return frame
    t = (frame_idx / max(fps, 1)) % period
    scale = 1.0 + (max_zoom - 1.0) * 0.5 * (1 - math.cos(2 * math.pi * t / period))
    if abs(scale - 1.0) < 1e-4: return frame
    h, w = frame.shape[:2]
    nw = max(int(w / scale), 2); nh = max(int(h / scale), 2)
    x0 = (w - nw) // 2; y0 = (h - nh) // 2
    return cv2.resize(frame[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LINEAR)


class DissolveBuffer:
    """Cross-dissolve over N frames at scene cuts."""
    def __init__(self, n=DISSOLVE_FRAMES):
        self.n = n; self._buf: Optional[np.ndarray] = None; self._rem = 0

    def on_cut(self, last_frame: np.ndarray):
        self._buf = last_frame.copy(); self._rem = self.n

    def blend(self, new_frame: np.ndarray) -> np.ndarray:
        if self._rem <= 0 or self._buf is None: return new_frame
        a = self._rem / self.n; self._rem -= 1
        return cv2.addWeighted(self._buf, a, new_frame, 1.0 - a, 0)

    @property
    def active(self): return self._rem > 0


# ── FFmpegVideoReader ─────────────────────────────────────────────────────────
class FFmpegVideoReader:
    def __init__(self, path, width, height,
                 seek_sec=0.0, n_frames=None, scale_w=None, scale_h=None):
        self.path  = path; self.width = width; self.height = height
        self.seek_sec = seek_sec; self.n_frames = n_frames
        self.out_w = scale_w or width; self.out_h = scale_h or height
        self._proc = None
        self._frame_bytes = self.out_w * self.out_h * 3
        self._leftover = b""

    def _build_cmd(self, extra=None):
        head = ["ffmpeg"]
        if self.seek_sec > 0: head += ["-ss", str(self.seek_sec)]
        if extra: head += extra
        tail = ["-i", self.path, "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-vf", f"scale={self.out_w}:{self.out_h}"]
        if self.n_frames is not None: tail += ["-vframes", str(self.n_frames)]
        return head + tail + ["pipe:1"]

    def _open(self):
        cmds = [self._build_cmd(["-vcodec", "libdav1d"]),
                self._build_cmd(["-hwaccel", "none"])]
        for cmd in cmds:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=max(self._frame_bytes * 4, 1 << 20))
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc = proc; self._leftover = test; return
                try: proc.stdout.close()
                except Exception: pass
                proc.wait()
            except Exception: pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")

    def close(self):
        if self._proc:
            try: self._proc.stdout.close()
            except Exception: pass
            self._proc.wait(); self._proc = None

    def __enter__(self): self._open(); return self
    def __exit__(self, *_): self.close()

    def __iter__(self):
        if not self._proc: self._open()
        buf = self._leftover; self._leftover = b""
        while True:
            needed = self._frame_bytes - len(buf)
            while needed > 0:
                chunk = self._proc.stdout.read(needed)
                if not chunk: return
                buf += chunk; needed -= len(chunk)
            yield np.frombuffer(buf[:self._frame_bytes],
                                 dtype=np.uint8).reshape(self.out_h, self.out_w, 3)
            buf = buf[self._frame_bytes:]


def _read_frame_at(path, width, height, t_sec, scale_w=None, scale_h=None):
    r = FFmpegVideoReader(path, width, height,
                           seek_sec=t_sec, n_frames=1, scale_w=scale_w, scale_h=scale_h)
    r._open(); frames = list(r); r.close()
    return frames[0] if frames else None


# ── FFmpeg helpers ────────────────────────────────────────────────────────────
def _check_ffmpeg():
    for t in ("ffmpeg", "ffprobe"):
        try: subprocess.run([t, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{t} not found. Install FFmpeg.")

def _has_audio(path):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a",
             "-show_entries","stream=codec_type","-of","csv=p=0", path],
            capture_output=True, text=True, timeout=15)
        return "audio" in r.stdout
    except Exception: return False

def _extract_audio_wav(vpath, wpath):
    r = subprocess.run(
        ["ffmpeg","-y","-i",vpath,"-ar","16000","-ac","1","-f","wav",wpath],
        capture_output=True)
    return r.returncode == 0 and os.path.exists(wpath)

def _trim_video(inp, out, start, end):
    r = subprocess.run(
        ["ffmpeg","-y","-hwaccel","none","-ss",str(start),"-to",str(end),"-i",inp,
         "-c:v","libx264","-preset","ultrafast","-crf","18",
         "-c:a","aac","-b:a","128k",
         "-avoid_negative_ts","make_zero","-reset_timestamps","1", out],
        capture_output=True)
    return r.returncode == 0 and os.path.exists(out)


def _open_ffmpeg_encoder(output_path, width, height, fps, audio_source,
                          crf=23, preset="fast", audio_bitrate="128k",
                          subtitle_path=None, subtitle_style=None):
    cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","bgr24",
           "-s", f"{width}x{height}","-r", str(fps),"-i","pipe:0"]
    has_aud = audio_source and _has_audio(audio_source)
    if has_aud: cmd += ["-hwaccel","none","-i", audio_source]

    vf = []
    if subtitle_path and os.path.exists(subtitle_path):
        s    = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        sesc = subtitle_path.replace("\\","/").replace(":","\\:")
        force = (f"Fontsize={s.get('fontsize',18)},"
                 f"PrimaryColour={s.get('primary_color','&H00FFFFFF')},"
                 f"OutlineColour={s.get('outline_color','&H00000000')},"
                 f"Outline={s.get('outline',2)},Bold={s.get('bold',1)},"
                 f"Shadow={s.get('shadow',0)},BackColour={s.get('back_color','&H00000000')},"
                 f"MarginV={s.get('margin_v',80)},Alignment=2")
        vf.append(f"subtitles='{sesc}':force_style='{force}'")

    cmd += ["-map","0:v:0"]
    if has_aud: cmd += ["-map","1:a:0?","-c:a","aac","-b:a",audio_bitrate,"-ac","2"]
    else: cmd += ["-an"]
    if vf: cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v","libx264","-preset",preset,"-crf",str(crf),
            "-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
            "-shortest","-movflags","+faststart", output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def _close_ffmpeg_encoder(proc, output_path):
    try: proc.stdin.close()
    except Exception: pass
    proc.wait()
    if proc.returncode != 0:
        try: err = proc.stderr.read(2000).decode(errors="replace")
        except Exception: err = ""
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")


# ── Video metadata ────────────────────────────────────────────────────────────
def get_video_info(path):
    cmd = ["ffprobe","-v","error","-select_streams","v:0",
           "-show_entries","stream=width,height,r_frame_rate,nb_frames",
           "-show_entries","format=duration",
           "-of","default=noprint_wrappers=1", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    kv = {}
    for line in r.stdout.splitlines():
        if "=" in line: k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    w = int(kv.get("width", 0) or 0); h = int(kv.get("height", 0) or 0)
    try:
        num, den = kv.get("r_frame_rate","30/1").split("/")
        fps = float(num) / float(den)
    except Exception: fps = 30.0
    dur = float(kv.get("duration", 0.0) or 0.0)
    if dur <= 0:
        nb = int(kv.get("nb_frames", 0) or 0)
        dur = nb / fps if fps > 0 and nb > 0 else 0.0
    if w == 0 or h == 0: raise ProcessingError(f"Cannot read dimensions: {path}")
    return {"fps": fps,
            "total_frames": min(int(dur*fps), MAX_FRAMES_GUARD),
            "width": w, "height": h,
            "duration_seconds": dur,
            "is_landscape": w > h}

def extract_thumbnail(path, t=1.0):
    info = get_video_info(path)
    frame = _read_frame_at(path, info["width"], info["height"], t, scale_w=320, scale_h=180)
    if frame is None: return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ── Resolution helpers ────────────────────────────────────────────────────────
def resolve_target_size(label, orig_w, orig_h):
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        cw = int(orig_h * 9 / 16)
        if cw > orig_w: cw = orig_w; ch = int(cw * 16 / 9)
        else: ch = orig_h
        return cw - (cw % 2), ch - (ch % 2)
    if th > orig_h: scale = orig_h/th; tw = int(tw*scale); th = int(orig_h)
    if tw > orig_w: scale = orig_w/tw; tw = int(orig_w); th = int(th*scale)
    return max(tw - (tw % 2), 2), max(th - (th % 2), 2)

def calculate_crop_dims(orig_w, orig_h, tw, th):
    ratio = tw / th
    if (orig_w / orig_h) > ratio: ch = orig_h; cw = int(round(ch * ratio))
    else: cw = orig_w; ch = int(round(cw / ratio))
    return min(cw, orig_w), min(ch, orig_h)


# ── YOLO model cache ──────────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}

def _get_model(weights="yolov8n.pt"):
    if not _YOLO_AVAILABLE: return None
    if weights in _model_cache: return _model_cache[weights]
    try:
        m = _YOLO(weights); _model_cache[weights] = m; return m
    except Exception as e:
        print(f"YOLO unavailable: {e}", file=sys.stderr); return None


# ── Face detection ────────────────────────────────────────────────────────────
_face_net = None; _haar_cascade = None
_FACE_PROTO = "deploy.prototxt"
_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

def _load_face_net():
    global _face_net
    if _face_net: return _face_net
    if os.path.exists(_FACE_PROTO) and os.path.exists(_FACE_MODEL):
        try: _face_net = cv2.dnn.readNetFromCaffe(_FACE_PROTO, _FACE_MODEL); return _face_net
        except Exception: pass
    return None

def _get_haar():
    global _haar_cascade
    if _haar_cascade: return _haar_cascade
    p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(p):
        c = cv2.CascadeClassifier(p)
        if not c.empty(): _haar_cascade = c; return c
    return None

def detect_faces(frame, confidence_thresh=0.6):
    h, w = frame.shape[:2]; net = _load_face_net()
    if net:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob); dets = net.forward(); faces = []
        for i in range(dets.shape[2]):
            if float(dets[0,0,i,2]) < confidence_thresh: continue
            x1 = max(0, int(dets[0,0,i,3]*w)); y1 = max(0, int(dets[0,0,i,4]*h))
            x2 = min(w, int(dets[0,0,i,5]*w)); y2 = min(h, int(dets[0,0,i,6]*h))
            if x2 > x1 and y2 > y1: faces.append((x1, y1, x2, y2))
        faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        return faces
    haar = _get_haar()
    if not haar: return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw  = haar.detectMultiScale(gray, 1.1, 5,
                                  minSize=(max(30, w//20), max(30, h//20)))
    if len(raw) == 0: return []
    faces2 = [(x, y, x+bw, y+bh) for x, y, bw, bh in raw]
    faces2.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
    return faces2


# ── Person & saliency detection ───────────────────────────────────────────────
def detect_persons_all(frame, model, confidence=0.45):
    """Return list of (x1,y1,x2,y2) for all detected persons."""
    if model is None: return []
    try: results = model(frame, verbose=False, conf=confidence)[0]
    except Exception: return []
    if results.boxes is None or len(results.boxes) == 0: return []
    persons = []
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            persons.append((x1, y1, x2, y2))
    return persons

def detect_high_prio(frame, model, confidence=0.45):
    """Weighted center of all high-priority objects (fallback when no persons)."""
    if model is None: return None
    try: results = model(frame, verbose=False, conf=confidence)[0]
    except Exception: return None
    if results.boxes is None or len(results.boxes) == 0: return None
    objs = []
    for box in results.boxes:
        if int(box.cls[0]) in HIGH_PRIO_CLASSES:
            c  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objs.append((c * (x2-x1), (x1+x2)//2, (y1+y2)//2))
    if not objs: return None
    tw = sum(o[0] for o in objs)
    if tw <= 0: return None
    return (int(sum(o[0]*o[1] for o in objs) / tw),
            int(sum(o[0]*o[2] for o in objs) / tw))

def saliency_center(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31,31), 0)
    sat  = cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32), (31,31), 0)
    sal  = lap / (lap.max()+1e-6) + sat / (sat.max()+1e-6)
    b = max(1, int(w*0.05))
    sal[:, :b] = sal[:, w-b:] = sal[:b, :] = sal[h-b:, :] = 0
    t = sal.sum()
    if t < 1e-6: return w//2, h//2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs*sal).sum()/t), int((ys*sal).sum()/t)

def is_scene_change(prev_gray, curr_gray, threshold=0.35):
    if prev_gray is None or curr_gray is None: return False
    try: return float(cv2.absdiff(prev_gray, curr_gray).mean()) / 255.0 > threshold
    except Exception: return False


# ── Smart crop center ─────────────────────────────────────────────────────────
def _persons_crop_center(persons, orig_w, orig_h, crop_w, crop_h):
    """
    Union bounding box of all detected persons → crop centre with headroom bias.
    Handles 1..N persons without splitting the frame.
    """
    if not persons: return None
    ux1 = min(p[0] for p in persons); uy1 = min(p[1] for p in persons)
    ux2 = max(p[2] for p in persons); uy2 = max(p[3] for p in persons)
    ucx = (ux1 + ux2) // 2
    ucy = (uy1 + uy2) // 2

    hw, hh = crop_w // 2, crop_h // 2
    # Slight upward bias for headroom
    ucy -= int(crop_h * 0.05)
    # Lower-third guard: don't push crop too far below subjects
    max_cy = ucy - int((1.0 - LOWER_THIRD_GUARD) * crop_h) + hh
    cy = min(ucy, max_cy)
    cx = ucx

    return (max(hw, min(cx, orig_w - hw)),
            max(hh, min(cy, orig_h - hh)))

def _faces_crop_center(faces, orig_w, orig_h, crop_w, crop_h, bias=0.28):
    """Crop centre from face union with upward bias for talking-head videos."""
    if not faces: return None
    ux1 = min(f[0] for f in faces); uy2 = max(f[3] for f in faces)
    ucx = sum((f[0]+f[2])//2 for f in faces) // len(faces)
    ucy = sum((f[1]+f[3])//2 for f in faces) // len(faces)
    cy  = int(ucy*(1-bias) + (ucy + crop_h//6)*bias)
    hw, hh = crop_w // 2, crop_h // 2
    return (max(hw, min(ucx, orig_w - hw)),
            max(hh, min(cy,  orig_h - hh)))


# ── SOI label (used in clip metadata) ────────────────────────────────────────
def _soi_region_label(cx, cy, w, h):
    col = "left" if cx < w//3 else ("right" if cx > 2*w//3 else "center")
    row = "upper" if cy < h//3 else ("lower" if cy > 2*h//3 else "mid")
    if row == "mid" and col == "center": return "center"
    if row == "mid": return col
    return f"{row}-{col}"


# ── Whisper subtitles ─────────────────────────────────────────────────────────
def _seconds_to_srt_time(s):
    h = int(s//3600); m = int((s%3600)//60); sc = int(s%60); ms = int((s-int(s))*1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def transcribe_to_srt(video_path, srt_path, whisper_model="base", language=None,
                       max_chars_per_line=42, progress_callback=None):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass
    if not whisper_available(): return False
    import whisper as _w
    _p(0.0, "Extracting audio...")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(wav_fd)
    try:
        if not _extract_audio_wav(video_path, wav_path): return False
        _p(0.2, f"Transcribing ({whisper_model})...")
        model  = _w.load_model(whisper_model)
        opts   = {"word_timestamps": True, "verbose": False}
        if language: opts["language"] = language
        result = model.transcribe(wav_path, **opts)
        _p(0.85, "Writing subtitles...")
        lines = []; idx = 1; words = []
        for seg in result.get("segments", []):
            for w_ in seg.get("words", []):
                words.append({"word": w_["word"].strip(),
                               "start": w_["start"], "end": w_["end"]})
        buf = []; buf_len = 0

        def flush():
            nonlocal idx, buf, buf_len
            if not buf: return
            lines.append(
                f"{idx}\n{_seconds_to_srt_time(buf[0]['start'])} --> "
                f"{_seconds_to_srt_time(buf[-1]['end'])}\n"
                f"{' '.join(x['word'] for x in buf)}\n")
            idx += 1; buf = []; buf_len = 0

        for w_ in words:
            wl = len(w_["word"]) + 1
            if buf_len + wl > max_chars_per_line and buf: flush()
            buf.append(w_); buf_len += wl
        flush()
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n".join(lines))
        _p(1.0, f"{len(lines)} subtitle lines"); return True
    except Exception as e:
        print(f"Whisper failed: {e}", file=sys.stderr); return False
    finally:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except OSError: pass

def translate_srt(srt_path, target_language, source_language="auto", progress_callback=None):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass
    if not translation_available() or not target_language: return bool(not target_language)
    try: from deep_translator import GoogleTranslator
    except ImportError: return False
    try:
        import re
        with open(srt_path, "r", encoding="utf-8") as f: content = f.read()
        blocks = re.split(r"\n\n+", content.strip()); out = []
        tr = GoogleTranslator(source=source_language, target=target_language)
        for i, block in enumerate(blocks):
            ls = block.strip().splitlines()
            if len(ls) < 3: out.append(block); continue
            try: translated = tr.translate(" ".join(ls[2:])) or " ".join(ls[2:])
            except Exception: translated = " ".join(ls[2:])
            out.append(f"{ls[0]}\n{ls[1]}\n{translated}")
            if i % 10 == 0: _p(i/max(len(blocks),1), f"{i}/{len(blocks)}")
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n\n".join(out)+"\n")
        _p(1.0, "Translation done"); return True
    except Exception as e:
        print(f"Translation failed: {e}", file=sys.stderr); return False


# ── Clip detection ────────────────────────────────────────────────────────────
def _frame_saliency_score(frame, prev_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_score = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)
    motion_score = 0.0
    if prev_frame is not None:
        motion_score = min(
            float(cv2.absdiff(gray, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)).mean()) / 30.0, 1.0)
    sat_score = min(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].mean()) / 128.0, 1.0)
    return 0.4*motion_score + 0.4*lap_score + 0.2*sat_score

def _compute_frame_scores(input_path, fps, total_frames, orig_w, orig_h,
                           sample_every=15, progress_callback=None):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass
    scores = []; scene_cuts = []; prev_gray = None; prev_frame = None
    sw = min(orig_w, 640); sh = max(1, int(sw*orig_h/orig_w))
    report_n = max(1, total_frames//20); fi = 0
    with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=sw, scale_h=sh) as reader:
        for frame in reader:
            if fi >= total_frames: break
            if fi % sample_every == 0:
                cg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if (prev_gray is not None and
                        float(cv2.absdiff(prev_gray, cg).mean())/255.0 > 0.30):
                    scene_cuts.append(fi)
                scores.append(_frame_saliency_score(frame, prev_frame))
                prev_gray = cg; prev_frame = frame.copy()
            if fi % report_n == 0: _p(fi/total_frames, f"Scanning {fi}/{total_frames}...")
            fi += 1
    return np.array(scores, dtype=float), scene_cuts

def _find_nearest_cut(scene_cuts_sec, target_sec, window):
    best = None; best_d = float("inf")
    for t in scene_cuts_sec:
        d = abs(t - target_sec)
        if d <= window and d < best_d: best_d = d; best = t
    return best

def _refine_clip_boundaries(start_sec, end_sec, duration, scene_cuts_sec,
                              min_dur, max_dur,
                              window=CLIP_BOUNDARY_SEARCH_SEC,
                              preroll=CLIP_PREROLL_PAD,
                              postroll=CLIP_POSTROLL_PAD):
    c0 = _find_nearest_cut(scene_cuts_sec, start_sec, window)
    c1 = _find_nearest_cut(scene_cuts_sec, end_sec,   window)
    new_s = max(0.0, (c0 + preroll)  if c0 is not None else start_sec)
    new_e = min(duration, (c1 - postroll) if c1 is not None else end_sec)
    d = new_e - new_s
    if d < min_dur:
        deficit = min_dur - d
        new_s = max(0.0, new_s - deficit/2)
        new_e = min(duration, new_s + min_dur)
        new_s = max(0.0, new_e - min_dur)
    elif d > max_dur:
        c = (new_s + new_e) / 2
        new_s = max(0.0, c - max_dur/2)
        new_e = min(duration, new_s + max_dur)
    return new_s, new_e

def detect_clips(input_path, min_duration_sec=25.0, max_duration_sec=65.0,
                  target_n_clips=10, model=None, confidence=0.45,
                  progress_callback=None):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass
    info = get_video_info(input_path)
    fps, total_frames = info["fps"], info["total_frames"]
    duration = info["duration_seconds"]; orig_w, orig_h = info["width"], info["height"]
    sample_every = max(1, int(fps)); _p(0.0, "Scanning...")
    scores, scene_cuts_frames = _compute_frame_scores(
        input_path, fps, total_frames, orig_w, orig_h,
        sample_every=sample_every,
        progress_callback=lambda v, m: _p(v*0.45, m))
    if len(scores) == 0: return []
    scene_cuts_sec = [sc/fps for sc in scene_cuts_frames]

    _p(0.45, "Computing arcs...")
    window = max(5, int(30 / (sample_every/fps)))
    ss = (np.convolve(scores, np.ones(window)/window, mode="same")
          if len(scores) >= window else scores.copy())
    if ss.max() > 0: ss /= ss.max()
    min_gap = max(1, int(min_duration_sec*fps/sample_every)); peaks = []
    for i in range(1, len(ss)-1):
        wh = min_gap//2; lo = max(0, i-wh); hi = min(len(ss), i+wh+1)
        if ss[i] == ss[lo:hi].max() and ss[i] > 0.3:
            if not peaks or i - peaks[-1] > min_gap//2: peaks.append(i)
    peaks.sort(key=lambda i: ss[i], reverse=True); peaks = peaks[:target_n_clips*2]

    def _arc(pi):
        ps = pi*sample_every/fps
        rs = max(0.0, ps - max_duration_sec*0.4)
        re = min(duration, rs + max_duration_sec)
        for sc in reversed(scene_cuts_frames):
            sc_s = sc/fps
            if 0 < ps - sc_s < 15.0: rs = max(0.0, sc_s - 1.0); break
        for sc in scene_cuts_frames:
            sc_s = sc/fps
            if 0 < sc_s - ps < 15.0: re = min(duration, sc_s + 0.5); break
        cd = re - rs
        if cd < min_duration_sec: re = min(duration, rs + min_duration_sec)
        elif cd > max_duration_sec:
            c = (rs+re)/2; rs = max(0.0, c-max_duration_sec/2)
            re = min(duration, rs + max_duration_sec)
        return rs, re

    cands = []
    for pi in peaks:
        s, e = _arc(pi); sc = float(ss[pi])
        if not any(min(e,ce)-max(s,cs) > min_duration_sec*0.5 for cs,ce,_ in cands):
            cands.append((s, e, sc))
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:target_n_clips]; cands.sort(key=lambda x: x[0])

    _p(0.55, "Refining boundaries..."); segments = []
    for ci, (ss2, se, score) in enumerate(cands):
        _p(0.55 + 0.35*(ci/max(len(cands),1)), f"Clip {ci+1}/{len(cands)}...")
        ref_s, ref_e = _refine_clip_boundaries(
            ss2, se, duration, scene_cuts_sec, min_duration_sec, max_duration_sec)
        # SOI probe
        soi_xs = []; soi_ys = []
        n_s = min(8, max(2, int(ref_e - ref_s)))
        for t in np.linspace(ref_s+1, ref_e-1, n_s):
            frame = _read_frame_at(input_path, orig_w, orig_h, t,
                                    scale_w=640, scale_h=max(1, int(640*orig_h/orig_w)))
            if frame is None: continue
            if model is not None:
                try:
                    res = model(frame, verbose=False, conf=confidence)[0]
                    if res.boxes is not None:
                        for box in res.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                            soi_xs.append((x1+x2)//2); soi_ys.append((y1+y2)//2)
                except Exception: pass
            else:
                scx, scy = saliency_center(frame)
                soi_xs.append(scx); soi_ys.append(scy)
        sr = _soi_region_label(int(np.median(soi_xs)), int(np.median(soi_ys)),
                                orig_w, orig_h) if soi_xs else "center"
        ms = int(ref_s//60); secs = int(ref_s%60)
        me = int(ref_e//60); sece = int(ref_e%60)
        segments.append(ClipSegment(
            start_sec=ref_s, end_sec=ref_e, score=score, soi_region=sr,
            peak_frame=int(np.linspace(ref_s+1, ref_e-1, n_s)[n_s//2]*fps),
            title=f"Clip {ci+1}  ({ms}:{secs:02d}–{me}:{sece:02d})"))
    _p(1.0, f"Found {len(segments)} clips"); return segments


# ══════════════════════════════════════════════════════════════════════════════
# process_video — main entry point
# ══════════════════════════════════════════════════════════════════════════════
def process_video(
    input_path, output_path,
    target_preset_label="Match source (no upscale)",
    # Tracking
    tracking_mode="subject",       # "subject" | "talking_head" | "saliency"
    talking_head_bias=0.28,
    sample_interval=None,          # detection interval in frames; None = auto (~8/s)
    confidence=0.45,
    scene_cut_threshold=0.35,
    # Encoding
    output_fps=None,
    crf=23, encoder_preset="fast", audio_bitrate="128k",
    yolo_weights="yolov8n.pt",
    # Subtitles
    burn_subtitles=False, whisper_model="base", whisper_language=None,
    subtitle_style_name="Bold White (TikTok)", subtitle_max_chars=42,
    subtitle_translate_to=None,
    # Visual FX (all optional)
    vignette_strength=VIGNETTE_STRENGTH,
    sharpen_strength=0.0,
    color_grade="none",
    ken_burns=False,
    dissolve_cuts=True,
    progress_callback=None,
):
    """
    Convert a landscape video to 9:16 vertical.

    Tracking strategy
    ─────────────────
    • Detect persons/faces every `sample_interval` frames (default ~8×/s).
    • On every frame, EMA-smooth the crop centre toward the current target.
    • EMA alpha adapts to scene motion:
        – low motion  → α ≈ 0.08  (smooth pan, good for interviews)
        – high motion → α ≈ 0.55  (responsive, good for sports)
      Alpha itself is EMA-smoothed to avoid sudden speed jumps.
    • On scene cuts: snap crop position immediately, then optionally dissolve.
    • Multiple persons → union bounding box crop (no frame splitting).
    """
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception: pass

    result_meta = {"output_path": output_path, "subtitle_path": None,
                   "clamped": False, "effective_size": (0, 0), "duration": 0.0}
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

    lbl = (target_preset_label if target_preset_label in RESOLUTION_PRESETS
           else "Match source (no upscale)")
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    req_w, req_h = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update(clamped=clamped, effective_size=(target_w, target_h), duration=duration)

    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)
    hw, hh = crop_w // 2, crop_h // 2

    # Detection runs at a smaller scale for speed
    det_scale = min(1.0, 640 / orig_w)
    det_w  = max(1, int(orig_w * det_scale))
    det_h  = max(1, int(orig_h * det_scale))
    sx, sy = orig_w / det_w, orig_h / det_h

    # Auto sample interval: ~8 detections/s
    if sample_interval is None:
        sample_interval = max(1, int(fps / 8))

    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    _p(0.01, f"Output {target_w}×{target_h}  src {orig_w}×{orig_h}  "
             f"det every {sample_interval} frames  mode={tracking_mode}")

    # ── Subtitles ──────────────────────────────────────────────────────────
    srt_path = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "Transcribing...")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt"); os.close(srt_fd)
        ok = transcribe_to_srt(
            input_path, srt_path, whisper_model=whisper_model,
            language=whisper_language, max_chars_per_line=subtitle_max_chars,
            progress_callback=lambda v, m: _p(0.02 + v*0.08, m))
        if not ok:
            if os.path.exists(srt_path): os.unlink(srt_path)
            srt_path = None
        else:
            if subtitle_translate_to:
                translate_srt(srt_path, target_language=subtitle_translate_to,
                               progress_callback=lambda v, m: _p(0.10 + v*0.04, m))
            result_meta["subtitle_path"] = srt_path

    # ── Load model ─────────────────────────────────────────────────────────
    model_obj = None
    if tracking_mode == "subject":
        _p(0.12, "Loading YOLO...")
        model_obj = _get_model(yolo_weights)
        if model_obj is None: _p(0.12, "YOLO unavailable — saliency fallback")
    elif tracking_mode == "talking_head":
        _p(0.12, "Loading face detector...")

    # ── Open encoder ───────────────────────────────────────────────────────
    style = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])
    proc  = _open_ffmpeg_encoder(
        output_path, target_w, target_h, render_fps,
        audio_source=input_path, crf=crf, preset=encoder_preset,
        audio_bitrate=audio_bitrate, subtitle_path=srt_path, subtitle_style=style)

    # Pre-build LUTs / vignette mask
    if vignette_strength > 0:  _build_vignette(target_w, target_h, vignette_strength)
    if color_grade not in (None, "none"): _build_lut(color_grade)

    dissolve_buf = DissolveBuffer(DISSOLVE_FRAMES) if dissolve_cuts else None

    # ── EMA tracking state ─────────────────────────────────────────────────
    # cur_cx / cur_cy : smoothed crop centre in full-frame coords (float)
    # target_cx / target_cy : most recent detection result
    cur_cx    = float(orig_w // 2); cur_cy    = float(orig_h // 2)
    target_cx = float(orig_w // 2); target_cy = float(orig_h // 2)
    ema_alpha = EMA_ALPHA_SLOW       # will adapt per-frame

    # Track last detection to re-use on non-sample frames when nothing new
    prev_gray = None; last_out_frame = None; scene_cuts = []
    rpt_n = max(1, total_frames // 40); fi = 0

    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames: break

                is_sample = (fi % sample_interval == 0)
                is_cut    = False   # set below if scene cut detected

                # ── Detection pass (sample frames only) ─────────────────
                if is_sample:
                    det_frame = cv2.resize(frame, (det_w, det_h),
                                           interpolation=cv2.INTER_LINEAR)
                    cg = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

                    # Scene cut?
                    if is_scene_change(prev_gray, cg, scene_cut_threshold):
                        scene_cuts.append(fi)
                        is_cut = True
                        # Snap crop position to last known target immediately
                        cur_cx = target_cx; cur_cy = target_cy

                    # Adapt EMA alpha from motion level
                    if prev_gray is not None:
                        motion = float(cv2.absdiff(prev_gray, cg).mean()) / 255.0
                        alpha_target = (EMA_ALPHA_SLOW
                                        + (EMA_ALPHA_FAST - EMA_ALPHA_SLOW)
                                        * min(motion / MOTION_SAT, 1.0))
                        # Smooth the alpha itself to avoid sudden jumps
                        ema_alpha = 0.7*ema_alpha + 0.3*alpha_target
                    prev_gray = cg

                    # ── Find crop target ────────────────────────────────
                    new_target = None

                    if tracking_mode == "subject" and model_obj is not None:
                        persons = detect_persons_all(det_frame, model_obj, confidence)
                        if persons:
                            persons_full = [
                                (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                for x1, y1, x2, y2 in persons]
                            new_target = _persons_crop_center(
                                persons_full, orig_w, orig_h, crop_w, crop_h)
                        if new_target is None:
                            # Any high-priority object (ball, vehicle, animal…)
                            hp = detect_high_prio(det_frame, model_obj, confidence)
                            if hp:
                                new_target = (
                                    max(hw, min(int(hp[0]*sx), orig_w-hw)),
                                    max(hh, min(int(hp[1]*sy), orig_h-hh)))

                    elif tracking_mode == "talking_head":
                        faces = detect_faces(det_frame, confidence_thresh=0.5)
                        if faces:
                            faces_full = [
                                (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                for x1, y1, x2, y2 in faces]
                            new_target = _faces_crop_center(
                                faces_full, orig_w, orig_h, crop_w, crop_h,
                                talking_head_bias)

                    # Saliency fallback (also covers "saliency" mode)
                    if new_target is None:
                        scx, scy = saliency_center(det_frame)
                        new_target = (max(hw, min(int(scx*sx), orig_w-hw)),
                                      max(hh, min(int(scy*sy), orig_h-hh)))

                    target_cx = float(new_target[0])
                    target_cy = float(new_target[1])

                # ── EMA smooth toward target — applied every frame ───────
                cur_cx = ema_alpha * target_cx + (1.0 - ema_alpha) * cur_cx
                cur_cy = ema_alpha * target_cy + (1.0 - ema_alpha) * cur_cy

                # Clamp to valid crop region
                cx = int(max(hw, min(cur_cx, orig_w - hw)))
                cy = int(max(hh, min(cur_cy, orig_h - hh)))

                # ── Crop ─────────────────────────────────────────────────
                left = cx - hw; top = cy - hh
                out_frame = frame[top:top+crop_h, left:left+crop_w]
                if out_frame.shape[0] != target_h or out_frame.shape[1] != target_w:
                    out_frame = cv2.resize(out_frame, (target_w, target_h),
                                           interpolation=cv2.INTER_LANCZOS4)

                # ── Visual FX ────────────────────────────────────────────
                if ken_burns:
                    out_frame = apply_ken_burns(out_frame, fi, fps)
                if sharpen_strength > 0:
                    out_frame = apply_sharpen(out_frame, sharpen_strength)
                if color_grade and color_grade != "none":
                    out_frame = apply_color_grade(out_frame, color_grade)
                if vignette_strength > 0:
                    out_frame = apply_vignette(out_frame, vignette_strength)

                # Dissolve: register cut, then blend on following frames
                if is_cut and dissolve_buf and last_out_frame is not None:
                    dissolve_buf.on_cut(last_out_frame)
                if dissolve_buf and dissolve_buf.active:
                    out_frame = dissolve_buf.blend(out_frame)

                last_out_frame = out_frame
                try: proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError: break

                fi += 1
                if fi % rpt_n == 0:
                    _p(0.15 + 0.75*(fi/total_frames), f"{fi}/{total_frames}…")

    finally:
        pass   # reader cleanup handled by context manager

    _p(0.91, "Finalizing...")
    _close_ffmpeg_encoder(proc, output_path)
    _p(1.0, "Done!")
    sz = os.path.getsize(output_path) / 1024**2
    print(f"Output: {output_path}  ({sz:.1f} MB)  cuts={len(scene_cuts)}", file=sys.stderr)
    return result_meta


# ── Batch clip pipeline ───────────────────────────────────────────────────────
def process_clips_batch(
    input_path, output_dir, clips,
    target_preset_label="720p   (720x1280  - HD)",
    tracking_mode="subject", talking_head_bias=0.28,
    confidence=0.45, sample_interval=None,
    crf=23, encoder_preset="fast", audio_bitrate="128k",
    yolo_weights="yolov8n.pt",
    burn_subtitles=False, whisper_model="base",
    subtitle_style_name="Bold White (TikTok)", subtitle_max_chars=42,
    vignette_strength=VIGNETTE_STRENGTH, sharpen_strength=0.0,
    color_grade="none", ken_burns=False, dissolve_cuts=True,
    progress_callback=None,
):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass
    os.makedirs(output_dir, exist_ok=True); results = []
    for i, clip in enumerate(clips):
        base_pct = i / max(len(clips), 1)
        next_pct = (i+1) / max(len(clips), 1)
        _p(base_pct, f"Clip {i+1}/{len(clips)}…")
        trimmed_path = None; out_path = None
        try:
            fd, trimmed_path = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
            if not _trim_video(input_path, trimmed_path, clip.start_sec, clip.end_sec):
                results.append({"clip": clip, "output_path": None, "error": "trim failed"})
                continue
            out_path = os.path.join(
                output_dir,
                f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s_vertical.mp4")

            def clip_cb(v, msg="", _b=base_pct, _n=next_pct):
                _p(_b + v*(_n-_b), msg)

            meta = process_video(
                trimmed_path, out_path,
                target_preset_label=target_preset_label,
                tracking_mode=tracking_mode, talking_head_bias=talking_head_bias,
                confidence=confidence, sample_interval=sample_interval,
                crf=crf, encoder_preset=encoder_preset, audio_bitrate=audio_bitrate,
                yolo_weights=yolo_weights,
                burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                vignette_strength=vignette_strength, sharpen_strength=sharpen_strength,
                color_grade=color_grade, ken_burns=ken_burns, dissolve_cuts=dissolve_cuts,
                progress_callback=clip_cb)
            meta["clip"] = clip; results.append(meta)

        except Exception as exc:
            results.append({"clip": clip, "output_path": out_path, "error": str(exc)})
        finally:
            if trimmed_path and os.path.exists(trimmed_path):
                try: os.unlink(trimmed_path)
                except OSError: pass

    n_ok = sum(1 for r in results if not r.get("error"))
    _p(1.0, f"{n_ok}/{len(results)} clips done"); return results
