"""
verticalize.py
──────────────
Convert landscape video to vertical format using AI subject tracking.
Features:
• Subject tracking  — YOLOv8 + optical flow + saliency
• Talking Head Mode — DNN face detector locks crop to face, upper-third framing
• Whisper subtitles — optional; transcribes audio and burns styled captions
• Multi-subject union framing
• Motion-aware adaptive smoothing
• Look-room + rule-of-thirds framing bias
• Upscale guard — output capped to source resolution
• Reliable MJPG temp → FFmpeg encode (no pipe)
Dependencies: opencv-python, ultralytics, numpy, ffmpeg (system)
Optional:     openai-whisper  (pip install openai-whisper)
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
import json
import time
from collections import namedtuple
from typing import Optional, Callable, List, Tuple, Dict, Any

class ProcessingError(Exception):
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PERSON_CLASS_ID   = 0
HIGH_PRIO_CLASSES = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB  = 500
MIN_FRAME_DIM     = 240
MAX_FRAMES_GUARD  = 216_000

VELOCITY_SMOOTH_TABLE: List[Tuple[float, int]] = [
    (0.0,   31), (5.0,   25), (15.0,  17),
    (30.0,  11), (60.0,   7), (120.0,  3),
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

# ─────────────────────────────────────────────────────────────────────────────
# Whisper availability check
# ─────────────────────────────────────────────────────────────────────────────
def whisper_available() -> bool:
    try:
        import whisper  # noqa
        return True
    except ImportError:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ProcessingError(f"{tool} not found. Install FFmpeg and add it to PATH.")

def _ffmpeg_encode(
    video_path: str,
    audio_source: Optional[str],
    output_path: str,
    fps: float,
    duration: float,
    crf: int = 23,
    preset: str = "fast",
    audio_bitrate: str = "128k",
    subtitle_path: Optional[str] = None,
    subtitle_style: Optional[Dict[str, Any]] = None,
) -> None:
    """Re-encode MJPG .avi → H.264 .mp4. Optionally mux audio and burn subtitles."""
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if audio_source:
        cmd += ["-i", audio_source]
    vf_chain = []

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
        srt_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
        force_style = (
            f"Fontsize={fs},PrimaryColour={pc},OutlineColour={oc},"
            f"Outline={ol},Bold={bold},Shadow={shad},"
            f"BackColour={bc},MarginV={mv},Alignment=2"
        )
        vf_chain.append(f"subtitles='{srt_escaped}':force_style='{force_style}'")

    cmd += ["-map", "0:v:0"]
    if audio_source:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    else:
        cmd += ["-an"]

    if vf_chain:
        cmd += ["-vf", ",".join(vf_chain)]

    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-t", str(duration), "-movflags", "+faststart",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ProcessingError(f"FFmpeg failed (rc={r.returncode}):\n{r.stderr[-2000:]}")

def _has_audio(path: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "audio" in r.stdout
    except Exception:
        return False

def _extract_audio_wav(video_path: str, wav_path: str) -> bool:
    """Extract mono 16kHz WAV for Whisper."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(wav_path)

# ─────────────────────────────────────────────────────────────────────────────
# Whisper transcription → SRT
# ─────────────────────────────────────────────────────────────────────────────
def _seconds_to_srt_time(s: float) -> str:
    h, m, sc, ms = int(s // 3600), int((s % 3600) // 60), int(s % 60), int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def transcribe_to_srt(
    video_path: str, srt_path: str, whisper_model: str = "base",
    language: Optional[str] = None, max_chars_per_line: int = 42,
    device: str = "auto",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    if not whisper_available():
        return False
    import whisper

    _p(0.0, "🎙️ Extracting audio for transcription…")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    try:
        if not _extract_audio_wav(video_path, wav_path):
            return False
        _p(0.2, f"📝 Transcribing with Whisper ({whisper_model})…")
        model = whisper.load_model(whisper_model, device=device)
        opts: Dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if language:
            opts["language"] = language
        result = model.transcribe(wav_path, **opts)

        _p(0.85, "✍️ Writing subtitles…")
        lines, idx, buf, buf_len = [], 1, [], 0

        def flush_buf():
            nonlocal idx, buf, buf_len
            if not buf: return
            text  = " ".join(w["word"] for w in buf)
            start = _seconds_to_srt_time(buf[0]["start"])
            end   = _seconds_to_srt_time(buf[-1]["end"])
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            idx += 1
            buf, buf_len = [], 0

        words = [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                 for seg in result.get("segments", []) for w in seg.get("words", [])]

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

# ─────────────────────────────────────────────────────────────────────────────
# Video metadata & Resolution resolver
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ProcessingError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "total_frames": min(nf, MAX_FRAMES_GUARD), "width": w, "height": h,
            "duration_seconds": nf / fps if fps > 0 else 0.0, "is_landscape": w > h}

def resolve_target_size(label: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    tw, th = RESOLUTION_PRESETS.get(label, (0, 0))
    if tw == 0 and th == 0:
        cw = int(orig_h * 9 / 16)
        if cw > orig_w: cw, ch = orig_w, int(orig_w * 16 / 9)
        else: ch = orig_h
        return cw - (cw % 2), ch - (ch % 2)
    if th > orig_h:
        scale = orig_h / th
        tw, th = int(tw * scale), int(orig_h)
        if tw > orig_w:
            scale = orig_w / tw
            tw, th = int(orig_w), int(th * scale)
    return max(tw - (tw % 2), 2), max(th - (th % 2), 2)

# ─────────────────────────────────────────────────────────────────────────────
# AI Models (YOLO & Face)
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, YOLO] = {}
def _get_model(weights: str = "yolov8n.pt", device: str = "auto") -> YOLO:
    key = f"{weights}_{device}"
    if key not in _model_cache:
        try: _model_cache[key] = YOLO(weights)
        except Exception as e: raise ProcessingError(f"Failed to load '{weights}': {e}")
    return _model_cache[key]

_face_net: Optional[cv2.dnn.Net] = None
_FACE_PROTO = "deploy.prototxt"
_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

def _load_face_net() -> Optional[cv2.dnn.Net]:
    global _face_net
    if _face_net is not None: return _face_net
    if os.path.exists(_FACE_PROTO) and os.path.exists(_FACE_MODEL):
        try:
            _face_net = cv2.dnn.readNetFromCaffe(_FACE_PROTO, _FACE_MODEL)
            return _face_net
        except Exception: pass
    return None

_haar_cascade: Optional[cv2.CascadeClassifier] = None
def _get_haar() -> Optional[cv2.CascadeClassifier]:
    global _haar_cascade
    if _haar_cascade is not None: return _haar_cascade
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(path):
        _haar_cascade = cv2.CascadeClassifier(path)
        if not _haar_cascade.empty(): return _haar_cascade
    return None

def detect_faces(frame: np.ndarray, confidence_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    net = _load_face_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
        net.setInput(blob)
        dets = net.forward()
        faces = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < confidence_thresh: continue
            x1, y1 = max(0, int(dets[0, 0, i, 3] * w)), max(0, int(dets[0, 0, i, 4] * h))
            x2, y2 = min(w, int(dets[0, 0, i, 5] * w)), min(h, int(dets[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1: faces.append((x1, y1, x2, y2))
        faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        return faces

    haar = _get_haar()
    if haar is None: return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(30, w//20), max(30, h//20)))
    faces = [(x, y, x+bw, y+bh) for (x, y, bw, bh) in dets]
    faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
    return faces

def talking_head_center(faces, orig_w, orig_h, crop_w, crop_h, upper_third_bias=0.30):
    if not faces: return None
    ux1 = min(f[0] for f in faces); uy1 = min(f[1] for f in faces)
    ux2 = max(f[2] for f in faces); uy2 = max(f[3] for f in faces)
    face_cx, face_cy = (ux1 + ux2) // 2, (uy1 + uy2) // 2
    target_cy = face_cy + crop_h // 6
    cy = int(face_cy * (1 - upper_third_bias) + target_cy * upper_third_bias)
    cx, hw, hh = face_cx, crop_w // 2, crop_h // 2
    return max(hw, min(cx, orig_w - hw)), max(hh, min(cy, orig_h - hh))

# ─────────────────────────────────────────────────────────────────────────────
# Subject detection, Optical Flow, Saliency, Framing
# ─────────────────────────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult", ["cx", "cy", "ux1", "uy1", "ux2", "uy2", "count"])

def detect_subjects(frame, model, confidence=0.45):
    try: results = model(frame, verbose=False, conf=confidence)[0]
    except Exception as e:
        print(f"⚠ Detection: {e}", file=sys.stderr)
        return None
    if results.boxes is None or len(results.boxes) == 0: return None
    person_pool, hiprio_pool, all_pool = [], [], []
    for box in results.boxes:
        cls, conf = int(box.cls[0]), float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        weight = max(1, (x2-x1)*(y2-y1)) * conf
        entry = (weight, x1, y1, x2, y2)
        if cls == PERSON_CLASS_ID: person_pool.append(entry)
        elif cls in HIGH_PRIO_CLASSES: hiprio_pool.append(entry)
        all_pool.append(entry)
    pool = person_pool or hiprio_pool or all_pool
    if not pool: return None
    tw = sum(e[0] for e in pool)
    if tw == 0: return None
    cx = int(sum(e[0]*(e[1]+e[3])/2 for e in pool)/tw)
    cy = int(sum(e[0]*(e[2]+e[4])/2 for e in pool)/tw)
    return DetectionResult(cx, cy, min(e[1] for e in pool), min(e[2] for e in pool),
                           max(e[3] for e in pool), max(e[4] for e in pool), len(pool))

def frame_for_union(ux1, uy1, ux2, uy2, orig_w, orig_h, crop_w, crop_h):
    ucx, ucy = (ux1+ux2)//2, (uy1+uy2)//2
    hw, hh = crop_w//2, crop_h//2
    return max(hw, min(ucx, orig_w-hw)), max(hh, min(ucy, orig_h-hh))

def optical_flow_center(prev, curr, w, h):
    if prev is None or curr is None: return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        b = max(1, int(w * 0.04))
        mag[:,:b] = mag[:,w-b:] = mag[:b,:] = mag[h-b:,:] = 0
        if mag.max() < 0.8: return None
        t = mag.sum()
        if t == 0: return None
        ys, xs = np.mgrid[0:h, 0:w]
        return int((xs * mag).sum()/t), int((ys * mag).sum()/t)
    except Exception: return None

def saliency_center(frame):
    h, w = frame.shape[:2]
    if w < MIN_FRAME_DIM or h < MIN_FRAME_DIM: return w//2, h//2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31,31), 0)
    sat = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32), (31,31), 0)
    sal = lap/(lap.max()+1e-6) + sat/(sat.max()+1e-6)
    b = max(1, int(w * 0.05))
    sal[:,:b] = sal[:,w-b:] = sal[:b,:] = sal[h-b:,:] = 0
    t = sal.sum()
    if t < 1e-6: return w//2, h//2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs * sal).sum()/t), int((ys * sal).sum()/t)

def is_scene_change(prev, curr, threshold=0.35):
    if prev is None: return False
    try: return float(cv2.absdiff(prev, curr).mean())/255.0 > threshold
    except Exception: return False

def apply_framing_bias(cx, cy, vx, vy, speed, orig_w, orig_h, crop_w, crop_h, look_room_frac=0.18, rot_bias=0.25):
    hw, hh = crop_w//2, crop_h//2
    look = min(speed/40.0, 1.0)
    if look > 0.05:
        n = speed + 1e-9
        lx = int(cx + (vx/n) * look_room_frac * crop_w * look)
        ly = int(cy + (vy/n) * look_room_frac * crop_h * look)
    else: lx, ly = cx, cy
    still = max(0.0, min(1.0, 1.0 - look * 2.0))
    if still > 0.01:
        tx = min([orig_w//3, 2*orig_w//3], key=lambda x: abs(x-cx))
        ty = min([orig_h//3, 2*orig_h//3], key=lambda y: abs(y-cy))
        rx = int(cx + rot_bias * still * (tx-cx))
        ry = int(cy + rot_bias * still * (ty-cy))
    else: rx, ry = cx, cy
    nx = int(lx * look + rx * (1.0 - look))
    ny = int(ly * look + ry * (1.0 - look))
    return max(hw, min(nx, orig_w-hw)), max(hh, min(ny, orig_h-hh))

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing & Interpolation
# ─────────────────────────────────────────────────────────────────────────────
def _compute_speeds(centers, smooth=5):
    n = len(centers)
    if n < 2: return [0.0]*n
    raw = [0.0] + [float(np.sqrt((centers[i][0]-centers[i-1][0])**2 + (centers[i][1]-centers[i-1][1])**2)) for i in range(1, n)]
    w = min(smooth, n)
    return np.convolve(raw, np.ones(w)/w, mode="same").tolist()

def _compute_vel_vecs(centers, look=4):
    n = len(centers); out = []
    for i in range(n):
        j, k, span = min(i+look, n-1), max(i-look, 0), 0
        span = j - k
        out.append(((centers[j][0]-centers[k][0])/span, (centers[j][1]-centers[k][1])/span) if span>0 else (0.0,0.0))
    return out

def _vel_to_window(speed):
    t = VELOCITY_SMOOTH_TABLE
    if speed <= t[0][0]: return t[0][1]
    if speed >= t[-1][0]: return t[-1][1]
    for i in range(len(t)-1):
        v0, w0 = t[i]; v1, w1 = t[i+1]
        if v0 <= speed <= v1:
            tt = (speed-v0)/(v1-v0+1e-9)
            w = int(w0 + tt*(w1-w0))
            return w if w%2==1 else w+1
    return 15

def _gauss_seg(xs, ys, window):
    n = len(xs)
    if n < 3: return xs.copy(), ys.copy()
    w = min(window, n-1); w = w if w%2==1 else w-1
    if w < 3: return xs.copy(), ys.copy()
    h2 = w//2; sigma = h2/2.0+1e-9
    k = np.exp(-0.5*(np.arange(-h2,h2+1)/sigma)**2); k /= k.sum()
    return np.convolve(np.pad(xs,h2,"reflect"),k,"valid")[:n], np.convolve(np.pad(ys,h2,"reflect"),k,"valid")[:n]

def smooth_centers(centers, speeds, base_window=15, adaptive=True, scene_cuts=None):
    if not centers or len(centers) < 3: return centers.copy() if centers else []
    n, xs, ys = len(centers), np.array([c[0] for c in centers], dtype=float), np.array([c[1] for c in centers], dtype=float)
    spd = np.array(speeds[:n], dtype=float)
    if len(spd) < n: spd = np.pad(spd, (0, n-len(spd)), mode="edge")
    cuts = set(scene_cuts or [])
    bounds = [0] + sorted(cuts) + [n]
    rx, ry = xs.copy(), ys.copy()
    for i in range(len(bounds)-1):
        s, e = bounds[i], bounds[i+1]
        if e-s < 3: continue
        w = _vel_to_window(float(np.median(spd[s:e]))) if adaptive else base_window
        xs_s, ys_s = _gauss_seg(xs[s:e], ys[s:e], w)
        rx[s:e] = xs_s; ry[s:e] = ys_s
    return [(int(x), int(y)) for x, y in zip(rx, ry)]

def interpolate_centers(centers, indices, total):
    if total <= 0: return []
    if not centers: return [(0,0)] * total
    n, result = len(indices), []
    for fi in range(total):
        if fi <= indices[0]: result.append(centers[0]); continue
        if fi >= indices[-1]: result.append(centers[-1]); continue
        r = bisect.bisect_right(indices, fi); l = r-1
        if r >= n: result.append(centers[-1]); continue
        span = max(indices[r] - indices[l], 1)
        t = (fi - indices[l]) / span
        result.append((int(centers[l][0] + t * (centers[r][0] - centers[l][0])),
                       int(centers[l][1] + t * (centers[r][1] - centers[l][1]))))
    while len(result) < total: result.append(result[-1] if result else (0,0))
    return result[:total]

def calculate_crop_dims(orig_w, orig_h, tw, th):
    ratio = tw/th
    if (orig_w/orig_h) > ratio: ch, cw = orig_h, int(round(orig_h * ratio))
    else: cw, ch = orig_w, int(round(orig_w / ratio))
    return min(cw, orig_w), min(ch, orig_h)

# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    input_path: str, output_path: str,
    target_preset_label: str = "Match source (no upscale)",
    tracking_mode: str = "subject",
    talking_head_bias: float = 0.30,
    sample_interval: Optional[int] = None,
    confidence: float = 0.45,
    use_optical_flow: bool = True,
    smooth_window: int = 15,
    adaptive_smoothing: bool = True,
    rule_of_thirds: bool = True,
    scene_cut_threshold: float = 0.35,
    output_fps: Optional[float] = None,
    crf: int = 23,
    encoder_preset: str = "fast",
    audio_bitrate: str = "128k",
    yolo_weights: str = "yolov8n.pt",
    device: str = "auto",
    burn_subtitles: bool = False,
    whisper_model: str = "base",
    whisper_language: Optional[str] = None,
    subtitle_style_name: str = "Bold White (TikTok)",
    subtitle_max_chars: int = 42,
    background_mode: str = "crop",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Returns metadata dict about the processing result."""
    def _p(v: float, msg: str = ""):
        if progress_callback:
            try: progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception: pass

    result_meta = {"output_path": output_path, "subtitle_path": None, "clamped": False,
                   "effective_size": (0, 0), "duration": 0.0}

    _check_ffmpeg()
    if not os.path.exists(input_path): raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path)/1024**2 > MAX_FILE_SIZE_MB: raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    info = get_video_info(input_path)
    fps, total_frames = info["fps"], info["total_frames"]
    orig_w, orig_h = info["width"], info["height"]
    duration = info["duration_seconds"]

    if total_frames <= 0 or orig_w <= 0 or orig_h <= 0: raise ProcessingError("Corrupt or unreadable video.")
    if not info["is_landscape"]: raise ProcessingError("Video is already vertical — upload a landscape video.")

    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    req_w, req_h = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update({"clamped": clamped, "effective_size": (target_w, target_h), "duration": duration})
    _p(0.01, f"📐 Output {target_w}×{target_h} (source {orig_w}×{orig_h})")

    if not sample_interval: sample_interval = max(1, int(fps/2))
    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)

    # ── Phase 0: Whisper ────────────────────────────────────────────────────────
    srt_path = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "🎙️ Transcribing audio with Whisper…")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt")
        os.close(srt_fd)
        def sub_cb(v, msg=""): _p(0.02 + v * 0.08, msg)
        ok = transcribe_to_srt(input_path, srt_path, whisper_model, whisper_language, subtitle_max_chars, device, sub_cb)
        if not ok:
            _p(0.10, "⚠️ Transcription failed — continuing without subtitles")
            if os.path.exists(srt_path): os.unlink(srt_path)
            srt_path = None
        else: result_meta["subtitle_path"] = srt_path

    # ── Load model ──────────────────────────────────────────────────────────────
    start_pct = 0.10
    if tracking_mode == "subject":
        _p(start_pct, "🤖 Loading AI model…")
        model = _get_model(yolo_weights, device)
    else:
        model = None
        _p(start_pct, "👤 Talking Head Mode — loading face detector…")
        if _get_haar() is None and _load_face_net() is None:
            raise ProcessingError("No face detector available. Reinstall opencv-python.")

    # ── Phase 1: Detection ──────────────────────────────────────────────────────
    _p(start_pct + 0.02, f"🔎 Analysing {total_frames} frames…")
    det_centers, det_indices, sal_centers, sal_indices, scene_cuts = [], [], [], [], []
    cap = cv2.VideoCapture(input_path)
    prev_gray, prev_flow, frame_idx = None, None, 0
    report_n = max(1, total_frames//25)
    det_phase_end = 0.42
    start_time = time.time()

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret: break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_scene_change(prev_gray, curr_gray, scene_cut_threshold):
            scene_cuts.append(frame_idx)
            prev_flow = None
        prev_gray = curr_gray

        if frame_idx % sample_interval == 0:
            center = None
            if tracking_mode == "talking_head":
                faces = detect_faces(frame, confidence_thresh=0.5)
                if faces: center = talking_head_center(faces, orig_w, orig_h, crop_w, crop_h, talking_head_bias)
                elif use_optical_flow:
                    small = cv2.resize(curr_gray, (orig_w//2, orig_h//2))
                    if prev_flow is not None:
                        fc = optical_flow_center(prev_flow, small, orig_w//2, orig_h//2)
                        if fc: center = (fc[0]*2, fc[1]*2)
                    prev_flow = small
            else:
                det = detect_subjects(frame, model, confidence)
                if det is not None:
                    center = frame_for_union(det.ux1, det.uy1, det.ux2, det.uy2, orig_w, orig_h, crop_w, crop_h)
                elif use_optical_flow:
                    small = cv2.resize(curr_gray, (orig_w//2, orig_h//2))
                    if prev_flow is not None:
                        fc = optical_flow_center(prev_flow, small, orig_w//2, orig_h//2)
                        if fc: center = (fc[0]*2, fc[1]*2)
                    prev_flow = small

            if center is not None: det_centers.append(center); det_indices.append(frame_idx)
            else: sal_centers.append(saliency_center(frame)); sal_indices.append(frame_idx)

        frame_idx += 1
        if frame_idx % report_n == 0:
            pct = start_pct + 0.02 + (det_phase_end - start_pct - 0.02) * (frame_idx/total_frames)
            elapsed = time.time() - start_time
            eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
            _p(pct, f"🔎 {frame_idx}/{total_frames} frames… (ETA: {eta:.0f}s)")

    cap.release()
    _p(det_phase_end, f"📍 {len(det_centers)} anchors · {len(scene_cuts)} scene cuts")

    if not det_centers:
        det_centers = sal_centers or [(orig_w//2, orig_h//2)]
        det_indices = sal_indices or [0]
    else:
        gap = sample_interval * 4
        for si, sc in zip(sal_indices, sal_centers):
            if min(abs(si-di) for di in det_indices) > gap:
                det_indices.append(si); det_centers.append(sc)
        pairs = sorted(zip(det_indices, det_centers))
        det_indices = [p[0] for p in pairs]; det_centers = [p[1] for p in pairs]

    # ── Phase 2: Path ─────────────────────────────────────────────────────────────
    _p(0.43, "📈 Computing crop path…")
    all_centers = interpolate_centers(det_centers, det_indices, total_frames)
    speeds = _compute_speeds(all_centers)
    all_centers = smooth_centers(all_centers, speeds, base_window=smooth_window, adaptive=adaptive_smoothing, scene_cuts=scene_cuts)

    if rule_of_thirds and tracking_mode != "talking_head":
        speeds = _compute_speeds(all_centers, smooth=3)
        vel_vecs = _compute_vel_vecs(all_centers, look=3)
        all_centers = [apply_framing_bias(cx, cy, vx, vy, speeds[i], orig_w, orig_h, crop_w, crop_h) for i, (cx, cy) in enumerate(all_centers)]
    elif tracking_mode == "talking_head" and rule_of_thirds:
        framed = []
        for cx, cy in all_centers:
            tx = min([orig_w//3, 2*orig_w//3], key=lambda x: abs(x-cx))
            nx = int(cx + 0.15*(tx-cx))
            hw, hh = crop_w//2, crop_h//2
            framed.append((max(hw, min(nx, orig_w-hw)), cy))
        all_centers = framed

    hw, hh = crop_w//2, crop_h//2
    all_centers = [(max(hw, min(cx, orig_w-hw)), max(hh, min(cy, orig_h-hh))) for cx, cy in all_centers]
    if len(all_centers) < total_frames: all_centers += [all_centers[-1]]*(total_frames-len(all_centers))
    all_centers = all_centers[:total_frames]

    # ── Phase 3: Render ───────────────────────────────────────────────────────────
    _p(0.46, "✂️ Rendering frames…")
    temp_avi, temp_mp4 = None, None
    try:
        fd, temp_avi = tempfile.mkstemp(suffix=".avi"); os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_avi, fourcc, render_fps, (target_w, target_h))
        if not writer.isOpened(): raise ProcessingError("cv2.VideoWriter failed to open.")

        cap = cv2.VideoCapture(input_path)
        rn2 = max(1, total_frames//40)
        for fn in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            cx, cy = all_centers[fn]
            left = max(0, min(cx-crop_w//2, orig_w-crop_w))
            top  = max(0, min(cy-crop_h//2, orig_h-crop_h))
            crop = frame[top:top+crop_h, left:left+crop_w]

            if background_mode != "crop" and (crop.shape[0] < target_h or crop.shape[1] < target_w):
                bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                if background_mode == "blur":
                    bg[:, :] = cv2.GaussianBlur(frame, (0,0), 21)
                x_off = (target_w - crop.shape[1]) // 2
                y_off = (target_h - crop.shape[0]) // 2
                bg[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
                crop = bg
            elif crop.shape[1] != target_w or crop.shape[0] != target_h:
                crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            writer.write(crop)
            if (fn+1) % rn2 == 0:
                _p(0.46 + 0.40*((fn+1)/total_frames), f"✂️ {fn+1}/{total_frames}…")
        cap.release(); writer.release()

        if not os.path.exists(temp_avi) or os.path.getsize(temp_avi) < 1000:
            raise ProcessingError("Rendered .avi is empty.")

        # ── Phase 4: FFmpeg encode ────────────────────────────────────────────────
        _p(0.87, "🎵 Encoding…" + (" Burning subtitles…" if srt_path else ""))
        fd, temp_mp4 = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
        style = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])
        _ffmpeg_encode(temp_avi, input_path if _has_audio(input_path) else None, temp_mp4,
                       fps=render_fps, duration=duration, crf=crf, preset=encoder_preset,
                       audio_bitrate=audio_bitrate, subtitle_path=srt_path, subtitle_style=style)

        if not os.path.exists(temp_mp4) or os.path.getsize(temp_mp4) < 1000:
            raise ProcessingError("FFmpeg produced empty output.")
        shutil.move(temp_mp4, output_path)
        temp_mp4 = None
        _p(1.0, "✅ Done!")
        print(f"✅ {output_path} ({os.path.getsize(output_path)/1024**2:.1f} MB)")
        return result_meta
    finally:
        for p in (temp_avi, temp_mp4):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass
