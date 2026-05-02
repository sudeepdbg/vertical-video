"""
verticalize.py
──────────────
Convert landscape video to vertical format using AI subject tracking.
Features:
• Subject tracking  — YOLOv8 + optical flow + saliency
• Talking Head Mode — DNN face detector locks crop to face, upper-third framing
• Whisper subtitles — optional; transcribes audio and burns styled captions
• Auto-Clip Detection — analyzes long videos for high-engagement segments
• Lower-Third Guard — keeps subjects above bottom 25% of frame
Dependencies: opencv-python, ultralytics, numpy, ffmpeg (system)
Optional:     openai-whisper, deep-translator
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
import re
from collections import namedtuple
from typing import Optional, Callable, List, Tuple, Dict, Any

# ────────────────────────────────────────────────────────────────────────
# Classes & Constants
# ─────────────────────────────────────────────────────────────────────────
class ProcessingError(Exception):
    pass

ClipSegment = namedtuple("ClipSegment", ["start_sec", "end_sec", "duration", "score", "soi_region"])

PERSON_CLASS_ID   = 0
HIGH_PRIO_CLASSES = {0, 2, 3, 5, 7, 15, 16}
MAX_FILE_SIZE_MB  = 2000
MIN_FRAME_DIM     = 240
MAX_FRAMES_GUARD  = 500_000
VELOCITY_SMOOTH_TABLE: List[Tuple[float, int]] = [
    (0.0, 31), (5.0, 25), (15.0, 17), (30.0, 11), (60.0, 7), (120.0, 3)
]
RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Match source (no upscale)":    (0, 0),
    "1080p  (1080×1920 — Full HD)": (1080, 1920),
    "720p   (720×1280  — HD)":      (720,  1280),
    "540p   (540×960   — SD)":      (540,  960),
    "480p   (480×854   — Low)":     (480,  854),
}

SUBTITLE_STYLES: Dict[str, Dict[str, Any]] = {
    "Bold White (TikTok)": {"fontsize": 18, "primary_color": "&H00FFFFFF", "outline_color": "&H00000000", "outline": 2, "bold": 1, "shadow": 0, "back_color": "&H00000000", "margin_v": 80},
    "Yellow (Classic)":    {"fontsize": 16, "primary_color": "&H0000FFFF", "outline_color": "&H00000000", "outline": 2, "bold": 1, "shadow": 1, "back_color": "&H00000000", "margin_v": 80},
    "Box (Accessible)":    {"fontsize": 15, "primary_color": "&H00FFFFFF", "outline_color": "&H00000000", "outline": 0, "bold": 0, "shadow": 0, "back_color": "&H80000000", "margin_v": 80},
}

TRANSLATION_LANGUAGES: Dict[str, str] = {
    "None (keep original)": "", "French 🇫🇷": "fr", "German 🇩🇪": "de", "Spanish 🇪🇸": "es",
    "Italian 🇮🇹": "it", "Portuguese 🇵🇹": "pt", "Dutch 🇳🇱": "nl", "Polish 🇵": "pl",
    "Russian 🇷🇺": "ru", "Japanese 🇯🇵": "ja", "Korean 🇷": "ko", "Chinese (Simplified) 🇨🇳": "zh-CN",
    "Arabic 🇸🇦": "ar", "Hindi 🇮": "hi", "Turkish 🇹": "tr", "Indonesian 🇩": "id",
    "Swedish 🇸🇪": "sv", "Norwegian 🇳🇴": "no", "Danish 🇩🇰": "da", "Finnish 🇫🇮": "fi",
    "Greek 🇬": "el", "Hebrew 🇱": "iw", "Thai 🇭": "th", "Vietnamese 🇳": "vi", "Malay 🇲🇾": "ms", "Ukrainian 🇺🇦": "uk"
}

# ─────────────────────────────────────────────────────────────────────────
# Helpers (FFmpeg, Whisper, Face, YOLO, Smoothing)
# ────────────────────────────────────────────────────────────────────────
def whisper_available() -> bool:
    try: import whisper; return True
    except ImportError: return False

def translation_available() -> bool:
    try: import deep_translator; return True
    except ImportError: return False

def _check_ffmpeg() -> None:
    for t in ("ffmpeg", "ffprobe"):
        try: subprocess.run([t, "-version"], check=True, capture_output=True, text=True)
        except: raise ProcessingError(f"{t} not found. Install FFmpeg & add to PATH.")

def _ffmpeg_encode(video_path: str, audio_source: Optional[str], output_path: str, fps: float, duration: float, crf: int=23, preset: str="fast", audio_bitrate: str="128k", subtitle_path: Optional[str]=None, subtitle_style: Optional[Dict]=None) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if audio_source: cmd += ["-i", audio_source]
    vf_chain = []
    if subtitle_path and os.path.exists(subtitle_path):
        s = subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        esc = subtitle_path.replace("\\", "/").replace(":", "\\:")
        fstyle = f"Fontsize={s['fontsize']},PrimaryColour={s['primary_color']},OutlineColour={s['outline_color']},Outline={s['outline']},Bold={s['bold']},Shadow={s['shadow']},BackColour={s['back_color']},MarginV={s['margin_v']},Alignment=2"
        vf_chain.append(f"subtitles='{esc}':force_style='{fstyle}'")
    cmd += ["-map", "0:v:0"]
    if audio_source: cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    else: cmd += ["-an"]
    if vf_chain: cmd += ["-vf", ", ".join(vf_chain)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p", "-r", str(fps), "-t", str(duration), "-movflags", "+faststart", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: raise ProcessingError(f"FFmpeg failed:\n{r.stderr[-1500:]}")

def _has_audio(p: str) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", p], capture_output=True, text=True, timeout=10)
        return "audio" in r.stdout
    except: return False

def _extract_audio_wav(v: str, w: str) -> bool:
    return subprocess.run(["ffmpeg", "-y", "-i", v, "-ar", "16000", "-ac", "1", "-f", "wav", w], capture_output=True, text=True).returncode == 0 and os.path.exists(w)

def _seconds_to_srt_time(s: float) -> str:
    h, m, sc, ms = int(s//3600), int((s%3600)//60), int(s%60), int((s-int(s))*1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def transcribe_to_srt(video_path: str, srt_path: str, whisper_model: str="base", language: Optional[str]=None, max_chars: int=42, progress_callback: Optional[Callable]=None) -> bool:
    def _p(v, m=""): 
        if progress_callback: progress_callback(v, m)
    if not whisper_available(): return False
    import whisper
    _p(0.0, "🎙️ Extracting audio…")
    _, wav = tempfile.mkstemp(suffix=".wav"); os.close(_)
    try:
        if not _extract_audio_wav(video_path, wav): return False
        _p(0.2, "📝 Transcribing…")
        model = whisper.load_model(whisper_model)
        res = model.transcribe(wav, word_timestamps=True, verbose=False, language=language)
        _p(0.85, "✍️ Formatting SRT…")
        lines, idx, buf, b_len = [], 1, [], 0
        def flush():
            nonlocal idx, buf, b_len
            if buf:
                lines.append(f"{idx}\n{_seconds_to_srt_time(buf[0]['start'])} --> {_seconds_to_srt_time(buf[-1]['end'])}\n{' '.join(w['word'] for w in buf)}\n")
                idx += 1; buf = []; b_len = 0
        for w in res.get("segments", []):
            for word in w.get("words", []):
                if b_len + len(word["word"]) > max_chars: flush()
                buf.append(word); b_len += len(word["word"]) + 1
        flush()
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n".join(lines))
        _p(1.0, f"✅ {len(lines)} lines written")
        return True
    except Exception as e:
        print(f"Whisper error: {e}", file=sys.stderr)
        return False
    finally:
        if os.path.exists(wav): os.unlink(wav)

def translate_srt(srt_path: str, target: str, progress_callback: Optional[Callable]=None) -> bool:
    if not target or not translation_available(): return target != ""
    try:
        from deep_translator import GoogleTranslator
        with open(srt_path, "r", encoding="utf-8") as f: txt = f.read()
        blocks = re.split(r"\n\n+", txt.strip())
        tr = GoogleTranslator(target=target)
        out = []
        for i, b in enumerate(blocks):
            parts = b.strip().splitlines()
            if len(parts) < 3: out.append(b); continue
            try: parts[2] = tr.translate(" ".join(parts[2:]))
            except: pass
            out.append("\n".join(parts))
            if progress_callback and i%10==0: progress_callback(i/len(blocks), f"🌐 Translating {i}/{len(blocks)}…")
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n\n".join(out) + "\n")
        if progress_callback: progress_callback(1.0, "✅ Translated")
        return True
    except: return False

_model_cache: Dict[str, YOLO] = {}
def _get_model(w: str="yolov8n.pt") -> YOLO:
    if w not in _model_cache:
        try: _model_cache[w] = YOLO(w)
        except Exception as e: raise ProcessingError(f"Model load failed: {e}")
    return _model_cache[w]

_face_net = None
_FACE_PROTO = "deploy.prototxt"
_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
def _load_face_net():
    global _face_net
    if _face_net is not None: return _face_net
    if os.path.exists(_FACE_PROTO) and os.path.exists(_FACE_MODEL):
        try: _face_net = cv2.dnn.readNetFromCaffe(_FACE_PROTO, _FACE_MODEL); return _face_net
        except: pass
    return None

_haar_cascade = None
def _get_haar():
    global _haar_cascade
    if _haar_cascade is not None: return _haar_cascade
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(path):
        _haar_cascade = cv2.CascadeClassifier(path)
        if not _haar_cascade.empty(): return _haar_cascade
    return None

def detect_faces(frame: np.ndarray, conf_thresh: float=0.6) -> List[Tuple[int,int,int,int]]:
    h, w = frame.shape[:2]
    net = _load_face_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
        net.setInput(blob)
        dets = net.forward()
        faces = []
        for i in range(dets.shape[2]):
            c = float(dets[0, 0, i, 2])
            if c < conf_thresh: continue
            x1, y1, x2, y2 = map(int, [dets[0,0,i,3]*w, dets[0,0,i,4]*h, dets[0,0,i,5]*w, dets[0,0,i,6]*h])
            if x2>x1 and y2>y1: faces.append((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
        faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        return faces
    haar = _get_haar()
    if not haar: return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(30,w//20), max(30,h//20)))
    return [(x,y,x+bw,y+bh) for x,y,bw,bh in dets]

def talking_head_center(faces, orig_w, orig_h, crop_w, crop_h, upper_third_bias=0.30):
    if not faces: return None
    ux1, uy1, ux2, uy2 = min(f[0] for f in faces), min(f[1] for f in faces), max(f[2] for f in faces), max(f[3] for f in faces)
    face_cx, face_cy = (ux1+ux2)//2, (uy1+uy2)//2
    target_cy = face_cy + crop_h // 6
    cy = int(face_cy * (1 - upper_third_bias) + target_cy * upper_third_bias)
    cx = face_cx
    hw, hh = crop_w//2, crop_h//2
    return max(hw, min(cx, orig_w-hw)), max(hh, min(cy, orig_h-hh))

def get_video_info(p: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(p)
    if not cap.isOpened(): raise ProcessingError(f"Cannot open: {p}")
    fps, nf, w, h = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "total_frames": min(nf, MAX_FRAMES_GUARD), "width": w, "height": h, "duration_seconds": nf/fps if fps>0 else 0.0, "is_landscape": w>h}

def resolve_target_size(lbl: str, ow: int, oh: int) -> Tuple[int, int]:
    tw, th = RESOLUTION_PRESETS.get(lbl, (0,0))
    if tw==0 and th==0:
        cw = int(oh * 9 / 16)
        if cw > ow: cw, ch = ow, int(ow * 16 / 9)
        else: cw, ch = cw, oh
        return max(cw-(cw%2),2), max(ch-(ch%2),2)
    if th > oh: tw, th = int(tw*(oh/th)), oh
    if tw > ow: tw, th = ow, int(th*(ow/tw))
    return max(tw-(tw%2),2), max(th-(th%2),2)

def calculate_crop_dims(ow, oh, tw, th) -> Tuple[int, int]:
    r = tw/th
    if (ow/oh)>r: ch, cw = oh, int(round(oh*r))
    else: cw, ch = ow, int(round(ow/r))
    return min(cw, ow), min(ch, oh)

# ─────────────────────────────────────────────────────────────────────────
# CORE TRACKING & PROCESSING LOGIC
# ─────────────────────────────────────────────────────────────────────────
DetectionResult = namedtuple("DetectionResult", ["cx","cy","ux1","uy1","ux2","uy2","count"])

def detect_subjects(frame: np.ndarray, model: YOLO, conf: float=0.45) -> Optional[DetectionResult]:
    try: res = model(frame, verbose=False, conf=conf)[0]
    except: return None
    if res.boxes is None or len(res.boxes)==0: return None
    pp, hp, ap = [], [], []
    for b in res.boxes:
        cls, c, x1,y1,x2,y2 = int(b.cls[0]), float(b.conf[0]), map(int, b.xyxy[0].tolist())
        w = max(1, (x2-x1)*(y2-y1)) * c
        e = (w, x1, y1, x2, y2)
        if cls==PERSON_CLASS_ID: pp.append(e)
        elif cls in HIGH_PRIO_CLASSES: hp.append(e)
        ap.append(e)
    pool = pp or hp or ap
    if not pool: return None
    tw = sum(e[0] for e in pool)
    if tw==0: return None
    cx = int(sum(e[0]*(e[1]+e[3])/2 for e in pool)/tw)
    cy = int(sum(e[0]*(e[2]+e[4])/2 for e in pool)/tw)
    return DetectionResult(cx, cy, min(e[1] for e in pool), min(e[2] for e in pool), max(e[3] for e in pool), max(e[4] for e in pool), len(pool))

def frame_for_union(ux1, uy1, ux2, uy2, ow, oh, cw, ch) -> Tuple[int, int]:
    hw, hh = cw//2, ch//2
    return max(hw, min((ux1+ux2)//2, ow-hw)), max(hh, min((uy1+uy2)//2, oh-hh))

def optical_flow_center(prev, curr, w, h):
    if prev is None: return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        b = max(1, int(w*0.04))
        mag[:,:b]=mag[:,w-b:]=mag[:b,:]=mag[h-b:,:]=0
        if mag.max()<0.8: return None
        t = mag.sum()
        if t==0: return None
        ys, xs = np.mgrid[0:h, 0:w]
        return int((xs*mag).sum()/t), int((ys*mag).sum()/t)
    except: return None

def saliency_center(frame: np.ndarray) -> Tuple[int,int]:
    h, w = frame.shape[:2]
    if w<MIN_FRAME_DIM or h<MIN_FRAME_DIM: return w//2, h//2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32), (31,31), 0)
    sat  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32), (31,31), 0)
    sal  = lap/(lap.max()+1e-6) + sat/(sat.max()+1e-6)
    b    = max(1, int(w*0.05))
    sal[:,:b]=sal[:,w-b:]=sal[:b,:]=sal[h-b:,:]=0
    t = sal.sum()
    if t<1e-6: return w//2, h//2
    ys, xs = np.mgrid[0:h, 0:w]
    return int((xs*sal).sum()/t), int((ys*sal).sum()/t)

def is_scene_change(prev, curr, thr=0.35):
    if prev is None: return False
    return float(cv2.absdiff(prev, curr).mean())/255.0 > thr

def smooth_centers(centers, speeds, base_w=15, adaptive=True, cuts=None):
    if len(centers)<3: return centers[:]
    n, xs, ys, spd = len(centers), np.array([c[0] for c in centers], float), np.array([c[1] for c in centers], float), np.array(speeds[:n], float)
    if len(spd)<n: spd=np.pad(spd,(0,n-len(spd)),'edge')
    bounds = [0] + sorted(cuts or []) + [n]
    rx, ry = xs.copy(), ys.copy()
    for i in range(len(bounds)-1):
        s, e = bounds[i], bounds[i+1]
        if e-s<3: continue
        w = base_w
        if adaptive and np.median(spd[s:e])>0:
            sp = float(np.median(spd[s:e]))
            t = VELOCITY_SMOOTH_TABLE
            if sp<=t[0][0]: w=t[0][1]
            elif sp>=t[-1][0]: w=t[-1][1]
            else:
                for j in range(len(t)-1):
                    if t[j][0]<=sp<=t[j+1][0]: w=int(t[j][1]+(sp-t[j][0])/(t[j+1][0]-t[j][0])*(t[j+1][1]-t[j][1])); break
        w = max(3, w-1 if w%2==0 else w)
        k = np.exp(-0.5*(np.arange(-w//2, w//2+1)/(w/4.0+1e-9))**2); k/=k.sum()
        rx[s:e] = np.convolve(np.pad(xs[s:e], w//2, 'reflect'), k, 'valid')[:e-s]
        ry[s:e] = np.convolve(np.pad(ys[s:e], w//2, 'reflect'), k, 'valid')[:e-s]
    return [(int(x),int(y)) for x,y in zip(rx,ry)]

def interpolate_centers(centers, indices, total):
    if total<=0: return []
    if not centers: return [(0,0)]*total
    n, res = len(indices), []
    for fi in range(total):
        if fi<=indices[0]: res.append(centers[0])
        elif fi>=indices[-1]: res.append(centers[-1])
        else:
            r = bisect.bisect_right(indices, fi); l = r-1
            span = indices[r]-indices[l]; t = (fi-indices[l])/span
            res.append((int(centers[l][0]+t*(centers[r][0]-centers[l][0])), int(centers[l][1]+t*(centers[r][1]-centers[l][1]))))
    return res[:total]

# ─────────────────────────────────────────────────────────────────────────
# LONG FORM: DETECT CLIPS & PROCESS BATCH
# ─────────────────────────────────────────────────────────────────────────
def detect_clips(video_path: str, min_duration_sec: float = 30.0, max_duration_sec: float = 60.0, target_n_clips: int = 8, confidence: float = 0.45, progress_callback: Optional[Callable] = None) -> List[ClipSegment]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ProcessingError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = range(0, total, step)
    try: model = _get_model("yolov8n.pt")
    except: model = None

    def _p(v,m=""):
        if progress_callback: progress_callback(v,m)
    _p(0.0, "🔍 Analyzing engagement…")
    
    idxs, scs = [], []
    prev_g = None
    for i, fi in enumerate(samples):
        ret, frm = cap.read()
        if not ret: break
        s = 0.0
        g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        if prev_g is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            s += float(np.sqrt(flow[...,0]**2+flow[...,1]**2).mean())
        prev_g = g
        if model is not None:
            try:
                r = model(frm, verbose=False, conf=confidence)[0]
                if r.boxes is not None and len(r.boxes)>0: s += float(np.mean(r.boxes.conf.cpu().numpy())) * 15.0
            except: pass
        idxs.append(fi); scs.append(s)
        if i % max(1, len(samples)//20) == 0: _p(i/len(samples), f"🔎 {i}/{len(samples)} frames")
    cap.release()
    if not scs: return []

    arr = np.array(scs)
    if arr.max()>0: arr /= arr.max()
    
    peaks = []
    thr = 0.35
    for i in range(1, len(arr)-1):
        if arr[i] > thr and arr[i]>=arr[i-1] and arr[i]>=arr[i+1]: peaks.append(i)
    if not peaks: peaks = [len(arr)//2]

    clips = []
    for p in peaks:
        center_t = p
        half = (min_duration_sec + max_duration_sec) / 2.0
        s, e = int(center_t - half), int(center_t + half)
        s, e = max(0, s), min(len(arr)-1, e)
        while s > 0 and arr[s] > 0.2: s -= 1
        while e < len(arr)-1 and arr[e] > 0.2: e += 1
        dur = (e - s)
        if dur < min_duration_sec:
            mid = (s+e)//2; s=max(0,mid-int(min_duration_sec/2)); e=min(len(arr)-1,mid+int(min_duration_sec/2))
        elif dur > max_duration_sec:
            mid = (s+e)//2; s=mid-int(max_duration_sec/2); e=mid+int(max_duration_sec/2)
        clips.append(ClipSegment(start_sec=s, end_sec=e, duration=(e-s), score=float(arr[p]), soi_region="center"))
    
    final = []
    last_e = -1
    for c in sorted(clips, key=lambda x: x.score, reverse=True):
        if c.start_sec > last_e + 5 and len(final) < target_n_clips:
            final.append(c)
            last_e = c.end_sec
    return sorted(final, key=lambda x: x.start_sec)

def process_clips_batch(input_path: str, output_dir: str, clips: List[ClipSegment], target_preset_label: str="Match source (no upscale)", tracking_mode: str="subject", talking_head_bias: float=0.30, confidence: float=0.45, smooth_window: int=15, adaptive_smoothing: bool=True, rule_of_thirds: bool=True, crf: int=23, encoder_preset: str="fast", audio_bitrate: str="128k", yolo_weights: str="yolov8n.pt", burn_subtitles: bool=False, whisper_model: str="base", subtitle_style_name: str="Bold White (TikTok)", subtitle_max_chars: int=42, progress_callback: Optional[Callable]=None) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, clip in enumerate(clips):
        def _cb(v,m):
            if progress_callback: progress_callback((i+v)/len(clips), f"✂️ Clip {i+1}/{len(clips)}: {m}")
        out = os.path.join(output_dir, f"clip_{i+1}_{int(clip.start_sec)}s.mp4")
        try:
            meta = process_video(input_path, out, start_sec=clip.start_sec, end_sec=clip.end_sec, target_preset_label=target_preset_label, tracking_mode=tracking_mode, talking_head_bias=talking_head_bias, confidence=confidence, smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing, rule_of_thirds=rule_of_thirds, crf=crf, encoder_preset=encoder_preset, audio_bitrate=audio_bitrate, yolo_weights=yolo_weights, burn_subtitles=burn_subtitles, whisper_model=whisper_model, subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars, avoid_lower_third=True, progress_callback=_cb)
            meta.update({"clip": clip, "index": i})
            results.append(meta)
        except Exception as e:
            results.append({"clip": clip, "index": i, "error": str(e)})
    return results

# ─────────────────────────────────────────────────────────────────────────
# MAIN PROCESS FUNCTION
# ─────────────────────────────────────────────────────────────────────────
def process_video(input_path: str, output_path: str, target_preset_label: str="Match source (no upscale)", tracking_mode: str="subject", talking_head_bias: float=0.30, sample_interval: Optional[int]=None, confidence: float=0.45, use_optical_flow: bool=True, smooth_window: int=15, adaptive_smoothing: bool=True, rule_of_thirds: bool=True, scene_cut_threshold: float=0.35, output_fps: Optional[float]=None, crf: int=23, encoder_preset: str="fast", audio_bitrate: str="128k", yolo_weights: str="yolov8n.pt", burn_subtitles: bool=False, whisper_model: str="base", whisper_language: Optional[str]=None, subtitle_style_name: str="Bold White (TikTok)", subtitle_max_chars: int=42, subtitle_translate_to: Optional[str]=None, progress_callback: Optional[Callable]=None, start_sec: float=0.0, end_sec: Optional[float]=None, avoid_lower_third: bool=True) -> Dict[str, Any]:
    def _p(v,m=""):
        if progress_callback: progress_callback(min(v,1.0), m)
    meta = {"output_path": output_path, "subtitle_path": None, "clamped": False, "effective_size": (0,0), "duration": 0.0}
    _check_ffmpeg()
    if not os.path.exists(input_path): raise ProcessingError(f"Input not found")
    if os.path.getsize(input_path)/1024**2 > MAX_FILE_SIZE_MB: raise ProcessingError(f"File > {MAX_FILE_SIZE_MB}MB")

    info = get_video_info(input_path)
    fps, tot = info["fps"], info["total_frames"]
    ow, oh = info["width"], info["height"]
    dur = info["duration_seconds"]

    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    tw, th = resolve_target_size(lbl, ow, oh)
    meta["effective_size"] = (tw, th)
    
    _p(0.01, f"📐 {tw}×{th}")
    if not sample_interval: sample_interval = max(1, int(fps/2))
    render_fps = float(output_fps) if output_fps and output_fps>0 else fps
    cw, ch = calculate_crop_dims(ow, oh, tw, th)

    seg_start_f = int(start_sec * fps)
    seg_end_f = int((end_sec or dur) * fps) if end_sec else tot
    if start_sec > 0:
        seg_start_f = max(0, seg_start_f)
        seg_end_f = min(tot, seg_end_f)
    process_frames = seg_end_f - seg_start_f
    
    srt_path = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "🎙️ Transcribing…")
        _, srt_path = tempfile.mkstemp(suffix=".srt"); os.close(_)
        ok = transcribe_to_srt(input_path, srt_path, whisper_model, whisper_language, subtitle_max_chars, lambda v,m: _p(0.02+v*0.08, m))
        if not ok: srt_path = None
        elif subtitle_translate_to:
            _p(0.10, f"🌐 Translating to {subtitle_translate_to}…")
            translate_srt(srt_path, subtitle_translate_to, lambda v,m: _p(0.10+v*0.05, m))
        meta["subtitle_path"] = srt_path

    _p(0.15, " Loading model…" if tracking_mode=="subject" else "👤 Loading face detector…")
    model = _get_model(yolo_weights) if tracking_mode=="subject" else None
    if tracking_mode=="talking_head" and not _get_haar() and not _load_face_net():
        raise ProcessingError("No face detector available.")

    _p(0.18, f"🔎 Analyzing {process_frames} frames…")
    det_c, det_i, sal_c, sal_i, cuts = [], [], [], [], []
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start_f)
    prev_g, prev_f = None, None
    fid = seg_start_f
    end_f = seg_end_f
    report_n = max(1, process_frames//25)
    while fid < end_f:
        ret, frm = cap.read()
        if not ret: break
        g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        if is_scene_change(prev_g, g, scene_cut_threshold): cuts.append(fid-seg_start_f)
        prev_g = g

        if (fid-seg_start_f) % sample_interval == 0:
            c = None
            if tracking_mode=="talking_head":
                faces = detect_faces(frm, conf_thresh=0.5)
                if faces: 
                    ux1,uy1,ux2,uy2 = min(f[0] for f in faces), min(f[1] for f in faces), max(f[2] for f in faces), max(f[3] for f in faces)
                    cx = (ux1+ux2)//2
                    cy = int((uy1+uy2)//2 + ch//6)
                    c = (cx, cy)
                elif use_optical_flow and prev_f is not None:
                    fc = optical_flow_center(prev_f, g, ow//2, oh//2)
                    if fc: c = (fc[0]*2, fc[1]*2)
                prev_f = g
            else:
                d = detect_subjects(frm, model, confidence)
                if d: c = frame_for_union(d.ux1, d.uy1, d.ux2, d.uy2, ow, oh, cw, ch)
                elif use_optical_flow and prev_f is not None:
                    fc = optical_flow_center(prev_f, g, ow//2, oh//2)
                    if fc: c = (fc[0]*2, fc[1]*2)
                prev_f = g
            if c: det_c.append(c); det_i.append(fid-seg_start_f)
            else: sal_c.append(saliency_center(frm)); sal_i.append(fid-seg_start_f)
        fid += 1
        if fid % report_n == 0: _p(0.18 + 0.35*((fid-seg_start_f)/process_frames), f"🔎 {fid-seg_start_f}/{process_frames}…")
    cap.release()
    
    if not det_c: det_c = sal_c or [(ow//2, oh//2)]; det_i = sal_i or [0]
    gap = sample_interval*4
    for si, sc in zip(sal_i, sal_c):
        if min(abs(si-di) for di in det_i) > gap: det_i.append(si); det_c.append(sc)
    pairs = sorted(zip(det_i, det_c))
    det_i, det_c = [p[0] for p in pairs], [p[1] for p in pairs]

    _p(0.53, f"📍 {len(det_c)} anchors")
    all_c = interpolate_centers(det_c, det_i, process_frames)
    spd = [0.0] + [float(np.sqrt((all_c[i][0]-all_c[i-1][0])**2+(all_c[i][1]-all_c[i-1][1])**2)) for i in range(1, len(all_c))]
    all_c = smooth_centers(all_c, spd, smooth_window, adaptive_smoothing, cuts)
    
    # ⬇️ LOWER-THIRD CONSTRAINT: Keep crop center in top 75% of frame
    if avoid_lower_third:
        safe_max_cy = int(oh * 0.75)
        all_c = [(x, min(y, safe_max_cy)) for x,y in all_c]

    if rule_of_thirds and tracking_mode != "talking_head":
        framed = []
        for cx, cy in all_c:
            tx = min([ow//3, 2*ow//3], key=lambda x: abs(x-cx))
            framed.append((int(cx + 0.15*(tx-cx)), cy))
        all_c = framed
        
    hw, hh = cw//2, ch//2
    all_c = [(max(hw, min(cx, ow-hw)), max(hh, min(cy, oh-hh))) for cx, cy in all_c]
    all_c = (all_c + [all_c[-1]]*max(0, process_frames-len(all_c)))[:process_frames]

    _p(0.56, "✂️ Rendering…")
    _, temp_avi = tempfile.mkstemp(suffix=".avi"); os.close(_)
    w = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*"MJPG"), render_fps, (tw, th))
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start_f)
    for i in range(process_frames):
        ret, frm = cap.read()
        if not ret: break
        cx, cy = all_c[i]
        crp = frm[max(0,cy-ch//2):min(oh,cy+ch//2), max(0,cx-cw//2):min(ow,cx+cw//2)]
        w.write(cv2.resize(crp, (tw, th), interpolation=cv2.INTER_LANCZOS4))
        if i % max(1, process_frames//40) == 0: _p(0.56 + 0.40*(i/process_frames), f"✂️ {i}/{process_frames}…")
    cap.release(); w.release()

    _p(0.96, "🎵 Encoding…")
    _, tmp_mp4 = tempfile.mkstemp(suffix=".mp4"); os.close(_)
    _ffmpeg_encode(temp_avi, input_path if _has_audio(input_path) else None, tmp_mp4, render_fps, process_frames/render_fps, crf, encoder_preset, audio_bitrate, srt_path, SUBTITLE_STYLES.get(subtitle_style_name))
    shutil.move(tmp_mp4, output_path)
    meta["duration"] = process_frames/render_fps
    _p(1.0, "✅ Done")
    if temp_avi and os.path.exists(temp_avi): os.unlink(temp_avi)
    return meta
