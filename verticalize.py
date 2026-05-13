"""
verticalize.py  —  AI Vertical Video converter  v2.8
──────────────────────────────────────────────────────
FIXES in v2.8 (Bug-fix pass):

  FIX-E1  get_video_info: indentation bug — kv parsing happened inside the
           for-line loop so w/h/fps/dur were computed on every line rather than
           after the loop.  Moved the post-loop extraction outside.

  FIX-E2  render_adaptive_frame / process_video: double color-grade + vignette.
           _tight_crop_for_group and _wide_crop_for_group already apply both;
           the outer loop in process_video applied them a second time, wrecking
           the image.  Removed the redundant outer applications.

  FIX-E3  render_adaptive_frame: fw/fh from frame.shape[:2] were backwards
           (shape returns (h, w, c)).  Fixed to fh, fw = frame.shape[:2].

  FIX-E4  FFmpegVideoReader._open: zombie-process leak when the first candidate
           command fails mid-read.  Added try/finally to guarantee proc.wait().
           Also reordered stdout.close() → wait() to avoid deadlock on stderr.

  FIX-E5  _close_ffmpeg_encoder: stderr.read() after proc.wait() can raise on
           some platforms because the pipe is already closed.  Read stderr before
           wait() instead.

  FIX-E6  _pillarbox_cache: module-level cache grows unboundedly across a batch
           run.  Replaced with a proper LRU-style bounded dict helper and wired
           it to reset between process_video calls.

  FIX-E7  process_video: sample_interval was resolved after it was first needed
           (optical-flow / saliency fallback path reads it at detection time).
           Moved resolution to immediately after video info is fetched.

  FIX-E8  process_video: the accumulated det_centers / det_indices were never
           fed into smooth_centers / interpolate_centers — the entire smoothing
           infrastructure was bypassed.  The per-frame EMA chain (TargetSmoother
           + CropAnchor) is the correct runtime path; removed the dead
           accumulation variables to avoid confusion and memory growth.

Inherited from v2.7 (Source-Aware Detection):
  FIX-D1..D3  Pillarbox detection, stricter _filter_persons, split threshold.

Inherited from v2.6 (Panel Close-up Correctness):
  FIX-C1..C5  Probe supermajority, mismatch guard, person-count override,
              empty-group placeholder, hysteresis 60→20 frames.

Inherited from v2.5 (Render Pipeline):
  FIX-B1..B8  Deduplicated render, stale cache fix, split threshold,
              hysteresis timing, slot cleanup, alpha overshoot, trio speaker,
              post-effects on all paths.

Inherited from v2.4 (Stability Engine):
  FIX-S1..S6  Double smoother, velocity clamp, persistent slots, duo framing.
"""

from __future__ import annotations
import bisect, subprocess, sys, os, tempfile, math
from collections import namedtuple
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
HIGH_PRIO_CLASSES  = {0,2,3,5,7,15,16}
MAX_FILE_SIZE_MB   = 2000
MIN_FRAME_DIM      = 240
MAX_FRAMES_GUARD   = 1_080_000
LOWER_THIRD_GUARD  = 0.80
PANEL_MIN_PERSONS  = 2

VELOCITY_SMOOTH_TABLE = [
    (0.0,51),(3.0,45),(8.0,37),(15.0,27),(30.0,19),(60.0,13),(120.0,7),
]

RESOLUTION_PRESETS = {
    "Match source (no upscale)":    (0,0),
    "1080p  (1080x1920 - Full HD)": (1080,1920),
    "720p   (720x1280  - HD)":      (720,1280),
    "540p   (540x960   - SD)":      (540,960),
    "480p   (480x854   - Low)":     (480,854),
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

# ENH constants
VIGNETTE_STRENGTH  = 0.55
VIGNETTE_FALLOFF   = 1.8
COLOR_GRADES       = ("none","warm","cool","vibrant","matte")
PANEL_SLOT_EMA     = 0.15
KEN_BURNS_MAX_ZOOM = 1.04
KEN_BURNS_PERIOD   = 8.0
DISSOLVE_FRAMES    = 3
PANEL_DIVIDER_PX   = 4
PANEL_DIVIDER_COLOR= (10,10,10)
PANEL_CROP_EXPAND  = 1.9

# Layout Constants
LAYOUT_SINGLE    = "single"
LAYOUT_DUO_SPLIT = "duo_split"
LAYOUT_DUO_WIDE  = "duo_wide"
LAYOUT_TRIO      = "trio"
LAYOUT_WIDE      = "wide"

LAYOUT_HYSTERESIS_FRAMES = 20
LAYOUT_PROBE_MIN_FRAC    = 0.40

CLIP_BOUNDARY_SEARCH_SEC = 3.0
CLIP_PREROLL_PAD  = 0.35
CLIP_POSTROLL_PAD = 0.35

_PILLARBOX_CACHE_MAX = 64


# ── Segment class ─────────────────────────────────────────────────────────────
class ClipSegment:
    def __init__(self,start_sec,end_sec,score,soi_region="center",peak_frame=0,title=""):
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


# ── ENH-1: Vignette ───────────────────────────────────────────────────────────
_vignette_cache: Dict[Tuple,np.ndarray] = {}

def _build_vignette(w,h,strength=VIGNETTE_STRENGTH,falloff=VIGNETTE_FALLOFF):
    key=(w,h,round(strength,3),round(falloff,3))
    if key in _vignette_cache: return _vignette_cache[key]
    xs=np.linspace(-1,1,w,dtype=np.float32); ys=np.linspace(-1,1,h,dtype=np.float32)
    xg,yg=np.meshgrid(xs,ys); dist=np.sqrt(xg**2+yg**2); dist/=dist.max()
    mask=np.clip(1.0-strength*(dist**falloff),0.0,1.0)[:,:,np.newaxis]
    _vignette_cache[key]=mask; return mask

def apply_vignette(frame,strength=VIGNETTE_STRENGTH):
    if strength<=0: return frame
    h,w=frame.shape[:2]; mask=_build_vignette(w,h,strength)
    return (frame.astype(np.float32)*mask).clip(0,255).astype(np.uint8)


# ── ENH-2: Unsharp mask ────────────────────────────────────────────────────────
def apply_sharpen(frame,strength=0.6,radius=1):
    if strength<=0: return frame
    ksize=radius*2+1; blurred=cv2.GaussianBlur(frame,(ksize,ksize),0)
    return cv2.addWeighted(frame,1+strength,blurred,-strength,0)


# ── ENH-3: Color grade LUT ────────────────────────────────────────────────────
_lut_cache: Dict[str,np.ndarray] = {}

def _build_lut(grade):
    if grade in _lut_cache: return _lut_cache[grade]
    x=np.arange(256,dtype=np.float32)
    if grade=="warm":
        r=np.clip(x*1.06+5,0,255); g=np.clip(x*1.02+2,0,255); b=np.clip(x*0.92-4,0,255)
    elif grade=="cool":
        r=np.clip(x*0.92-4,0,255); g=np.clip(x*1.01+1,0,255); b=np.clip(x*1.07+6,0,255)
    elif grade=="vibrant":
        def sc(v): n=v/255; s=n*n*(3-2*n); return np.clip((n*0.6+s*0.4)*255,0,255)
        r=sc(x*1.04); g=sc(x*1.02); b=sc(x)
    elif grade=="matte":
        r=np.clip(x*0.88+18,0,255); g=np.clip(x*0.86+16,0,255); b=np.clip(x*0.84+22,0,255)
    else:
        r=g=b=x.copy()
    lut=np.stack([b,g,r],axis=1).astype(np.uint8).reshape(256,1,3)
    _lut_cache[grade]=lut; return lut

def apply_color_grade(frame,grade="none"):
    if not grade or grade=="none": return frame
    return cv2.LUT(frame,_build_lut(grade))


# ── ENH-6: Ken Burns ──────────────────────────────────────────────────────────
def apply_ken_burns(frame,frame_idx,fps,max_zoom=KEN_BURNS_MAX_ZOOM,period=KEN_BURNS_PERIOD):
    if max_zoom<=1.0: return frame
    t=(frame_idx/max(fps,1))%period
    scale=1.0+(max_zoom-1.0)*0.5*(1-math.cos(2*math.pi*t/period))
    if abs(scale-1.0)<1e-4: return frame
    h,w=frame.shape[:2]; nw=max(int(w/scale),2); nh=max(int(h/scale),2)
    x0=(w-nw)//2; y0=(h-nh)//2
    return cv2.resize(frame[y0:y0+nh,x0:x0+nw],(w,h),interpolation=cv2.INTER_LINEAR)


# ── ENH-7: Cross-dissolve ─────────────────────────────────────────────────────
class DissolveBuffer:
    def __init__(self,n=DISSOLVE_FRAMES):
        self.n=n; self._buf=None; self._rem=0
    def on_cut(self,last_frame):
        self._buf=last_frame.copy(); self._rem=self.n
    def blend(self,new_frame):
        if self._rem<=0 or self._buf is None: return new_frame
        a=self._rem/self.n; self._rem-=1
        return cv2.addWeighted(self._buf,a,new_frame,1.0-a,0)
    @property
    def active(self): return self._rem>0


# ── ENH-8: FFmpeg post-filter ────────────────────────────────────────────────
def _build_ffmpeg_vf(color_grade="none",ffmpeg_sharpen=False):
    filters=[]
    eq_map={"warm":"brightness=0.02:saturation=1.12:gamma_r=1.05:gamma_b=0.95",
            "cool":"brightness=0.01:saturation=1.08:gamma_r=0.95:gamma_b=1.05",
            "vibrant":"brightness=0.0:saturation=1.25:contrast=1.05",
            "matte":"brightness=0.03:saturation=0.85:contrast=0.92"}
    if color_grade in eq_map: filters.append(f"eq={eq_map[color_grade]}")
    if ffmpeg_sharpen: filters.append("unsharp=5:5:0.8:3:3:0.0")
    return filters


# ── FFmpegVideoReader ──────────────────────────────────────────────────────────
class FFmpegVideoReader:
    def __init__(self,path,width,height,seek_sec=0.0,n_frames=None,scale_w=None,scale_h=None):
        self.path=path; self.width=width; self.height=height
        self.seek_sec=seek_sec; self.n_frames=n_frames
        self.out_w=scale_w or width; self.out_h=scale_h or height
        self._proc=None; self._frame_bytes=self.out_w*self.out_h*3; self._leftover=b""

    def _candidate_cmds(self):
        head=["ffmpeg"]
        if self.seek_sec>0: head+=["-ss",str(self.seek_sec)]
        tail=["-i",self.path,"-f","rawvideo","-pix_fmt","bgr24","-vf",f"scale={self.out_w}:{self.out_h}"]
        if self.n_frames is not None: tail+=["-vframes",str(self.n_frames)]
        tail+=["pipe:1"]
        return [head+["-vcodec","libdav1d"]+tail, head+["-hwaccel","none"]+tail]

    def _open(self):
        # FIX-E4: guarantee proc.wait() even when an exception occurs mid-read,
        # and close stdout before waiting to avoid deadlock on stderr backpressure.
        for cmd in self._candidate_cmds():
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=max(self._frame_bytes * 4, 1 << 20),
                )
                test = proc.stdout.read(self._frame_bytes)
                if len(test) == self._frame_bytes:
                    self._proc = proc
                    self._leftover = test
                    return
                # Not enough data — this candidate failed; clean up.
                try: proc.stdout.close()
                except Exception: pass
                proc.wait()
            except Exception:
                if proc is not None:
                    try: proc.stdout.close()
                    except Exception: pass
                    try: proc.wait()
                    except Exception: pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")

    def close(self):
        if self._proc:
            try: self._proc.stdout.close()
            except Exception: pass
            try: self._proc.wait()
            except Exception: pass
            self._proc = None

    def __enter__(self): self._open(); return self
    def __exit__(self,*_): self.close()

    def __iter__(self):
        if not self._proc: self._open()
        buf=self._leftover; self._leftover=b""
        while True:
            needed=self._frame_bytes-len(buf)
            while needed>0:
                chunk=self._proc.stdout.read(needed)
                if not chunk: return
                buf+=chunk; needed-=len(chunk)
            yield np.frombuffer(buf[:self._frame_bytes],dtype=np.uint8).reshape(self.out_h,self.out_w,3)
            buf=buf[self._frame_bytes:]


def _read_frame_at(path,width,height,t_sec,scale_w=None,scale_h=None):
    r=FFmpegVideoReader(path,width,height,seek_sec=t_sec,n_frames=1,scale_w=scale_w,scale_h=scale_h)
    r._open(); frames=list(r); r.close()
    return frames[0] if frames else None


# ── FFmpeg helpers ────────────────────────────────────────────────────────────
def _check_ffmpeg():
    for t in ("ffmpeg","ffprobe"):
        try: subprocess.run([t,"-version"],check=True,capture_output=True)
        except (subprocess.CalledProcessError,FileNotFoundError):
            raise ProcessingError(f"{t} not found. Install FFmpeg.")

def _has_audio(path):
    try:
        r=subprocess.run(["ffprobe","-v","error","-select_streams","a",
                           "-show_entries","stream=codec_type","-of","csv=p=0",path],
                          capture_output=True,text=True,timeout=15)
        return "audio" in r.stdout
    except Exception: return False

def _extract_audio_wav(vpath,wpath):
    r=subprocess.run(["ffmpeg","-y","-i",vpath,"-ar","16000","-ac","1","-f","wav",wpath],capture_output=True)
    return r.returncode==0 and os.path.exists(wpath)

def _trim_video(inp,out,start,end):
    r=subprocess.run(["ffmpeg","-y","-hwaccel","none","-ss",str(start),"-to",str(end),"-i",inp,
                       "-c:v","libx264","-preset","ultrafast","-crf","18","-c:a","aac","-b:a","128k",
                       "-avoid_negative_ts","make_zero","-reset_timestamps","1",out],capture_output=True)
    return r.returncode==0 and os.path.exists(out)


# ── Encoder ────────────────────────────────────────────────────────────────────
def _open_ffmpeg_encoder(output_path,width,height,fps,audio_source,crf=23,preset="fast",
                          audio_bitrate="128k",subtitle_path=None,subtitle_style=None,
                          extra_vf=None):
    cmd=["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","bgr24",
         "-s",f"{width}x{height}","-r",str(fps),"-i","pipe:0"]
    has_aud=audio_source and _has_audio(audio_source)
    if has_aud: cmd+=["-hwaccel","none","-i",audio_source]
    vf=[]
    if subtitle_path and os.path.exists(subtitle_path):
        s=subtitle_style or SUBTITLE_STYLES["Bold White (TikTok)"]
        sesc=subtitle_path.replace("\\","/").replace(":","\\:")
        force=(f"Fontsize={s.get('fontsize',18)},PrimaryColour={s.get('primary_color','&H00FFFFFF')},"
               f"OutlineColour={s.get('outline_color','&H00000000')},Outline={s.get('outline',2)},"
               f"Bold={s.get('bold',1)},Shadow={s.get('shadow',0)},BackColour={s.get('back_color','&H00000000')},"
               f"MarginV={s.get('margin_v',80)},Alignment=2")
        vf.append(f"subtitles='{sesc}':force_style='{force}'")
    if extra_vf: vf.extend(extra_vf)
    cmd+=["-map","0:v:0"]
    if has_aud: cmd+=["-map","1:a:0?","-c:a","aac","-b:a",audio_bitrate,"-ac","2"]
    else: cmd+=["-an"]
    if vf: cmd+=["-vf",",".join(vf)]
    cmd+=["-c:v","libx264","-preset",preset,"-crf",str(crf),
          "-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
          "-shortest","-movflags","+faststart",output_path]
    # FIX-E5: keep stderr as PIPE so we can read it before wait() on failure.
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)

def _close_ffmpeg_encoder(proc, output_path):
    try: proc.stdin.close()
    except Exception: pass
    # FIX-E5: read stderr BEFORE wait() — the pipe may be flushed/closed by wait().
    try: err = proc.stderr.read(4000).decode(errors="replace")
    except Exception: err = ""
    proc.wait()
    if proc.returncode != 0:
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")


# ── Video metadata ────────────────────────────────────────────────────────────
def get_video_info(path):
    cmd=["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=width,height,r_frame_rate,nb_frames",
         "-show_entries","format=duration","-of","default=noprint_wrappers=1",path]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)

    # FIX-E1: build the full kv dict FIRST, then extract values.
    # The original code computed w/h/fps/dur inside the loop body on every
    # iteration, so the final values depended on whichever line happened to be
    # last — producing garbage dimensions for most real files.
    kv: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    w   = int(kv.get("width",  0) or 0)
    h   = int(kv.get("height", 0) or 0)
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
    return {"fps": fps, "total_frames": min(int(dur * fps), MAX_FRAMES_GUARD),
            "width": w, "height": h, "duration_seconds": dur, "is_landscape": w > h}

def extract_thumbnail(path,t=1.0):
    info=get_video_info(path)
    frame=_read_frame_at(path,info["width"],info["height"],t,scale_w=320,scale_h=180)
    if frame is None: return None
    ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
    return buf.tobytes() if ok else None


# ── Resolution helpers ────────────────────────────────────────────────────────
def resolve_target_size(label,orig_w,orig_h):
    tw,th=RESOLUTION_PRESETS.get(label,(0,0))
    if tw==0 and th==0:
        cw=int(orig_h*9/16)
        if cw>orig_w: cw=orig_w; ch=int(cw*16/9)
        else: ch=orig_h
        return cw-(cw%2),ch-(ch%2)
    if th>orig_h: scale=orig_h/th; tw=int(tw*scale); th=int(orig_h)
    if tw>orig_w: scale=orig_w/tw; tw=int(orig_w); th=int(th*scale)
    return max(tw-(tw%2),2),max(th-(th%2),2)

def calculate_crop_dims(orig_w,orig_h,tw,th):
    ratio=tw/th
    if (orig_w/orig_h)>ratio: ch=orig_h; cw=int(round(ch*ratio))
    else: cw=orig_w; ch=int(round(cw/ratio))
    return min(cw,orig_w),min(ch,orig_h)


# ── YOLO ──────────────────────────────────────────────────────────────────────
_model_cache: Dict[str,Any]={}

def _get_model(weights="yolov8n.pt"):
    if not _YOLO_AVAILABLE: return None
    if weights in _model_cache: return _model_cache[weights]
    try: m=_YOLO(weights); _model_cache[weights]=m; return m
    except Exception as e: print(f"YOLO unavailable: {e}",file=sys.stderr); return None


# ── Face detection ────────────────────────────────────────────────────────────
_face_net=None; _haar_cascade=None
_FACE_PROTO="deploy.prototxt"; _FACE_MODEL="res10_300x300_ssd_iter_140000.caffemodel"

def _load_face_net():
    global _face_net
    if _face_net: return _face_net
    if os.path.exists(_FACE_PROTO) and os.path.exists(_FACE_MODEL):
        try: _face_net=cv2.dnn.readNetFromCaffe(_FACE_PROTO,_FACE_MODEL); return _face_net
        except Exception: pass
    return None

def _get_haar():
    global _haar_cascade
    if _haar_cascade: return _haar_cascade
    p=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    if os.path.exists(p):
        c=cv2.CascadeClassifier(p)
        if not c.empty(): _haar_cascade=c; return c
    return None

def detect_faces(frame,confidence_thresh=0.6):
    h,w=frame.shape[:2]; net=_load_face_net()
    if net:
        blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,123))
        net.setInput(blob); dets=net.forward(); faces=[]
        for i in range(dets.shape[2]):
            if float(dets[0,0,i,2])<confidence_thresh: continue
            x1,y1=max(0,int(dets[0,0,i,3]*w)),max(0,int(dets[0,0,i,4]*h))
            x2,y2=min(w,int(dets[0,0,i,5]*w)),min(h,int(dets[0,0,i,6]*h))
            if x2>x1 and y2>y1: faces.append((x1,y1,x2,y2))
        faces.sort(key=lambda f:(f[2]-f[0])*(f[3]-f[1]),reverse=True); return faces
    haar=_get_haar()
    if not haar: return []
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    raw=haar.detectMultiScale(gray,1.1,5,minSize=(max(30,w//20),max(30,h//20)))
    if len(raw)==0: return []
    faces2=[(x,y,x+bw,y+bh) for x,y,bw,bh in raw]
    faces2.sort(key=lambda f:(f[2]-f[0])*(f[3]-f[1]),reverse=True); return faces2


# ── Subject detection ─────────────────────────────────────────────────────────
DetectionResult=namedtuple("DetectionResult",["cx","cy","ux1","uy1","ux2","uy2","count"])

def detect_subjects(frame,model,confidence=0.45):
    if model is None: return None
    try: results=model(frame,verbose=False,conf=confidence)[0]
    except Exception as e: print(f"det err: {e}",file=sys.stderr); return None
    if results.boxes is None or len(results.boxes)==0: return None
    pp=[]; hp=[]; ap=[]
    for box in results.boxes:
        cls=int(box.cls[0]); conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        w_=(x2-x1)*conf; e=(w_,x1,y1,x2,y2)
        if cls==PERSON_CLASS_ID: pp.append(e)
        elif cls in HIGH_PRIO_CLASSES: hp.append(e)
        ap.append(e)
    pool=pp or hp or ap
    if not pool: return None
    tw=sum(e[0] for e in pool)
    if tw==0: return None
    cx=int(sum(e[0]*(e[1]+e[3])/2 for e in pool)/tw)
    cy=int(sum(e[0]*(e[2]+e[4])/2 for e in pool)/tw)
    return DetectionResult(cx,cy,min(e[1] for e in pool),min(e[2] for e in pool),
                            max(e[3] for e in pool),max(e[4] for e in pool),len(pool))

def _detect_pillarbox(frame, blur_thresh=8.0, min_pad_frac=0.08):
    """
    FIX-D3: Detect if the frame has blurred/darkened pillarbox padding.
    Returns (x1, y1, x2, y2) inner active region, or None if no pillarbox found.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sharpness(region):
        if region.size == 0: return 0.0
        return float(cv2.Laplacian(region, cv2.CV_64F).var())

    pad = max(int(w * min_pad_frac), 8)
    centre_sharp = sharpness(gray[h//4:3*h//4, w//4:3*w//4])
    left_sharp   = sharpness(gray[h//4:3*h//4, :pad])
    right_sharp  = sharpness(gray[h//4:3*h//4, w-pad:])

    if centre_sharp < 1.0: return None
    left_ratio  = left_sharp  / centre_sharp
    right_ratio = right_sharp / centre_sharp

    if left_ratio < 0.15 and right_ratio < 0.15:
        col_sharp = np.array([float(cv2.Laplacian(gray[:, c:c+4], cv2.CV_64F).var())
                               for c in range(0, w, 4)])
        threshold = centre_sharp * 0.25
        active_cols = np.where(col_sharp > threshold)[0] * 4
        if len(active_cols) < 2: return None
        x1 = max(0, int(active_cols[0]) - 4)
        x2 = min(w, int(active_cols[-1]) + 8)
        if (x2 - x1) < w * 0.4: return None
        if x1 <= 4 and x2 >= w - 4: return None
        return (x1, 0, x2, h)
    return None


# FIX-E6: Proper bounded pillarbox cache — evict oldest entry when full.
class _BoundedCache:
    def __init__(self, maxsize: int):
        self._data: Dict = {}
        self._maxsize = maxsize

    def get(self, key, sentinel=object()):
        return self._data.get(key, sentinel)

    def set(self, key, value):
        if key not in self._data and len(self._data) >= self._maxsize:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value

    def clear(self):
        self._data.clear()

_pillarbox_cache = _BoundedCache(_PILLARBOX_CACHE_MAX)

def _get_active_region(frame):
    """Return (x1,y1,x2,y2) of active content area, caching per frame shape+edge-hash."""
    h, w = frame.shape[:2]
    edge_hash = hash(frame[:4, :20].tobytes() + frame[:4, -20:].tobytes())
    key = (w, h, edge_hash)
    _SENTINEL = object()
    cached = _pillarbox_cache.get(key, _SENTINEL)
    if cached is _SENTINEL:
        result = _detect_pillarbox(frame)
        _pillarbox_cache.set(key, result)
        return result
    return cached


def detect_persons_all(frame, model, confidence=0.45):
    """FIX-D3: Crops to inner active region before detection."""
    if model is None: return []
    region = _get_active_region(frame)
    if region is not None:
        rx1, ry1, rx2, ry2 = region
        det_frame = frame[ry1:ry2, rx1:rx2]
        ox, oy = rx1, ry1
    else:
        det_frame = frame
        ox, oy = 0, 0
    try: results = model(det_frame, verbose=False, conf=confidence)[0]
    except Exception: return []
    if results.boxes is None or len(results.boxes) == 0: return []
    p = []
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            p.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    p.sort(key=lambda b: b[0])
    return p


# ── Framing helpers ───────────────────────────────────────────────────────────
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

def talking_head_center(faces,orig_w,orig_h,crop_w,crop_h,bias=0.30):
    if not faces: return None
    ux1=min(f[0] for f in faces); uy1=min(f[1] for f in faces)
    ux2=max(f[2] for f in faces); uy2=max(f[3] for f in faces)
    face_cx=(ux1+ux2)//2; face_cy=(uy1+uy2)//2
    cy=int(face_cy*(1-bias)+(face_cy+crop_h//6)*bias)
    hw,hh=crop_w//2,crop_h//2
    cx=max(hw,min(face_cx,orig_w-hw)); cy=max(hh,min(cy,orig_h-hh))
    cy=_apply_lower_third_guard(cy,crop_h,face_cy,orig_h)
    return cx,max(hh,min(cy,orig_h-hh))


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _group_union(persons):
    if not persons: return (0,0,0,0)
    return (min(p[0] for p in persons), min(p[1] for p in persons),
            max(p[2] for p in persons), max(p[3] for p in persons))

def _filter_persons(persons, fw, fh, min_w_frac=0.06, min_h_frac=0.18):
    """
    FIX-D1: Stricter thresholds + two spatial guards.
    """
    edge_guard  = fw * 0.08
    lower_guard = fh * 0.80
    out = []
    for x1, y1, x2, y2 in persons:
        if (x2 - x1) < fw * min_w_frac: continue
        if (y2 - y1) < fh * min_h_frac: continue
        cx = (x1 + x2) / 2
        if cx < edge_guard or cx > fw - edge_guard: continue
        if y1 > lower_guard: continue
        out.append((x1, y1, x2, y2))
    return out

def _should_split(p0, p1, fw):
    x10,_,x20,_ = p0
    x11,_,x21,_ = p1
    overlap = max(0, min(x20, x21) - max(x10, x11))
    w0 = max(x20 - x10, 1)
    w1 = max(x21 - x11, 1)
    min_w = min(w0, w1)
    overlap_ratio = overlap / min_w
    cx0 = (x10 + x20) / 2
    cx1 = (x11 + x21) / 2
    dist_ratio = abs(cx1 - cx0) / fw
    if (x20 - x10) < fw * 0.10 or (x21 - x11) < fw * 0.10:
        return False
    return overlap_ratio < 0.10 and dist_ratio > 0.70

def _classify_layout(persons, fw, fh):
    n = len(persons)
    if n == 0 or n == 1:
        return LAYOUT_SINGLE
    if n >= 4:
        return LAYOUT_WIDE
    persons_s = sorted(persons, key=lambda p: (p[0]+p[2])//2)
    if n == 2:
        if _should_split(persons_s[0], persons_s[1], fw):
            return LAYOUT_DUO_SPLIT
        return LAYOUT_DUO_WIDE
    if n == 3:
        ux1 = min(p[0] for p in persons_s)
        ux2 = max(p[2] for p in persons_s)
        if (ux2 - ux1) > fw * 0.80:
            return LAYOUT_WIDE
        return LAYOUT_TRIO
    return LAYOUT_SINGLE


def _probe_dominant_layout(input_path, model, fps, total_frames,
                            orig_w, orig_h, confidence=0.45, n_probe=24):
    """FIX-C1: Requires ≥60% supermajority before committing to multi-person layout."""
    if model is None: return LAYOUT_SINGLE
    sw = 640; sh = max(1, int(640 * orig_h / orig_w))
    sx = orig_w / sw; sy = orig_h / sh
    dur = total_frames / max(fps, 1)
    probe_ts = np.linspace(max(2.0, dur * 0.05), max(3.0, dur * 0.95), n_probe)
    counts = {LAYOUT_SINGLE: 0, LAYOUT_DUO_SPLIT: 0, LAYOUT_DUO_WIDE: 0, LAYOUT_TRIO: 0, LAYOUT_WIDE: 0}

    for t in probe_ts:
        frame = _read_frame_at(input_path, orig_w, orig_h, t, scale_w=sw, scale_h=sh)
        if frame is None: continue
        raw = detect_persons_all(frame, model, confidence)
        raw = _filter_persons(raw, sw, sh)
        persons_full = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)) for x1, y1, x2, y2 in raw]
        layout = _classify_layout(persons_full, orig_w, orig_h)
        counts[layout] += 1

    total_probed = max(sum(counts.values()), 1)
    best_multi = max((l for l in counts if l != LAYOUT_SINGLE), key=counts.get)
    if counts[best_multi] / total_probed >= 0.60:
        return best_multi
    return LAYOUT_SINGLE


# ── Per-strip tight crop ───────────────────────────────────────────────────────
def _tight_crop_for_group(frame, group, out_w, out_h,
                           expand=PANEL_CROP_EXPAND,
                           vignette_strength=0.0, color_grade="none"):
    fh, fw = frame.shape[:2]
    ratio = out_w / out_h

    if not group:
        placeholder = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        placeholder = cv2.GaussianBlur(placeholder, (31, 31), 0)
        placeholder = (placeholder * 0.25).astype(np.uint8)
        return placeholder
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        ph = max(uy2 - uy1, 1)
        pw = max(ux2 - ux1, 1)
        ucx = (ux1 + ux2) // 2
        ucy = (uy1 + uy2) // 2

        src_h = int(ph * expand)
        src_w = int(src_h * ratio)
        max_src_w = int(pw * 1.9)
        if src_w > max_src_w > 4:
            src_w = max_src_w
            src_h = int(src_w / ratio)

        src_h = max(min(src_h, fh), 4)
        src_w = max(min(src_w, fw), 4)

        if src_w / max(src_h, 1) > ratio:
            src_h = max(int(src_w / ratio), 4)
        else:
            src_w = max(int(src_h * ratio), 4)
        src_h = min(src_h, fh); src_w = min(src_w, fw)

        head_bias = int(src_h * 0.08)
        x0 = max(0, min(int(ucx - src_w / 2), fw - src_w))
        y0 = max(0, min(int(ucy - head_bias - src_h / 2), fh - src_h))

    x1 = min(x0 + src_w, fw); y1 = min(y0 + src_h, fh)
    x0 = max(0, x1 - src_w);  y0 = max(0, y1 - src_h)

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0: crop = frame
    result = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    if color_grade and color_grade != "none": result = apply_color_grade(result, color_grade)
    if vignette_strength > 0: result = apply_vignette(result, vignette_strength)
    return result


def _wide_crop_for_group(frame, group, out_w, out_h,
                          vignette_strength=0.0, color_grade="none"):
    fh, fw = frame.shape[:2]
    ratio = out_w / out_h

    if not group:
        ucx, ucy = fw // 2, fh // 2
        pw, ph   = fw, fh
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        ucx = (ux1 + ux2) // 2
        ucy = (uy1 + uy2) // 2
        pw  = max(ux2 - ux1, 1)
        ph  = max(uy2 - uy1, 1)

    src_h = min(int(ph * 1.35), fh)
    src_w = min(int(src_h * ratio), fw)
    if src_w / max(src_h, 1) < ratio:
        src_h = max(int(src_w / ratio), 4)

    src_h = max(min(src_h, fh), 4)
    src_w = max(min(src_w, fw), 4)

    head_bias = int(src_h * 0.06)
    x0 = max(0, min(int(ucx - src_w / 2), fw - src_w))
    y0 = max(0, min(int(ucy - head_bias - src_h / 2), fh - src_h))
    x1 = min(x0 + src_w, fw); y1 = min(y0 + src_h, fh)
    x0 = max(0, x1 - src_w);  y0 = max(0, y1 - src_h)

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0: crop = frame
    result = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    if color_grade and color_grade != "none": result = apply_color_grade(result, color_grade)
    if vignette_strength > 0: result = apply_vignette(result, vignette_strength)
    return result


def _assemble_strips(strips, out_w, out_h):
    div = PANEL_DIVIDER_PX
    n   = len(strips)
    total_div = div * (n - 1)
    avail_h   = out_h - total_div
    heights   = [avail_h // n] * n
    heights[-1] += avail_h - sum(heights)
    heights = [(h & ~1) for h in heights]

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    y = 0
    for i, (strip, h) in enumerate(zip(strips, heights)):
        if strip.shape[0] != h or strip.shape[1] != out_w:
            strip = cv2.resize(strip, (out_w, h), interpolation=cv2.INTER_LINEAR)
        canvas[y:y+h, :] = strip
        y += h
        if i < n - 1:
            canvas[y:y+div, :] = PANEL_DIVIDER_COLOR
            y += div
    if y < out_h:
        canvas[y:, :] = PANEL_DIVIDER_COLOR
    return canvas


# ── Stability Engine ──────────────────────────────────────────────────────────

class TargetSmoother:
    def __init__(self, alpha=0.2):
        self.cx = None; self.cy = None; self.alpha = alpha

    def update(self, x, y):
        if self.cx is None:
            self.cx, self.cy = float(x), float(y)
        else:
            a = self.alpha
            self.cx = self.cx*(1-a) + x*a
            self.cy = self.cy*(1-a) + y*a
        return int(self.cx), int(self.cy)

class CropAnchor:
    def __init__(self, alpha=0.15, max_step=8):
        self._cx = None; self._cy = None
        self.alpha = alpha; self.max_step = max_step

    def update(self, tcx, tcy):
        if self._cx is None:
            self._cx, self._cy = float(tcx), float(tcy)
            return int(tcx), int(tcy)
        dx = max(-self.max_step, min(self.max_step, tcx - self._cx))
        dy = max(-self.max_step, min(self.max_step, tcy - self._cy))
        self._cx += dx * self.alpha
        self._cy += dy * self.alpha
        return int(self._cx), int(self._cy)


class StablePanelSmoother:
    def __init__(self, max_slots=3, alpha=PANEL_SLOT_EMA):
        self.alpha = alpha
        self._slots = [None] * max_slots
        self._last_n = 0

    def _ema(self, prev, new_val):
        if prev is None: return new_val
        a = self.alpha
        return prev*(1-a) + new_val*a

    def smooth(self, groups):
        out = []
        n = len(groups)
        if n != self._last_n:
            if n > self._last_n:
                for i in range(self._last_n, min(n, len(self._slots))):
                    self._slots[i] = None
            else:
                for i in range(n, self._last_n):
                    if i < len(self._slots):
                        self._slots[i] = None
        self._last_n = n

        for i, group in enumerate(groups):
            if i >= len(self._slots): out.append(group); continue
            if not group: out.append([]); continue
            u = _group_union(group)
            ucx = (u[0]+u[2])/2; ucy = (u[1]+u[3])/2
            uw  = u[2]-u[0];     uh  = u[3]-u[1]
            if self._slots[i] is None:
                self._slots[i] = (ucx, ucy, uw, uh)
            else:
                pcx, pcy, pw, ph = self._slots[i]
                self._slots[i] = (self._ema(pcx, ucx), self._ema(pcy, ucy),
                                   self._ema(pw, uw),   self._ema(ph, uh))
            scx, scy, sw, sh = self._slots[i]
            out.append([(int(scx-sw/2), int(scy-sh/2), int(scx+sw/2), int(scy+sh/2))])
        return out


class LayoutState:
    def __init__(self):
        self.current   = LAYOUT_SINGLE
        self._locked   = 0
        self.smoother  = StablePanelSmoother(max_slots=3)
        self.prev_groups = []

    def update(self, proposed, frame_idx, n_persons=None):
        if n_persons is not None and n_persons <= 1:
            if self.current in (LAYOUT_DUO_SPLIT, LAYOUT_TRIO, LAYOUT_WIDE, LAYOUT_DUO_WIDE):
                self.current = LAYOUT_SINGLE
                self._locked = 0
                return LAYOUT_SINGLE
        if self._locked > 0:
            self._locked -= 1
            return self.current
        if proposed != self.current:
            self._locked = LAYOUT_HYSTERESIS_FRAMES
            self.current = proposed
        return self.current


def _get_active_speaker_index(scores, prev_idx, history_exists):
    if not history_exists:
        return int(np.argmax(scores))
    max_score = max(scores)
    prev_score = scores[prev_idx]
    if prev_score > max_score * 0.85:
        return prev_idx
    return int(np.argmax(scores))


def render_adaptive_frame(frame, persons_full, out_w, out_h,
                           layout_state,
                           vignette_strength=VIGNETTE_STRENGTH*0.7,
                           color_grade="none",
                           frame_idx=0):
    # FIX-E3: frame.shape is (h, w, c) — was incorrectly unpacked as (fw, fh).
    fh, fw = frame.shape[:2]
    persons_full = _filter_persons(persons_full, fw, fh)

    proposed = _classify_layout(persons_full, fw, fh)
    layout   = layout_state.update(proposed, frame_idx, n_persons=len(persons_full))

    persons_s = sorted(persons_full, key=lambda p: (p[0]+p[2])//2)

    if not persons_s and layout_state.prev_groups:
        persons_s = [b for g in layout_state.prev_groups for b in g]

    if layout == LAYOUT_DUO_SPLIT and len(persons_s) < 2:
        layout = LAYOUT_SINGLE
    if layout == LAYOUT_TRIO and len(persons_s) < 2:
        layout = LAYOUT_SINGLE

    if layout == LAYOUT_SINGLE or len(persons_s) == 0:
        groups = [persons_s] if persons_s else [[]]
    elif layout == LAYOUT_DUO_SPLIT:
        split = len(persons_s) // 2
        for k in range(1, len(persons_s)):
            if _should_split(persons_s[k-1], persons_s[k], fw):
                split = k; break
        groups = [persons_s[:split], persons_s[split:]]
    elif layout == LAYOUT_DUO_WIDE:
        groups = [persons_s]
    elif layout == LAYOUT_TRIO:
        main_idx = max(range(len(persons_s)),
                       key=lambda i: (persons_s[i][2]-persons_s[i][0])*(persons_s[i][3]-persons_s[i][1]))
        main  = [persons_s[main_idx]]
        rest  = sorted([p for i, p in enumerate(persons_s) if i != main_idx],
                       key=lambda p: (p[0]+p[2])//2)
        mid   = len(rest) // 2 or 1
        groups = [main, rest[:mid], rest[mid:]]
    else:
        groups = [persons_s]

    groups = layout_state.smoother.smooth(groups)
    layout_state.prev_groups = groups

    # FIX-E2: pass vignette/color_grade down into crop helpers — do NOT apply
    # them again in the outer loop.  kw is used by crop helpers only.
    kw = dict(vignette_strength=vignette_strength, color_grade=color_grade)

    if layout == LAYOUT_SINGLE:
        out_frame = _wide_crop_for_group(frame, groups[0], out_w, out_h, **kw)

    elif layout == LAYOUT_DUO_SPLIT:
        div    = PANEL_DIVIDER_PX
        sh_top = ((out_h - div) // 2) & ~1
        sh_bot = out_h - sh_top - div
        top = _tight_crop_for_group(frame, groups[0], out_w, sh_top, **kw)
        bot = _tight_crop_for_group(frame, groups[1], out_w, sh_bot, **kw)
        out_frame = _assemble_strips([top, bot], out_w, out_h)

    elif layout == LAYOUT_TRIO:
        div      = PANEL_DIVIDER_PX
        sh_main  = int((out_h - 2*div) * 0.60) & ~1
        sh_side  = ((out_h - sh_main - 2*div) // 2) & ~1
        main_strip = _tight_crop_for_group(frame, groups[0], out_w, sh_main, **kw)
        side_l     = _tight_crop_for_group(frame, groups[1] if len(groups)>1 else [],
                                            out_w, sh_side, **kw)
        side_r     = _tight_crop_for_group(frame, groups[2] if len(groups)>2 else [],
                                            out_w, sh_side, **kw)
        bottom_w = out_w // 2
        sl = cv2.resize(side_l, (bottom_w, sh_side), interpolation=cv2.INTER_LINEAR)
        sr = cv2.resize(side_r, (out_w - bottom_w, sh_side), interpolation=cv2.INTER_LINEAR)
        bottom_row = np.concatenate([sl, sr], axis=1)
        canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
        y = 0
        canvas[y:y+sh_main, :] = main_strip;  y += sh_main
        canvas[y:y+div, :]     = PANEL_DIVIDER_COLOR; y += div
        canvas[y:y+sh_side, :] = bottom_row;  y += sh_side
        if y < out_h: canvas[y:, :] = PANEL_DIVIDER_COLOR
        out_frame = canvas

    else:
        out_frame = _wide_crop_for_group(frame, groups[0], out_w, out_h, **kw)

    return out_frame, layout


# ── Optical flow / saliency ───────────────────────────────────────────────────
def optical_flow_center(prev,curr,w,h):
    if prev is None or curr is None: return None
    try:
        flow=cv2.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)
        mag=np.sqrt(flow[...,0]**2+flow[...,1]**2)
        b=max(1,int(w*0.04)); mag[:,:b]=mag[:,w-b:]=mag[:b,:]=mag[h-b:,:]=0
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


# ── Smooth camera path ────────────────────────────────────────────────────────
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
            if i==0: m[i]=v[1]-v[0] if len(v)>1 else 0.0
            elif i==len(v)-1: m[i]=v[-1]-v[-2]
            else: m[i]=0.5*(v[i+1]-v[i-1])
        return m
    mx=_tan(xs); my=_tan(ys); result=[]
    for fi in range(total):
        if fi<=indices[0]: result.append(centers[0]); continue
        if fi>=indices[-1]: result.append(centers[-1]); continue
        r=bisect.bisect_right(indices,fi); l=r-1
        if r>=n: result.append(centers[-1]); continue
        span=max(indices[r]-indices[l],1); t=(fi-indices[l])/span
        result.append((int(_cubic_hermite(xs[l],xs[r],mx[l]*span,mx[r]*span,t)),
                       int(_cubic_hermite(ys[l],ys[r],my[l]*span,my[r]*span,t))))
    while len(result)<total: result.append(result[-1] if result else (0,0))
    return result[:total]

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


# ── Whisper / translate ───────────────────────────────────────────────────────
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
    import whisper as _w; _p(0.0,"Extracting audio...")
    wav_fd,wav_path=tempfile.mkstemp(suffix=".wav"); os.close(wav_fd)
    try:
        if not _extract_audio_wav(video_path,wav_path): return False
        _p(0.2,f"Transcribing ({whisper_model})...")
        model=_w.load_model(whisper_model)
        opts={"word_timestamps":True,"verbose":False}
        if language: opts["language"]=language
        result=model.transcribe(wav_path,**opts)
        _p(0.85,"Writing subtitles..."); lines=[]; idx=1; words=[]
        for seg in result.get("segments",[]):
            for w_ in seg.get("words",[]): words.append({"word":w_["word"].strip(),"start":w_["start"],"end":w_["end"]})
        buf=[]; buf_len=0
        def flush():
            nonlocal idx,buf,buf_len
            if not buf: return
            lines.append(f"{idx}\n{_seconds_to_srt_time(buf[0]['start'])} --> {_seconds_to_srt_time(buf[-1]['end'])}\n{' '.join(x['word'] for x in buf)}\n")
            idx+=1; buf=[]; buf_len=0
        for w_ in words:
            wl=len(w_["word"])+1
            if buf_len+wl>max_chars_per_line and buf: flush()
            buf.append(w_); buf_len+=wl
        flush()
        with open(srt_path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
        _p(1.0,f"{len(lines)} subtitle lines"); return True
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
        blocks=re.split(r"\n\n+",content.strip()); out=[]
        tr=GoogleTranslator(source=source_language,target=target_language)
        for i,block in enumerate(blocks):
            ls=block.strip().splitlines()
            if len(ls)<3: out.append(block); continue
            try: translated=tr.translate(" ".join(ls[2:])) or " ".join(ls[2:])
            except Exception: translated=" ".join(ls[2:])
            out.append(f"{ls[0]}\n{ls[1]}\n{translated}")
            if i%10==0: _p(i/max(len(blocks),1),f"{i}/{len(blocks)}")
        with open(srt_path,"w",encoding="utf-8") as f: f.write("\n\n".join(out)+"\n")
        _p(1.0,"Translation done"); return True
    except Exception as e: print(f"Translation failed: {e}",file=sys.stderr); return False


# ═══════════════════════════════════════════════════════════════════════════════
# Clip boundary refinement
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_motion_profile(input_path, fps, total_frames, orig_w, orig_h,
                             sample_every=None, scale_w=320):
    if sample_every is None: sample_every = max(1, int(fps / 2))
    sh = max(1, int(scale_w * orig_h / orig_w))
    scores = {}
    prev_gray = None
    fi = 0
    with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=scale_w, scale_h=sh) as rdr:
        for frame in rdr:
            if fi > total_frames: break
            if fi % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    scores[fi] = float(cv2.absdiff(prev_gray, gray).mean()) / 255.0
                else:
                    scores[fi] = 0.0
                prev_gray = gray
            fi += 1
    return scores

def _find_nearest_scene_cut(scene_cuts_sec, target_sec, search_window_sec):
    best = None; best_dist = float("inf")
    for t in scene_cuts_sec:
        d = abs(t - target_sec)
        if d <= search_window_sec and d < best_dist:
            best_dist = d; best = t
    return best

def _find_low_motion_valley(motion_profile_sec, target_sec, search_window_sec, fps):
    candidates = {t: s for t, s in motion_profile_sec.items()
                  if abs(t - target_sec) <= search_window_sec}
    if not candidates: return target_sec
    return min(candidates, key=candidates.get)

def _refine_clip_boundaries(start_sec, end_sec, duration,
                             scene_cuts_sec, motion_profile_sec,
                             min_duration, max_duration,
                             search_window=CLIP_BOUNDARY_SEARCH_SEC,
                             preroll=CLIP_PREROLL_PAD,
                             postroll=CLIP_POSTROLL_PAD):
    cut_start = _find_nearest_scene_cut(scene_cuts_sec, start_sec, search_window)
    if cut_start is not None: new_start = max(0.0, cut_start + preroll)
    else:
        valley = _find_low_motion_valley(motion_profile_sec, start_sec, search_window, fps=1)
        new_start = max(0.0, valley - preroll * 0.5)

    cut_end = _find_nearest_scene_cut(scene_cuts_sec, end_sec, search_window)
    if cut_end is not None: new_end = min(duration, cut_end - postroll)
    else:
        valley = _find_low_motion_valley(motion_profile_sec, end_sec, search_window, fps=1)
        new_end = min(duration, valley + postroll * 0.5)

    new_dur = new_end - new_start
    if new_dur < min_duration:
        deficit = min_duration - new_dur
        new_start = max(0.0, new_start - deficit / 2)
        new_end   = min(duration, new_start + min_duration)
        new_start = max(0.0, new_end - min_duration)
    if new_dur > max_duration:
        centre = (new_start + new_end) / 2
        new_start = max(0.0, centre - max_duration / 2)
        new_end   = min(duration, new_start + max_duration)
    return new_start, new_end


# ── Clip detection ─────────────────────────────────────────────────────────────
def _frame_saliency_score(frame,prev_frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    lap_score=min(float(cv2.Laplacian(gray,cv2.CV_64F).var())/3000.0,1.0)
    motion_score=0.0
    if prev_frame is not None:
        motion_score=min(float(cv2.absdiff(gray,cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)).mean())/30.0,1.0)
    sat_score=min(float(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1].mean())/128.0,1.0)
    return 0.4*motion_score+0.4*lap_score+0.2*sat_score

def _compute_frame_scores(input_path,fps,total_frames,orig_w,orig_h,sample_every=15,progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    scores=[]; scene_cuts=[]; prev_gray=None; prev_frame=None
    sw=min(orig_w,640); sh=max(1,int(sw*orig_h/orig_w)); report_n=max(1,total_frames//20); fi=0
    with FFmpegVideoReader(input_path,orig_w,orig_h,scale_w=sw,scale_h=sh) as reader:
        for frame in reader:
            if fi>=total_frames: break
            if fi%sample_every==0:
                cg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and float(cv2.absdiff(prev_gray,cg).mean())/255.0>0.30: scene_cuts.append(fi)
                scores.append(_frame_saliency_score(frame,prev_frame)); prev_gray=cg; prev_frame=frame.copy()
            if fi%report_n==0: _p(fi/total_frames,f"Scanning {fi}/{total_frames}...")
            fi+=1
    return np.array(scores,dtype=float),scene_cuts

def detect_clips(input_path,min_duration_sec=25.0,max_duration_sec=65.0,
                  target_n_clips=10,model=None,confidence=0.45,progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    info=get_video_info(input_path); fps,total_frames=info["fps"],info["total_frames"]
    duration=info["duration_seconds"]; orig_w,orig_h=info["width"],info["height"]
    sample_every=max(1,int(fps)); _p(0.0,"Scanning...")
    scores,scene_cuts_frames=_compute_frame_scores(input_path,fps,total_frames,orig_w,orig_h,
        sample_every=sample_every,progress_callback=lambda v,m:_p(v*0.45,m))
    if len(scores)==0: return []

    motion_profile_sec = {fi * sample_every / fps: float(scores[fi]) for fi in range(len(scores))}
    scene_cuts_sec = [sc / fps for sc in scene_cuts_frames]

    _p(0.45,"Computing arcs..."); window=max(5,int(30/(sample_every/fps)))
    ss=(np.convolve(scores,np.ones(window)/window,mode="same") if len(scores)>=window else scores.copy())
    if ss.max()>0: ss=ss/ss.max()
    min_gap=max(1,int(min_duration_sec*fps/sample_every)); peaks=[]
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
        if not any(min(e,ce)-max(s,cs)>min_duration_sec*0.5 for cs,ce,_ in cands): cands.append((s,e,sc))
    cands.sort(key=lambda x:x[2],reverse=True); cands=cands[:target_n_clips]; cands.sort(key=lambda x:x[0])

    _p(0.55,"Refining boundaries + SOI per clip..."); segments=[]
    for ci,(ss2,se,score) in enumerate(cands):
        _p(0.55+0.35*(ci/max(len(cands),1)),f"Clip {ci+1}/{len(cands)}...")
        refined_start, refined_end = _refine_clip_boundaries(
            ss2, se, duration, scene_cuts_sec, motion_profile_sec,
            min_duration=min_duration_sec, max_duration=max_duration_sec,
        )
        soi_xs=[]; soi_ys=[]; n_s=min(8,max(2,int(refined_end-refined_start)))
        for t in np.linspace(refined_start+1,refined_end-1,n_s):
            frame=_read_frame_at(input_path,orig_w,orig_h,t,scale_w=640,scale_h=max(1,int(640*orig_h/orig_w)))
            if frame is None: continue
            if model is not None:
                try:
                    res=model(frame,verbose=False,conf=confidence)[0]
                    if res.boxes is not None:
                        for box in res.boxes: x1,y1,x2,y2=map(int,box.xyxy[0].tolist()); soi_xs.append((x1+x2)//2); soi_ys.append((y1+y2)//2)
                except Exception: pass
            else:
                scx,scy=saliency_center(frame); soi_xs.append(scx); soi_ys.append(scy)
        sr="center"
        if soi_xs: sr=_soi_region_label(int(np.median(soi_xs)),int(np.median(soi_ys)),orig_w,orig_h)
        ms=int(refined_start//60); secs=int(refined_start%60)
        me=int(refined_end//60);   sece=int(refined_end%60)
        segments.append(ClipSegment(
            start_sec=refined_start, end_sec=refined_end, score=score,
            soi_region=sr,
            peak_frame=int(np.linspace(refined_start+1,refined_end-1,n_s)[n_s//2]*fps),
            title=f"Clip {ci+1}  ({ms}:{secs:02d} - {me}:{sece:02d})",
        ))
    _p(1.0,f"Found {len(segments)} clips"); return segments


# ── process_video — main entry point ──────────────────────────────────────────
def process_video(
    input_path, output_path,
    target_preset_label="Match source (no upscale)",
    tracking_mode="subject", talking_head_bias=0.30,
    sample_interval=None, confidence=0.45, use_optical_flow=True,
    smooth_window=27, adaptive_smoothing=True, rule_of_thirds=True,
    scene_cut_threshold=0.35, output_fps=None, crf=23, encoder_preset="fast",
    audio_bitrate="128k", yolo_weights="yolov8n.pt",
    burn_subtitles=False, whisper_model="base", whisper_language=None,
    subtitle_style_name="Bold White (TikTok)", subtitle_max_chars=42,
    subtitle_translate_to=None,
    vignette_strength=VIGNETTE_STRENGTH,
    sharpen_strength=0.0,
    color_grade="none",
    ken_burns=False,
    dissolve_cuts=True,
    ffmpeg_sharpen=False,
    progress_callback=None,
):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(min(max(v, 0.0), 1.0), msg)
            except Exception: pass

    result_meta = {"output_path": output_path, "subtitle_path": None, "clamped": False,
                   "effective_size": (0, 0), "duration": 0.0, "panel_mode": False}

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

    # FIX-E7: resolve sample_interval immediately after we have fps, before
    # anything else that might reference it.
    if not sample_interval:
        sample_interval = max(1, int(fps / 5))

    lbl = target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w, target_h = resolve_target_size(lbl, orig_w, orig_h)
    req_w, req_h = RESOLUTION_PRESETS.get(lbl, (0, 0))
    clamped = req_h > 0 and (target_h < req_h or target_w < req_w)
    result_meta.update(clamped=clamped, effective_size=(target_w, target_h), duration=duration)
    _p(0.01, f"Output {target_w}x{target_h}  source {orig_w}x{orig_h}")

    render_fps = float(output_fps) if output_fps and output_fps > 0 else fps
    crop_w, crop_h = calculate_crop_dims(orig_w, orig_h, target_w, target_h)
    det_scale = min(1.0, 640 / orig_w)
    det_w, det_h = max(1, int(orig_w * det_scale)), max(1, int(orig_h * det_scale))
    sx, sy = orig_w / det_w, orig_h / det_h

    # FIX-E6: clear pillarbox cache between jobs so stale entries from a
    # different video don't ghost into this one.
    _pillarbox_cache.clear()

    srt_path = None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02, "Transcribing...")
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt"); os.close(srt_fd)
        ok = transcribe_to_srt(input_path, srt_path, whisper_model=whisper_model,
                                language=whisper_language, max_chars_per_line=subtitle_max_chars,
                                progress_callback=lambda v, m: _p(0.02 + v * 0.08, m))
        if not ok:
            if os.path.exists(srt_path): os.unlink(srt_path)
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
        if model_obj is None: _p(start_pct, "YOLO unavailable - saliency fallback")
    elif tracking_mode == "talking_head":
        _p(start_pct, "Loading face detector...")
        if _get_haar() is None and _load_face_net() is None:
            _p(start_pct, "No face detector - saliency fallback")

    layout_state    = LayoutState()
    dominant_layout = LAYOUT_SINGLE
    persons_cache   = []

    if tracking_mode == "subject" and model_obj is not None:
        _p(start_pct + 0.01, "Probing scene layout...")
        dominant_layout = _probe_dominant_layout(
            input_path, model_obj, fps, total_frames,
            orig_w, orig_h, confidence, n_probe=18,
        )
        layout_state.current = dominant_layout
        result_meta["panel_mode"] = dominant_layout != LAYOUT_SINGLE
        _p(start_pct + 0.02, f"Dominant layout: {dominant_layout}")

    extra_vf = _build_ffmpeg_vf(color_grade=color_grade, ffmpeg_sharpen=ffmpeg_sharpen)
    _p(0.12, f"Single-pass detect+render ({total_frames} frames)  layout={dominant_layout}")
    style = SUBTITLE_STYLES.get(subtitle_style_name, SUBTITLE_STYLES["Bold White (TikTok)"])
    proc = _open_ffmpeg_encoder(
        output_path, target_w, target_h, render_fps, audio_source=input_path,
        crf=crf, preset=encoder_preset, audio_bitrate=audio_bitrate,
        subtitle_path=srt_path, subtitle_style=style,
        extra_vf=extra_vf if extra_vf else None,
    )
    if vignette_strength > 0: _build_vignette(target_w, target_h, vignette_strength)
    if color_grade and color_grade != "none": _build_lut(color_grade)
    dissolve_buf = DissolveBuffer(DISSOLVE_FRAMES) if dissolve_cuts else None

    # FIX-E8: removed dead det_centers / det_indices accumulators.
    # The TargetSmoother + CropAnchor chain IS the runtime smoothing path;
    # the post-pass interpolate_centers / smooth_centers pipeline was never
    # wired up and was accumulating data for the entire video duration in RAM.
    scene_cuts: List[int] = []
    prev_gray = None; prev_flow = None
    last_det = None; det_dropout = 0; MAX_DROPOUT = int(fps * 1.5)
    cur_cx = orig_w // 2; cur_cy = orig_h // 2
    hw, hh = crop_w // 2, crop_h // 2
    last_out_frame = None
    rpt_n = max(1, total_frames // 40)
    fi = 0
    last_sample_fi = 0        # for inter-frame interpolation (FIX-B6)
    last_sample_cx = cur_cx
    last_sample_cy = cur_cy
    prev_sample_cx = cur_cx
    prev_sample_cy = cur_cy

    target_smoother  = TargetSmoother(alpha=0.2)
    crop_anchor      = CropAnchor(alpha=0.15, max_step=8)
    history_exists   = False

    try:
        with FFmpegVideoReader(input_path, orig_w, orig_h) as reader:
            for frame in reader:
                if fi >= total_frames: break
                is_sample = (fi % sample_interval == 0)

                if is_sample:
                    det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                    cg = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
                    cut = is_scene_change(prev_gray, cg, scene_cut_threshold)
                    if cut:
                        scene_cuts.append(fi); prev_flow = None; det_dropout = 0
                        if dissolve_buf and last_out_frame is not None:
                            dissolve_buf.on_cut(last_out_frame)
                    prev_gray = cg

                    if tracking_mode == "subject" and model_obj is not None:
                        raw_dets = detect_persons_all(det_frame, model_obj, confidence)
                        persons_cache = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                         for x1, y1, x2, y2 in raw_dets]

                    anchor_cx = anchor_cy = None
                    if tracking_mode == "talking_head":
                        faces = detect_faces(det_frame, confidence_thresh=0.5)
                        if faces:
                            faces_orig = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                                          for x1, y1, x2, y2 in faces]
                            r = talking_head_center(faces_orig, orig_w, orig_h, crop_w, crop_h, talking_head_bias)
                            if r: anchor_cx, anchor_cy = r; det_dropout = 0
                        if anchor_cx is None and use_optical_flow:
                            sm = cv2.resize(cg, (max(1, det_w//2), max(1, det_h//2)))
                            if prev_flow is not None:
                                fc = optical_flow_center(prev_flow, sm, det_w//2, det_h//2)
                                if fc: anchor_cx, anchor_cy = int(fc[0]*2*sx), int(fc[1]*2*sy)
                            prev_flow = sm; det_dropout += sample_interval
                    elif tracking_mode == "subject":
                        if persons_cache:
                            det = detect_subjects(det_frame, model_obj, confidence)
                            if det is not None:
                                anchor_cx, anchor_cy = frame_for_union(
                                    int(det.ux1*sx), int(det.uy1*sy),
                                    int(det.ux2*sx), int(det.uy2*sy),
                                    orig_w, orig_h, crop_w, crop_h)
                                last_det = (anchor_cx, anchor_cy); det_dropout = 0
                        if anchor_cx is None and use_optical_flow:
                            sm = cv2.resize(cg, (max(1, det_w//2), max(1, det_h//2)))
                            if prev_flow is not None:
                                fc = optical_flow_center(prev_flow, sm, det_w//2, det_h//2)
                                if fc: anchor_cx, anchor_cy = int(fc[0]*2*sx), int(fc[1]*2*sy)
                            prev_flow = sm; det_dropout += sample_interval
                        if anchor_cx is None:
                            if last_det and det_dropout < MAX_DROPOUT:
                                anchor_cx, anchor_cy = last_det
                            else:
                                sc_ = saliency_center(det_frame)
                                anchor_cx, anchor_cy = int(sc_[0]*sx), int(sc_[1]*sy)

                    if anchor_cx is not None:
                        ts_cx, ts_cy = target_smoother.update(anchor_cx, anchor_cy)
                        prev_sample_cx, prev_sample_cy = last_sample_cx, last_sample_cy
                        cur_cx, cur_cy = crop_anchor.update(ts_cx, ts_cy)
                        last_sample_cx, last_sample_cy = cur_cx, cur_cy
                        last_sample_fi = fi
                        history_exists = True

                # ── Interpolate between sample frames (FIX-B6) ─────────────
                if not is_sample:
                    if history_exists:
                        span = max(sample_interval, 1)
                        alpha = min(1.0, (fi - last_sample_fi) / span)
                        cur_cx = int(prev_sample_cx * (1 - alpha) + last_sample_cx * alpha)
                        cur_cy = int(prev_sample_cy * (1 - alpha) + last_sample_cy * alpha)

                cur_cx = max(hw, min(cur_cx, orig_w - hw))
                cur_cy = max(hh, min(cur_cy, orig_h - hh))

                # ══ Render ════════════════════════════════════════════════
                if tracking_mode == "subject" and model_obj is not None:
                    pf = persons_cache if is_sample else [b for g in layout_state.prev_groups for b in g]

                    out_frame, active_layout = render_adaptive_frame(
                        frame, pf, target_w, target_h,
                        layout_state=layout_state,
                        # FIX-E2: vignette/color_grade applied inside crop helpers;
                        # do NOT pass them to the outer loop again.
                        vignette_strength=vignette_strength * 0.7,
                        color_grade=color_grade,
                        frame_idx=fi,
                    )

                    if active_layout == LAYOUT_SINGLE:
                        left = max(0, min(cur_cx - crop_w // 2, orig_w - crop_w))
                        top_ = max(0, min(cur_cy - crop_h // 2, orig_h - crop_h))
                        crop = frame[top_:top_ + crop_h, left:left + crop_w]
                        if crop.shape[1] != target_w or crop.shape[0] != target_h:
                            crop = cv2.resize(crop, (target_w, target_h),
                                              interpolation=cv2.INTER_LANCZOS4)
                        out_frame = crop
                        # For SINGLE layout the crop helpers are bypassed, so
                        # apply effects here (these are the only application).
                        if color_grade and color_grade != "none":
                            out_frame = apply_color_grade(out_frame, color_grade)
                        if vignette_strength > 0:
                            out_frame = apply_vignette(out_frame, vignette_strength)

                else:
                    left = max(0, min(cur_cx - crop_w // 2, orig_w - crop_w))
                    top_ = max(0, min(cur_cy - crop_h // 2, orig_h - crop_h))
                    crop = frame[top_:top_ + crop_h, left:left + crop_w]
                    if crop.shape[1] != target_w or crop.shape[0] != target_h:
                        crop = cv2.resize(crop, (target_w, target_h),
                                          interpolation=cv2.INTER_LANCZOS4)
                    out_frame = crop
                    # Talking-head / saliency path — apply effects once here.
                    if color_grade and color_grade != "none":
                        out_frame = apply_color_grade(out_frame, color_grade)
                    if vignette_strength > 0:
                        out_frame = apply_vignette(out_frame, vignette_strength)

                # Post-effects that are cheap and non-duplicating.
                if ken_burns:            out_frame = apply_ken_burns(out_frame, fi, fps)
                if sharpen_strength > 0: out_frame = apply_sharpen(out_frame, sharpen_strength)

                if dissolve_buf and dissolve_buf.active:
                    out_frame = dissolve_buf.blend(out_frame)
                last_out_frame = out_frame

                try: proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError: break

                fi += 1
                if fi % rpt_n == 0:
                    _p(0.12 + 0.75 * (fi / total_frames), f"{fi}/{total_frames}...")

    finally:
        pass

    _p(0.88, "Encoding...")
    _close_ffmpeg_encoder(proc, output_path)
    _p(1.0, "Done!")
    print(f"Output: {output_path}  ({os.path.getsize(output_path)/1024**2:.1f} MB)"
          f"  cuts={len(scene_cuts)}", file=sys.stderr)
    return result_meta


# ── Batch clip pipeline ───────────────────────────────────────────────────────
def process_clips_batch(
    input_path, output_dir, clips,
    target_preset_label="720p   (720x1280  - HD)",
    tracking_mode="subject", talking_head_bias=0.30,
    confidence=0.45, smooth_window=27, adaptive_smoothing=True,
    use_optical_flow=True, rule_of_thirds=True, crf=23, encoder_preset="fast",
    audio_bitrate="128k", yolo_weights="yolov8n.pt",
    burn_subtitles=False, whisper_model="base",
    subtitle_style_name="Bold White (TikTok)", subtitle_max_chars=42,
    vignette_strength=VIGNETTE_STRENGTH, sharpen_strength=0.0, color_grade="none",
    ken_burns=False, dissolve_cuts=True, ffmpeg_sharpen=False,
    progress_callback=None,
):
    def _p(v, msg=""):
        if progress_callback:
            try: progress_callback(v, msg)
            except Exception: pass

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, clip in enumerate(clips):
        base_pct = i / max(len(clips), 1)
        next_pct = (i + 1) / max(len(clips), 1)
        _p(base_pct, f"Clip {i+1}/{len(clips)}...")
        trimmed_path = None; out_path = None
        try:
            fd, trimmed_path = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
            if not _trim_video(input_path, trimmed_path, clip.start_sec, clip.end_sec):
                results.append({"clip": clip, "output_path": None, "error": "trim failed"})
                continue
            out_path = os.path.join(
                output_dir,
                f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s_vertical.mp4",
            )
            def clip_cb(v, msg="", _b=base_pct, _n=next_pct): _p(_b + v * (_n - _b), msg)
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
                try: os.unlink(trimmed_path)
                except OSError: pass

    n_ok = sum(1 for r in results if not r.get("error"))
    _p(1.0, f"{n_ok}/{len(results)} clips done")
    return results
