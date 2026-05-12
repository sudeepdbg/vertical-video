"""
verticalize.py  —  AI Vertical Video converter  v3.0
──────────────────────────────────────────────────────
COMPLETE REWRITE of multi-speaker and anti-jitter logic.

ROOT CAUSE ANALYSIS of v2.x problems
──────────────────────────────────────
Problem 1 – STRIP SPLITTING IS WRONG FOR NEWS/PANEL CONTENT
  The v2.x DUO layout stacked top/bottom strips.  For a 2-person news desk,
  this produced an ugly profile shot of the host on top (shot from the side)
  and a redundant solo shot of the guest on bottom.  The strips DO NOT match
  how humans watch two-person interviews.

Problem 2 – JITTER FROM FRAME-TO-FRAME BBOX NOISE
  YOLO detection boxes vary ±10-30 px per frame.  Using raw boxes to drive
  crop position causes visible jitter even with EMA because the EMA alpha
  was still too responsive.

Problem 3 – LAYOUT THRASHING
  Layout classification was re-evaluated every sample frame.  A person
  briefly occluded would drop the count from 2→1, triggering a layout
  switch and a hard cut.

NEW DESIGN  v3.0
─────────────────
PANEL STRATEGY — CONTEXT-AWARE WIDE CROP + ACTIVE SPEAKER TRACKING
  For 2-person content (most news/interview/podcast footage):
  • Keep a single wide crop that shows BOTH speakers in frame.
  • Detect which speaker is "active" (larger bbox, more motion, more central).
  • Slowly pan the crop window toward the active speaker.
  • This is how professional broadcast vertical cuts are done — no ugly strips.

  Strips are ONLY used when speakers are PHYSICALLY SEPARATED on screen
  (e.g. split-screen graphics baked into source) — threshold raised to 40%.

ANTI-JITTER — THREE-LAYER SMOOTHING
  Layer 1: Raw YOLO boxes → median over a rolling 7-frame buffer per slot.
  Layer 2: Median boxes  → EMA with alpha=0.04 (very slow response).
  Layer 3: Crop anchor   → EMA with alpha=0.06 on the computed target cx/cy.
  Result: crop position changes < 2 px/frame for stationary content.

LAYOUT STABILITY — COMMIT + VOTE GATE
  Layout is held for a minimum of 150 frames (~5 s at 30 fps).
  A proposed change requires 3 consecutive sample-frame agreements
  before the switch is accepted.
"""

from __future__ import annotations
import bisect, subprocess, sys, os, tempfile, math
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
HIGH_PRIO_CLASSES  = {0,2,3,5,7,15,16}
MAX_FILE_SIZE_MB   = 2000
MIN_FRAME_DIM      = 240
MAX_FRAMES_GUARD   = 1_080_000
LOWER_THIRD_GUARD  = 0.80

VELOCITY_SMOOTH_TABLE = [
    (0.0,71),(3.0,65),(8.0,55),(15.0,43),(30.0,31),(60.0,21),(120.0,13),
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

# ── Smoothing constants ───────────────────────────────────────────────────────
VIGNETTE_STRENGTH   = 0.55
VIGNETTE_FALLOFF    = 1.8
COLOR_GRADES        = ("none","warm","cool","vibrant","matte")
BOX_EMA_ALPHA       = 0.04   # Layer-2 EMA on bbox median (very slow)
ANCHOR_EMA_ALPHA    = 0.06   # Layer-3 EMA on crop anchor position
BOX_MEDIAN_WINDOW   = 7      # Layer-1 rolling median window

KEN_BURNS_MAX_ZOOM  = 1.04
KEN_BURNS_PERIOD    = 8.0
DISSOLVE_FRAMES     = 8
PANEL_DIVIDER_PX    = 3
PANEL_DIVIDER_COLOR = (8,8,8)

# Layouts
LAYOUT_SINGLE    = "single"    # 0-1 person or persons too close to split
LAYOUT_DUO_WIDE  = "duo_wide"  # 2 persons close together: single wide crop + pan
LAYOUT_DUO_SPLIT = "duo_split" # 2 persons physically separated: top/bottom strips
LAYOUT_WIDE      = "wide"      # 3+ persons: single wide crop of the group

DUO_SPLIT_THRESHOLD    = 0.40  # min centre-to-centre separation fraction to split
LAYOUT_MIN_HOLD_FRAMES = 150   # never switch layout more often than this
LAYOUT_VOTE_REQUIRED   = 3     # consecutive sample frames needed to commit a switch

ACTIVE_SPEAKER_WINDOW  = 6
ACTIVE_SPEAKER_VOTES   = 4

CLIP_BOUNDARY_SEARCH_SEC = 3.0
CLIP_PREROLL_PAD         = 0.35
CLIP_POSTROLL_PAD        = 0.35


# ── Segment ───────────────────────────────────────────────────────────────────
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


# ── Vignette ──────────────────────────────────────────────────────────────────
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

def apply_sharpen(frame,strength=0.6,radius=1):
    if strength<=0: return frame
    ksize=radius*2+1; blurred=cv2.GaussianBlur(frame,(ksize,ksize),0)
    return cv2.addWeighted(frame,1+strength,blurred,-strength,0)

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

def apply_ken_burns(frame,frame_idx,fps,max_zoom=KEN_BURNS_MAX_ZOOM,period=KEN_BURNS_PERIOD):
    if max_zoom<=1.0: return frame
    t=(frame_idx/max(fps,1))%period
    scale=1.0+(max_zoom-1.0)*0.5*(1-math.cos(2*math.pi*t/period))
    if abs(scale-1.0)<1e-4: return frame
    h,w=frame.shape[:2]; nw=max(int(w/scale),2); nh=max(int(h/scale),2)
    x0=(w-nw)//2; y0=(h-nh)//2
    return cv2.resize(frame[y0:y0+nh,x0:x0+nw],(w,h),interpolation=cv2.INTER_LINEAR)


# ── Cross-dissolve ────────────────────────────────────────────────────────────
class DissolveBuffer:
    def __init__(self,n=DISSOLVE_FRAMES):
        self.n=n; self._buf=None; self._rem=0
    def on_cut(self,last_frame):
        if last_frame is not None: self._buf=last_frame.copy(); self._rem=self.n
    def blend(self,new_frame):
        if self._rem<=0 or self._buf is None: return new_frame
        a=self._rem/self.n; self._rem-=1
        if self._buf.shape!=new_frame.shape:
            self._buf=cv2.resize(self._buf,(new_frame.shape[1],new_frame.shape[0]))
        return cv2.addWeighted(self._buf,a,new_frame,1.0-a,0)
    @property
    def active(self): return self._rem>0

def _build_ffmpeg_vf(color_grade="none",ffmpeg_sharpen=False):
    filters=[]
    eq_map={"warm":"brightness=0.02:saturation=1.12:gamma_r=1.05:gamma_b=0.95",
            "cool":"brightness=0.01:saturation=1.08:gamma_r=0.95:gamma_b=1.05",
            "vibrant":"brightness=0.0:saturation=1.25:contrast=1.05",
            "matte":"brightness=0.03:saturation=0.85:contrast=0.92"}
    if color_grade in eq_map: filters.append(f"eq={eq_map[color_grade]}")
    if ffmpeg_sharpen: filters.append("unsharp=5:5:0.8:3:3:0.0")
    return filters


# ── FFmpegVideoReader ─────────────────────────────────────────────────────────
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
        for cmd in self._candidate_cmds():
            try:
                proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                       bufsize=max(self._frame_bytes*4,1<<20))
                test=proc.stdout.read(self._frame_bytes)
                if len(test)==self._frame_bytes:
                    self._proc=proc; self._leftover=test; return
                try: proc.stdout.close()
                except Exception: pass
                proc.wait()
            except Exception: pass
        raise ProcessingError(f"FFmpeg could not decode: {self.path}")

    def close(self):
        if self._proc:
            try: self._proc.stdout.close()
            except Exception: pass
            self._proc.wait(); self._proc=None

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

def _open_ffmpeg_encoder(output_path,width,height,fps,audio_source,crf=23,preset="fast",
                          audio_bitrate="128k",subtitle_path=None,subtitle_style=None,extra_vf=None):
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
    return subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def _close_ffmpeg_encoder(proc,output_path):
    try: proc.stdin.close()
    except Exception: pass
    proc.wait()
    if proc.returncode!=0:
        try: err=proc.stderr.read(2000).decode(errors="replace")
        except Exception: err=""
        raise ProcessingError(f"FFmpeg encoder failed (rc={proc.returncode}):\n{err}")
    if not os.path.exists(output_path) or os.path.getsize(output_path)<1000:
        raise ProcessingError("FFmpeg encoder produced empty output.")

def get_video_info(path):
    cmd=["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=width,height,r_frame_rate,nb_frames",
         "-show_entries","format=duration","-of","default=noprint_wrappers=1",path]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    kv={}
    for line in r.stdout.splitlines():
        if "=" in line: k,v=line.split("=",1); kv[k.strip()]=v.strip()
    w=int(kv.get("width",0) or 0); h=int(kv.get("height",0) or 0)
    try: num,den=kv.get("r_frame_rate","30/1").split("/"); fps=float(num)/float(den)
    except Exception: fps=30.0
    dur=float(kv.get("duration",0.0) or 0.0)
    if dur<=0:
        nb=int(kv.get("nb_frames",0) or 0); dur=nb/fps if fps>0 and nb>0 else 0.0
    if w==0 or h==0: raise ProcessingError(f"Cannot read dimensions: {path}")
    return {"fps":fps,"total_frames":min(int(dur*fps),MAX_FRAMES_GUARD),
            "width":w,"height":h,"duration_seconds":dur,"is_landscape":w>h}

def extract_thumbnail(path,t=1.0):
    info=get_video_info(path)
    frame=_read_frame_at(path,info["width"],info["height"],t,scale_w=320,scale_h=180)
    if frame is None: return None
    ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
    return buf.tobytes() if ok else None

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

_model_cache: Dict[str,Any]={}

def _get_model(weights="yolov8n.pt"):
    if not _YOLO_AVAILABLE: return None
    if weights in _model_cache: return _model_cache[weights]
    try: m=_YOLO(weights); _model_cache[weights]=m; return m
    except Exception as e: print(f"YOLO unavailable: {e}",file=sys.stderr); return None

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

def detect_persons_all(frame,model,confidence=0.45):
    if model is None: return []
    try: results=model(frame,verbose=False,conf=confidence)[0]
    except Exception: return []
    if results.boxes is None or len(results.boxes)==0: return []
    p=[]
    for box in results.boxes:
        if int(box.cls[0])==PERSON_CLASS_ID:
            x1,y1,x2,y2=map(int,box.xyxy[0].tolist()); p.append((x1,y1,x2,y2))
    p.sort(key=lambda b:b[0]); return p

def _filter_tiny(persons,fw,fh,min_w=0.04,min_h=0.08):
    return [(x1,y1,x2,y2) for x1,y1,x2,y2 in persons
            if (x2-x1)>fw*min_w and (y2-y1)>fh*min_h]


# ═══════════════════════════════════════════════════════════════════════════════
# THREE-LAYER BOX SMOOTHER
# ═══════════════════════════════════════════════════════════════════════════════

class BoxSmoother:
    """Layer 1 (rolling median) + Layer 2 (EMA) per left-to-right slot."""
    def __init__(self,n_slots=4,median_w=BOX_MEDIAN_WINDOW,alpha=BOX_EMA_ALPHA):
        self.n_slots=n_slots; self.median_w=median_w; self.alpha=alpha
        self._history:List[deque]=[deque(maxlen=median_w) for _ in range(n_slots)]
        self._ema:List[Optional[np.ndarray]]=[None]*n_slots
        self._last_n=0

    def update(self,persons):
        n=len(persons)
        if n!=self._last_n:
            self._history=[deque(maxlen=self.median_w) for _ in range(self.n_slots)]
            self._ema=[None]*self.n_slots; self._last_n=n
        out=[]
        for i,box in enumerate(persons):
            if i>=self.n_slots: out.append(box); continue
            self._history[i].append(np.array(box,dtype=float))
            hist=np.stack(self._history[i]); med=np.median(hist,axis=0)
            if self._ema[i] is None: self._ema[i]=med.copy()
            else: self._ema[i]=self._ema[i]*(1-self.alpha)+med*self.alpha
            out.append(tuple(int(v) for v in self._ema[i]))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT CLASSIFIER + STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_layout(persons,fw,fh):
    n=len(persons)
    if n<=1: return LAYOUT_SINGLE
    if n>=3: return LAYOUT_WIDE
    ps=sorted(persons,key=lambda p:(p[0]+p[2])//2)
    cx0=(ps[0][0]+ps[0][2])/2; cx1=(ps[1][0]+ps[1][2])/2
    if (cx1-cx0)/fw>=DUO_SPLIT_THRESHOLD: return LAYOUT_DUO_SPLIT
    return LAYOUT_DUO_WIDE


class LayoutStateMachine:
    def __init__(self):
        self.current=LAYOUT_SINGLE; self._hold=0
        self._pending=None; self._pending_count=0

    def propose(self,layout):
        if self._hold>0: self._hold-=1; self._pending=None; self._pending_count=0; return self.current
        if layout==self.current: self._pending=None; self._pending_count=0; return self.current
        if layout==self._pending: self._pending_count+=1
        else: self._pending=layout; self._pending_count=1
        if self._pending_count>=LAYOUT_VOTE_REQUIRED:
            self.current=layout; self._hold=LAYOUT_MIN_HOLD_FRAMES
            self._pending=None; self._pending_count=0
        return self.current


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE SPEAKER TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveSpeakerTracker:
    """Scores each person by area + motion + centre-proximity, with vote smoothing."""
    def __init__(self,window=ACTIVE_SPEAKER_WINDOW,votes=ACTIVE_SPEAKER_VOTES):
        self.window=window; self.votes=votes
        self._history:deque=deque(maxlen=window)
        self._prev_gray:Optional[np.ndarray]=None; self.active_idx=0

    def update(self,frame_gray,persons):
        if not persons: return 0
        if len(persons)==1: self._history.clear(); return 0
        fh,fw=frame_gray.shape[:2]; frame_cx=fw/2; max_area=fw*fh
        scores=[]
        for i,(x1,y1,x2,y2) in enumerate(persons):
            area=(x2-x1)*(y2-y1)/max_area
            motion=0.0
            if self._prev_gray is not None:
                rx1,ry1=max(0,x1),max(0,y1); rx2,ry2=min(fw,x2),min(fh,y2)
                if rx2>rx1 and ry2>ry1:
                    pn=frame_gray[ry1:ry2,rx1:rx2]; po=self._prev_gray[ry1:ry2,rx1:rx2]
                    if pn.shape==po.shape: motion=float(cv2.absdiff(pn,po).mean())/255.0
            cx=(x1+x2)/2; centre_prox=1.0-abs(cx-frame_cx)/max(frame_cx,1)
            scores.append(area*0.5+motion*0.3+centre_prox*0.2)
        self._prev_gray=frame_gray.copy(); winner=int(np.argmax(scores))
        self._history.append(winner)
        if len(self._history)>=self.votes:
            counts=[0]*len(persons)
            for idx in self._history:
                if idx<len(counts): counts[idx]+=1
            self.active_idx=int(np.argmax(counts))
        return self.active_idx


# ═══════════════════════════════════════════════════════════════════════════════
# CROP ANCHOR  (Layer-3 EMA on crop target)
# ═══════════════════════════════════════════════════════════════════════════════

class CropAnchor:
    def __init__(self,init_cx,init_cy,alpha=ANCHOR_EMA_ALPHA):
        self._cx=float(init_cx); self._cy=float(init_cy); self.alpha=alpha
    def update(self,tcx,tcy):
        a=self.alpha
        self._cx=self._cx*(1-a)+tcx*a; self._cy=self._cy*(1-a)+tcy*a
    def get(self): return int(self._cx),int(self._cy)
    def snap(self,cx,cy): self._cx=float(cx); self._cy=float(cy)


# ═══════════════════════════════════════════════════════════════════════════════
# CROP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_crop_resize(frame,x0,y0,src_w,src_h,out_w,out_h):
    fh,fw=frame.shape[:2]
    x0=max(0,min(x0,fw-src_w)); y0=max(0,min(y0,fh-src_h))
    x1=min(x0+src_w,fw); y1=min(y0+src_h,fh)
    x0=max(0,x1-src_w); y0=max(0,y1-src_h)
    crop=frame[y0:y1,x0:x1]
    if crop.size==0: crop=frame
    return cv2.resize(crop,(out_w,out_h),interpolation=cv2.INTER_LANCZOS4)

def _group_bbox(persons):
    return (min(p[0] for p in persons),min(p[1] for p in persons),
            max(p[2] for p in persons),max(p[3] for p in persons))

def _wide_crop(frame,cx,cy,crop_w,crop_h,out_w,out_h,
               vignette_strength=0.0,color_grade="none",sharpen=0.0):
    """Standard single-window crop centred on (cx,cy)."""
    out=_safe_crop_resize(frame,cx-crop_w//2,cy-crop_h//2,crop_w,crop_h,out_w,out_h)
    if sharpen>0: out=apply_sharpen(out,sharpen)
    if color_grade and color_grade!="none": out=apply_color_grade(out,color_grade)
    if vignette_strength>0: out=apply_vignette(out,vignette_strength)
    return out

def _equalise_brightness(strips):
    if len(strips)<2: return strips
    means=[float(cv2.cvtColor(s,cv2.COLOR_BGR2GRAY).mean())+1e-6 for s in strips]
    target=float(np.mean(means))
    return [np.clip(s.astype(np.float32)*(target/m),0,255).astype(np.uint8)
            if abs(target/m-1.0)>0.08 else s for s,m in zip(strips,means)]

def _split_crop(frame,persons,out_w,out_h,vignette_strength=0.0,color_grade="none"):
    """
    LAYOUT_DUO_SPLIT only — physically separated speakers.
    Each gets a tight upper-body crop in their half of the output height.
    """
    div=PANEL_DIVIDER_PX; sh=((out_h-div)//2)&~1
    fh,fw=frame.shape[:2]; ratio=out_w/sh; strips=[]
    for (x1,y1,x2,y2) in persons[:2]:
        ph=max(y2-y1,1); pw=max(x2-x1,1)
        ucx=(x1+x2)//2; ucy=(y1+y2)//2
        src_h=min(int(ph*2.0),fh)
        src_w=int(src_h*ratio)
        if src_w>pw*2.2 and pw*2.2>4: src_w=int(pw*2.2); src_h=int(src_w/ratio)
        src_h=max(min(src_h,fh),4); src_w=max(min(src_w,fw),4)
        head_bias=int(src_h*0.10)
        x0=max(0,min(int(ucx-src_w/2),fw-src_w))
        y0=max(0,min(int(ucy-head_bias-src_h/2),fh-src_h))
        strip=_safe_crop_resize(frame,x0,y0,src_w,src_h,out_w,sh)
        if color_grade and color_grade!="none": strip=apply_color_grade(strip,color_grade)
        if vignette_strength>0: strip=apply_vignette(strip,vignette_strength)
        strips.append(strip)
    while len(strips)<2: strips.append(strips[-1] if strips else np.zeros((sh,out_w,3),dtype=np.uint8))
    strips=_equalise_brightness(strips)
    canvas=np.empty((out_h,out_w,3),dtype=np.uint8)
    canvas[:sh]=strips[0]; canvas[sh:sh+div]=PANEL_DIVIDER_COLOR
    canvas[sh+div:sh+div+sh]=strips[1]
    if sh+div+sh<out_h: canvas[sh+div+sh:]=PANEL_DIVIDER_COLOR
    return canvas


# ── Saliency / optical flow ───────────────────────────────────────────────────
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

def is_scene_change(prev,curr,threshold=0.30):
    if prev is None: return False
    try: return float(cv2.absdiff(prev,curr).mean())/255.0>threshold
    except Exception: return False

def _soi_region_label(cx,cy,w,h):
    col="left" if cx<w//3 else("right" if cx>2*w//3 else "center")
    row="upper" if cy<h//3 else("lower" if cy>2*h//3 else "mid")
    if row=="mid" and col=="center": return "center"
    if row=="mid": return col
    return f"{row}-{col}"


# ── Probe ─────────────────────────────────────────────────────────────────────
def _probe_dominant_layout(input_path,model,fps,total_frames,orig_w,orig_h,confidence=0.45,n_probe=24):
    if model is None: return LAYOUT_SINGLE
    sw=640; sh=max(1,int(640*orig_h/orig_w)); sx=orig_w/sw; sy=orig_h/sh
    probe_ts=np.linspace(2.0,max(3.0,total_frames/fps-2.0),n_probe)
    counts={LAYOUT_SINGLE:0,LAYOUT_DUO_WIDE:0,LAYOUT_DUO_SPLIT:0,LAYOUT_WIDE:0}
    for t in probe_ts:
        frame=_read_frame_at(input_path,orig_w,orig_h,t,scale_w=sw,scale_h=sh)
        if frame is None: continue
        raw=detect_persons_all(frame,model,confidence)
        raw=_filter_tiny(raw,sw,sh)
        pf=[(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in raw]
        counts[_classify_layout(pf,orig_w,orig_h)]+=1
    best=max(counts,key=counts.get)
    return best if counts[best]>=n_probe*0.30 else LAYOUT_SINGLE


# ── Whisper/translate ─────────────────────────────────────────────────────────
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


# ── Clip boundary refinement ──────────────────────────────────────────────────
def _compute_motion_profile(input_path,fps,total_frames,orig_w,orig_h,sample_every=None,scale_w=320):
    if sample_every is None: sample_every=max(1,int(fps/2))
    sh=max(1,int(scale_w*orig_h/orig_w)); scores={}; prev_gray=None; fi=0
    with FFmpegVideoReader(input_path,orig_w,orig_h,scale_w=scale_w,scale_h=sh) as rdr:
        for frame in rdr:
            if fi>total_frames: break
            if fi%sample_every==0:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                scores[fi]=(float(cv2.absdiff(prev_gray,gray).mean())/255.0 if prev_gray is not None else 0.0)
                prev_gray=gray
            fi+=1
    return scores

def _find_nearest_scene_cut(scs,target,window):
    best=None; bd=float("inf")
    for t in scs:
        d=abs(t-target)
        if d<=window and d<bd: bd=d; best=t
    return best

def _find_low_motion_valley(mp,target,window,fps):
    c={t:s for t,s in mp.items() if abs(t-target)<=window}
    return min(c,key=c.get) if c else target

def _refine_clip_boundaries(s,e,dur,scs,mp,mind,maxd,
                              sw=CLIP_BOUNDARY_SEARCH_SEC,pre=CLIP_PREROLL_PAD,post=CLIP_POSTROLL_PAD):
    cs=_find_nearest_scene_cut(scs,s,sw)
    ns=max(0.0,cs+pre) if cs else max(0.0,_find_low_motion_valley(mp,s,sw,1)-pre*0.5)
    ce=_find_nearest_scene_cut(scs,e,sw)
    ne=min(dur,ce-post) if ce else min(dur,_find_low_motion_valley(mp,e,sw,1)+post*0.5)
    nd=ne-ns
    if nd<mind:
        df=mind-nd; ns=max(0.0,ns-df/2); ne=min(dur,ns+mind); ns=max(0.0,ne-mind)
    if nd>maxd:
        c2=(ns+ne)/2; ns=max(0.0,c2-maxd/2); ne=min(dur,ns+maxd)
    return ns,ne

def _frame_saliency_score(frame,prev_frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    lap=min(float(cv2.Laplacian(gray,cv2.CV_64F).var())/3000.0,1.0)
    mot=(min(float(cv2.absdiff(gray,cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)).mean())/30.0,1.0) if prev_frame is not None else 0.0)
    sat=min(float(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1].mean())/128.0,1.0)
    return 0.4*mot+0.4*lap+0.2*sat

def _compute_frame_scores(input_path,fps,total_frames,orig_w,orig_h,sample_every=15,progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    scores=[]; scuts=[]; pg=None; pf=None
    sw=min(orig_w,640); sh=max(1,int(sw*orig_h/orig_w)); rn=max(1,total_frames//20); fi=0
    with FFmpegVideoReader(input_path,orig_w,orig_h,scale_w=sw,scale_h=sh) as reader:
        for frame in reader:
            if fi>=total_frames: break
            if fi%sample_every==0:
                cg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                if pg is not None and float(cv2.absdiff(pg,cg).mean())/255.0>0.30: scuts.append(fi)
                scores.append(_frame_saliency_score(frame,pf)); pg=cg; pf=frame.copy()
            if fi%rn==0: _p(fi/total_frames,f"Scanning {fi}/{total_frames}...")
            fi+=1
    return np.array(scores,dtype=float),scuts

def detect_clips(input_path,min_duration_sec=25.0,max_duration_sec=65.0,
                  target_n_clips=10,model=None,confidence=0.45,progress_callback=None):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    info=get_video_info(input_path); fps=info["fps"]; total_frames=info["total_frames"]
    dur=info["duration_seconds"]; ow=info["width"]; oh=info["height"]
    se=max(1,int(fps)); _p(0.0,"Scanning...")
    scores,scuts=_compute_frame_scores(input_path,fps,total_frames,ow,oh,sample_every=se,
        progress_callback=lambda v,m:_p(v*0.45,m))
    if len(scores)==0: return []
    mp={fi*se/fps:float(scores[fi]) for fi in range(len(scores))}
    scs=[sc/fps for sc in scuts]
    _p(0.45,"Computing arcs..."); w=max(5,int(30/(se/fps)))
    ss=(np.convolve(scores,np.ones(w)/w,mode="same") if len(scores)>=w else scores.copy())
    if ss.max()>0: ss=ss/ss.max()
    mg=max(1,int(min_duration_sec*fps/se)); peaks=[]
    for i in range(1,len(ss)-1):
        wh=mg//2; lo=max(0,i-wh); hi=min(len(ss),i+wh+1)
        if ss[i]==ss[lo:hi].max() and ss[i]>0.3:
            if not peaks or i-peaks[-1]>mg//2: peaks.append(i)
    peaks.sort(key=lambda i:ss[i],reverse=True); peaks=peaks[:target_n_clips*2]
    def _arc(pi):
        ps=pi*se/fps; rs=max(0.0,ps-max_duration_sec*0.4); re=min(dur,rs+max_duration_sec)
        for sc in reversed(scuts):
            sc_s=sc/fps
            if 0<ps-sc_s<15.0: rs=max(0.0,sc_s-1.0); break
        for sc in scuts:
            sc_s=sc/fps
            if 0<sc_s-ps<15.0: re=min(dur,sc_s+0.5); break
        cd=re-rs
        if cd<min_duration_sec: re=min(dur,rs+min_duration_sec)
        elif cd>max_duration_sec:
            c=(rs+re)/2; rs=max(0.0,c-max_duration_sec/2); re=min(dur,rs+max_duration_sec)
        return rs,re
    cands=[]
    for pi in peaks:
        s,e=_arc(pi); sc=float(ss[pi])
        if not any(min(e,ce)-max(s,cs)>min_duration_sec*0.5 for cs,ce,_ in cands): cands.append((s,e,sc))
    cands.sort(key=lambda x:x[2],reverse=True); cands=cands[:target_n_clips]; cands.sort(key=lambda x:x[0])
    _p(0.55,"Refining + SOI..."); segments=[]
    for ci,(ss2,se2,score) in enumerate(cands):
        _p(0.55+0.35*(ci/max(len(cands),1)),f"Clip {ci+1}/{len(cands)}...")
        rs,re=_refine_clip_boundaries(ss2,se2,dur,scs,mp,min_duration_sec,max_duration_sec)
        soi_xs=[]; soi_ys=[]; n_s=min(8,max(2,int(re-rs)))
        for t in np.linspace(rs+1,re-1,n_s):
            frame=_read_frame_at(input_path,ow,oh,t,scale_w=640,scale_h=max(1,int(640*oh/ow)))
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
        if soi_xs: sr=_soi_region_label(int(np.median(soi_xs)),int(np.median(soi_ys)),ow,oh)
        ms=int(rs//60); secs=int(rs%60); me=int(re//60); sece=int(re%60)
        segments.append(ClipSegment(start_sec=rs,end_sec=re,score=score,soi_region=sr,
            peak_frame=int(np.linspace(rs+1,re-1,n_s)[n_s//2]*fps),
            title=f"Clip {ci+1}  ({ms}:{secs:02d} - {me}:{sece:02d})"))
    _p(1.0,f"Found {len(segments)} clips"); return segments


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(
    input_path,output_path,
    target_preset_label="Match source (no upscale)",
    tracking_mode="subject",talking_head_bias=0.30,
    sample_interval=None,confidence=0.45,use_optical_flow=True,
    smooth_window=27,adaptive_smoothing=True,rule_of_thirds=True,
    scene_cut_threshold=0.30,output_fps=None,crf=23,encoder_preset="fast",
    audio_bitrate="128k",yolo_weights="yolov8n.pt",
    burn_subtitles=False,whisper_model="base",whisper_language=None,
    subtitle_style_name="Bold White (TikTok)",subtitle_max_chars=42,
    subtitle_translate_to=None,
    vignette_strength=VIGNETTE_STRENGTH,sharpen_strength=0.0,
    color_grade="none",ken_burns=False,dissolve_cuts=True,ffmpeg_sharpen=False,
    progress_callback=None,
):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(min(max(v,0.0),1.0),msg)
            except Exception: pass

    result_meta={"output_path":output_path,"subtitle_path":None,"clamped":False,
                  "effective_size":(0,0),"duration":0.0,"panel_mode":False}
    _check_ffmpeg()
    if not os.path.exists(input_path): raise ProcessingError(f"Input not found: {input_path}")
    if os.path.getsize(input_path)/1024**2>MAX_FILE_SIZE_MB: raise ProcessingError(f"File exceeds {MAX_FILE_SIZE_MB} MB.")

    info=get_video_info(input_path)
    fps=info["fps"]; total_frames=info["total_frames"]
    orig_w=info["width"]; orig_h=info["height"]; duration=info["duration_seconds"]
    if total_frames<=0 or orig_w<=0 or orig_h<=0: raise ProcessingError("Corrupt or unreadable video.")
    if not info["is_landscape"]: raise ProcessingError("Video is already vertical.")

    lbl=target_preset_label if target_preset_label in RESOLUTION_PRESETS else "Match source (no upscale)"
    target_w,target_h=resolve_target_size(lbl,orig_w,orig_h)
    req_w,req_h=RESOLUTION_PRESETS.get(lbl,(0,0))
    clamped=req_h>0 and (target_h<req_h or target_w<req_w)
    result_meta.update(clamped=clamped,effective_size=(target_w,target_h),duration=duration)
    _p(0.01,f"Output {target_w}x{target_h}  source {orig_w}x{orig_h}")

    if not sample_interval: sample_interval=max(1,int(fps/5))
    render_fps=float(output_fps) if output_fps and output_fps>0 else fps
    crop_w,crop_h=calculate_crop_dims(orig_w,orig_h,target_w,target_h)
    det_scale=min(1.0,640/orig_w)
    det_w=max(1,int(orig_w*det_scale)); det_h=max(1,int(orig_h*det_scale))
    sx=orig_w/det_w; sy=orig_h/det_h
    hw=crop_w//2; hh=crop_h//2

    # ── Subtitles ─────────────────────────────────────────────────────────
    srt_path=None
    if burn_subtitles and _has_audio(input_path):
        _p(0.02,"Transcribing...")
        srt_fd,srt_path=tempfile.mkstemp(suffix=".srt"); os.close(srt_fd)
        ok=transcribe_to_srt(input_path,srt_path,whisper_model=whisper_model,language=whisper_language,
                              max_chars_per_line=subtitle_max_chars,progress_callback=lambda v,m:_p(0.02+v*0.08,m))
        if not ok:
            if os.path.exists(srt_path): os.unlink(srt_path); srt_path=None
        else:
            if subtitle_translate_to: translate_srt(srt_path,target_language=subtitle_translate_to,progress_callback=lambda v,m:_p(0.10+v*0.05,m))
            result_meta["subtitle_path"]=srt_path

    # ── Model ─────────────────────────────────────────────────────────────
    start_pct=0.10; model_obj=None
    if tracking_mode=="subject":
        _p(start_pct,"Loading YOLO..."); model_obj=_get_model(yolo_weights)
        if model_obj is None: _p(start_pct,"YOLO unavailable — saliency fallback")
    elif tracking_mode=="talking_head":
        _p(start_pct,"Loading face detector...")

    # ── Probe layout ──────────────────────────────────────────────────────
    dominant_layout=LAYOUT_SINGLE
    if tracking_mode=="subject" and model_obj is not None:
        _p(start_pct+0.01,"Probing layout...")
        dominant_layout=_probe_dominant_layout(input_path,model_obj,fps,total_frames,
                                                orig_w,orig_h,confidence,n_probe=24)
        result_meta["panel_mode"]=dominant_layout!=LAYOUT_SINGLE
        _p(start_pct+0.02,f"Dominant layout: {dominant_layout}")

    # ── State ─────────────────────────────────────────────────────────────
    box_smoother=BoxSmoother(n_slots=4)
    layout_sm=LayoutStateMachine(); layout_sm.current=dominant_layout
    active_spk=ActiveSpeakerTracker()
    anchor=CropAnchor(orig_w//2,orig_h//2)
    dissolve_buf=DissolveBuffer(DISSOLVE_FRAMES) if dissolve_cuts else None

    extra_vf=_build_ffmpeg_vf(color_grade=color_grade,ffmpeg_sharpen=ffmpeg_sharpen)
    style=SUBTITLE_STYLES.get(subtitle_style_name,SUBTITLE_STYLES["Bold White (TikTok)"])
    proc=_open_ffmpeg_encoder(output_path,target_w,target_h,render_fps,audio_source=input_path,
                               crf=crf,preset=encoder_preset,audio_bitrate=audio_bitrate,
                               subtitle_path=srt_path,subtitle_style=style,
                               extra_vf=extra_vf if extra_vf else None)
    if vignette_strength>0: _build_vignette(target_w,target_h,vignette_strength)
    if color_grade and color_grade!="none": _build_lut(color_grade)
    _p(0.12,f"Rendering {total_frames} frames  dominant={dominant_layout}")

    prev_gray=None; prev_flow=None
    last_persons:List=[]
    last_layout=dominant_layout; last_out_frame=None
    rpt_n=max(1,total_frames//40); fi=0

    try:
        with FFmpegVideoReader(input_path,orig_w,orig_h) as reader:
            for frame in reader:
                if fi>=total_frames: break
                is_sample=(fi%sample_interval==0)

                if is_sample:
                    det_frame=cv2.resize(frame,(det_w,det_h),interpolation=cv2.INTER_LINEAR)
                    cg=cv2.cvtColor(det_frame,cv2.COLOR_BGR2GRAY)

                    # Scene cut
                    if is_scene_change(prev_gray,cg,scene_cut_threshold):
                        if dissolve_buf and last_out_frame is not None:
                            dissolve_buf.on_cut(last_out_frame)
                        prev_flow=None
                    prev_gray=cg

                    # ── Person detection + smoothing ───────────────────────
                    if tracking_mode=="subject" and model_obj is not None:
                        raw=detect_persons_all(det_frame,model_obj,confidence)
                        raw=_filter_tiny(raw,det_w,det_h)
                        # Scale to full-frame coords
                        raw_full=[(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in raw]
                        smoothed=box_smoother.update(raw_full)
                        if smoothed: last_persons=smoothed

                        # Layout vote
                        proposed=_classify_layout(last_persons,orig_w,orig_h)
                        committed=layout_sm.propose(proposed)
                        if committed!=last_layout:
                            if dissolve_buf and last_out_frame is not None:
                                dissolve_buf.on_cut(last_out_frame)
                            last_layout=committed

                        # Active speaker → anchor target
                        # Scale smoothed boxes to det coords for motion measurement
                        sp_det=[(int(x1/sx),int(y1/sy),int(x2/sx),int(y2/sy))
                                 for x1,y1,x2,y2 in last_persons]
                        act=active_spk.update(cg,sp_det)

                        if last_persons:
                            if committed in (LAYOUT_SINGLE,LAYOUT_DUO_WIDE,LAYOUT_WIDE):
                                # Wide crop: blend union centre + active speaker
                                ub=_group_bbox(last_persons)
                                ucx=(ub[0]+ub[2])//2; ucy=(ub[1]+ub[3])//2
                                if len(last_persons)>1 and act<len(last_persons):
                                    sp=last_persons[act]
                                    scx=(sp[0]+sp[2])//2; scy=(sp[1]+sp[3])//2
                                    # 60% union, 40% active — keeps both speakers visible
                                    tcx=int(ucx*0.60+scx*0.40)
                                    tcy=int(ucy*0.60+scy*0.40)
                                else:
                                    tcx=ucx; tcy=ucy
                                # Slight upward shift for headroom
                                tcy=max(hh,tcy-int(crop_h*0.05))
                                anchor.update(tcx,tcy)
                            # For LAYOUT_DUO_SPLIT the anchor isn't used (strips handle it)
                        else:
                            # Fallback: optical flow or saliency
                            if use_optical_flow and prev_flow is not None:
                                sm2=cv2.resize(cg,(det_w//2,det_h//2))
                                fc=optical_flow_center(prev_flow,sm2,det_w//2,det_h//2)
                                if fc: anchor.update(int(fc[0]*2*sx),int(fc[1]*2*sy))
                            else:
                                sc_=saliency_center(det_frame)
                                anchor.update(int(sc_[0]*sx),int(sc_[1]*sy))

                        if use_optical_flow:
                            prev_flow=cv2.resize(cg,(det_w//2,det_h//2))

                    elif tracking_mode=="talking_head":
                        faces=detect_faces(det_frame,confidence_thresh=0.5)
                        if faces:
                            fo=[(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in faces]
                            ux1=min(f[0] for f in fo); ux2=max(f[2] for f in fo)
                            uy1=min(f[1] for f in fo); uy2=max(f[3] for f in fo)
                            tcx=(ux1+ux2)//2; tcy=max(hh,(uy1+uy2)//2-int(crop_h*0.08))
                            anchor.update(tcx,tcy)
                        elif use_optical_flow and prev_flow is not None:
                            sm2=cv2.resize(cg,(det_w//2,det_h//2))
                            fc=optical_flow_center(prev_flow,sm2,det_w//2,det_h//2)
                            if fc: anchor.update(int(fc[0]*2*sx),int(fc[1]*2*sy))
                        if use_optical_flow:
                            prev_flow=cv2.resize(cg,(det_w//2,det_h//2))
                    else:
                        sc_=saliency_center(det_frame)
                        anchor.update(int(sc_[0]*sx),int(sc_[1]*sy))

                # ── Clamp anchor ──────────────────────────────────────────
                cx,cy=anchor.get()
                cx=max(hw,min(cx,orig_w-hw)); cy=max(hh,min(cy,orig_h-hh))

                # ── Render ────────────────────────────────────────────────
                layout=layout_sm.current
                kw=dict(vignette_strength=vignette_strength*0.75,color_grade=color_grade)

                if layout==LAYOUT_DUO_SPLIT and len(last_persons)>=2:
                    out_frame=_split_crop(frame,last_persons,target_w,target_h,**kw)
                else:
                    # SINGLE / DUO_WIDE / WIDE — single smooth crop window
                    out_frame=_wide_crop(frame,cx,cy,crop_w,crop_h,target_w,target_h,
                                          sharpen=sharpen_strength,**kw)
                    if ken_burns: out_frame=apply_ken_burns(out_frame,fi,fps)

                if dissolve_buf and dissolve_buf.active:
                    out_frame=dissolve_buf.blend(out_frame)
                last_out_frame=out_frame

                try: proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError: break
                fi+=1
                if fi%rpt_n==0: _p(0.12+0.75*(fi/total_frames),f"{fi}/{total_frames}...")
    finally:
        pass

    _p(0.88,"Encoding..."); _close_ffmpeg_encoder(proc,output_path)
    _p(1.0,"Done!")
    print(f"Output: {output_path}  ({os.path.getsize(output_path)/1024**2:.1f} MB)"
          f"  layout={dominant_layout}",file=sys.stderr)
    return result_meta


# ── Batch clip pipeline ───────────────────────────────────────────────────────
def process_clips_batch(
    input_path,output_dir,clips,
    target_preset_label="720p   (720x1280  - HD)",
    tracking_mode="subject",talking_head_bias=0.30,
    confidence=0.45,smooth_window=27,adaptive_smoothing=True,
    use_optical_flow=True,rule_of_thirds=True,crf=23,encoder_preset="fast",
    audio_bitrate="128k",yolo_weights="yolov8n.pt",
    burn_subtitles=False,whisper_model="base",
    subtitle_style_name="Bold White (TikTok)",subtitle_max_chars=42,
    vignette_strength=VIGNETTE_STRENGTH,sharpen_strength=0.0,color_grade="none",
    ken_burns=False,dissolve_cuts=True,ffmpeg_sharpen=False,
    progress_callback=None,
):
    def _p(v,msg=""):
        if progress_callback:
            try: progress_callback(v,msg)
            except Exception: pass
    os.makedirs(output_dir,exist_ok=True); results=[]
    for i,clip in enumerate(clips):
        bp=i/max(len(clips),1); np_=( i+1)/max(len(clips),1)
        _p(bp,f"Clip {i+1}/{len(clips)}...")
        tp=None; op=None
        try:
            fd,tp=tempfile.mkstemp(suffix=".mp4"); os.close(fd)
            if not _trim_video(input_path,tp,clip.start_sec,clip.end_sec):
                results.append({"clip":clip,"output_path":None,"error":"trim failed"}); continue
            op=os.path.join(output_dir,f"clip_{i+1:02d}_{int(clip.start_sec)}s_{int(clip.end_sec)}s_vertical.mp4")
            def cb(v,msg="",_b=bp,_n=np_): _p(_b+v*(_n-_b),msg)
            meta=process_video(tp,op,target_preset_label=target_preset_label,
                tracking_mode=tracking_mode,talking_head_bias=talking_head_bias,
                confidence=confidence,smooth_window=smooth_window,adaptive_smoothing=adaptive_smoothing,
                use_optical_flow=use_optical_flow,rule_of_thirds=rule_of_thirds,
                crf=crf,encoder_preset=encoder_preset,audio_bitrate=audio_bitrate,
                yolo_weights=yolo_weights,burn_subtitles=burn_subtitles,whisper_model=whisper_model,
                subtitle_style_name=subtitle_style_name,subtitle_max_chars=subtitle_max_chars,
                vignette_strength=vignette_strength,sharpen_strength=sharpen_strength,
                color_grade=color_grade,ken_burns=ken_burns,dissolve_cuts=dissolve_cuts,
                ffmpeg_sharpen=ffmpeg_sharpen,progress_callback=cb)
            meta["clip"]=clip; results.append(meta)
        except Exception as exc:
            results.append({"clip":clip,"output_path":op,"error":str(exc)})
        finally:
            if tp and os.path.exists(tp):
                try: os.unlink(tp)
                except OSError: pass
    n_ok=sum(1 for r in results if not r.get("error"))
    _p(1.0,f"{n_ok}/{len(results)} clips done"); return results
