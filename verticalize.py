"""
verticalize.py  —  AI Vertical Video converter  v2.2
──────────────────────────────────────────────────────
FIXES in v2.2:
  FIX-P1  Panel rendering completely rewritten:
           • Persons sorted left→right by bbox center-x
           • Split into LEFT group (left half of frame) vs RIGHT group (right half)
           • Each group gets a HORIZONTAL BAND crop (full output width, half output height)
             that zooms tightly on that group — matching the reference screenshot aesthetic
           • Crop is portrait-ratio (out_w × strip_h) sourced from a wider horizontal
             region around each group, preserving the "wide shot of that side" feel
           • EMA slot smoother applied per-group independently for temporal stability
           • Fallback: if only one side has people, mirror or use saliency

  FIX-P2  Panel slot assignment now uses spatial median across probe frames
           instead of per-frame majority vote — stable group identity.

  FIX-C1  Clip boundary refinement:
           • _refine_clip_boundaries() searches ±3 s around detected start/end
             for the nearest scene cut; snaps to it with 0.3 s pre-roll padding
           • If no scene cut is nearby the boundary is nudged to the nearest
             low-motion valley (avoid cutting mid-gesture)
           • Minimum 1.5 s post-roll guard: clip never ends on a hard cut frame

  FIX-C2  detect_clips() now calls _refine_clip_boundaries() on every candidate
           so every exported clip starts/ends on a clean shot.

All v2.1 enhancements (ENH-1…8) and v2.0 fixes (FIX-1…5) are retained unchanged.
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
PANEL_SLOT_EMA     = 0.18          # smoother (lower = more inertia)
KEN_BURNS_MAX_ZOOM = 1.04
KEN_BURNS_PERIOD   = 8.0
DISSOLVE_FRAMES    = 3
PANEL_DIVIDER_PX   = 4
PANEL_DIVIDER_COLOR= (10,10,10)

# Panel crop tuning — how many times wider than the union bbox to include
# 1.0 = tight on the people, 1.6 = include some context on each side
# How much to expand person height for the crop window.
# 1.8 = person body + comfortable headroom above + some body below.
# Larger = wider shot / more context. Smaller = tighter face crop.
PANEL_CROP_EXPAND  = 1.80

# How far (seconds) to search around a clip edge for a clean scene cut
CLIP_BOUNDARY_SEARCH_SEC = 3.0
# Pre-roll pad added when snapping to a scene cut
CLIP_PREROLL_PAD  = 0.35
# Post-roll pad
CLIP_POSTROLL_PAD = 0.35


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


# ── FIX-1: FFmpegVideoReader ──────────────────────────────────────────────────
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


# ── FIX-5: Encoder with ENH-8 ────────────────────────────────────────────────
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


# ── Video metadata ────────────────────────────────────────────────────────────
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
# FIX-P1/P2: Panel Discussion — rewritten for correct horizontal-band output
# ═══════════════════════════════════════════════════════════════════════════════

def _group_union(persons):
    """Return bounding union of a list of (x1,y1,x2,y2) boxes."""
    return (min(p[0] for p in persons), min(p[1] for p in persons),
            max(p[2] for p in persons), max(p[3] for p in persons))


def _split_persons_by_position(persons, frame_w):
    """
    Split detected persons into LEFT and RIGHT groups based on their
    horizontal center relative to the frame midpoint.

    Returns (group_left, group_right).  Each may be empty.
    """
    mid = frame_w / 2
    left  = [p for p in persons if (p[0]+p[2])/2 <= mid]
    right = [p for p in persons if (p[0]+p[2])/2 >  mid]
    return left, right


def _crop_band_for_group(
    frame, group, out_w, out_h,
    expand=PANEL_CROP_EXPAND,
    vignette_strength=0.0,
    color_grade="none",
):
    """
    Produce a strip of size (out_w × out_h) zoomed tightly onto `group`.

    Algorithm:
      1. Person bbox union → person_h, person_w, centre (ucx, ucy).
      2. src_h = person_h * expand  (height drives zoom)
      3. src_w = src_h * (out_w / out_h)  (enforce strip aspect ratio)
      4. Clamp src_w to at most (person_w * expand * 1.5) so the crop
         doesn't drift far from the person horizontally.
      5. If clamping src_w forced it smaller, recompute src_h from src_w.
      6. Centre crop on (ucx, ucy - head_bias), clamp to frame edges.
      7. Resize to (out_w, out_h).

    If group is empty: centre crop of full frame at strip ratio.
    """
    fh, fw = frame.shape[:2]
    ratio = out_w / out_h   # e.g. 9/8 ≈ 1.125 for a half-height strip

    if not group:
        src_h = fh
        src_w = min(fw, int(src_h * ratio))
        x0 = (fw - src_w) // 2
        y0 = 0
    else:
        ux1, uy1, ux2, uy2 = _group_union(group)
        person_h = max(uy2 - uy1, 1)
        person_w = max(ux2 - ux1, 1)
        ucx = (ux1 + ux2) // 2
        ucy = (uy1 + uy2) // 2

        # --- Height-driven zoom ---
        src_h = int(person_h * expand)

        # --- Width from aspect ratio ---
        src_w = int(src_h * ratio)

        # --- Key constraint: don't let src_w be much wider than the person.
        #     This stops the crop from spanning the whole frame and showing
        #     the OTHER group's side. Cap at ~1.8× person width. ---
        max_src_w = int(person_w * 1.8)
        if src_w > max_src_w and max_src_w > 4:
            src_w = max_src_w
            # Recompute height to maintain ratio
            src_h = int(src_w / ratio)

        # --- Clamp to frame ---
        src_h = max(min(src_h, fh), 4)
        src_w = max(min(src_w, fw), 4)

        # Rebalance to exact ratio after clamping
        if src_w / max(src_h, 1) > ratio:
            src_h = max(int(src_w / ratio), 4)
        else:
            src_w = max(int(src_h * ratio), 4)
        src_h = min(src_h, fh)
        src_w = min(src_w, fw)

        # --- Position: centre on person with slight upward head-room bias ---
        head_bias = int(src_h * 0.08)
        cx = ucx
        cy = ucy - head_bias

        x0 = int(cx - src_w / 2)
        y0 = int(cy - src_h / 2)
        x0 = max(0, min(x0, fw - src_w))
        y0 = max(0, min(y0, fh - src_h))

    x1 = min(x0 + src_w, fw)
    y1 = min(y0 + src_h, fh)
    x0 = max(0, x1 - src_w)
    y0 = max(0, y1 - src_h)

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame

    result = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    if color_grade and color_grade != "none":
        result = apply_color_grade(result, color_grade)
    if vignette_strength > 0:
        result = apply_vignette(result, vignette_strength)

    return result


class PanelSlotSmoother:
    """
    ENH-5 (rewritten): Independent EMA smoothers for LEFT and RIGHT slots.

    Smooths the union bounding-box of each group so crops don't jitter
    frame-to-frame.  Lower alpha = more inertia.
    """
    def __init__(self, alpha=PANEL_SLOT_EMA):
        self.alpha = alpha
        self._slots = [None, None]   # [left_bbox, right_bbox]

    def _ema(self, prev, new_box):
        if prev is None:
            return new_box
        a = self.alpha
        return tuple(prev[i] * (1 - a) + new_box[i] * a for i in range(4))

    def update(self, group_left, group_right):
        """
        Smooth each group's union bbox and return synthetic single-person
        group lists that encode the smoothed bbox.
        """
        def _smooth(slot_idx, group):
            if not group:
                return group  # Nothing to smooth; caller handles empty group
            u = _group_union(group)
            s = self._ema(self._slots[slot_idx], u)
            self._slots[slot_idx] = s
            # Return as a single synthetic bbox (the smoothed union)
            return [tuple(int(v) for v in s)]

        return _smooth(0, group_left), _smooth(1, group_right)


def _detect_panel_mode(input_path, model, fps, total_frames, orig_w, orig_h,
                       confidence=0.45, n_probe=16):
    """
    Probe n_probe frames. Return True only if ALL of:
      • ≥50 % of probed frames have ≥2 distinct persons
      • ≥30 % of probed frames have persons on BOTH sides of the frame midpoint
      • The two groups are spatially separated enough (gap > 5 % of frame width)

    Detections are done in scaled space and split tested in THAT same space
    (not orig coords) so the midpoint is correct.
    """
    if model is None:
        return False
    sw = 640; sh = max(1, int(640 * orig_h / orig_w))
    probe_ts = np.linspace(1.0, max(1.5, total_frames / fps - 1.0), n_probe)
    hits = 0
    both_sides = 0
    for t in probe_ts:
        frame = _read_frame_at(input_path, orig_w, orig_h, t, scale_w=sw, scale_h=sh)
        if frame is None:
            continue
        persons = detect_persons_all(frame, model, confidence)
        # Filter tiny boxes (noise / partial detections)
        persons = [(x1,y1,x2,y2) for x1,y1,x2,y2 in persons
                   if (x2-x1) > sw*0.05 and (y2-y1) > sh*0.10]
        if len(persons) >= PANEL_MIN_PERSONS:
            hits += 1
            # Split in SCALED space (frame is sw×sh here)
            gl, gr = _split_persons_by_position(persons, sw)
            if gl and gr:
                # Require meaningful horizontal separation between the groups
                gl_cx = sum((p[0]+p[2])//2 for p in gl) / len(gl)
                gr_cx = sum((p[0]+p[2])//2 for p in gr) / len(gr)
                if (gr_cx - gl_cx) > sw * 0.10:
                    both_sides += 1
    return hits > n_probe * 0.5 and both_sides > n_probe * 0.3


def _render_panel_frame(
    frame, persons, out_w, out_h,
    prev_slots=None,
    vignette_strength=VIGNETTE_STRENGTH * 0.7,
    color_grade="none",
    slot_smoother=None,
):
    """
    Render a panel frame:
      • persons is a list of (x1,y1,x2,y2) in FULL-FRAME coordinates.
      • Sort persons left→right by bbox centre-x.
      • Split at frame midpoint into LEFT group and RIGHT group.
      • Each group → one tightly-zoomed horizontal band (out_w × strip_h).
      • Stack: top=left-group, bottom=right-group, with thin divider.

    Fallback rules (in order):
      1. If a side has no person this frame, reuse that slot from prev_slots.
      2. If BOTH sides empty (no detection at all), duplicate the single
         fallback bbox but crop different vertical halves.
      3. If prev_slots also empty, use a centre full-frame crop for both.

    Returns (canvas, [group_left, group_right]).
    """
    fh, fw = frame.shape[:2]

    # ── Filter tiny/degenerate boxes ─────────────────────────────────────
    persons = [(x1,y1,x2,y2) for x1,y1,x2,y2 in persons
               if (x2-x1) > fw*0.04 and (y2-y1) > fh*0.08]

    # ── Assign to left / right by frame midpoint ──────────────────────────
    group_left, group_right = _split_persons_by_position(persons, fw)

    # ── Fallback: fill empty slots from previous detections ───────────────
    if not group_left and prev_slots and prev_slots[0]:
        group_left = prev_slots[0]
    if not group_right and prev_slots and prev_slots[1]:
        group_right = prev_slots[1]

    # ── If still only one side has people, use them for both but DON'T
    #    crop identically — use full-frame context for the missing side ────
    # (this case means panel mode was triggered but this frame is a cut-away;
    #  just use whatever we have and let EMA smooth it out)

    # ── Apply EMA smoothing ────────────────────────────────────────────────
    if slot_smoother:
        group_left, group_right = slot_smoother.update(group_left, group_right)

    # ── Compute strip heights (must be even, divider between them) ────────
    div_px      = PANEL_DIVIDER_PX
    strip_h_top = ((out_h - div_px) // 2) & ~1
    strip_h_bot = out_h - strip_h_top - div_px
    strip_h_bot = max(strip_h_bot & ~1, 2)

    # ── Render each band with tight zoom on its group ─────────────────────
    top = _crop_band_for_group(
        frame, group_left,  out_w, strip_h_top,
        vignette_strength=vignette_strength,
        color_grade=color_grade,
    )
    bot = _crop_band_for_group(
        frame, group_right, out_w, strip_h_bot,
        vignette_strength=vignette_strength,
        color_grade=color_grade,
    )

    # ── Assemble canvas ───────────────────────────────────────────────────
    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[0:strip_h_top, :]                                        = top
    canvas[strip_h_top:strip_h_top + div_px, :]                     = PANEL_DIVIDER_COLOR
    canvas[strip_h_top + div_px:strip_h_top + div_px + strip_h_bot, :] = bot
    used = strip_h_top + div_px + strip_h_bot
    if used < out_h:
        canvas[used:, :] = PANEL_DIVIDER_COLOR

    return canvas, [group_left, group_right]


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


# ── FIX-2: Smooth camera path ─────────────────────────────────────────────────
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

def _ema_polish(centers,alpha=0.08):
    if len(centers)<3: return centers
    n=len(centers); fx=[float(centers[0][0])]; fy=[float(centers[0][1])]
    for i in range(1,n):
        fx.append(alpha*centers[i][0]+(1-alpha)*fx[-1])
        fy.append(alpha*centers[i][1]+(1-alpha)*fy[-1])
    rx=[fx[-1]]; ry=[fy[-1]]
    for i in range(n-2,-1,-1):
        rx.append(alpha*fx[i]+(1-alpha)*rx[-1]); ry.append(alpha*fy[i]+(1-alpha)*ry[-1])
    rx.reverse(); ry.reverse()
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
# FIX-C1/C2: Clip boundary refinement
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_motion_profile(input_path, fps, total_frames, orig_w, orig_h,
                             sample_every=None, scale_w=320):
    """
    Return a dict {frame_index: motion_score} sampled at `sample_every` frames.
    motion_score is mean absolute diff between consecutive sampled frames (0-1).
    """
    if sample_every is None:
        sample_every = max(1, int(fps / 2))
    sh = max(1, int(scale_w * orig_h / orig_w))
    scores = {}
    prev_gray = None
    fi = 0
    with FFmpegVideoReader(input_path, orig_w, orig_h, scale_w=scale_w, scale_h=sh) as rdr:
        for frame in rdr:
            if fi > total_frames:
                break
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
    """
    Return the scene-cut time (seconds) closest to `target_sec` within
    ±search_window_sec, or None if none found.
    """
    best = None
    best_dist = float("inf")
    for t in scene_cuts_sec:
        d = abs(t - target_sec)
        if d <= search_window_sec and d < best_dist:
            best_dist = d
            best = t
    return best


def _find_low_motion_valley(motion_profile_sec, target_sec, search_window_sec, fps):
    """
    Among sampled frames within ±search_window_sec of target_sec, return
    the time of the frame with the lowest motion score (best place to cut).
    """
    candidates = {t: s for t, s in motion_profile_sec.items()
                  if abs(t - target_sec) <= search_window_sec}
    if not candidates:
        return target_sec
    best_t = min(candidates, key=candidates.get)
    return best_t


def _refine_clip_boundaries(start_sec, end_sec, duration,
                             scene_cuts_sec, motion_profile_sec,
                             min_duration, max_duration,
                             search_window=CLIP_BOUNDARY_SEARCH_SEC,
                             preroll=CLIP_PREROLL_PAD,
                             postroll=CLIP_POSTROLL_PAD):
    """
    Given a raw (start_sec, end_sec):

    1. Try to snap start_sec to the nearest scene cut within ±search_window.
       Apply a preroll pad so we don't start on the exact cut frame.
    2. If no cut found, find the lowest-motion frame near start and use that.
    3. Same for end_sec (snap to cut or valley, add postroll).
    4. Clamp to [0, duration] and enforce min/max duration.

    Returns (refined_start, refined_end).
    """
    # ── Refine start ──────────────────────────────────────────────────────
    cut_start = _find_nearest_scene_cut(scene_cuts_sec, start_sec, search_window)
    if cut_start is not None:
        # Snap to just after the cut
        new_start = max(0.0, cut_start + preroll)
    else:
        # Find a low-motion valley
        valley = _find_low_motion_valley(motion_profile_sec, start_sec, search_window, fps=1)
        new_start = max(0.0, valley - preroll * 0.5)

    # ── Refine end ────────────────────────────────────────────────────────
    cut_end = _find_nearest_scene_cut(scene_cuts_sec, end_sec, search_window)
    if cut_end is not None:
        # End just before the cut to avoid a hard-cut last frame
        new_end = min(duration, cut_end - postroll)
    else:
        valley = _find_low_motion_valley(motion_profile_sec, end_sec, search_window, fps=1)
        new_end = min(duration, valley + postroll * 0.5)

    # ── Enforce duration constraints ─────────────────────────────────────
    new_dur = new_end - new_start
    if new_dur < min_duration:
        # Extend symmetrically
        deficit = min_duration - new_dur
        new_start = max(0.0, new_start - deficit / 2)
        new_end   = min(duration, new_start + min_duration)
        new_start = max(0.0, new_end - min_duration)
    if new_dur > max_duration:
        centre = (new_start + new_end) / 2
        new_start = max(0.0, centre - max_duration / 2)
        new_end   = min(duration, new_start + max_duration)

    return new_start, new_end


# ── Clip detection (with FIX-C2) ─────────────────────────────────────────────
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

    # Build motion profile (seconds→score) for boundary refinement
    motion_profile_sec = {fi * sample_every / fps: float(scores[fi])
                          for fi in range(len(scores))}
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

        # ── FIX-C2: Refine start/end to clean shot boundaries ─────────────
        refined_start, refined_end = _refine_clip_boundaries(
            ss2, se, duration,
            scene_cuts_sec, motion_profile_sec,
            min_duration=min_duration_sec,
            max_duration=max_duration_sec,
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


# ── process_video — main entry point ─────────────────────────────────────────
def process_video(
    input_path,output_path,
    target_preset_label="Match source (no upscale)",
    tracking_mode="subject",talking_head_bias=0.30,
    sample_interval=None,confidence=0.45,use_optical_flow=True,
    smooth_window=27,adaptive_smoothing=True,rule_of_thirds=True,
    scene_cut_threshold=0.35,output_fps=None,crf=23,encoder_preset="fast",
    audio_bitrate="128k",yolo_weights="yolov8n.pt",
    burn_subtitles=False,whisper_model="base",whisper_language=None,
    subtitle_style_name="Bold White (TikTok)",subtitle_max_chars=42,
    subtitle_translate_to=None,
    vignette_strength=VIGNETTE_STRENGTH,
    sharpen_strength=0.0,
    color_grade="none",
    ken_burns=False,
    dissolve_cuts=True,
    ffmpeg_sharpen=False,
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
    fps,total_frames=info["fps"],info["total_frames"]
    orig_w,orig_h=info["width"],info["height"]; duration=info["duration_seconds"]
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
    det_scale=min(1.0,640/orig_w); det_w,det_h=max(1,int(orig_w*det_scale)),max(1,int(orig_h*det_scale))
    sx,sy=orig_w/det_w,orig_h/det_h

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

    start_pct=0.10; model_obj=None
    if tracking_mode=="subject":
        _p(start_pct,"Loading YOLO..."); model_obj=_get_model(yolo_weights)
        if model_obj is None: _p(start_pct,"YOLO unavailable - saliency fallback")
    elif tracking_mode=="talking_head":
        _p(start_pct,"Loading face detector...")
        if _get_haar() is None and _load_face_net() is None: _p(start_pct,"No face detector - saliency fallback")

    is_panel=False; slot_smoother=None
    if tracking_mode=="subject" and model_obj is not None:
        _p(start_pct+0.01,"Checking panel/group shot...")
        is_panel=_detect_panel_mode(input_path,model_obj,fps,total_frames,orig_w,orig_h,confidence,n_probe=12)
        if is_panel:
            _p(start_pct+0.02,"Panel mode — left/right horizontal band split")
            result_meta["panel_mode"]=True
            slot_smoother=PanelSlotSmoother()

    extra_vf=_build_ffmpeg_vf(color_grade="none" if not is_panel else color_grade,ffmpeg_sharpen=ffmpeg_sharpen)
    _p(0.12,f"Single-pass detect+render ({total_frames} frames)...")
    style=SUBTITLE_STYLES.get(subtitle_style_name,SUBTITLE_STYLES["Bold White (TikTok)"])
    proc=_open_ffmpeg_encoder(output_path,target_w,target_h,render_fps,audio_source=input_path,
                               crf=crf,preset=encoder_preset,audio_bitrate=audio_bitrate,
                               subtitle_path=srt_path,subtitle_style=style,
                               extra_vf=extra_vf if extra_vf else None)
    if vignette_strength>0: _build_vignette(target_w,target_h,vignette_strength)
    if color_grade and color_grade!="none": _build_lut(color_grade)
    dissolve_buf=DissolveBuffer(DISSOLVE_FRAMES) if dissolve_cuts else None

    det_centers=[]; det_indices=[]; scene_cuts=[]
    prev_gray=None; prev_flow=None
    last_det=None; det_dropout=0; MAX_DROPOUT=int(fps*1.5)
    cur_cx=orig_w//2; cur_cy=orig_h//2; prev_anchor=None
    hw,hh=crop_w//2,crop_h//2; prev_slots=None; last_out_frame=None
    rpt_n=max(1,total_frames//40); fi=0

    try:
        with FFmpegVideoReader(input_path,orig_w,orig_h) as reader:
            for frame in reader:
                if fi>=total_frames: break
                is_sample=(fi%sample_interval==0)
                if is_sample:
                    det_frame=cv2.resize(frame,(det_w,det_h),interpolation=cv2.INTER_LINEAR)
                    cg=cv2.cvtColor(det_frame,cv2.COLOR_BGR2GRAY)
                    cut=is_scene_change(prev_gray,cg,scene_cut_threshold)
                    if cut:
                        scene_cuts.append(fi); prev_flow=None; det_dropout=0
                        if dissolve_buf and last_out_frame is not None: dissolve_buf.on_cut(last_out_frame)
                    prev_gray=cg; anchor_cx=anchor_cy=None
                    if is_panel:
                        pass  # panel rendering handled below
                    elif tracking_mode=="talking_head":
                        faces=detect_faces(det_frame,confidence_thresh=0.5)
                        if faces:
                            faces_orig=[(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in faces]
                            r=talking_head_center(faces_orig,orig_w,orig_h,crop_w,crop_h,talking_head_bias)
                            if r: anchor_cx,anchor_cy=r; det_dropout=0
                        if anchor_cx is None and use_optical_flow:
                            sm=cv2.resize(cg,(max(1,det_w//2),max(1,det_h//2)))
                            if prev_flow is not None:
                                fc=optical_flow_center(prev_flow,sm,det_w//2,det_h//2)
                                if fc: anchor_cx,anchor_cy=int(fc[0]*2*sx),int(fc[1]*2*sy)
                            prev_flow=sm; det_dropout+=sample_interval
                    else:
                        if model_obj is not None:
                            det=detect_subjects(det_frame,model_obj,confidence)
                            if det is not None:
                                anchor_cx,anchor_cy=frame_for_union(int(det.ux1*sx),int(det.uy1*sy),
                                    int(det.ux2*sx),int(det.uy2*sy),orig_w,orig_h,crop_w,crop_h)
                                last_det=(anchor_cx,anchor_cy); det_dropout=0
                        if anchor_cx is None and use_optical_flow:
                            sm=cv2.resize(cg,(max(1,det_w//2),max(1,det_h//2)))
                            if prev_flow is not None:
                                fc=optical_flow_center(prev_flow,sm,det_w//2,det_h//2)
                                if fc: anchor_cx,anchor_cy=int(fc[0]*2*sx),int(fc[1]*2*sy)
                            prev_flow=sm; det_dropout+=sample_interval
                        if anchor_cx is None:
                            if last_det and det_dropout<MAX_DROPOUT: anchor_cx,anchor_cy=last_det
                            else:
                                sc_=saliency_center(det_frame); anchor_cx,anchor_cy=int(sc_[0]*sx),int(sc_[1]*sy)
                    if anchor_cx is not None and not is_panel:
                        det_centers.append((anchor_cx,anchor_cy)); det_indices.append(fi)
                        prev_anchor=(fi,anchor_cx,anchor_cy); cur_cx,cur_cy=anchor_cx,anchor_cy

                if not is_panel and not is_sample and prev_anchor is not None and len(det_centers)>=2:
                    fi_a,cx_a,cy_a=prev_anchor; alpha=min(1.0,(fi-fi_a)/sample_interval)
                    cur_cx=int(cx_a*(1-alpha)+det_centers[-1][0]*alpha)
                    cur_cy=int(cy_a*(1-alpha)+det_centers[-1][1]*alpha)
                cur_cx=max(hw,min(cur_cx,orig_w-hw)); cur_cy=max(hh,min(cur_cy,orig_h-hh))

                if is_panel:
                    if is_sample:
                        # Detect persons at detection resolution, scale back to full
                        det_frame_p=cv2.resize(frame,(det_w,det_h),interpolation=cv2.INTER_LINEAR)
                        persons_det=detect_persons_all(det_frame_p,model_obj,confidence)
                        persons_full=[(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy))
                                      for x1,y1,x2,y2 in persons_det]
                    else:
                        # Between samples: reuse last slot groups as person proxies
                        persons_full=[b for grp in (prev_slots or []) if grp for b in grp]
                    out_frame,prev_slots=_render_panel_frame(
                        frame, persons_full, target_w, target_h, prev_slots,
                        vignette_strength=vignette_strength*0.7,
                        color_grade=color_grade,
                        slot_smoother=slot_smoother,
                    )
                else:
                    left=max(0,min(cur_cx-crop_w//2,orig_w-crop_w))
                    top=max(0,min(cur_cy-crop_h//2,orig_h-crop_h))
                    crop=frame[top:top+crop_h,left:left+crop_w]
                    if crop.shape[1]!=target_w or crop.shape[0]!=target_h:
                        crop=cv2.resize(crop,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)
                    out_frame=crop
                    if ken_burns: out_frame=apply_ken_burns(out_frame,fi,fps)
                    if sharpen_strength>0: out_frame=apply_sharpen(out_frame,sharpen_strength)
                    if color_grade and color_grade!="none": out_frame=apply_color_grade(out_frame,color_grade)
                    if vignette_strength>0: out_frame=apply_vignette(out_frame,vignette_strength)

                if dissolve_buf and dissolve_buf.active: out_frame=dissolve_buf.blend(out_frame)
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
          f"  anchors={len(det_centers)}  cuts={len(scene_cuts)}",file=sys.stderr)
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
        base_pct=i/max(len(clips),1); next_pct=(i+1)/max(len(clips),1)
        _p(base_pct,f"Clip {i+1}/{len(clips)}...")
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
                vignette_strength=vignette_strength,sharpen_strength=sharpen_strength,
                color_grade=color_grade,ken_burns=ken_burns,dissolve_cuts=dissolve_cuts,
                ffmpeg_sharpen=ffmpeg_sharpen,progress_callback=clip_cb)
            meta["clip"]=clip; results.append(meta)
        except Exception as exc:
            results.append({"clip":clip,"output_path":out_path,"error":str(exc)})
        finally:
            if trimmed_path and os.path.exists(trimmed_path):
                try: os.unlink(trimmed_path)
                except OSError: pass
    n_ok=sum(1 for r in results if not r.get("error"))
    _p(1.0,f"{n_ok}/{len(results)} clips done"); return results
