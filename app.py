"""
app.py — Reframe · AI Vertical Studio
Mobile-first · Light Theme · Long-Form Auto-Clips
"""
import streamlit as st
import tempfile
import os
import zipfile
import io
from verticalize import (
    process_video, detect_clips, process_clips_batch, get_video_info,
    RESOLUTION_PRESETS, SUBTITLE_STYLES, TRANSLATION_LANGUAGES,
    resolve_target_size, whisper_available, translation_available,
    ClipSegment
)

st.set_page_config(page_title="Reframe · AI", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────
# MOBILE-FIRST LIGHT THEME
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg: #f8fafc; --surf: #ffffff; --surf2: #f1f5f9; --bdr: #e2e8f0;
  --ink: #0f172a; --ink2: #475569; --ink3: #94a3b8;
  --acc: #ea580c; --acc-bg: #fff7ed; --acc-hv: #c2410c;
  --grn: #059669; --grn-bg: #edf7f2; --prp: #7c3aed; --prp-bg: #f3eeff;
  --r: 12px; --sh: 0 2px 8px rgba(0,0,0,0.05);
}
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; background: var(--bg) !important; color: var(--ink) !important; }
.stApp { background: var(--bg) !important; padding-top: 0 !important; }
.main .block-container { max-width: 1200px !important; padding: 0 !important; }

.rf-hd { background: var(--surf); border-bottom: 1px solid var(--bdr); padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
.rf-hd-t { font-weight: 800; font-size: 1.1rem; display: flex; align-items: center; gap: 8px; }
.rf-hd-tag { font-size: 10px; font-weight: 700; background: var(--acc-bg); color: var(--acc); padding: 3px 8px; border-radius: 99px; }

.rf-modes { display: flex; gap: 10px; padding: 16px 16px 0; flex-wrap: wrap; }
.rf-mode { flex: 1; min-width: 150px; padding: 14px; border: 2px solid var(--bdr); background: var(--surf); border-radius: var(--r); cursor: pointer; transition: 0.2s; }
.rf-mode.sel { border-color: var(--acc); background: var(--acc-bg); box-shadow: var(--sh); }
.rf-mode-t { font-weight: 700; font-size: 13px; margin-bottom: 2px; }
.rf-mode-d { font-size: 11px; color: var(--ink3); }

.rf-sec { padding: 0 16px; margin-top: 20px; }
.rf-upload { border: 2px dashed var(--bdr); background: var(--surf); border-radius: var(--r); padding: 24px 16px; text-align: center; cursor: pointer; transition: 0.2s; }
.rf-upload:hover { border-color: var(--acc); background: var(--acc-bg); }
.rf-card { background: var(--surf); border: 1px solid var(--bdr); border-radius: var(--r); padding: 14px; margin-top: 12px; }
.rf-lbl { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink3); margin-bottom: 6px; }

.stButton > button { height: 48px !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 14px !important; width: 100%; }
.stButton > button[kind="primary"] { background: var(--acc) !important; color: #fff !important; }
.stButton > button[kind="primary"]:hover { background: var(--acc-hv) !important; transform: translateY(-1px) !important; }
.stButton > button[kind="secondary"] { background: var(--surf) !important; border: 1.5px solid var(--bdr) !important; color: var(--ink2) !important; }
.stDownloadButton > button { background: var(--grn) !important; color: #fff !important; border: none !important; height: 44px !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; }

.rf-clips { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; margin-top: 20px; }
.rf-clip { background: var(--surf); border: 1px solid var(--bdr); border-radius: var(--r); overflow: hidden; transition: 0.2s; }
.rf-clip:hover { border-color: var(--acc); box-shadow: var(--sh); }
.rf-clip-info { padding: 12px; }
.rf-clip-t { font-weight: 700; font-size: 13px; }
.rf-clip-d { font-size: 11px; color: var(--ink3); margin-top: 4px; }
.rf-badge { display: inline-block; background: var(--grn-bg); color: var(--grn); font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-top: 6px; }

@media (max-width: 600px) {
  .rf-modes { flex-direction: column; }
  .rf-clips { grid-template-columns: 1fr; }
  .rf-sec { padding: 0 12px; }
  .stSelectbox, .stSlider, .stToggle { font-size: 13px !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# STATE & INIT
# ─────────────────────────────────────────────────────────────────────────
if "mode" not in st.session_state: st.session_state.mode = "single"
if "clips" not in st.session_state: st.session_state.clips = []
if "results" not in st.session_state: st.session_state.results = []
if "busy" not in st.session_state: st.session_state.busy = False
if "out_bytes" not in st.session_state: st.session_state.out_bytes = None

def reset(): st.session_state.clips, st.session_state.results, st.session_state.busy, st.session_state.out_bytes = [], [], False, None

# ─────────────────────────────────────────────────────────────────────────
# HEADER & MODES
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="rf-hd"><div class="rf-hd-t">🎬 Reframe <span class="rf-hd-tag">AI</span></div><div style="font-size:11px;color:var(--ink3)">v2.1 Mobile</div></div>', unsafe_allow_html=True)

st.markdown('<div class="rf-modes">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: st.button("📱 Single Clip", type="primary" if st.session_state.mode=="single" else "secondary", use_container_width=True, on_click=lambda: setattr(st.session_state, "mode", "single") or reset())
with c2: st.button("🎬 Long Video → Clips", type="primary" if st.session_state.mode=="clips" else "secondary", use_container_width=True, on_click=lambda: setattr(st.session_state, "mode", "clips") or reset())
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# WORKSPACE
# ─────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="rf-sec">', unsafe_allow_html=True)
    
    up = st.file_uploader("Upload landscape video (MP4/MOV/MKV)", type=["mp4","mov","mkv"], label_visibility="collapsed")
    if up and len(up.getvalue()) > 500*1024**2:
        st.error("⚠️ Max 500MB"); up = None
        
    if not up:
        st.markdown('<div class="rf-upload"><div style="font-size:36px;margin-bottom:6px">📤</div><div style="font-weight:600;font-size:15px">Drop video to start</div><div style="font-size:12px;color:var(--ink3)">Works best with 16:9 or 4:3</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(up.getvalue()); tf.close()
    try: info = get_video_info(tf.name)
    except: info = None
    
    if info and not info["is_landscape"]:
        st.warning("⚠️ Already vertical. Upload landscape."); st.stop()
        
    st.markdown('<div class="rf-card">', unsafe_allow_html=True)
    r1, r2 = st.columns([2, 2])
    with r1:
        st.markdown('<div class="rf-lbl">Output Resolution</div>')
        res = st.selectbox("", list(RESOLUTION_PRESETS.keys()), index=0, label_visibility="collapsed")
    with r2:
        st.markdown('<div class="rf-lbl">Tracking</div>')
        trk = "subject" if st.selectbox("", ["Subject Tracking", "Talking Head"], index=0, label_visibility="collapsed")=="Subject Tracking" else "talking_head"
    subs = st.toggle("Burn Captions (Whisper)", value=False, disabled=not whisper_available())
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    
    if st.session_state.mode == "single":
        if st.button("▶ Convert to Vertical", type="primary", disabled=st.session_state.busy, use_container_width=True):
            st.session_state.busy = True
            out = tempfile.mktemp(suffix=".mp4")
            prog, stat = st.progress(0), st.empty()
            try:
                meta = process_video(tf.name, out, target_preset_label=res, tracking_mode=trk, burn_subtitles=subs, avoid_lower_third=True, progress_callback=lambda v,m: (prog.progress(v), stat.info(m) if m else None))
                with open(out, "rb") as f: st.session_state.out_bytes = f.read()
                stat.success("✅ Ready!"); prog.progress(1.0)
            except Exception as e: stat.error(f"Error: {e}")
            finally: st.session_state.busy = False
            
        if "out_bytes" in st.session_state and st.session_state.out_bytes:
            st.video(st.session_state.out_bytes)
            st.download_button("⬇️ Download", st.session_state.out_bytes, "vertical.mp4", "video/mp4", use_container_width=True)
            
    else:
        st.markdown(f'<div class="rf-lbl">Source: {info["duration_seconds"]//60}m · Target: 30-60s clips</div>', unsafe_allow_html=True)
        out_dir = tempfile.mkdtemp()
        
        if not st.session_state.clips:
            if st.button("🔍 Scan & Extract Clips", type="primary", disabled=st.session_state.busy, use_container_width=True):
                st.session_state.busy = True
                prog, stat = st.progress(0), st.empty()
                try:
                    clips = detect_clips(tf.name, min_duration_sec=30, max_duration_sec=60, target_n_clips=8, progress_callback=lambda v,m: (prog.progress(v*0.3), stat.info(m) if m else None))
                    st.session_state.clips = clips
                    stat.success(f"✅ Found {len(clips)} high-engagement segments")
                    prog.progress(0.3)
                except Exception as e: stat.error(f"Scan failed: {e}")
                finally: st.session_state.busy = False
        else:
            if not st.session_state.results:
                sel = [c for c in st.session_state.clips]
                if st.button(f"▶ Verticalize {len(sel)} Clips", type="primary", disabled=st.session_state.busy or not sel, use_container_width=True):
                    st.session_state.busy = True
                    prog, stat = st.progress(0.3), st.empty()
                    try:
                        res_batch = process_clips_batch(tf.name, out_dir, sel, target_preset_label=res, tracking_mode=trk, burn_subtitles=subs, progress_callback=lambda v,m: (prog.progress(0.3 + v*0.7), stat.info(m) if m else None))
                        st.session_state.results = res_batch
                        stat.success("✅ Clips processed!")
                    except Exception as e: stat.error(f"Batch error: {e}")
                    finally: st.session_state.busy = False
                    
            if st.session_state.results:
                st.markdown('<div class="rf-clips">', unsafe_allow_html=True)
                for i, r in enumerate(st.session_state.results):
                    if r.get("error"):
                        st.markdown(f'<div class="rf-clip"><div class="rf-clip-info"><div class="rf-clip-t">Clip {i+1}</div><div style="color:var(--acc);font-size:12px">⚠️ {r["error"]}</div></div></div>', unsafe_allow_html=True)
                    elif os.path.exists(r["output_path"]):
                        with open(r["output_path"], "rb") as f: b = f.read()
                        st.markdown(f'''
                        <div class="rf-clip">
                          <video controls preload="metadata" style="width:100%;border-bottom:1px solid var(--bdr)"><source src="data:video/mp4;base64,{b}" type="video/mp4"></video>
                          <div class="rf-clip-info">
                            <div class="rf-clip-t">Clip {i+1}</div>
                            <div class="rf-clip-d">{r["clip"].start_sec:.0f}s → {r["clip"].end_sec:.0f}s · {os.path.getsize(r["output_path"])//1024} KB</div>
                            <span class="rf-badge">SOI Locked · Lower-Third Safe</span>
                            <div style="margin-top:8px">
                              <a href="data:video/mp4;base64,{b}" download="clip_{i+1}.mp4" style="display:block;padding:8px;background:var(--surf2);border:1px solid var(--bdr);border-radius:8px;text-align:center;text-decoration:none;color:var(--ink);font-size:12px;font-weight:600">⬇️ Download</a>
                            </div>
                          </div>
                        </div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for r in st.session_state.results:
                        if not r.get("error") and os.path.exists(r["output_path"]):
                            zf.write(r["output_path"], os.path.basename(r["output_path"]))
                st.download_button("⬇️ Download All (ZIP)", zip_buf.getvalue(), "clips.zip", use_container_width=True)
                
    st.markdown('</div>', unsafe_allow_html=True)
