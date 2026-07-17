"""
app.py — Reframe · AI Vertical Video Studio
Mobile-first · Light theme · Single Clip + Auto-Clip modes

v5.1: Added Cinematic Mode for actor/dialogue-first reframing.
v5.0: Enhanced panel mode with N-person support, speaker focus,
head normalization, lower-third awareness, and portrait extraction.
"""
import streamlit as st
import tempfile
import os
import shutil
from verticalize import (
    process_video, process_sports_video, process_cinematic_video, get_video_info, detect_clips, process_clips_batch,
    RESOLUTION_PRESETS, SUBTITLE_STYLES, TRANSLATION_LANGUAGES,
    resolve_target_size, whisper_available, translation_available,
    PanelModeConfig,
)

st.set_page_config(page_title="Reframe", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root{--bg:#080a08;--surf:#0d100d;--surf2:#111511;--surf3:#161b16;--ink:#39ff14;--ink2:#7fdc7f;--ink3:#3f7a3f;--bdr:#1c2a1c;--bdr2:#2c452c;--acc:#39ff14;--acc-l:#0e1c0e;--acc-d:#28cc0e;--grn:#39ff14;--grn-l:#0e1c0e;--pur:#39ff14;--pur-l:#0e1c0e;--amb:#ffb000;--amb-l:#231a05;--r:0px;--rs:0px}
*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Courier New',ui-monospace,Consolas,monospace!important;background:var(--bg)!important;color:var(--ink)!important}
.stApp{background:var(--bg)!important}.main .block-container{padding:0!important;max-width:100%!important}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="collapsedControl"],section[data-testid="stSidebar"]{display:none!important}
.rf-top{height:44px;background:var(--surf);border-bottom:1px solid var(--bdr);display:flex;align-items:center;justify-content:space-between;padding:0 20px;position:sticky;top:0;z-index:200}
.rf-logo{display:flex;align-items:center;gap:9px}.rf-mark{width:22px;height:22px;border:1px solid var(--ink);background:transparent;display:flex;align-items:center;justify-content:center}
.rf-mark svg rect{fill:var(--ink)!important}
.rf-name{font-family:'Courier New',monospace;font-size:15px;font-weight:700;color:var(--ink);letter-spacing:0.04em;text-transform:uppercase}.rf-tag{font-size:11px;color:var(--ink3)}
.rf-sec{font-size:10px;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:var(--ink3);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.rf-sec::before{content:'>';color:var(--ink)}
.rf-sec::after{content:'';flex:1;height:1px;background:var(--bdr)}
.rf-mode-box{border-radius:var(--r);padding:10px 14px;display:flex;gap:10px;align-items:flex-start;margin-top:8px}
.rf-mode-box.acc{background:var(--surf2);border:1px solid var(--bdr2)}.rf-mode-box.pur{background:var(--surf2);border:1px solid var(--ink)}
.rf-mode-h{font-size:12px;font-weight:700;margin-bottom:2px;text-transform:uppercase;letter-spacing:0.05em}.rf-mode-h.acc{color:var(--ink)}.rf-mode-h.pur{color:var(--ink)}
.rf-mode-s{font-size:11px;color:var(--ink2);line-height:1.5}
[data-testid="stFileUploader"]{background:var(--surf)!important;border:1px dashed var(--bdr2)!important;border-radius:var(--r)!important}
[data-testid="stFileUploader"]:hover{border-color:var(--acc)!important;background:var(--acc-l)!important}
[data-testid="stFileUploadDropzone"]{padding:22px 14px!important}[data-testid="stFileUploadDropzone"] *{color:var(--ink3)!important;font-family:'Courier New',monospace!important;font-size:12px!important}
[data-testid="stVideo"]{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--bdr)!important}video{border-radius:var(--r)!important;width:100%!important}
.rf-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--bdr);border:1px solid var(--bdr);border-radius:var(--r);overflow:hidden}
.rf-m{background:var(--surf);padding:10px 12px}.rf-ml{font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--ink3);margin-bottom:3px}
.rf-mv{font-family:'Courier New',monospace;font-size:16px;font-weight:700;color:var(--ink);letter-spacing:0}.rf-mv.a{color:var(--acc)}
.rf-analytics{background:var(--surf);border:1px solid var(--bdr);border-radius:var(--r);padding:16px;margin-top:12px}
.rf-an-title{font-size:11px;font-weight:700;color:var(--ink3);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.rf-an-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.rf-an-item{background:var(--surf2);padding:10px;border-radius:var(--rs);border:1px solid var(--bdr)}
.rf-an-label{font-size:10px;color:var(--ink2);margin-bottom:4px}.rf-an-val{font-family:'Courier New',monospace;font-size:15px;font-weight:700;color:var(--ink)}
.rf-an-val.good{color:var(--grn)}.rf-an-val.bad{color:#ff4444}.rf-an-sub{font-size:9px;color:var(--ink3);margin-top:2px}
.rf-ok{background:var(--surf2);border:1px solid var(--grn);border-radius:var(--rs);padding:9px 12px;font-size:12px;color:var(--grn);display:flex;align-items:center;gap:8px;font-weight:600;margin-bottom:8px}
.rf-warn{background:var(--amb-l);border:1px solid var(--amb);border-radius:var(--rs);padding:9px 12px;font-size:12px;color:var(--amb);margin-bottom:8px}
.rf-info{background:var(--surf2);border:1px solid var(--bdr2);border-radius:var(--rs);padding:9px 12px;font-size:12px;color:var(--ink2);margin-bottom:8px}
.rf-purp{background:var(--surf2);border:1px solid var(--bdr2);border-radius:var(--rs);padding:9px 12px;font-size:12px;color:var(--ink2);margin-bottom:8px}
.rf-empty{background:var(--surf);border:1px dashed var(--bdr2);border-radius:var(--r);padding:40px 20px;text-align:center;display:flex;flex-direction:column;align-items:center;gap:7px;min-height:180px;justify-content:center}
.rf-empty-icon{width:40px;height:40px;background:var(--surf2);border:1px solid var(--bdr);border-radius:0;font-size:18px;display:flex;align-items:center;justify-content:center;margin-bottom:3px}
.rf-empty-h{font-family:'Courier New',monospace;font-size:14px;font-weight:700;color:var(--ink3);text-transform:uppercase}.rf-empty-s{font-size:11px;color:var(--ink3);opacity:0.8}
.rf-ccard{background:var(--surf);border:1px solid var(--bdr);border-radius:var(--r);padding:12px;margin-bottom:8px;position:relative}
.rf-ccard.sel{border-color:var(--acc);background:var(--acc-l)}.rf-ccard.done{border-color:var(--grn);background:var(--surf2)}
.rf-cscore{position:absolute;top:10px;right:10px;font-size:10px;font-weight:700;padding:2px 7px;border-radius:0;color:var(--bg);background:var(--ink)}
.rf-cscore.h{background:var(--acc)}.rf-cscore.m{background:var(--amb)}
.rf-ctitle{font-size:11px;font-weight:700;color:var(--ink);margin-bottom:4px;padding-right:50px}.rf-cmeta{font-size:10px;color:var(--ink3);line-height:1.5}
.rf-cdur{display:inline-block;background:var(--surf2);border:1px solid var(--bdr);border-radius:0;font-size:10px;font-weight:700;color:var(--ink2);padding:1px 6px;margin-top:5px}
.rf-csoi{display:inline-block;background:var(--surf2);border:1px solid var(--bdr2);border-radius:0;font-size:10px;font-weight:600;color:var(--ink2);padding:1px 6px;margin-top:5px;margin-left:4px}
.stButton >button{font-family:'Courier New',monospace!important;border-radius:var(--rs)!important;font-weight:700!important;font-size:13px!important;text-transform:uppercase!important;letter-spacing:0.03em!important;transition:none!important}
.stButton >button[kind="primary"]{background:var(--ink)!important;color:var(--bg)!important;border:1px solid var(--ink)!important;padding:10px 20px!important}
.stButton >button[kind="primary"]:hover{background:var(--bg)!important;color:var(--ink)!important;transform:none!important}
.stButton >button[kind="primary"]:disabled{background:var(--surf2)!important;color:var(--ink3)!important;border-color:var(--bdr)!important;transform:none!important}
.stButton >button[kind="secondary"]{background:var(--surf)!important;color:var(--ink2)!important;border:1px solid var(--bdr2)!important}
.stButton >button[kind="secondary"]:hover{border-color:var(--ink)!important;color:var(--ink)!important}
.stDownloadButton >button{background:var(--ink)!important;color:var(--bg)!important;border:1px solid var(--ink)!important;border-radius:var(--rs)!important;font-family:'Courier New',monospace!important;font-weight:700!important;font-size:13px!important;text-transform:uppercase!important;letter-spacing:0.03em!important;padding:10px 18px!important;width:100%!important;transition:none!important}
.stDownloadButton >button:hover{background:var(--bg)!important;color:var(--ink)!important;transform:none!important}
.stProgress >div >div >div{background:var(--ink)!important;border-radius:0}.stProgress >div >div{background:var(--bdr)!important;border-radius:0;height:3px!important}.stProgress >div{height:3px!important}
[data-baseweb="select"] >div{background:var(--surf)!important;border-color:var(--bdr2)!important;border-radius:var(--rs)!important;font-family:'Courier New',monospace!important;font-size:13px!important;color:var(--ink)!important}
[data-baseweb="select"] *{color:var(--ink)!important}[data-baseweb="popover"],[data-baseweb="menu"]{background:var(--surf)!important;border:1px solid var(--bdr)!important;border-radius:0!important}
[data-baseweb="option"]{background:var(--surf)!important;color:var(--ink2)!important;font-size:13px!important}[data-baseweb="option"]:hover{background:var(--acc-l)!important;color:var(--acc)!important}
[data-baseweb="tab-list"]{background:var(--surf2)!important;border-radius:var(--rs)!important;padding:3px!important;gap:2px!important;border:1px solid var(--bdr)!important}
[data-baseweb="tab"]{background:transparent!important;border-radius:0!important;font-family:'Courier New',monospace!important;font-size:12px!important;font-weight:700!important;text-transform:uppercase!important;color:var(--ink3)!important;padding:6px 11px!important;border:none!important}
[aria-selected="true"][data-baseweb="tab"]{background:var(--bg)!important;color:var(--ink)!important}[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
.stSlider label{font-size:12px!important;color:var(--ink2)!important;font-weight:600!important;font-family:'Courier New',monospace!important}.stSlider [role="slider"]{background:var(--ink)!important;border:2px solid var(--bg)!important;border-radius:0!important}
.stSlider [data-testid="stSliderTrackFill"]{background:var(--ink)!important}.stSlider >div >div{background:var(--bdr)!important}
[data-testid="stSliderValue"]{color:var(--ink)!important;font-size:11px!important;font-weight:700!important}
[data-testid="stToggleSwitch"] >div{background:var(--bdr2)!important;border-radius:0!important}[data-testid="stToggleSwitch"][aria-checked="true"] >div{background:var(--ink)!important}
.rf-chip{display:inline-flex;align-items:center;gap:6px;background:var(--surf2);border:1px solid var(--bdr);border-radius:0;padding:4px 9px;font-size:11px;color:var(--ink2)}
.rf-chip strong{color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px}
.rf-safe{display:inline-flex;align-items:center;gap:4px;background:var(--surf2);border:1px solid var(--grn);border-radius:0;padding:2px 7px;font-size:10px;font-weight:600;color:var(--grn)}
.rf-foot{margin-top:32px;padding:12px 20px;border-top:1px solid var(--bdr);display:flex;align-items:center;justify-content:space-between}
.rf-tech{display:flex;gap:5px;flex-wrap:wrap}.rf-tech span{font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:3px 7px;border:1px solid var(--bdr);border-radius:0;color:var(--ink3)}
.rf-panel{padding:14px 20px}.rf-panelr{padding:14px 20px 14px 10px}
@media(max-width:768px){.rf-panel,.rf-panelr{padding:12px 14px}.rf-metrics{grid-template-columns:repeat(2,1fr)}.rf-an-grid{grid-template-columns:1fr}}
.stCaption,small{color:var(--ink3)!important;font-size:10px!important;font-family:'Courier New',monospace!important}[data-testid="stHorizontalBlock"]{gap:10px!important}
[data-testid="stRadio"] label{font-size:12px!important;color:var(--ink2)!important}[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p{font-size:12px!important}[data-testid="stRadio"] > div{gap:6px!important}
.rf-vplayer{width:202px;flex-shrink:0}.rf-vplayer [data-testid="stVideo"]{border-radius:0!important;overflow:hidden!important;height:360px!important;border:1px solid var(--bdr)!important}
.rf-vplayer video{width:202px!important;height:360px!important;object-fit:cover!important;border-radius:0!important;display:block!important}
h1,h2,h3,h4,h5,h6,p,span,div,label{font-family:'Courier New',ui-monospace,Consolas,monospace!important}
::selection{background:var(--ink);color:var(--bg)}
</style>
""", unsafe_allow_html=True)

_DEFAULTS = dict(
    input_path=None, uploaded_file_name=None, video_info=None,
    app_mode="single", tracking_mode="subject", sport_type="auto",
    output_path=None, processing_done=False,
    output_bytes=None, srt_bytes=None, last_settings=None, analytics_data=None,
    detected_clips=None, selected_clip_indices=None,
    clip_results=None, scan_done=False, clip_out_dir=None,
    playing_clip_idx=-1,
    panel_mode_override="auto",
    panel_max_motion=20.0, panel_min_area=0.03,
    panel_max_variance=2.5, panel_stability=0.60,
    panel_layout_mode="equal", panel_speaker_focus_ratio=0.60,
    panel_head_normalize=False, panel_lower_third_aware=False,
    panel_portrait_mode=False, panel_max_slots=4,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _cleanup():
    for key in ("input_path", "output_path"):
        p = st.session_state.get(key)
        if p and os.path.exists(p):
            try: os.unlink(p)
            except OSError: pass
    clip_dir = st.session_state.get("clip_out_dir")
    if clip_dir and os.path.isdir(clip_dir):
        try: shutil.rmtree(clip_dir)
        except OSError: pass
    st.session_state.update(
        input_path=None, output_path=None, output_bytes=None,
        srt_bytes=None, video_info=None, processing_done=False,
        detected_clips=None, selected_clip_indices=None,
        clip_results=None, scan_done=False, clip_out_dir=None,
        playing_clip_idx=-1, analytics_data=None)

def _new_out():
    fd, p = tempfile.mkstemp(suffix=".mp4"); os.close(fd); os.unlink(p)
    st.session_state.output_path = p

def _invalidate_if_changed(cur):
    if st.session_state.processing_done and st.session_state.last_settings != cur:
        st.session_state.processing_done = False
        st.session_state.output_bytes = None
        st.session_state.srt_bytes = None
        st.session_state.analytics_data = None

_whisper_ok = whisper_available()
_translate_ok = translation_available()

st.markdown("""
<div class="rf-top"><div class="rf-logo"><div class="rf-mark">
<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
<rect x="1" y="1" width="5" height="12" rx="1.5" fill="white"/>
<rect x="8" y="4" width="5" height="9" rx="1.5" fill="white" opacity="0.5"/>
</svg></div><span class="rf-name">Reframe</span></div>
<span class="rf-tag">AI Vertical Video</span></div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

st.markdown("<div style='padding:0 20px'>", unsafe_allow_html=True)
st.markdown('<div class="rf-sec">Mode</div>', unsafe_allow_html=True)
mc1, mc2 = st.columns(2, gap="small")
with mc1:
    if st.button("Single Clip", type="secondary", use_container_width=True):
        st.session_state.app_mode = "single"
with mc2:
    if st.button("Auto-Clip", type="secondary", use_container_width=True):
        st.session_state.app_mode = "autoClip"
app_mode = st.session_state.app_mode
if app_mode == "single":
    st.markdown("""
<div class="rf-mode-box acc"><div>
<div class="rf-mode-h acc">Single Clip</div>
<div class="rf-mode-s">Upload any landscape video. AI tracks your subject and converts to 9:16 in one pass.</div>
</div></div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="rf-mode-box pur"><div>
<div class="rf-mode-h pur">Auto-Clip <span style="background:var(--acc);color:#fff;font-size:9px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;padding:2px 6px;border-radius:99px;">AI</span></div>
<div class="rf-mode-s">Upload a 30–90 min video. AI scans for saliency peaks, detects narrative arcs, then verticalizes every selected clip.</div>
</div></div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("<div style='padding:0 20px'>", unsafe_allow_html=True)
st.markdown('<div class="rf-sec">Tracking Mode</div>', unsafe_allow_html=True)
tm1, tm2, tm3 = st.columns(3, gap="small")
with tm1:
    if st.button("Subject", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "subject"
with tm2:
    if st.button("Talking Head", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "talking_head"
with tm3:
    if st.button("Cinematic", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "cinematic"
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
if st.button("Sports Action · Ball-aware · Kalman", type="secondary", use_container_width=True):
    st.session_state.tracking_mode = "sports_action"

tracking_mode = st.session_state.tracking_mode
if tracking_mode == "sports_action":
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    sport_type = st.selectbox("Sport Type", ["auto","basketball","football","soccer","hockey"],
        index=["auto","basketball","football","soccer","hockey"].index(st.session_state.get("sport_type","auto")),
        help="Auto-detects playing surface.")
    st.session_state.sport_type = sport_type
else:
    sport_type = st.session_state.get("sport_type", "auto")

if tracking_mode == "sports_action":
    sd = st.session_state.get("sport_type","auto").title()
    st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px;margin-bottom:4px;"><span style="background:var(--acc);color:#fff;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;">Sports Action · {sd}</span><span style="font-size:11px;color:var(--ink3);">Ball-aware · Kalman tracking</span></div>', unsafe_allow_html=True)
elif tracking_mode == "talking_head":
    st.markdown('<div style="margin-top:6px;margin-bottom:4px;"><span style="background:var(--pur);color:#fff;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;">Talking Head</span></div>', unsafe_allow_html=True)
elif tracking_mode == "cinematic":
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-top:6px;margin-bottom:4px;"><span style="background:var(--amb);color:#fff;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;">Cinematic Mode</span><span style="font-size:11px;color:var(--ink3);">Actor/dialogue-first · sports disabled</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="margin-top:6px;margin-bottom:4px;"><span style="background:var(--ink);color:#fff;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;">Subject Tracking</span></div>', unsafe_allow_html=True)

if tracking_mode == "subject":
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("Panel Mode Settings", expanded=False):
        st.caption("For news panels, podcasts, interviews with 2+ people")
        panel_mode_override = st.radio("Panel mode", ["auto","force_on","force_off"],
            format_func={"auto":"Auto-detect","force_on":"[OK] Force ON","force_off":"[X] Force OFF"}.get,
            index=["auto","force_on","force_off"].index(st.session_state.get("panel_mode_override","auto")),
            help="Auto = detect panel layout automatically.")
        st.session_state.panel_mode_override = panel_mode_override
        if panel_mode_override != "force_off":
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rf-sec">Layout & Features</div>', unsafe_allow_html=True)
            layout_mode = st.radio("Layout mode", ["equal","speaker_focus","solo_spotlight","auto"],
                format_func={"equal":"Equal Split","speaker_focus":"Speaker Focus","solo_spotlight":"Solo Spotlight","auto":"Auto"}.get,
                index=["equal","speaker_focus","solo_spotlight","auto"].index(st.session_state.get("panel_layout_mode","equal")),
                help="How to distribute screen space among detected persons.")
            st.session_state.panel_layout_mode = layout_mode
            if layout_mode == "speaker_focus":
                st.session_state.panel_speaker_focus_ratio = st.slider("Speaker focus ratio", 0.50, 0.80,
                    float(st.session_state.get("panel_speaker_focus_ratio", 0.60)), 0.05, format="%.2f",
                    help="How much space the active speaker gets")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.session_state.panel_head_normalize = st.toggle("Equal head sizing", value=st.session_state.get("panel_head_normalize", False), help="Normalize crop scale so all faces are ~same pixel height")
                st.session_state.panel_lower_third_aware = st.toggle("Lower-third awareness", value=st.session_state.get("panel_lower_third_aware", False), help="Detect text banners and avoid cropping into them")
            with fc2:
                st.session_state.panel_portrait_mode = st.toggle("Portrait extraction", value=st.session_state.get("panel_portrait_mode", False), help="Head-and-shoulders crop from face")
                st.session_state.panel_max_slots = st.slider("Max persons", 2, 4, int(st.session_state.get("panel_max_slots", 4)), 1, help="Max people to track in panel mode")
        if panel_mode_override == "auto":
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rf-sec">Detection Sensitivity</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.panel_max_motion = st.slider("Max motion px", 5.0, 40.0, float(st.session_state.get("panel_max_motion", 20.0)), 1.0)
                st.session_state.panel_min_area = st.slider("Min person area pct", 0.01, 0.10, float(st.session_state.get("panel_min_area", 0.03)), 0.01, format="%.2f")
            with c2:
                st.session_state.panel_max_variance = st.slider("Max count variance", 0.5, 5.0, float(st.session_state.get("panel_max_variance", 2.5)), 0.5)
                st.session_state.panel_stability = st.slider("Stability fraction", 0.30, 0.90, float(st.session_state.get("panel_stability", 0.60)), 0.05, format="%.2f")

tab_list = ["Output", "Tracking", "Subtitles", "Advanced"]
if app_mode == "autoClip": tab_list += ["Clips", "Analytics"]
if app_mode == "autoClip":
    tab_out, tab_trk, tab_sub, tab_adv, tab_clip, tab_analytics = st.tabs(tab_list)
else:
    tab_out, tab_trk, tab_sub, tab_adv = st.tabs(tab_list)
    tab_clip = tab_analytics = None

with tab_out:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    o1, o2 = st.columns(2, gap="medium")
    with o1:
        resolution_label = st.selectbox("Resolution", list(RESOLUTION_PRESETS.keys()), index=0)
        fps_label = st.selectbox("Frame rate", ["Source - keep original","60 fps","30 fps","25 fps","24 fps"], index=0)
    with o2:
        crf = st.slider("Quality CRF", 15, 35, 23, 1)
        st.caption("18 = near-lossless · 28 = compact")
        encoder_preset_label = st.selectbox("Speed", ["ultrafast","fast","medium","slow"], index=1)
_fps_map = {"Source - keep original":None,"60 fps":60.0,"30 fps":30.0,"25 fps":25.0,"24 fps":24.0}
output_fps = _fps_map[fps_label]

with tab_trk:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if tracking_mode == "talking_head":
        th1, th2 = st.columns(2, gap="medium")
        with th1:
            talking_head_bias = st.slider("Upper-third pull", 0.0, 1.0, 0.30, 0.05)
            st.caption("0 = centered face · 1 = upper third")
            smooth_window = st.slider("Smoothness", 3, 31, 21, 2)
        with th2:
            adaptive_smoothing = st.toggle("Adaptive smoothing", value=False)
            use_optical_flow = st.toggle("Optical flow bridge", value=True)
            rule_of_thirds = st.toggle("Horizontal rule-of-thirds", value=True)
            confidence = 0.5; scene_cut_threshold = 0.35
        use_ball_tracking = False; use_kalman = False
    elif tracking_mode == "cinematic":
        st.markdown('<div class="rf-info" style="margin-bottom:12px;"> <b>Cinematic Mode Active</b> — actor/dialogue-first framing. Sports mode, ball tracking and overlays are disabled.</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            confidence = st.slider("Actor / face detection confidence", 0.10, 0.95, 0.45, 0.05)
            smooth_window = st.slider("Cinematic smoothness hint", 7, 41, 27, 2, help="Backend uses long-lens cinematic smoothing; this value is retained for settings compatibility.")
        with t2:
            scene_cut_threshold = st.slider("Shot-cut sensitivity", 0.10, 0.60, 0.32, 0.05)
            st.caption("Cinematic mode uses shot-aware smoothing and composition analysis.")
        adaptive_smoothing = True
        use_optical_flow = True
        rule_of_thirds = True
        talking_head_bias = 0.30
        use_ball_tracking = False
        use_kalman = False
    elif tracking_mode == "sports_action":
        st.markdown('<div class="rf-info" style="margin-bottom:12px;"> <b>Sports Mode Active</b> — Ball-aware tracking with Kalman smoothing.</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            adaptive_smoothing = st.toggle("Adaptive smoothing", value=True)
            smooth_window = st.slider("Smoothness", 3, 15, 5, 1)
            confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
            use_ball_tracking = st.toggle("Ball tracking", value=True, help="Prioritize ball carrier")
        with t2:
            use_optical_flow = st.toggle("Optical flow fallback", value=True)
            rule_of_thirds = st.toggle("Look-room / Rule-of-thirds", value=True)
            scene_cut_threshold = st.slider("Scene-cut sensitivity", 0.10, 0.60, 0.22, 0.05)
            use_kalman = st.toggle("Kalman prediction", value=True, help="Zero-lag predictive tracking")
        talking_head_bias = 0.30
    else:
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            adaptive_smoothing = st.toggle("Adaptive smoothing", value=True)
            smooth_window = st.slider("Smoothness", 3, 31, 15, 2)
            confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
        with t2:
            use_optical_flow = st.toggle("Optical flow fallback", value=True)
            rule_of_thirds = st.toggle("Look-room / Rule-of-thirds", value=True)
            scene_cut_threshold = st.slider("Scene-cut sensitivity", 0.10, 0.60, 0.35, 0.05)
        talking_head_bias = 0.30; use_ball_tracking = False; use_kalman = False

with tab_sub:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if not _whisper_ok:
        st.markdown('<div class="rf-purp">[!] Install <code>openai-whisper</code> to enable subtitles.</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2, gap="medium")
    with s1:
        burn_subtitles = st.toggle("Burn subtitles", value=False, disabled=not _whisper_ok)
        if not _whisper_ok: burn_subtitles = False
        translate_subtitles = st.toggle("Translate", value=False, disabled=(not _whisper_ok or not _translate_ok or not burn_subtitles))
        if not burn_subtitles or not _translate_ok: translate_subtitles = False
        whisper_model = st.selectbox("Whisper model", ["tiny","base","small","medium"], index=1, disabled=not _whisper_ok)
    with s2:
        subtitle_style_name = st.selectbox("Style", list(SUBTITLE_STYLES.keys()), disabled=not _whisper_ok)
        whisper_language_raw = st.selectbox("Audio language", ["Auto-detect","en","hi","es","fr","de","ja","zh","pt","ar"], disabled=not _whisper_ok)
        whisper_language = None if whisper_language_raw == "Auto-detect" else whisper_language_raw
        subtitle_max_chars = st.slider("Max chars/line", 20, 60, 42, 2, disabled=not _whisper_ok)
        subtitle_translate_label = st.selectbox("Translate to", list(TRANSLATION_LANGUAGES.keys()), index=0, disabled=(not _whisper_ok or not _translate_ok or not burn_subtitles or not translate_subtitles))
        subtitle_translate_to = TRANSLATION_LANGUAGES[subtitle_translate_label] or None
        if not translate_subtitles: subtitle_translate_to = None

with tab_adv:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="medium")
    with a1: audio_bitrate_label = st.selectbox("Audio bitrate", ["64k","96k","128k","192k"], index=2)
    with a2:
        yolo_weights = st.selectbox("YOLO model", ["yolov8n.pt","yolov8s.pt","yolov8m.pt"], index=0) if tracking_mode in ("subject", "cinematic") else "yolov8n.pt"
        st.markdown('<div class="rf-purp">Talking Head uses OpenCV face detector — YOLO not needed.</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-safe">✓ Lower-third guard — subjects kept above bottom 20% of frame</div>', unsafe_allow_html=True)

_CLIP_PRESETS = {"15 sec (snappy highlight)":(13,17),"30 sec (short reel)":(25,35),"60 sec (full segment)":(50,65)}
clip_min_dur = 25; clip_max_dur = 60; clip_target_n = 8
if app_mode == "autoClip" and tab_clip is not None:
    with tab_clip:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        cl1, cl2 = st.columns(2, gap="medium")
        with cl1:
            preset_label = st.radio("Clip length preset", list(_CLIP_PRESETS.keys()), index=2)
            clip_min_dur, clip_max_dur = _CLIP_PRESETS[preset_label]
        with cl2: clip_target_n = st.slider("Target # clips", 3, 20, 8, 1)
if app_mode == "autoClip" and tab_analytics is not None:
    with tab_analytics:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.session_state.clip_results:
            vr = [r for r in st.session_state.clip_results if not r.get("error") and "analytics" in r]
            if vr:
                ti = sum(r["analytics"]["input_size_mb"] for r in vr); to = sum(r["analytics"]["output_size_mb"] for r in vr)
                # Aggregate resource stats across clips
                total_cpu_avg = sum(r["analytics"].get("cpu_avg_pct", 0) for r in vr) / len(vr) if vr else 0
                total_cpu_max = max((r["analytics"].get("cpu_max_pct", 0) for r in vr), default=0)
                total_ram_avg = sum(r["analytics"].get("ram_avg_mb", 0) for r in vr) / len(vr) if vr else 0
                total_ram_max = max((r["analytics"].get("ram_max_mb", 0) for r in vr), default=0)
                total_proc_time = sum(r["analytics"].get("processing_time_sec", 0) for r in vr)
                has_batch_res = total_cpu_avg > 0 or total_ram_avg > 0

                res_html = ""
                if has_batch_res:
                    res_html = f"""<div class="rf-an-item"><div class="rf-an-label">Avg CPU</div><div class="rf-an-val">{total_cpu_avg:.1f}% <span style="font-size:11px;color:var(--ink3)">(max {total_cpu_max:.1f}%)</span></div></div><div class="rf-an-item"><div class="rf-an-label">Avg RAM</div><div class="rf-an-val">{total_ram_avg:.1f} MB <span style="font-size:11px;color:var(--ink3)">(max {total_ram_max:.1f} MB)</span></div></div><div class="rf-an-item"><div class="rf-an-label">Total Time</div><div class="rf-an-val">{total_proc_time:.1f}s</div></div>"""

                st.markdown(f'<div class="rf-analytics"><div class="rf-an-title">Batch Analytics</div><div class="rf-an-grid"><div class="rf-an-item"><div class="rf-an-label">Total Input</div><div class="rf-an-val">{ti:.1f} MB</div></div><div class="rf-an-item"><div class="rf-an-label">Total Output</div><div class="rf-an-val good">{to:.1f} MB</div></div><div class="rf-an-item"><div class="rf-an-label">Clips</div><div class="rf-an-val">{len(vr)}</div></div>{res_html}</div></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="rf-empty" style="min-height:120px;padding:20px;"><div class="rf-empty-s">Process clips to view analytics</div></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

def _build_panel_config():
    """Build PanelModeConfig — works with both old and new backends."""
    base = dict(
        split_mode=st.session_state.get("panel_mode_override", "auto"),
        max_person_motion=st.session_state.get("panel_max_motion", 20.0),
        min_person_area_frac=st.session_state.get("panel_min_area", 0.03),
        max_count_variance=st.session_state.get("panel_max_variance", 2.5),
        stability_frac=st.session_state.get("panel_stability", 0.60),
    )
    v5 = dict(
        layout_mode=st.session_state.get("panel_layout_mode", "equal"),
        speaker_focus_ratio=st.session_state.get("panel_speaker_focus_ratio", 0.60),
        head_normalize=st.session_state.get("panel_head_normalize", False),
        lower_third_aware=st.session_state.get("panel_lower_third_aware", False),
        portrait_mode=st.session_state.get("panel_portrait_mode", False),
        max_slots=st.session_state.get("panel_max_slots", 4),
    )
    try: return PanelModeConfig(**base, **v5)
    except TypeError: return PanelModeConfig(**base)

current_settings = dict(
    app_mode=app_mode, tracking_mode=tracking_mode,
    sport_type=st.session_state.get("sport_type","auto"),
    resolution_label=resolution_label, fps_label=fps_label, crf=crf,
    encoder_preset_label=encoder_preset_label,
    smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
    confidence=confidence, use_optical_flow=use_optical_flow,
    rule_of_thirds=rule_of_thirds, scene_cut_threshold=scene_cut_threshold,
    talking_head_bias=talking_head_bias,
    burn_subtitles=burn_subtitles,
    whisper_model=whisper_model if burn_subtitles else "",
    audio_bitrate_label=audio_bitrate_label, yolo_weights=yolo_weights,
    subtitle_style_name=subtitle_style_name if burn_subtitles else "",
    subtitle_max_chars=subtitle_max_chars if burn_subtitles else 0,
    whisper_language=whisper_language if burn_subtitles else None,
    subtitle_translate_to=subtitle_translate_to if burn_subtitles else None,
    panel_mode_override=st.session_state.get("panel_mode_override","auto"),
    panel_max_motion=st.session_state.get("panel_max_motion",20.0),
    panel_min_area=st.session_state.get("panel_min_area",0.03),
    panel_max_variance=st.session_state.get("panel_max_variance",2.5),
    panel_stability=st.session_state.get("panel_stability",0.60),
    panel_layout_mode=st.session_state.get("panel_layout_mode","equal"),
    panel_speaker_focus_ratio=st.session_state.get("panel_speaker_focus_ratio",0.60),
    panel_head_normalize=st.session_state.get("panel_head_normalize",False),
    panel_lower_third_aware=st.session_state.get("panel_lower_third_aware",False),
    panel_portrait_mode=st.session_state.get("panel_portrait_mode",False),
    panel_max_slots=st.session_state.get("panel_max_slots",4),
)
_invalidate_if_changed(current_settings)
st.markdown("<div style='height:2px;background:var(--bdr);margin:10px 0 0'></div>", unsafe_allow_html=True)

col_src, col_out = st.columns(2, gap="small")
with col_src:
    st.markdown("<div class='rf-panel'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec">Source Video</div>', unsafe_allow_html=True)
    max_mb = 2000 if app_mode == "autoClip" else 500
    uploaded_file = st.file_uploader("Drop video", type=["mp4","mov","avi","mkv"], label_visibility="collapsed")
    if uploaded_file is not None:
        mb = len(uploaded_file.getvalue())/(1024**2)
        if mb > max_mb: st.markdown(f'<div class="rf-warn">[!] {mb:.1f} MB — max {max_mb} MB.</div>', unsafe_allow_html=True); uploaded_file = None
    if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
        _cleanup()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getvalue()); st.session_state.input_path = tmp.name
        _new_out(); st.session_state.uploaded_file_name = uploaded_file.name
        try: st.session_state.video_info = get_video_info(st.session_state.input_path)
        except Exception: st.session_state.video_info = None
    if uploaded_file is not None and st.session_state.input_path:
        info = st.session_state.video_info
        if info and not info["is_landscape"]: st.markdown('<div class="rf-warn">[!] Video is already vertical.</div>', unsafe_allow_html=True)
        mb_str = f"{len(uploaded_file.getvalue())/(1024**2):.1f} MB"
        st.markdown(f'<div class="rf-chip"><span></span><strong>{uploaded_file.name}</strong><span style="color:var(--bdr2)">·</span><span>{mb_str}</span></div>', unsafe_allow_html=True)
        st.video(uploaded_file)
        if info:
            dur = info["duration_seconds"]; mins, secs = int(dur//60), int(dur%60)
            dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
            st.markdown(f"<div class='rf-metrics' style='margin-top:10px;'><div class='rf-m'><div class='rf-ml'>Duration</div><div class='rf-mv'>{dur_str}</div></div><div class='rf-m'><div class='rf-ml'>Source</div><div class='rf-mv'>{info['width']}×{info['height']}</div></div><div class='rf-m'><div class='rf-ml'>Output</div><div class='rf-mv a'>{eff_w}×{eff_h}</div></div><div class='rf-m'><div class='rf-ml'>FPS</div><div class='rf-mv'>{info['fps']:.0f}</div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    st.markdown("<div class='rf-panelr'>", unsafe_allow_html=True)
    if app_mode == "single":
        st.markdown('<div class="rf-sec">Output · 9:16</div>', unsafe_allow_html=True)
        if st.session_state.processing_done and st.session_state.output_bytes:
            out_mb = len(st.session_state.output_bytes)/(1024**2)
            st.markdown(f'<div class="rf-ok">✓ Done — {out_mb:.1f} MB</div>', unsafe_allow_html=True)
            st.video(st.session_state.output_bytes, format="video/mp4")
            if st.session_state.analytics_data:
                a = st.session_state.analytics_data
                if a.get("panel_mode"): st.markdown('<div class="rf-ok">Panel mode active</div>', unsafe_allow_html=True)
                if a.get("cinematic_mode"): st.markdown(f'<div class="rf-ok">Cinematic mode active · {int(a.get("scene_cuts", 0))} shot cut(s) · Sports disabled</div>', unsafe_allow_html=True)
                sp = a.get("smoothness_pct",0); sc_v = "var(--grn)" if sp>80 else ("var(--amb)" if sp>50 else "var(--acc)")
                # Build analytics grid with CPU/RAM metrics if available
                cpu_avg = a.get("cpu_avg_pct", 0)
                cpu_max = a.get("cpu_max_pct", 0)
                ram_avg = a.get("ram_avg_mb", 0)
                ram_max = a.get("ram_max_mb", 0)
                proc_time = a.get("processing_time_sec", 0)
                has_resource = cpu_avg > 0 or ram_avg > 0

                # Determine resource color coding
                cpu_color = "var(--grn)" if cpu_max < 50 else ("var(--amb)" if cpu_max < 80 else "var(--acc)")
                ram_color = "var(--grn)" if ram_max < 512 else ("var(--amb)" if ram_max < 1024 else "var(--acc)")

                resource_html = ""
                if has_resource:
                    resource_html = f"""<div class="rf-an-item"><div class="rf-an-label">CPU Usage</div><div class="rf-an-val" style="color:{cpu_color}">{cpu_avg:.1f}% <span style="font-size:11px;color:var(--ink3)">(max {cpu_max:.1f}%)</span></div></div><div class="rf-an-item"><div class="rf-an-label">RAM Usage</div><div class="rf-an-val" style="color:{ram_color}">{ram_avg:.1f} MB <span style="font-size:11px;color:var(--ink3)">(max {ram_max:.1f} MB)</span></div></div><div class="rf-an-item"><div class="rf-an-label">Processing Time</div><div class="rf-an-val">{proc_time:.1f}s</div></div>"""

                st.markdown(f'<div class="rf-analytics"><div class="rf-an-title">Analytics</div><div class="rf-an-grid"><div class="rf-an-item"><div class="rf-an-label">Size Reduction</div><div class="rf-an-val">{a.get("file_size_reduction_pct",0):.1f}%</div><div class="rf-an-sub">{a.get("input_size_mb",0):.1f} → {a.get("output_size_mb",0):.1f} MB</div></div><div class="rf-an-item"><div class="rf-an-label">Smoothness</div><div class="rf-an-val" style="color:{sc_v}">{sp:.1f}%</div></div><div class="rf-an-item"><div class="rf-an-label">Resolution</div><div class="rf-an-val">{a.get("output_resolution","")}</div></div>{resource_html}</div></div>', unsafe_allow_html=True)
            stem = os.path.splitext(st.session_state.uploaded_file_name or "video")[0]
            st.download_button("↓ Download vertical video", data=st.session_state.output_bytes, file_name=f"{stem}_vertical.mp4", mime="video/mp4", use_container_width=True)
            if st.session_state.srt_bytes: st.download_button("↓ Download subtitles (.srt)", data=st.session_state.srt_bytes, file_name=f"{stem}.srt", mime="text/plain", use_container_width=True)
        else:
            st.markdown('<div class="rf-empty"><div class="rf-empty-icon"></div><div class="rf-empty-h">Vertical output</div><div class="rf-empty-s">appears here after conversion</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rf-sec">Detected Clips</div>', unsafe_allow_html=True)
        if not st.session_state.scan_done:
            st.markdown('<div class="rf-empty"><div class="rf-empty-icon"></div><div class="rf-empty-h">Scan first</div><div class="rf-empty-s">AI will detect high-engagement segments</div></div>', unsafe_allow_html=True)
        elif st.session_state.detected_clips:
            clips = st.session_state.detected_clips
            if st.session_state.selected_clip_indices is None: st.session_state.selected_clip_indices = set(range(len(clips)))
            sel = st.session_state.selected_clip_indices
            st.markdown(f"<div style='font-size:11px;color:var(--ink3);margin-bottom:8px;'>{len(clips)} clips found · {len(sel)} selected</div>", unsafe_allow_html=True)
            crm = {}
            if st.session_state.clip_results:
                for r in st.session_state.clip_results:
                    co = r.get("clip")
                    if co: crm[(round(co.start_sec,1), round(co.end_sec,1))] = r
            pi = st.session_state.playing_clip_idx
            for ci, clip in enumerate(clips):
                sp2 = int(clip.score*100); sc2 = "h" if clip.score>0.7 else ("m" if clip.score>0.4 else "")
                is_s = ci in sel; is_p = pi == ci
                ck = (round(clip.start_sec,1), round(clip.end_sec,1)); rfc = crm.get(ck)
                is_d = rfc is not None and not rfc.get("error") and rfc.get("output_path") and os.path.exists(rfc["output_path"])
                ts = f"{int(clip.start_sec//60)}:{int(clip.start_sec%60):02d} → {int(clip.end_sec//60)}:{int(clip.end_sec%60):02d}"
                cc = "rf-ccard" + (" done" if is_d else (" sel" if is_s else ""))
                dt = "<div style='margin-top:5px;font-size:10px;color:var(--grn);font-weight:700;'>✓ Converted</div>" if is_d else ""
                st.markdown(f'<div class="{cc}"><span class="rf-cscore {sc2}">{sp2}%</span><div class="rf-ctitle">Clip {ci+1}</div><div class="rf-cmeta">{ts}</div><span class="rf-cdur">{clip.duration:.0f}s</span><span class="rf-csoi">SOI: {clip.soi_region}</span>{dt}</div>', unsafe_allow_html=True)
                if is_d:
                    bc, dc = st.columns([1,1])
                    with bc:
                        if st.button("⏹ Close" if is_p else "▶ Play 9:16", key=f"play_{ci}", type="secondary", use_container_width=True):
                            st.session_state.playing_clip_idx = -1 if is_p else ci; st.rerun()
                    with dc:
                        try:
                            with open(rfc["output_path"],"rb") as f: cb = f.read()
                            st.download_button("↓ Download", data=cb, file_name=f"clip_{ci+1}_vertical.mp4", mime="video/mp4", key=f"dl_{ci}", use_container_width=True)
                        except Exception: pass
                    if is_p:
                        try:
                            with open(rfc["output_path"],"rb") as f: cpb = f.read()
                            vc, _ = st.columns([202,400])
                            with vc: st.markdown('<div class="rf-vplayer">', unsafe_allow_html=True); st.video(cpb, format="video/mp4"); st.markdown('</div>', unsafe_allow_html=True)
                        except Exception: pass
                else:
                    cbc, _ = st.columns([2,1])
                    with cbc:
                        tog = st.checkbox("✓ Selected" if is_s else "Include", value=is_s, key=f"csel_{ci}")
                        if tog != is_s:
                            if tog: st.session_state.selected_clip_indices.add(ci)
                            else: st.session_state.selected_clip_indices.discard(ci)
                            st.rerun()
        else:
            st.markdown('<div class="rf-empty"><div class="rf-empty-icon"></div><div class="rf-empty-h">No clips found</div><div class="rf-empty-s">try adjusting clip duration</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None and st.session_state.input_path:
    info = st.session_state.video_info
    can_go = bool(info and info.get("is_landscape", True))
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:var(--bdr);margin:0 20px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if app_mode == "single":
        if not st.session_state.processing_done:
            a1, a2, a3 = st.columns([4,5,2])
            with a1: go = st.button("▶ Convert to Vertical", type="primary", use_container_width=True, disabled=not can_go)
            with a2:
                if info:
                    eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
                    mt = "Talking Head" if tracking_mode=="talking_head" else ("Sports" if tracking_mode=="sports_action" else ("Cinematic" if tracking_mode=="cinematic" else "Subject"))
                    st.markdown(f"<p style='color:var(--ink3);font-size:11px;margin-top:12px;'>{mt} · {eff_w}×{eff_h} · CRF {crf}</p>", unsafe_allow_html=True)
            with a3:
                if st.button("Clear", type="secondary", use_container_width=True): _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
            if go:
                st.session_state.last_settings = current_settings
                prog = st.progress(0.0); status = st.empty(); status.info("Starting…")
                try:
                    def _cb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg:
                            status.info(msg)
                    if tracking_mode == "cinematic":
                        meta = process_cinematic_video(st.session_state.input_path, st.session_state.output_path,
                            target_preset_label=resolution_label, confidence=confidence,
                            output_fps=output_fps, crf=crf, encoder_preset=encoder_preset_label,
                            audio_bitrate=audio_bitrate_label, yolo_weights=yolo_weights,
                            burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                            whisper_language=whisper_language,
                            subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                            subtitle_translate_to=subtitle_translate_to,
                            color_grade="warm", progress_callback=_cb)
                    elif tracking_mode == "sports_action":
                        meta = process_sports_video(st.session_state.input_path, st.session_state.output_path,
                            sport_type=st.session_state.get("sport_type","auto"),
                            target_preset_label=resolution_label, confidence=confidence,
                            smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
                            use_optical_flow=use_optical_flow, rule_of_thirds=rule_of_thirds,
                            scene_cut_threshold=scene_cut_threshold,
                            output_fps=output_fps, crf=crf, encoder_preset=encoder_preset_label,
                            audio_bitrate=audio_bitrate_label, yolo_weights=yolo_weights,
                            burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                            whisper_language=whisper_language,
                            subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                            subtitle_translate_to=subtitle_translate_to,
                            use_ball_tracking=use_ball_tracking, use_kalman=use_kalman,
                            progress_callback=_cb)
                    else:
                        meta = process_video(st.session_state.input_path, st.session_state.output_path,
                            target_preset_label=resolution_label, tracking_mode=tracking_mode,
                            talking_head_bias=talking_head_bias, confidence=confidence,
                            smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
                            use_optical_flow=use_optical_flow, rule_of_thirds=rule_of_thirds,
                            scene_cut_threshold=scene_cut_threshold,
                            output_fps=output_fps, crf=crf, encoder_preset=encoder_preset_label,
                            audio_bitrate=audio_bitrate_label, yolo_weights=yolo_weights,
                            burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                            whisper_language=whisper_language,
                            subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                            subtitle_translate_to=subtitle_translate_to,
                            panel_config=_build_panel_config(), progress_callback=_cb)
                    prog.progress(1.0)
                    out_p = st.session_state.output_path
                    if os.path.exists(out_p) and os.path.getsize(out_p) > 0:
                        with open(out_p,"rb") as f: st.session_state.output_bytes = f.read()
                        if "analytics" in meta: st.session_state.analytics_data = meta["analytics"]
                        srt_p = meta.get("subtitle_path")
                        if srt_p and os.path.exists(srt_p):
                            with open(srt_p,"rb") as f: st.session_state.srt_bytes = f.read()
                            try: os.unlink(srt_p)
                            except OSError: pass
                        st.session_state.processing_done = True; status.success("[OK] Done!"); st.rerun()
                    else: status.error("Output is empty — check FFmpeg.")
                except Exception as exc: status.error(f"Error: {exc}")
        else:
            r1, _, r2 = st.columns([2,5,2])
            with r1:
                if st.button("← Start over", type="secondary", use_container_width=True): _cleanup(); st.session_state.uploaded_file_name=None; st.session_state.processing_done=False; st.rerun()
    else:
        if st.session_state.detected_clips is None: st.session_state.scan_done = False
        if not st.session_state.scan_done:
            b1, b2, b3 = st.columns([4,4,2])
            with b1: scan_btn = st.button("Scan for Clips", type="primary", use_container_width=True, disabled=not can_go)
            with b3:
                if st.button("Clear", type="secondary", use_container_width=True): _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
            if scan_btn:
                prog=st.progress(0.0); status=st.empty(); status.info("Scanning…")
                try:
                    def _scb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg:
                            status.info(msg)
                    clips = detect_clips(st.session_state.input_path, min_duration_sec=float(clip_min_dur), max_duration_sec=float(clip_max_dur), target_n_clips=int(clip_target_n), model=None, confidence=confidence, progress_callback=_scb)
                    prog.progress(1.0)
                    if not clips: status.warning("[!] No clips detected."); st.session_state.detected_clips=[]; st.session_state.selected_clip_indices=set()
                    else: st.session_state.detected_clips=clips; st.session_state.selected_clip_indices=set(range(len(clips))); st.session_state.clip_results=None; status.success(f"[OK] Found {len(clips)} clips!")
                    st.session_state.scan_done=True; st.rerun()
                except Exception as exc: status.error(f"Scan error: {exc}")
        else:
            clips = st.session_state.detected_clips or []
            if not clips:
                st.markdown('<div class="rf-warn">[!] No clips detected. Adjust settings and re-scan.</div>', unsafe_allow_html=True)
                if st.button("Re-scan", type="secondary"): st.session_state.scan_done=False; st.session_state.detected_clips=None; st.rerun()
                st.stop()
            if st.session_state.selected_clip_indices is None: st.session_state.selected_clip_indices = set(range(len(clips)))
            sel = st.session_state.selected_clip_indices
            if not st.session_state.clip_results:
                p1,p2,p3 = st.columns([4,3,2])
                with p1:
                    ns = len(sel)
                    pb = st.button(f"▶ Verticalize {ns} Clip{'s' if ns!=1 else ''}", type="primary", use_container_width=True, disabled=ns==0)
                with p2:
                    if st.button("Re-scan", type="secondary", use_container_width=True): st.session_state.scan_done=False; st.session_state.detected_clips=None; st.session_state.clip_results=None; st.rerun()
                with p3:
                    if st.button("Clear", type="secondary", use_container_width=True): _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
                if pb and sel:
                    sc = [clips[i] for i in sorted(sel)]; od = tempfile.mkdtemp(); st.session_state.clip_out_dir = od
                    prog=st.progress(0.0); status=st.empty(); status.info(f"Processing {len(sc)} clips…")
                    def _bcb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg:
                            status.info(msg)
                    try:
                        results = process_clips_batch(input_path=st.session_state.input_path, output_dir=od, clips=sc,
                            target_preset_label=resolution_label, tracking_mode=tracking_mode,
                            talking_head_bias=talking_head_bias, confidence=confidence,
                            smooth_window=smooth_window, adaptive_smoothing=adaptive_smoothing,
                            use_optical_flow=use_optical_flow, rule_of_thirds=rule_of_thirds,
                            scene_cut_threshold=scene_cut_threshold,
                            crf=crf, encoder_preset=encoder_preset_label,
                            audio_bitrate=audio_bitrate_label, yolo_weights=yolo_weights,
                            burn_subtitles=burn_subtitles, whisper_model=whisper_model,
                            whisper_language=whisper_language,
                            subtitle_style_name=subtitle_style_name, subtitle_max_chars=subtitle_max_chars,
                            subtitle_translate_to=subtitle_translate_to,
                            sport_type=st.session_state.get("sport_type","auto"),
                            panel_config=_build_panel_config(), progress_callback=_bcb)
                        prog.progress(1.0); st.session_state.clip_results=results
                        nk = sum(1 for r in results if not r.get("error"))
                        status.success(f"[OK] {nk}/{len(results)} clips converted!"); st.rerun()
                    except Exception as exc: status.error(f"Error: {exc}")
            else:
                results = st.session_state.clip_results; nk = sum(1 for r in results if not r.get("error"))
                if clips: st.markdown(f'<div class="rf-ok">✓ {nk} clip{"s" if nk!=1 else ""} ready — download from cards above</div>', unsafe_allow_html=True)
                rc1,rc2,rc3 = st.columns(3)
                with rc1:
                    if st.button("← New scan", type="secondary", use_container_width=True): st.session_state.scan_done=False; st.session_state.detected_clips=None; st.session_state.clip_results=None; st.session_state.playing_clip_idx=-1; st.rerun()
                with rc2:
                    if st.button("← New video", type="secondary", use_container_width=True): _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
                with rc3:
                    if st.button("Clear cache", type="secondary", use_container_width=True): _cleanup(); st.session_state.uploaded_file_name=None; st.cache_data.clear(); st.rerun()
else:
    st.markdown("""\n<div style='padding:0 20px 44px;margin-top:16px;'>\n<div style='background:var(--surf);border:2px dashed var(--bdr);border-radius:var(--r);padding:48px 28px;text-align:center;'>\n<div style='font-family:"DM Serif Display",serif;font-size:clamp(1.5rem,3.5vw,2.2rem);font-weight:400;color:var(--bdr2);letter-spacing:-0.03em;margin-bottom:10px;line-height:1.1;'>Drop a video to begin.</div>\n<p style='font-size:12px;color:var(--ink3);margin-bottom:16px;'>Landscape MP4 · MOV · AVI · MKV</p>\n</div></div>\n""", unsafe_allow_html=True)

st.markdown("""\n<div class="rf-foot"><div class="rf-tech"><span>YOLOv8</span><span>OpenCV</span><span>Whisper</span><span>FFmpeg</span></div>\n<div style='font-size:10px;color:var(--bdr2);'>Reframe · AI Vertical Video</div></div>\n""", unsafe_allow_html=True)
