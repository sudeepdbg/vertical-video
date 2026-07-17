"""
app.py -- Reframe :: AI Vertical Video Studio
v6.0 :: HACKER EDITION :: Terminal UI

[>] Monochrome. Monospace. No fluff.
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

st.set_page_config(page_title="Reframe", page_icon="[>]", layout="wide",
                   initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL CSS -- Zero rounded corners, monospace, dark theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root{
  --bg0:#0a0a0a;--bg1:#111111;--bg2:#1a1a1a;--bg3:#222222;
  --fg0:#e0e0e0;--fg1:#a0a0a0;--fg2:#666666;--fg3:#444444;
  --acc:#00ff41;--acc-d:#00cc33;--warn:#ffaa00;--err:#ff3333;
  --info:#00aaff;--pur:#aa66ff;--bdr:#333333;--bdr2:#444444;
}

*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace!important;background:var(--bg0)!important;color:var(--fg0)!important}
.stApp{background:var(--bg0)!important}
.main .block-container{padding:0!important;max-width:100%!important}

/* Kill Streamlit chrome */
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="collapsedControl"],section[data-testid="stSidebar"]{display:none!important}

/* ── Header ── */
.rf-top{height:40px;background:var(--bg1);border-bottom:1px solid var(--bdr);display:flex;align-items:center;justify-content:space-between;padding:0 16px;position:sticky;top:0;z-index:200}
.rf-logo{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:700;letter-spacing:0.05em;color:var(--acc)}
.rf-logo::before{content:'[>]';color:var(--acc);font-weight:700}
.rf-tag{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:0.15em}

/* ── Section dividers ── */
.rf-sec{font-size:9px;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:var(--fg2);margin:16px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--bdr)}

/* ── Mode boxes ── */
.rf-mode-box{border:1px solid var(--bdr);background:var(--bg1);padding:10px 12px;margin-top:6px;font-size:11px;line-height:1.6}
.rf-mode-box.sel{border-color:var(--acc);background:var(--bg2)}
.rf-mode-h{font-size:11px;font-weight:700;margin-bottom:2px;color:var(--fg0)}
.rf-mode-s{font-size:10px;color:var(--fg1)}

/* ── File uploader ── */
[data-testid="stFileUploader"]{background:var(--bg1)!important;border:1px dashed var(--fg3)!important;border-radius:0!important}
[data-testid="stFileUploader"]:hover{border-color:var(--acc)!important;background:var(--bg2)!important}
[data-testid="stFileUploadDropzone"]{padding:18px 12px!important}
[data-testid="stFileUploadDropzone"] *{color:var(--fg2)!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important}

/* ── Video ── */
[data-testid="stVideo"]{border-radius:0!important;overflow:hidden!important;border:1px solid var(--bdr)}
video{border-radius:0!important;width:100%!important}

/* ── Metrics grid ── */
.rf-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--bdr);border:1px solid var(--bdr);margin-top:8px}
.rf-m{background:var(--bg1);padding:8px 10px}
.rf-ml{font-size:8px;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:var(--fg2);margin-bottom:2px}
.rf-mv{font-size:13px;font-weight:700;color:var(--fg0)}
.rf-mv.a{color:var(--acc)}

/* ── Analytics ── */
.rf-analytics{background:var(--bg1);border:1px solid var(--bdr);padding:12px;margin-top:10px}
.rf-an-title{font-size:9px;font-weight:700;color:var(--fg2);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:10px}
.rf-an-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.rf-an-item{background:var(--bg2);padding:8px;border:1px solid var(--bdr)}
.rf-an-label{font-size:9px;color:var(--fg2);margin-bottom:2px}
.rf-an-val{font-size:12px;font-weight:700;color:var(--fg0)}
.rf-an-val.good{color:var(--acc)}
.rf-an-val.bad{color:var(--err)}
.rf-an-sub{font-size:8px;color:var(--fg3);margin-top:2px}

/* ── Status boxes ── */
.rf-ok{background:var(--bg2);border-left:3px solid var(--acc);padding:8px 10px;font-size:11px;color:var(--acc);margin-bottom:6px}
.rf-warn{background:var(--bg2);border-left:3px solid var(--warn);padding:8px 10px;font-size:11px;color:var(--warn);margin-bottom:6px}
.rf-info{background:var(--bg2);border-left:3px solid var(--info);padding:8px 10px;font-size:11px;color:var(--info);margin-bottom:6px}
.rf-purp{background:var(--bg2);border-left:3px solid var(--pur);padding:8px 10px;font-size:11px;color:var(--pur);margin-bottom:6px}

/* ── Empty states ── */
.rf-empty{background:var(--bg1);border:1px dashed var(--fg3);padding:32px 16px;text-align:center;display:flex;flex-direction:column;align-items:center;gap:6px;min-height:140px;justify-content:center}
.rf-empty-h{font-size:12px;font-weight:700;color:var(--fg2);text-transform:uppercase;letter-spacing:0.1em}
.rf-empty-s{font-size:10px;color:var(--fg3)}

/* ── Clip cards ── */
.rf-ccard{background:var(--bg1);border:1px solid var(--bdr);padding:10px;margin-bottom:6px;position:relative}
.rf-ccard.sel{border-color:var(--acc);background:var(--bg2)}
.rf-ccard.done{border-left:3px solid var(--acc)}
.rf-cscore{position:absolute;top:8px;right:8px;font-size:9px;font-weight:700;padding:1px 5px;background:var(--bg3);color:var(--fg1);border:1px solid var(--bdr)}
.rf-cscore.h{background:var(--acc);color:var(--bg0);border-color:var(--acc)}
.rf-cscore.m{background:var(--warn);color:var(--bg0);border-color:var(--warn)}
.rf-ctitle{font-size:10px;font-weight:700;color:var(--fg0);margin-bottom:3px;padding-right:40px}
.rf-cmeta{font-size:9px;color:var(--fg2);line-height:1.5}
.rf-cdur{display:inline-block;background:var(--bg2);border:1px solid var(--bdr);font-size:9px;font-weight:700;color:var(--fg1);padding:1px 5px;margin-top:4px}
.rf-csoi{display:inline-block;background:var(--bg2);border:1px solid var(--bdr);font-size:9px;font-weight:600;color:var(--pur);padding:1px 5px;margin-top:4px;margin-left:4px}

/* ── Buttons ── */
.stButton >button{font-family:'JetBrains Mono',monospace!important;border-radius:0!important;font-weight:700!important;font-size:11px!important;letter-spacing:0.05em!important;transition:none!important;text-transform:uppercase!important}
.stButton >button[kind="primary"]{background:var(--acc)!important;color:var(--bg0)!important;border:none!important;padding:8px 16px!important}
.stButton >button[kind="primary"]:hover{background:var(--acc-d)!important}
.stButton >button[kind="primary"]:disabled{background:var(--bg3)!important;color:var(--fg3)!important}
.stButton >button[kind="secondary"]{background:var(--bg1)!important;color:var(--fg1)!important;border:1px solid var(--bdr)!important}
.stButton >button[kind="secondary"]:hover{border-color:var(--acc)!important;color:var(--acc)!important}

.stDownloadButton >button{background:var(--bg2)!important;color:var(--acc)!important;border:1px solid var(--acc)!important;border-radius:0!important;font-family:'JetBrains Mono',monospace!important;font-weight:700!important;font-size:11px!important;padding:8px 14px!important;width:100%!important;text-transform:uppercase!important;letter-spacing:0.05em!important}
.stDownloadButton >button:hover{background:var(--acc)!important;color:var(--bg0)!important}

/* ── Progress ── */
.stProgress >div >div >div{background:var(--acc)!important;border-radius:0!important}
.stProgress >div >div{background:var(--bg3)!important;border-radius:0!important;height:2px!important}
.stProgress >div{height:2px!important}

/* ── Inputs ── */
[data-baseweb="select"] >div{background:var(--bg1)!important;border-color:var(--bdr)!important;border-radius:0!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;color:var(--fg0)!important}
[data-baseweb="select"] *{color:var(--fg0)!important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:var(--bg1)!important;border:1px solid var(--bdr)!important;border-radius:0!important}
[data-baseweb="option"]{background:var(--bg1)!important;color:var(--fg1)!important;font-size:11px!important}
[data-baseweb="option"]:hover{background:var(--bg2)!important;color:var(--acc)!important}

[data-baseweb="tab-list"]{background:var(--bg1)!important;border-radius:0!important;padding:2px!important;gap:1px!important;border:1px solid var(--bdr)!important}
[data-baseweb="tab"]{background:transparent!important;border-radius:0!important;font-family:'JetBrains Mono',monospace!important;font-size:10px!important;font-weight:700!important;color:var(--fg2)!important;padding:5px 10px!important;border:none!important;text-transform:uppercase!important;letter-spacing:0.08em!important}
[aria-selected="true"][data-baseweb="tab"]{background:var(--bg2)!important;color:var(--acc)!important}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}

.stSlider label{font-size:10px!important;color:var(--fg1)!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:0.08em!important}
.stSlider [role="slider"]{background:var(--acc)!important;border:2px solid var(--bg0)!important;border-radius:0!important}
.stSlider [data-testid="stSliderTrackFill"]{background:var(--acc)!important}
.stSlider >div >div{background:var(--bg3)!important}
[data-testid="stSliderValue"]{color:var(--acc)!important;font-size:9px!important;font-weight:700!important}

[data-testid="stToggleSwitch"] >div{background:var(--bg3)!important;border-radius:0!important}
[data-testid="stToggleSwitch"][aria-checked="true"] >div{background:var(--acc)!important}

/* ── Chips & tags ── */
.rf-chip{display:inline-flex;align-items:center;gap:5px;background:var(--bg1);border:1px solid var(--bdr);padding:3px 8px;font-size:10px;color:var(--fg1)}
.rf-chip strong{color:var(--fg0);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px}
.rf-safe{display:inline-flex;align-items:center;gap:3px;background:var(--bg2);border:1px solid var(--bdr);padding:2px 6px;font-size:9px;font-weight:700;color:var(--acc);text-transform:uppercase}

/* ── Footer ── */
.rf-foot{margin-top:24px;padding:10px 16px;border-top:1px solid var(--bdr);display:flex;align-items:center;justify-content:space-between}
.rf-tech{display:flex;gap:4px;flex-wrap:wrap}
.rf-tech span{font-size:8px;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;padding:2px 6px;border:1px solid var(--bdr);color:var(--fg2)}

/* ── Panels ── */
.rf-panel{padding:12px 16px}
.rf-panelr{padding:12px 16px 12px 8px}

@media(max-width:768px){.rf-panel,.rf-panelr{padding:10px 12px}.rf-metrics{grid-template-columns:repeat(2,1fr)}.rf-an-grid{grid-template-columns:1fr}}

.stCaption,small{color:var(--fg2)!important;font-size:9px!important}
[data-testid="stHorizontalBlock"]{gap:8px!important}

[data-testid="stRadio"] label{font-size:11px!important;color:var(--fg1)!important}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p{font-size:11px!important}
[data-testid="stRadio"] > div{gap:4px!important}

.rf-vplayer{width:180px;flex-shrink:0}
.rf-vplayer [data-testid="stVideo"]{border-radius:0!important;overflow:hidden!important;height:320px!important;border:1px solid var(--bdr)}
.rf-vplayer video{width:180px!important;height:320px!important;object-fit:cover!important;border-radius:0!important;display:block!important}

/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg0)}
::-webkit-scrollbar-thumb{background:var(--bg3);border:1px solid var(--bdr)}
::-webkit-scrollbar-thumb:hover{background:var(--fg3)}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="rf-top">
  <div class="rf-logo">REFRAME</div>
  <span class="rf-tag">AI Vertical Video :: v6.0</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE SELECT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='padding:0 16px'>", unsafe_allow_html=True)
st.markdown('<div class="rf-sec">// MODE</div>', unsafe_allow_html=True)
mc1, mc2 = st.columns(2, gap="small")
with mc1:
    if st.button("[1] SINGLE CLIP", type="secondary", use_container_width=True):
        st.session_state.app_mode = "single"
with mc2:
    if st.button("[2] AUTO-CLIP", type="secondary", use_container_width=True):
        st.session_state.app_mode = "autoClip"
app_mode = st.session_state.app_mode

if app_mode == "single":
    st.markdown("""
<div class="rf-mode-box sel">
<div class="rf-mode-h">[1] SINGLE CLIP</div>
<div class="rf-mode-s">Upload landscape video. AI tracks subject, outputs 9:16.</div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="rf-mode-box sel">
<div class="rf-mode-h">[2] AUTO-CLIP  ::  AI</div>
<div class="rf-mode-s">Upload 30-90min video. AI scans saliency peaks, detects arcs, verticalizes clips.</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TRACKING MODE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='padding:0 16px'>", unsafe_allow_html=True)
st.markdown('<div class="rf-sec">// TRACKING</div>', unsafe_allow_html=True)
tm1, tm2, tm3 = st.columns(3, gap="small")
with tm1:
    if st.button("[S] SUBJECT", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "subject"
with tm2:
    if st.button("[T] TALKING HEAD", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "talking_head"
with tm3:
    if st.button("[C] CINEMATIC", type="secondary", use_container_width=True):
        st.session_state.tracking_mode = "cinematic"
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
if st.button("[B] SPORTS ACTION  ::  Ball-aware / Kalman", type="secondary", use_container_width=True):
    st.session_state.tracking_mode = "sports_action"

tracking_mode = st.session_state.tracking_mode
if tracking_mode == "sports_action":
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    sport_type = st.selectbox("SPORT_TYPE", ["auto","basketball","football","soccer","hockey"],
        index=["auto","basketball","football","soccer","hockey"].index(st.session_state.get("sport_type","auto")),
        help="Auto-detects playing surface.")
    st.session_state.sport_type = sport_type
else:
    sport_type = st.session_state.get("sport_type", "auto")

# Status line
status_labels = {
    "sports_action": ("SPORTS", "Ball-aware / Kalman tracking"),
    "talking_head": ("TALKING_HEAD", "Face detection / upper-third bias"),
    "cinematic": ("CINEMATIC", "Actor/dialogue-first / sports disabled"),
    "subject": ("SUBJECT", "YOLOv8 person detection"),
}
label, desc = status_labels.get(tracking_mode, ("UNKNOWN", ""))
st.markdown(f'<div style="margin-top:6px;margin-bottom:4px;"><span style="background:var(--bg2);border:1px solid var(--bdr);color:var(--acc);font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:2px 8px;">{label}</span><span style="font-size:10px;color:var(--fg2);margin-left:8px;">{desc}</span></div>', unsafe_allow_html=True)

# Panel mode settings
if tracking_mode == "subject":
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.expander("// PANEL_MODE SETTINGS", expanded=False):
        st.caption("News panels, podcasts, interviews (2+ people)")
        panel_mode_override = st.radio("PANEL_MODE", ["auto","force_on","force_off"],
            format_func={"auto":"AUTO-DETECT","force_on":"FORCE ON","force_off":"FORCE OFF"}.get,
            index=["auto","force_on","force_off"].index(st.session_state.get("panel_mode_override","auto")),
            help="Auto = detect panel layout automatically.")
        st.session_state.panel_mode_override = panel_mode_override
        if panel_mode_override != "force_off":
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rf-sec">// LAYOUT</div>', unsafe_allow_html=True)
            layout_mode = st.radio("LAYOUT", ["equal","speaker_focus","solo_spotlight","auto"],
                format_func={"equal":"EQUAL SPLIT","speaker_focus":"SPEAKER FOCUS","solo_spotlight":"SOLO SPOTLIGHT","auto":"AUTO"}.get,
                index=["equal","speaker_focus","solo_spotlight","auto"].index(st.session_state.get("panel_layout_mode","equal")),
                help="Screen space distribution among detected persons.")
            st.session_state.panel_layout_mode = layout_mode
            if layout_mode == "speaker_focus":
                st.session_state.panel_speaker_focus_ratio = st.slider("SPEAKER_RATIO", 0.50, 0.80,
                    float(st.session_state.get("panel_speaker_focus_ratio", 0.60)), 0.05, format="%.2f",
                    help="Active speaker screen share")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.session_state.panel_head_normalize = st.toggle("HEAD_NORMALIZE", value=st.session_state.get("panel_head_normalize", False), help="Normalize crop scale for equal face pixel height")
                st.session_state.panel_lower_third_aware = st.toggle("LOWER_THIRD", value=st.session_state.get("panel_lower_third_aware", False), help="Detect text banners, avoid cropping")
            with fc2:
                st.session_state.panel_portrait_mode = st.toggle("PORTRAIT", value=st.session_state.get("panel_portrait_mode", False), help="Head-and-shoulders crop")
                st.session_state.panel_max_slots = st.slider("MAX_SLOTS", 2, 4, int(st.session_state.get("panel_max_slots", 4)), 1, help="Max people to track")
        if panel_mode_override == "auto":
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rf-sec">// DETECTION SENSITIVITY</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.panel_max_motion = st.slider("MAX_MOTION_PX", 5.0, 40.0, float(st.session_state.get("panel_max_motion", 20.0)), 1.0)
                st.session_state.panel_min_area = st.slider("MIN_AREA_PCT", 0.01, 0.10, float(st.session_state.get("panel_min_area", 0.03)), 0.01, format="%.2f")
            with c2:
                st.session_state.panel_max_variance = st.slider("MAX_VARIANCE", 0.5, 5.0, float(st.session_state.get("panel_max_variance", 2.5)), 0.5)
                st.session_state.panel_stability = st.slider("STABILITY_FRAC", 0.30, 0.90, float(st.session_state.get("panel_stability", 0.60)), 0.05, format="%.2f")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_list = ["OUTPUT", "TRACKING", "SUBTITLES", "ADVANCED"]
if app_mode == "autoClip": tab_list += ["CLIPS", "ANALYTICS"]
if app_mode == "autoClip":
    tab_out, tab_trk, tab_sub, tab_adv, tab_clip, tab_analytics = st.tabs(tab_list)
else:
    tab_out, tab_trk, tab_sub, tab_adv = st.tabs(tab_list)
    tab_clip = tab_analytics = None

with tab_out:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    o1, o2 = st.columns(2, gap="medium")
    with o1:
        resolution_label = st.selectbox("RESOLUTION", list(RESOLUTION_PRESETS.keys()), index=0)
        fps_label = st.selectbox("FRAME_RATE", ["Source - keep original","60 fps","30 fps","25 fps","24 fps"], index=0)
    with o2:
        crf = st.slider("QUALITY_CRF", 15, 35, 23, 1)
        st.caption("18 = near-lossless  |  28 = compact")
        encoder_preset_label = st.selectbox("SPEED", ["ultrafast","fast","medium","slow"], index=1)
_fps_map = {"Source - keep original":None,"60 fps":60.0,"30 fps":30.0,"25 fps":25.0,"24 fps":24.0}
output_fps = _fps_map[fps_label]

with tab_trk:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if tracking_mode == "talking_head":
        th1, th2 = st.columns(2, gap="medium")
        with th1:
            talking_head_bias = st.slider("UPPER_THIRD_PULL", 0.0, 1.0, 0.30, 0.05)
            st.caption("0 = centered face  |  1 = upper third")
            smooth_window = st.slider("SMOOTHNESS", 3, 31, 21, 2)
        with th2:
            adaptive_smoothing = st.toggle("ADAPTIVE", value=False)
            use_optical_flow = st.toggle("OPTICAL_FLOW", value=True)
            rule_of_thirds = st.toggle("RULE_OF_THIRDS", value=True)
            confidence = 0.5; scene_cut_threshold = 0.35
        use_ball_tracking = False; use_kalman = False
    elif tracking_mode == "cinematic":
        st.markdown('<div class="rf-info" style="margin-bottom:10px;"><b>[CINEMATIC MODE]</b> -- actor/dialogue-first framing. Sports disabled.</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            confidence = st.slider("FACE_CONFIDENCE", 0.10, 0.95, 0.45, 0.05)
            smooth_window = st.slider("SMOOTH_HINT", 7, 41, 27, 2, help="Backend uses long-lens cinematic smoothing")
        with t2:
            scene_cut_threshold = st.slider("SHOT_CUT_SENS", 0.10, 0.60, 0.32, 0.05)
            st.caption("Shot-aware smoothing + composition analysis")
        adaptive_smoothing = True; use_optical_flow = True; rule_of_thirds = True
        talking_head_bias = 0.30; use_ball_tracking = False; use_kalman = False
    elif tracking_mode == "sports_action":
        st.markdown('<div class="rf-info" style="margin-bottom:10px;"><b>[SPORTS MODE]</b> -- Ball-aware tracking / Kalman smoothing</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            adaptive_smoothing = st.toggle("ADAPTIVE", value=True)
            smooth_window = st.slider("SMOOTHNESS", 3, 15, 5, 1)
            confidence = st.slider("DETECTION_CONF", 0.10, 0.95, 0.45, 0.05)
            use_ball_tracking = st.toggle("BALL_TRACKING", value=True, help="Prioritize ball carrier")
        with t2:
            use_optical_flow = st.toggle("OPTICAL_FLOW_FB", value=True)
            rule_of_thirds = st.toggle("LOOK_ROOM", value=True)
            scene_cut_threshold = st.slider("SCENE_CUT_SENS", 0.10, 0.60, 0.22, 0.05)
            use_kalman = st.toggle("KALMAN", value=True, help="Zero-lag predictive tracking")
        talking_head_bias = 0.30
    else:
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            adaptive_smoothing = st.toggle("ADAPTIVE", value=True)
            smooth_window = st.slider("SMOOTHNESS", 3, 31, 15, 2)
            confidence = st.slider("DETECTION_CONF", 0.10, 0.95, 0.45, 0.05)
        with t2:
            use_optical_flow = st.toggle("OPTICAL_FLOW", value=True)
            rule_of_thirds = st.toggle("RULE_OF_THIRDS", value=True)
            scene_cut_threshold = st.slider("SCENE_CUT_SENS", 0.10, 0.60, 0.35, 0.05)
        talking_head_bias = 0.30; use_ball_tracking = False; use_kalman = False

with tab_sub:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if not _whisper_ok:
        st.markdown('<div class="rf-purp">[!] Install openai-whisper to enable subtitles</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2, gap="medium")
    with s1:
        burn_subtitles = st.toggle("BURN_SUBTITLES", value=False, disabled=not _whisper_ok)
        if not _whisper_ok: burn_subtitles = False
        translate_subtitles = st.toggle("TRANSLATE", value=False, disabled=(not _whisper_ok or not _translate_ok or not burn_subtitles))
        if not burn_subtitles or not _translate_ok: translate_subtitles = False
        whisper_model = st.selectbox("WHISPER_MODEL", ["tiny","base","small","medium"], index=1, disabled=not _whisper_ok)
    with s2:
        subtitle_style_name = st.selectbox("STYLE", list(SUBTITLE_STYLES.keys()), disabled=not _whisper_ok)
        whisper_language_raw = st.selectbox("AUDIO_LANG", ["Auto-detect","en","hi","es","fr","de","ja","zh","pt","ar"], disabled=not _whisper_ok)
        whisper_language = None if whisper_language_raw == "Auto-detect" else whisper_language_raw
        subtitle_max_chars = st.slider("MAX_CHARS", 20, 60, 42, 2, disabled=not _whisper_ok)
        subtitle_translate_label = st.selectbox("TARGET_LANG", list(TRANSLATION_LANGUAGES.keys()), index=0, disabled=(not _whisper_ok or not _translate_ok or not burn_subtitles or not translate_subtitles))
        subtitle_translate_to = TRANSLATION_LANGUAGES[subtitle_translate_label] or None
        if not translate_subtitles: subtitle_translate_to = None

with tab_adv:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="medium")
    with a1: audio_bitrate_label = st.selectbox("AUDIO_BITRATE", ["64k","96k","128k","192k"], index=2)
    with a2:
        yolo_weights = st.selectbox("YOLO_WEIGHTS", ["yolov8n.pt","yolov8s.pt","yolov8m.pt"], index=0) if tracking_mode in ("subject", "cinematic") else "yolov8n.pt"
        st.markdown('<div class="rf-purp">[!] Talking Head uses OpenCV face detector -- YOLO not required</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-safe">[OK] Lower-third guard -- subjects kept above bottom 20%</div>', unsafe_allow_html=True)

_CLIP_PRESETS = {"15s (snappy)":(13,17),"30s (short)":(25,35),"60s (full)":(50,65)}
clip_min_dur = 25; clip_max_dur = 60; clip_target_n = 8
if app_mode == "autoClip" and tab_clip is not None:
    with tab_clip:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        cl1, cl2 = st.columns(2, gap="medium")
        with cl1:
            preset_label = st.radio("CLIP_PRESET", list(_CLIP_PRESETS.keys()), index=2)
            clip_min_dur, clip_max_dur = _CLIP_PRESETS[preset_label]
        with cl2: clip_target_n = st.slider("TARGET_CLIPS", 3, 20, 8, 1)
if app_mode == "autoClip" and tab_analytics is not None:
    with tab_analytics:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.session_state.clip_results:
            vr = [r for r in st.session_state.clip_results if not r.get("error") and "analytics" in r]
            if vr:
                ti = sum(r["analytics"]["input_size_mb"] for r in vr); to = sum(r["analytics"]["output_size_mb"] for r in vr)
                total_cpu_avg = sum(r["analytics"].get("cpu_avg_pct", 0) for r in vr) / len(vr) if vr else 0
                total_cpu_max = max((r["analytics"].get("cpu_max_pct", 0) for r in vr), default=0)
                total_ram_avg = sum(r["analytics"].get("ram_avg_mb", 0) for r in vr) / len(vr) if vr else 0
                total_ram_max = max((r["analytics"].get("ram_max_mb", 0) for r in vr), default=0)
                total_proc_time = sum(r["analytics"].get("processing_time_sec", 0) for r in vr)
                has_batch_res = total_cpu_avg > 0 or total_ram_avg > 0
                res_html = ""
                if has_batch_res:
                    res_html = f"""<div class="rf-an-item"><div class="rf-an-label">AVG_CPU</div><div class="rf-an-val">{total_cpu_avg:.1f}% <span style="font-size:10px;color:var(--fg3)">(max {total_cpu_max:.1f}%)</span></div></div><div class="rf-an-item"><div class="rf-an-label">AVG_RAM</div><div class="rf-an-val">{total_ram_avg:.1f} MB <span style="font-size:10px;color:var(--fg3)">(max {total_ram_max:.1f} MB)</span></div></div><div class="rf-an-item"><div class="rf-an-label">TOTAL_TIME</div><div class="rf-an-val">{total_proc_time:.1f}s</div></div>"""
                st.markdown(f'<div class="rf-analytics"><div class="rf-an-title">// BATCH ANALYTICS</div><div class="rf-an-grid"><div class="rf-an-item"><div class="rf-an-label">TOTAL_IN</div><div class="rf-an-val">{ti:.1f} MB</div></div><div class="rf-an-item"><div class="rf-an-label">TOTAL_OUT</div><div class="rf-an-val good">{to:.1f} MB</div></div><div class="rf-an-item"><div class="rf-an-label">CLIPS</div><div class="rf-an-val">{len(vr)}</div></div>{res_html}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rf-empty" style="min-height:100px;padding:16px;"><div class="rf-empty-s">Process clips to view analytics</div></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

def _build_panel_config():
    """Build PanelModeConfig -- works with both old and new backends."""
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
st.markdown("<div style='height:1px;background:var(--bdr);margin:8px 0 0'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE / OUTPUT COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
col_src, col_out = st.columns(2, gap="small")
with col_src:
    st.markdown("<div class='rf-panel'>", unsafe_allow_html=True)
    st.markdown('<div class="rf-sec">// SOURCE VIDEO</div>', unsafe_allow_html=True)
    max_mb = 2000 if app_mode == "autoClip" else 500
    uploaded_file = st.file_uploader("DROP_VIDEO", type=["mp4","mov","avi","mkv"], label_visibility="collapsed")
    if uploaded_file is not None:
        mb = len(uploaded_file.getvalue())/(1024**2)
        if mb > max_mb:
            st.markdown(f'<div class="rf-warn">[!] {mb:.1f} MB -- max {max_mb} MB</div>', unsafe_allow_html=True)
            uploaded_file = None
    if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
        _cleanup()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getvalue()); st.session_state.input_path = tmp.name
        _new_out(); st.session_state.uploaded_file_name = uploaded_file.name
        try: st.session_state.video_info = get_video_info(st.session_state.input_path)
        except Exception: st.session_state.video_info = None
    if uploaded_file is not None and st.session_state.input_path:
        info = st.session_state.video_info
        if info and not info["is_landscape"]:
            st.markdown('<div class="rf-warn">[!] Video is already vertical</div>', unsafe_allow_html=True)
        mb_str = f"{len(uploaded_file.getvalue())/(1024**2):.1f} MB"
        st.markdown(f'<div class="rf-chip"><span>&gt;</span><strong>{uploaded_file.name}</strong><span style="color:var(--fg3)">|</span><span>{mb_str}</span></div>', unsafe_allow_html=True)
        st.video(uploaded_file)
        if info:
            dur = info["duration_seconds"]; mins, secs = int(dur//60), int(dur%60)
            dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
            st.markdown(f"<div class='rf-metrics'><div class='rf-m'><div class='rf-ml'>DURATION</div><div class='rf-mv'>{dur_str}</div></div><div class='rf-m'><div class='rf-ml'>SOURCE</div><div class='rf-mv'>{info['width']}x{info['height']}</div></div><div class='rf-m'><div class='rf-ml'>OUTPUT</div><div class='rf-mv a'>{eff_w}x{eff_h}</div></div><div class='rf-m'><div class='rf-ml'>FPS</div><div class='rf-mv'>{info['fps']:.0f}</div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    st.markdown("<div class='rf-panelr'>", unsafe_allow_html=True)
    if app_mode == "single":
        st.markdown('<div class="rf-sec">// OUTPUT :: 9:16</div>', unsafe_allow_html=True)
        if st.session_state.processing_done and st.session_state.output_bytes:
            out_mb = len(st.session_state.output_bytes)/(1024**2)
            st.markdown(f'<div class="rf-ok">[OK] DONE -- {out_mb:.1f} MB</div>', unsafe_allow_html=True)
            st.video(st.session_state.output_bytes, format="video/mp4")
            if st.session_state.analytics_data:
                a = st.session_state.analytics_data
                if a.get("panel_mode"): st.markdown('<div class="rf-ok">[OK] PANEL MODE ACTIVE</div>', unsafe_allow_html=True)
                if a.get("cinematic_mode"): st.markdown(f'<div class="rf-ok">[OK] CINEMATIC MODE -- {int(a.get("scene_cuts", 0))} shot cut(s)</div>', unsafe_allow_html=True)
                sp = a.get("smoothness_pct",0)
                sc_v = "var(--acc)" if sp>80 else ("var(--warn)" if sp>50 else "var(--err)")
                cpu_avg = a.get("cpu_avg_pct", 0); cpu_max = a.get("cpu_max_pct", 0)
                ram_avg = a.get("ram_avg_mb", 0); ram_max = a.get("ram_max_mb", 0)
                proc_time = a.get("processing_time_sec", 0)
                has_resource = cpu_avg > 0 or ram_avg > 0
                cpu_color = "var(--acc)" if cpu_max < 50 else ("var(--warn)" if cpu_max < 80 else "var(--err)")
                ram_color = "var(--acc)" if ram_max < 512 else ("var(--warn)" if ram_max < 1024 else "var(--err)")
                resource_html = ""
                if has_resource:
                    resource_html = f"""<div class="rf-an-item"><div class="rf-an-label">CPU_USAGE</div><div class="rf-an-val" style="color:{cpu_color}">{cpu_avg:.1f}% <span style="font-size:10px;color:var(--fg3)">(max {cpu_max:.1f}%)</span></div></div><div class="rf-an-item"><div class="rf-an-label">RAM_USAGE</div><div class="rf-an-val" style="color:{ram_color}">{ram_avg:.1f} MB <span style="font-size:10px;color:var(--fg3)">(max {ram_max:.1f} MB)</span></div></div><div class="rf-an-item"><div class="rf-an-label">PROC_TIME</div><div class="rf-an-val">{proc_time:.1f}s</div></div>"""
                st.markdown(f'<div class="rf-analytics"><div class="rf-an-title">// ANALYTICS</div><div class="rf-an-grid"><div class="rf-an-item"><div class="rf-an-label">SIZE_REDUCTION</div><div class="rf-an-val">{a.get("file_size_reduction_pct",0):.1f}%</div><div class="rf-an-sub">{a.get("input_size_mb",0):.1f} &rarr; {a.get("output_size_mb",0):.1f} MB</div></div><div class="rf-an-item"><div class="rf-an-label">SMOOTHNESS</div><div class="rf-an-val" style="color:{sc_v}">{sp:.1f}%</div></div><div class="rf-an-item"><div class="rf-an-label">RESOLUTION</div><div class="rf-an-val">{a.get("output_resolution","")}</div></div>{resource_html}</div></div>', unsafe_allow_html=True)
            stem = os.path.splitext(st.session_state.uploaded_file_name or "video")[0]
            st.download_button("[>] DOWNLOAD VERTICAL", data=st.session_state.output_bytes, file_name=f"{stem}_vertical.mp4", mime="video/mp4", use_container_width=True)
            if st.session_state.srt_bytes:
                st.download_button("[>] DOWNLOAD SRT", data=st.session_state.srt_bytes, file_name=f"{stem}.srt", mime="text/plain", use_container_width=True)
        else:
            st.markdown('<div class="rf-empty"><div class="rf-empty-h">VERTICAL OUTPUT</div><div class="rf-empty-s">appears here after conversion</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rf-sec">// DETECTED CLIPS</div>', unsafe_allow_html=True)
        if not st.session_state.scan_done:
            st.markdown('<div class="rf-empty"><div class="rf-empty-h">SCAN FIRST</div><div class="rf-empty-s">AI will detect high-engagement segments</div></div>', unsafe_allow_html=True)
        elif st.session_state.detected_clips:
            clips = st.session_state.detected_clips
            if st.session_state.selected_clip_indices is None:
                st.session_state.selected_clip_indices = set(range(len(clips)))
            sel = st.session_state.selected_clip_indices
            st.markdown(f"<div style='font-size:10px;color:var(--fg2);margin-bottom:6px;'>{len(clips)} clips found | {len(sel)} selected</div>", unsafe_allow_html=True)
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
                ts = f"{int(clip.start_sec//60)}:{int(clip.start_sec%60):02d} &rarr; {int(clip.end_sec//60)}:{int(clip.end_sec%60):02d}"
                cc = "rf-ccard" + (" done" if is_d else (" sel" if is_s else ""))
                dt = "<div style='margin-top:4px;font-size:9px;color:var(--acc);font-weight:700;'>[OK] CONVERTED</div>" if is_d else ""
                st.markdown(f'<div class="{cc}"><span class="rf-cscore {sc2}">{sp2}%</span><div class="rf-ctitle">CLIP_{ci+1:03d}</div><div class="rf-cmeta">{ts}</div><span class="rf-cdur">{clip.duration:.0f}s</span><span class="rf-csoi">SOI:{clip.soi_region}</span>{dt}</div>', unsafe_allow_html=True)
                if is_d:
                    bc, dc = st.columns([1,1])
                    with bc:
                        if st.button("[X] CLOSE" if is_p else "[>] PLAY 9:16", key=f"play_{ci}", type="secondary", use_container_width=True):
                            st.session_state.playing_clip_idx = -1 if is_p else ci; st.rerun()
                    with dc:
                        try:
                            with open(rfc["output_path"],"rb") as f: cb = f.read()
                            st.download_button("[>] DOWNLOAD", data=cb, file_name=f"clip_{ci+1:03d}_vertical.mp4", mime="video/mp4", key=f"dl_{ci}", use_container_width=True)
                        except Exception: pass
                    if is_p:
                        try:
                            with open(rfc["output_path"],"rb") as f: cpb = f.read()
                            vc, _ = st.columns([180,400])
                            with vc:
                                st.markdown('<div class="rf-vplayer">', unsafe_allow_html=True)
                                st.video(cpb, format="video/mp4")
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception: pass
                else:
                    cbc, _ = st.columns([2,1])
                    with cbc:
                        tog = st.checkbox("[OK] SELECTED" if is_s else "INCLUDE", value=is_s, key=f"csel_{ci}")
                        if tog != is_s:
                            if tog: st.session_state.selected_clip_indices.add(ci)
                            else: st.session_state.selected_clip_indices.discard(ci)
                            st.rerun()
        else:
            st.markdown('<div class="rf-empty"><div class="rf-empty-h">NO CLIPS FOUND</div><div class="rf-empty-s">adjust clip duration settings</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ACTION BAR
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None and st.session_state.input_path:
    info = st.session_state.video_info
    can_go = bool(info and info.get("is_landscape", True))
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:var(--bdr);margin:0 16px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if app_mode == "single":
        if not st.session_state.processing_done:
            a1, a2, a3 = st.columns([4,5,2])
            with a1:
                go = st.button("[>] CONVERT TO VERTICAL", type="primary", use_container_width=True, disabled=not can_go)
            with a2:
                if info:
                    eff_w, eff_h = resolve_target_size(resolution_label, info["width"], info["height"])
                    mt = "TALKING_HEAD" if tracking_mode=="talking_head" else ("SPORTS" if tracking_mode=="sports_action" else ("CINEMATIC" if tracking_mode=="cinematic" else "SUBJECT"))
                    st.markdown(f"<p style='color:var(--fg2);font-size:10px;margin-top:10px;'>{mt} | {eff_w}x{eff_h} | CRF {crf}</p>", unsafe_allow_html=True)
            with a3:
                if st.button("[X] CLEAR", type="secondary", use_container_width=True):
                    _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
            if go:
                st.session_state.last_settings = current_settings
                prog = st.progress(0.0); status = st.empty()
                status.info("[>] INITIALIZING...")
                try:
                    def _cb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(f"[>] {msg.upper()}")
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
                        st.session_state.processing_done = True; status.success("[OK] DONE"); st.rerun()
                    else: status.error("[!] OUTPUT EMPTY -- CHECK FFMPEG")
                except Exception as exc: status.error(f"[!] ERROR: {exc}")
        else:
            r1, _, r2 = st.columns([2,5,2])
            with r1:
                if st.button("[<] START OVER", type="secondary", use_container_width=True):
                    _cleanup(); st.session_state.uploaded_file_name=None; st.session_state.processing_done=False; st.rerun()
    else:
        if st.session_state.detected_clips is None: st.session_state.scan_done = False
        if not st.session_state.scan_done:
            b1, b2, b3 = st.columns([4,4,2])
            with b1:
                scan_btn = st.button("[>] SCAN FOR CLIPS", type="primary", use_container_width=True, disabled=not can_go)
            with b3:
                if st.button("[X] CLEAR", type="secondary", use_container_width=True):
                    _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
            if scan_btn:
                prog=st.progress(0.0); status=st.empty()
                status.info("[>] SCANNING...")
                try:
                    def _scb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(f"[>] {msg.upper()}")
                    clips = detect_clips(st.session_state.input_path, min_duration_sec=float(clip_min_dur), max_duration_sec=float(clip_max_dur), target_n_clips=int(clip_target_n), model=None, confidence=confidence, progress_callback=_scb)
                    prog.progress(1.0)
                    if not clips:
                        status.warning("[!] NO CLIPS DETECTED")
                        st.session_state.detected_clips=[]; st.session_state.selected_clip_indices=set()
                    else:
                        st.session_state.detected_clips=clips; st.session_state.selected_clip_indices=set(range(len(clips))); st.session_state.clip_results=None
                        status.success(f"[OK] FOUND {len(clips)} CLIPS")
                    st.session_state.scan_done=True; st.rerun()
                except Exception as exc: status.error(f"[!] SCAN ERROR: {exc}")
        else:
            clips = st.session_state.detected_clips or []
            if not clips:
                st.markdown('<div class="rf-warn">[!] NO CLIPS DETECTED. ADJUST SETTINGS AND RE-SCAN.</div>', unsafe_allow_html=True)
                if st.button("[>] RE-SCAN", type="secondary"):
                    st.session_state.scan_done=False; st.session_state.detected_clips=None; st.rerun()
                st.stop()
            if st.session_state.selected_clip_indices is None:
                st.session_state.selected_clip_indices = set(range(len(clips)))
            sel = st.session_state.selected_clip_indices
            if not st.session_state.clip_results:
                p1,p2,p3 = st.columns([4,3,2])
                with p1:
                    ns = len(sel)
                    pb = st.button(f"[>] VERTICALIZE {ns} CLIP{'S' if ns!=1 else ''}", type="primary", use_container_width=True, disabled=ns==0)
                with p2:
                    if st.button("[>] RE-SCAN", type="secondary", use_container_width=True):
                        st.session_state.scan_done=False; st.session_state.detected_clips=None; st.session_state.clip_results=None; st.rerun()
                with p3:
                    if st.button("[X] CLEAR", type="secondary", use_container_width=True):
                        _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
                if pb and sel:
                    sc = [clips[i] for i in sorted(sel)]; od = tempfile.mkdtemp(); st.session_state.clip_out_dir = od
                    prog=st.progress(0.0); status=st.empty()
                    status.info(f"[>] PROCESSING {len(sc)} CLIPS...")
                    def _bcb(v, msg=""):
                        prog.progress(min(v, 1.0))
                        if msg: status.info(f"[>] {msg.upper()}")
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
                        status.success(f"[OK] {nk}/{len(results)} CLIPS CONVERTED"); st.rerun()
                    except Exception as exc: status.error(f"[!] ERROR: {exc}")
            else:
                results = st.session_state.clip_results; nk = sum(1 for r in results if not r.get("error"))
                if clips:
                    st.markdown(f'<div class="rf-ok">[OK] {nk} CLIP{"S" if nk!=1 else ""} READY -- DOWNLOAD FROM CARDS ABOVE</div>', unsafe_allow_html=True)
                rc1,rc2,rc3 = st.columns(3)
                with rc1:
                    if st.button("[<] NEW SCAN", type="secondary", use_container_width=True):
                        st.session_state.scan_done=False; st.session_state.detected_clips=None; st.session_state.clip_results=None; st.session_state.playing_clip_idx=-1; st.rerun()
                with rc2:
                    if st.button("[<] NEW VIDEO", type="secondary", use_container_width=True):
                        _cleanup(); st.session_state.uploaded_file_name=None; st.rerun()
                with rc3:
                    if st.button("[!] CLEAR CACHE", type="secondary", use_container_width=True):
                        _cleanup(); st.session_state.uploaded_file_name=None; st.cache_data.clear(); st.rerun()
else:
    st.markdown("""
<div style='padding:0 16px 32px;margin-top:12px;'>
<div style='background:var(--bg1);border:1px dashed var(--fg3);padding:40px 24px;text-align:center;'>
<div style='font-family:"JetBrains Mono",monospace;font-size:clamp(1.2rem,3vw,1.8rem);font-weight:700;color:var(--fg3);letter-spacing:0.05em;margin-bottom:8px;'>DROP A VIDEO TO BEGIN</div>
<p style='font-size:10px;color:var(--fg2);margin-bottom:12px;text-transform:uppercase;letter-spacing:0.1em;'>Landscape MP4 | MOV | AVI | MKV</p>
</div></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rf-foot">
<div class="rf-tech"><span>YOLOv8</span><span>OpenCV</span><span>Whisper</span><span>FFmpeg</span></div>
<div style='font-size:9px;color:var(--fg3);text-transform:uppercase;letter-spacing:0.1em;'>Reframe :: AI Vertical Video :: v6.0</div>
</div>
""", unsafe_allow_html=True)
