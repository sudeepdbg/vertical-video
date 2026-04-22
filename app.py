"""
app.py  —  Reframe · AI Video Converter
Streamlit frontend for verticalize.py — world-class redesign
"""

import streamlit as st
import tempfile
import os
from verticalize import process_video, get_video_info, extract_thumbnail, RESOLUTION_PRESETS

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reframe",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

/* ── Reset ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #0a0a0a !important;
    color: #f0ece4 !important;
}
.stApp { background: #0a0a0a !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Page shell ─────────────────────────────────────────────── */
.rf-shell {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: #0a0a0a;
}

/* ── Topbar ─────────────────────────────────────────────────── */
.rf-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 48px;
    border-bottom: 1px solid #1e1e1e;
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10,10,10,0.92);
    backdrop-filter: blur(12px);
}
.rf-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    letter-spacing: -0.04em;
    color: #f0ece4;
    display: flex;
    align-items: center;
    gap: 10px;
}
.rf-logo-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #e8ff47;
    display: inline-block;
}
.rf-badge {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a4a4a;
    padding: 4px 12px;
    border: 1px solid #1e1e1e;
    border-radius: 99px;
}

/* ── Hero section ─────────────────────────────────────────────── */
.rf-hero {
    padding: 80px 48px 60px;
    max-width: 900px;
}
.rf-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e8ff47;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rf-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1.5px;
    background: #e8ff47;
}
.rf-h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.035em;
    color: #f0ece4;
    margin-bottom: 20px;
}
.rf-h1 em {
    font-style: normal;
    color: #e8ff47;
}
.rf-sub {
    font-size: 16px;
    color: #6a6a6a;
    line-height: 1.6;
    max-width: 520px;
    font-weight: 400;
}

/* ── Main content grid ────────────────────────────────────────── */
.rf-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    flex: 1;
    border-top: 1px solid #1e1e1e;
}
.rf-panel {
    padding: 48px;
}
.rf-panel-left { border-right: 1px solid #1e1e1e; }
.rf-panel-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3a3a3a;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rf-panel-label span {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #2a2a2a;
    display: inline-block;
}
.rf-panel-label.active span { background: #e8ff47; }

/* ── Upload zone ─────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0f0f0f !important;
    border: 1.5px dashed #252525 !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #e8ff47 !important;
    background: #111008 !important;
}
[data-testid="stFileUploadDropzone"] {
    padding: 56px 32px !important;
    text-align: center !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: #3a3a3a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stFileUploadDropzone"] small {
    color: #2a2a2a !important;
    font-size: 12px !important;
}
/* Upload icon */
[data-testid="stFileUploadDropzone"] svg {
    color: #3a3a3a !important;
    width: 28px !important;
    height: 28px !important;
}

/* ── Metrics strip ────────────────────────────────────────────── */
.rf-metrics {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #1a1a1a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 32px;
}
.rf-metric {
    background: #0f0f0f;
    padding: 16px;
}
.rf-metric-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a3a3a;
    margin-bottom: 6px;
}
.rf-metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #f0ece4;
    letter-spacing: -0.02em;
}
.rf-metric-value.accent { color: #e8ff47; }

/* ── Settings panel ───────────────────────────────────────────── */
.rf-settings {
    display: flex;
    flex-direction: column;
    gap: 0;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 32px;
}
.rf-setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    background: #0f0f0f;
    border-bottom: 1px solid #1a1a1a;
    gap: 16px;
}
.rf-setting-row:last-child { border-bottom: none; }
.rf-setting-info { flex: 1; min-width: 0; }
.rf-setting-name {
    font-size: 13px;
    font-weight: 500;
    color: #c0bdb5;
    margin-bottom: 2px;
}
.rf-setting-desc {
    font-size: 11px;
    color: #3a3a3a;
}
.rf-setting-control { flex-shrink: 0; }

/* ── Streamlit widget overrides ───────────────────────────────── */
[data-baseweb="select"] > div {
    background: #161616 !important;
    border-color: #252525 !important;
    border-radius: 8px !important;
    color: #f0ece4 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    min-width: 180px;
}
[data-baseweb="select"] * { color: #f0ece4 !important; }
[data-baseweb="popover"] { background: #161616 !important; }
[data-baseweb="menu"] { background: #161616 !important; border: 1px solid #252525 !important; }
[data-baseweb="option"] { background: #161616 !important; color: #c0bdb5 !important; }
[data-baseweb="option"]:hover { background: #1e1e1e !important; }

.stSlider { padding: 0 !important; }
.stSlider label { 
    font-size: 12px !important; 
    color: #4a4a4a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] { background: #e8ff47 !important; border: none !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: #e8ff47 !important; }
.stSlider [data-baseweb="slider"] > div > div { background: #252525 !important; }
[data-testid="stTickBar"] { color: #2a2a2a !important; }

[data-testid="stToggleSwitch"] > div {
    background: #252525 !important;
}
[data-testid="stToggleSwitch"][aria-checked="true"] > div {
    background: #e8ff47 !important;
}
[data-testid="stToggleSwitch"] span { color: #4a4a4a !important; font-size: 13px !important; }

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em;
    transition: all 0.16s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: #e8ff47 !important;
    color: #0a0a0a !important;
    padding: 14px 28px !important;
    box-shadow: 0 0 0 0 rgba(232,255,71,0) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #d4eb2a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(232,255,71,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: #141414 !important;
    color: #6a6a6a !important;
    border: 1px solid #252525 !important;
    padding: 12px 20px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #3a3a3a !important;
    color: #c0bdb5 !important;
}

/* ── Download button ──────────────────────────────────────────── */
.stDownloadButton > button {
    background: #e8ff47 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.02em;
    transition: all 0.16s ease !important;
}
.stDownloadButton > button:hover {
    background: #d4eb2a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(232,255,71,0.25) !important;
}

/* ── Progress ─────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: #e8ff47 !important;
    border-radius: 99px;
}
.stProgress > div > div {
    background: #1a1a1a !important;
    border-radius: 99px;
    height: 3px !important;
}
.stProgress > div { height: 3px !important; }

/* ── Alerts ───────────────────────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid #1e1e1e !important;
    background: #0f0f0f !important;
}
[data-baseweb="notification"] {
    background: #0f0f0f !important;
    border-radius: 10px !important;
    border: 1px solid #1e1e1e !important;
}

/* ── Video ────────────────────────────────────────────────────── */
video {
    border-radius: 12px !important;
    width: 100% !important;
    background: #000;
    display: block;
}
[data-testid="stVideo"] { border-radius: 12px; overflow: hidden; }

/* ── Caption / small ──────────────────────────────────────────── */
.stCaption, small { color: #3a3a3a !important; font-size: 11px !important; }

/* ── Empty state ──────────────────────────────────────────────── */
.rf-empty {
    background: #0c0c0c;
    border: 1.5px dashed #1e1e1e;
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 60px 32px;
    min-height: 300px;
    gap: 12px;
}
.rf-empty-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: #141414;
    border: 1px solid #1e1e1e;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    margin-bottom: 4px;
}
.rf-empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #2a2a2a;
}
.rf-empty-sub { font-size: 12px; color: #222; }

/* ── Process steps ────────────────────────────────────────────── */
.rf-steps {
    display: flex;
    gap: 6px;
    margin-bottom: 32px;
}
.rf-step {
    flex: 1;
    height: 3px;
    border-radius: 99px;
    background: #1a1a1a;
}
.rf-step.done { background: #e8ff47; }
.rf-step.active { background: #5a6620; }

/* ── Success banner ───────────────────────────────────────────── */
.rf-success {
    background: #0d1208;
    border: 1px solid #2a3a10;
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.rf-success-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #e8ff47;
    flex-shrink: 0;
}
.rf-success-text { font-size: 13px; color: #8aab3a; font-weight: 500; }

/* ── Validation warning ───────────────────────────────────────── */
.rf-warn {
    background: #120d08;
    border: 1px solid #3a2010;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #9a5a2a;
    margin-bottom: 16px;
}

/* ── File chip ────────────────────────────────────────────────── */
.rf-file-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #6a6a6a;
    margin-bottom: 16px;
    max-width: 100%;
    overflow: hidden;
}
.rf-file-chip strong { color: #c0bdb5; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Footer ───────────────────────────────────────────────────── */
.rf-footer {
    padding: 24px 48px;
    border-top: 1px solid #111;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.rf-footer-stack {
    display: flex;
    gap: 8px;
}
.rf-tech-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2a2a2a;
    padding: 4px 10px;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
}
.rf-footer-copy { font-size: 11px; color: #222; }

/* ── Status text ──────────────────────────────────────────────── */
.stInfo, .stSuccess, .stError, .stWarning {
    border-radius: 10px !important;
}
[data-testid="stMarkdownContainer"] p { color: #6a6a6a; font-size: 13px; }

/* Slider value display */
[data-testid="stSliderValue"] {
    color: #e8ff47 !important;
    font-size: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        input_path=None,
        output_path=None,
        uploaded_file_name=None,
        processing_done=False,
        output_bytes=None,
        video_info=None,
        # Store last-used settings so output invalidates on change
        last_resolution=None,
        last_smoothness=None,
        last_confidence=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _cleanup():
    for key in ("input_path", "output_path"):
        p = st.session_state.get(key)
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
        st.session_state[key] = None
    st.session_state.output_bytes = None
    st.session_state.video_info   = None


def _invalidate_output():
    """Call when settings change that would affect the output."""
    st.session_state.processing_done = False
    st.session_state.output_bytes    = None
    out = st.session_state.get("output_path")
    if out and os.path.exists(out):
        try:
            os.unlink(out)
        except OSError:
            pass
    # Re-create the output temp path
    if st.session_state.input_path:
        import tempfile
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        os.unlink(out_path)
        st.session_state.output_path = out_path


_init()


# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-topbar">
    <div class="rf-logo">
        <span class="rf-logo-dot"></span>
        Reframe
    </div>
    <div class="rf-badge">AI Video Converter</div>
</div>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-hero">
    <div class="rf-eyebrow">Landscape to Vertical</div>
    <h1 class="rf-h1">Convert. Track.<br><em>Reframe.</em></h1>
    <p class="rf-sub">
        AI-powered subject tracking converts your landscape footage into
        scroll-stopping vertical video — automatically.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Settings (above grid, inline) ─────────────────────────────────────────────
with st.container():
    st.markdown("""
    <div style="padding: 0 48px; margin-bottom: 0;">
        <div style="font-size:11px;font-weight:600;letter-spacing:0.14em;
             text-transform:uppercase;color:#3a3a3a;margin-bottom:16px;">
            ⚙ Settings
        </div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns([2, 1.5, 1.5, 1, 1], gap="small")

    with s1:
        resolution_label = st.selectbox(
            "Output resolution",
            list(RESOLUTION_PRESETS.keys()),
            index=0,
        )
    with s2:
        smooth_window = st.slider("Smoothness", 3, 31, 15, 2,
            help="Higher = steadier pan")
    with s3:
        confidence = st.slider("AI confidence", 0.10, 0.95, 0.50, 0.05,
            help="Detection sensitivity")
    with s4:
        use_optical_flow = st.toggle("Motion tracking", value=True)
    with s5:
        rule_of_thirds = st.toggle("Rule of thirds", value=True)

    target_size = RESOLUTION_PRESETS[resolution_label]

    # Invalidate output if key settings changed after a completed conversion
    if st.session_state.processing_done:
        if (st.session_state.last_resolution  != resolution_label or
            st.session_state.last_smoothness  != smooth_window   or
            st.session_state.last_confidence  != confidence):
            _invalidate_output()

st.markdown("<div style='height:1px;background:#1a1a1a;margin:0 48px 0'></div>",
            unsafe_allow_html=True)


# ── Main grid ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="small")

with col_left:
    st.markdown("""
    <div style='padding:32px 48px 0 48px;'>
        <div class='rf-panel-label active'>
            <span></span> Source · Landscape
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        # Inner padding hack via columns
        pad_l, content, pad_r = st.columns([0.08, 10, 0.08])
        with content:
            uploaded_file = st.file_uploader(
                "upload",
                type=["mp4", "mov", "avi", "mkv"],
                label_visibility="collapsed",
                help="Landscape (wider than tall) · max 500 MB",
            )

            # Validate file size
            if uploaded_file is not None:
                mb = len(uploaded_file.getvalue()) / (1024 ** 2)
                if mb > 500:
                    st.markdown(
                        f'<div class="rf-warn">⚠ File is {mb:.1f} MB — '
                        f'please upload a file under 500 MB.</div>',
                        unsafe_allow_html=True
                    )
                    uploaded_file = None

            # New upload → reset
            if (uploaded_file is not None and
                    st.session_state.uploaded_file_name != uploaded_file.name):
                _cleanup()
                st.session_state.processing_done = False

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded_file.getvalue())
                    st.session_state.input_path = tmp_in.name

                fd, out_path = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                os.unlink(out_path)
                st.session_state.output_path        = out_path
                st.session_state.uploaded_file_name = uploaded_file.name

                try:
                    st.session_state.video_info = get_video_info(st.session_state.input_path)
                except Exception:
                    st.session_state.video_info = None

            # Show original video + metadata
            if uploaded_file is not None and st.session_state.input_path:
                info = st.session_state.video_info

                # Landscape warning
                if info and not info["is_landscape"]:
                    st.markdown(
                        '<div class="rf-warn">⚠ This video is already vertical — '
                        'please upload a landscape (wider than tall) video.</div>',
                        unsafe_allow_html=True
                    )

                # File chip
                mb_str = f"{len(uploaded_file.getvalue()) / (1024**2):.1f} MB"
                st.markdown(
                    f'<div class="rf-file-chip">'
                    f'<span>▶</span>'
                    f'<strong>{uploaded_file.name}</strong>'
                    f'<span style="color:#2a2a2a">·</span>'
                    f'<span>{mb_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.video(uploaded_file)

with col_right:
    st.markdown("""
    <div style='padding:32px 48px 0 48px;'>
        <div class='rf-panel-label'>
            <span></span> Output · Vertical
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        pad_l, content, pad_r = st.columns([0.08, 10, 0.08])
        with content:
            if st.session_state.processing_done and st.session_state.output_bytes:
                # Success banner
                out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
                st.markdown(
                    f'<div class="rf-success">'
                    f'<div class="rf-success-dot"></div>'
                    f'<div class="rf-success-text">'
                    f'Conversion complete · {target_size[0]}×{target_size[1]} · {out_mb:.1f} MB'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                st.video(st.session_state.output_bytes, format="video/mp4")

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                stem = os.path.splitext(
                    st.session_state.uploaded_file_name or "video"
                )[0]
                st.download_button(
                    label="↓  Download vertical video",
                    data=st.session_state.output_bytes,
                    file_name=f"{stem}_vertical.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                st.markdown("""
                <div class="rf-empty">
                    <div class="rf-empty-icon">📱</div>
                    <div class="rf-empty-title">Your vertical video</div>
                    <div class="rf-empty-sub">will appear here after conversion</div>
                </div>
                """, unsafe_allow_html=True)


# ── Metrics + Action bar ──────────────────────────────────────────────────────
if uploaded_file is not None and st.session_state.input_path:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1a1a1a;margin:0 48px'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    info = st.session_state.video_info

    # ── Metrics strip ──────────────────────────────────────────────────
    if info:
        dur     = info["duration_seconds"]
        mins, secs = int(dur // 60), int(dur % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        est_sec = max(10, dur * 0.6 + 8)
        est_min = est_sec / 60
        est_str = (f"~{int(est_min)}m {int(est_sec % 60):02d}s"
                   if est_min >= 1 else f"~{int(est_sec)}s")

        st.markdown(f"""
        <div style='padding: 0 48px; margin-bottom: 32px;'>
        <div class="rf-metrics">
            <div class="rf-metric">
                <div class="rf-metric-label">Duration</div>
                <div class="rf-metric-value">{dur_str}</div>
            </div>
            <div class="rf-metric">
                <div class="rf-metric-label">Source</div>
                <div class="rf-metric-value">{info['width']}×{info['height']}</div>
            </div>
            <div class="rf-metric">
                <div class="rf-metric-label">Frame rate</div>
                <div class="rf-metric-value">{info['fps']:.0f} fps</div>
            </div>
            <div class="rf-metric">
                <div class="rf-metric-label">Output</div>
                <div class="rf-metric-value accent">{target_size[0]}×{target_size[1]}</div>
            </div>
            <div class="rf-metric">
                <div class="rf-metric-label">Est. time</div>
                <div class="rf-metric-value">{est_str}</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Action row ──────────────────────────────────────────────────────
    with st.container():
        _, action_area, __ = st.columns([0.06, 10, 0.06])
        with action_area:
            if not st.session_state.processing_done:
                btn_col, info_col, clear_col = st.columns([3, 5, 1.5])
                with btn_col:
                    can_process = info and info.get("is_landscape", True)
                    go = st.button(
                        "▶  Convert to vertical",
                        type="primary",
                        use_container_width=True,
                        disabled=not can_process,
                    )
                with info_col:
                    if info:
                        st.markdown(
                            f"<p style='color:#3a3a3a;font-size:12px;"
                            f"margin-top:14px;font-family:Space Grotesk,sans-serif'>"
                            f"{target_size[0]}×{target_size[1]}  ·  "
                            f"{resolution_label.split('(')[0].strip()}  ·  "
                            f"Est. {est_str}</p>",
                            unsafe_allow_html=True,
                        )
                with clear_col:
                    if st.button("✕  Clear", type="secondary", use_container_width=True):
                        _cleanup()
                        st.session_state.uploaded_file_name = None
                        st.rerun()

                if go:
                    # Save settings used
                    st.session_state.last_resolution = resolution_label
                    st.session_state.last_smoothness = smooth_window
                    st.session_state.last_confidence = confidence

                    progress_bar = st.progress(0.0)
                    status_text  = st.empty()
                    status_text.info("⚡ Starting…")

                    try:
                        def _cb(prog: float, msg: str = ""):
                            progress_bar.progress(min(prog, 1.0))
                            if msg:
                                status_text.info(msg)

                        process_video(
                            st.session_state.input_path,
                            st.session_state.output_path,
                            target_size=target_size,
                            confidence=confidence,
                            smooth_window=smooth_window,
                            use_optical_flow=use_optical_flow,
                            rule_of_thirds=rule_of_thirds,
                            progress_callback=_cb,
                        )

                        progress_bar.progress(1.0)
                        out = st.session_state.output_path
                        if os.path.exists(out) and os.path.getsize(out) > 0:
                            with open(out, "rb") as f:
                                st.session_state.output_bytes = f.read()
                            st.session_state.processing_done = True
                            status_text.success("✅ Conversion complete!")
                            st.rerun()
                        else:
                            status_text.error("❌ Output file is empty — something went wrong.")

                    except Exception as exc:
                        status_text.error(f"❌ {exc}")

            else:
                # Post-conversion actions
                reset_col, _, size_col = st.columns([2, 5, 2])
                with reset_col:
                    if st.button("← Start over", type="secondary", use_container_width=True):
                        _cleanup()
                        st.session_state.uploaded_file_name = None
                        st.session_state.processing_done    = False
                        st.rerun()
                with size_col:
                    if info and st.session_state.output_bytes:
                        in_mb  = len(uploaded_file.getvalue()) / (1024 ** 2)
                        out_mb = len(st.session_state.output_bytes) / (1024 ** 2)
                        delta  = out_mb - in_mb
                        st.markdown(
                            f"<p style='color:#3a3a3a;font-size:12px;"
                            f"text-align:right;margin-top:14px;"
                            f"font-family:Space Grotesk,sans-serif'>"
                            f"Size: {out_mb:.1f} MB  "
                            f"<span style='color:{\"#6aab3a\" if delta<0 else \"#9a5a2a\"}'>"
                            f"({delta:+.1f} MB)</span></p>",
                            unsafe_allow_html=True,
                        )


# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='padding: 0 48px 80px; margin-top: 32px;'>
        <div style='
            border: 1.5px dashed #1a1a1a;
            border-radius: 20px;
            padding: 80px 48px;
            text-align: center;
            background: #0c0c0c;
        '>
            <div style='
                font-family: Syne, sans-serif;
                font-size: 3rem;
                font-weight: 800;
                color: #1e1e1e;
                letter-spacing: -0.04em;
                margin-bottom: 16px;
                line-height: 1;
            '>Drop a video<br>to begin.</div>
            <p style='font-size:13px;color:#2a2a2a;margin-bottom:24px;'>
                Landscape MP4, MOV, AVI, or MKV · up to 500 MB
            </p>
            <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
                <span style='font-size:11px;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#222;padding:5px 12px;
                    border:1px solid #1a1a1a;border-radius:4px;'>MP4</span>
                <span style='font-size:11px;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#222;padding:5px 12px;
                    border:1px solid #1a1a1a;border-radius:4px;'>MOV</span>
                <span style='font-size:11px;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#222;padding:5px 12px;
                    border:1px solid #1a1a1a;border-radius:4px;'>AVI</span>
                <span style='font-size:11px;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#222;padding:5px 12px;
                    border:1px solid #1a1a1a;border-radius:4px;'>MKV</span>
                <span style='font-size:11px;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#222;padding:5px 12px;
                    border:1px solid #1a1a1a;border-radius:4px;'>≤ 500 MB</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rf-footer">
    <div class="rf-footer-stack">
        <span class="rf-tech-tag">YOLOv8</span>
        <span class="rf-tech-tag">OpenCV</span>
        <span class="rf-tech-tag">FFmpeg</span>
        <span class="rf-tech-tag">Streamlit</span>
    </div>
    <div class="rf-footer-copy">Reframe · AI Video Converter</div>
</div>
""", unsafe_allow_html=True)
