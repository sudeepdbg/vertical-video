import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import tempfile
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 0
DEFAULT_TARGET_SIZE = (1080, 1920)   # (width, height)  →  9 : 16
MAX_FILE_SIZE_MB = 500


# ---------------------------------------------------------------------------
# Model loader (cached at module level so it is only loaded once per process)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _get_model(weights: str = "yolov8n.pt") -> YOLO:
    """Load YOLO model once and cache it."""
    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_largest_person(frame: np.ndarray, model: YOLO, confidence: float = 0.5):
    """Return (cx, cy) of the largest detected person, or None."""
    results = model(frame, verbose=False)[0]
    best, best_area = None, 0
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS_ID:
            continue
        if float(box.conf[0]) < confidence:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    if best is None:
        return None
    return ((best[0] + best[2]) // 2, (best[1] + best[3]) // 2)


def saliency_center(frame: np.ndarray) -> tuple[int, int]:
    """
    Compute a visually salient center using edge density in a sliding window.
    Falls back to frame center if the frame is uniform (e.g. black leader).
    Much better than raw frame center for title cards and non-person scenes.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Divide frame into a grid and find the column with most edge content
    grid_cols = 16
    col_w = w // grid_cols
    col_scores = []
    for i in range(grid_cols):
        region = edges[:, i * col_w: (i + 1) * col_w]
        col_scores.append(float(region.sum()))

    total = sum(col_scores)
    if total < 1e-3:
        # Blank / uniform frame — dead center
        return (w // 2, h // 2)

    # Weighted centroid of edge activity across columns
    weighted_x = sum((i + 0.5) * col_w * s for i, s in enumerate(col_scores)) / total
    return (int(weighted_x), h // 2)


# ---------------------------------------------------------------------------
# Center smoothing / interpolation
# ---------------------------------------------------------------------------

def smooth_centers(centers: list[tuple], window: int = 7) -> list[tuple]:
    """Gaussian-weighted moving average to reduce jitter."""
    if not centers:
        return []
    n = len(centers)
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)

    # Build a 1-D Gaussian kernel
    half = window // 2
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / (half / 2 + 1e-6)) ** 2)
    k /= k.sum()

    # Reflect-pad and convolve
    xs_s = np.convolve(np.pad(xs, half, mode="reflect"), k, mode="valid")
    ys_s = np.convolve(np.pad(ys, half, mode="reflect"), k, mode="valid")

    return [(int(x), int(y)) for x, y in zip(xs_s, ys_s)]


def interpolate_centers(
    detected_centers: list[tuple],
    detected_indices: list[int],
    total_frames: int,
) -> list[tuple]:
    """Linearly interpolate subject centers for frames without a detection."""
    if not detected_centers:
        return [(0, 0)] * total_frames

    all_centers: list[tuple] = []
    di = detected_indices
    dc = detected_centers

    for i in range(total_frames):
        if i <= di[0]:
            all_centers.append(dc[0])
            continue
        if i >= di[-1]:
            all_centers.append(dc[-1])
            continue
        # Binary-search for the surrounding pair
        lo, hi = 0, len(di) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if di[mid] <= i:
                lo = mid
            else:
                hi = mid
        t = (i - di[lo]) / max(di[hi] - di[lo], 1)
        cx = int(dc[lo][0] + t * (dc[hi][0] - dc[lo][0]))
        cy = int(dc[lo][1] + t * (dc[hi][1] - dc[lo][1]))
        all_centers.append((cx, cy))

    return all_centers


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------

def _crop_geometry(orig_w: int, orig_h: int, target_w: int, target_h: int):
    """
    Return (crop_w, crop_h) that fits inside the original frame
    while matching target_w / target_h aspect ratio.
    """
    target_ratio = target_w / target_h   # e.g. 1080/1920 ≈ 0.5625  (portrait)
    crop_h = orig_h
    crop_w = int(round(crop_h * target_ratio))
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(round(crop_w / target_ratio))
    return crop_w, crop_h


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_video(
    input_path: str,
    output_path: str,
    sample_interval: int | None = None,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    confidence: float = 0.5,
    smooth_window: int = 15,
    yolo_weights: str = "yolov8n.pt",
    progress_callback=None,
) -> None:
    """
    Convert a landscape video to vertical format using AI subject tracking.

    Parameters
    ----------
    input_path      : Path to the source video.
    output_path     : Path for the converted output video.
    sample_interval : Detect subjects every N frames. Defaults to fps/2 (2 detections/sec).
    target_size     : Output (width, height). Default is 1080×1920.
    confidence      : YOLO detection confidence threshold.
    smooth_window   : Temporal smoothing radius in frames.
    yolo_weights    : YOLO model weights file.
    progress_callback : Optional callable(float) receiving 0.0–1.0 progress.
    """
    # --- Validate input -------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    file_mb = os.path.getsize(input_path) / (1024 ** 2)
    if file_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size {file_mb:.1f} MB exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # --- Open video -----------------------------------------------------------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not determine frame count — the file may be corrupt.")

    if sample_interval is None:
        # ~2 detections per second, minimum every frame for very short clips
        sample_interval = max(1, int(fps / 2))

    target_w, target_h = target_size
    crop_w, crop_h = _crop_geometry(orig_w, orig_h, target_w, target_h)

    # --- Phase 1: subject detection on sampled frames -------------------------
    model = _get_model(yolo_weights)
    detected_centers: list[tuple] = []
    detected_indices: list[int] = []
    # Saliency fallback: richer per-frame visual centers for non-person scenes
    saliency_centers_fb: list[tuple] = []
    saliency_indices_fb: list[int] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            center = detect_largest_person(frame, model, confidence)
            if center:
                detected_centers.append(center)
                detected_indices.append(frame_idx)
            else:
                sc = saliency_center(frame)
                saliency_centers_fb.append(sc)
                saliency_indices_fb.append(frame_idx)
        frame_idx += 1
    cap.release()

    if not detected_centers:
        # No people found — use edge-saliency centers if available
        if saliency_centers_fb:
            detected_centers = saliency_centers_fb
            detected_indices = saliency_indices_fb
        else:
            detected_centers = [(orig_w // 2, orig_h // 2)]
            detected_indices = [0]

    # --- Phase 2: interpolate + smooth ----------------------------------------
    all_centers = interpolate_centers(detected_centers, detected_indices, total_frames)
    all_centers = smooth_centers(all_centers, window=smooth_window)

    # Ensure list is exactly total_frames long (guard against edge cases)
    if len(all_centers) < total_frames:
        all_centers += [all_centers[-1]] * (total_frames - len(all_centers))

    # --- Phase 3: render cropped frames to temp file --------------------------
    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_w, target_h))

        cap = cv2.VideoCapture(input_path)
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cx, cy = all_centers[frame_num]
            left = cx - crop_w // 2
            top = cy - crop_h // 2
            # Clamp to frame bounds
            left = max(0, min(left, orig_w - crop_w))
            top = max(0, min(top, orig_h - crop_h))

            cropped = frame[top: top + crop_h, left: left + crop_w]
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)

            frame_num += 1
            if progress_callback:
                # Reserve last 10 % for audio mux
                progress_callback(min(frame_num / total_frames * 0.9, 0.9))

        cap.release()
        out.release()

        # --- Phase 4: mux audio -----------------------------------------------
        video_clip = VideoFileClip(temp_video_path)
        source_clip = VideoFileClip(input_path)
        audio = source_clip.audio

        if audio is not None:
            # Trim audio to match video length (prevents mismatch crashes)
            audio = audio.subclip(0, min(audio.duration, video_clip.duration))
            final_clip = video_clip.set_audio(audio)
        else:
            final_clip = video_clip

        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

        final_clip.close()
        video_clip.close()
        source_clip.close()

        if progress_callback:
            progress_callback(1.0)

        print(f"✅ Saved to {output_path}")

    finally:
        # Always clean up the temp file even if an error occurred
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except OSError:
                pass
