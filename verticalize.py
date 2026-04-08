import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import tempfile
import os

def smooth_centers(centers, window=7):
    """Apply moving average to subject center coordinates."""
    if not centers:
        return []
    smoothed = []
    for i in range(len(centers)):
        start = max(0, i - window)
        end = min(len(centers), i + window + 1)
        avg_x = np.mean([c[0] for c in centers[start:end]])
        avg_y = np.mean([c[1] for c in centers[start:end]])
        smoothed.append((int(avg_x), int(avg_y)))
    return smoothed

def detect_largest_person(frame, model, confidence=0.5):
    """Return (x, y) center of the largest person in frame, or None."""
    results = model(frame, verbose=False)[0]
    persons = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            if conf > confidence:
                persons.append((x1, y1, x2, y2))
    if not persons:
        return None
    largest = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    center_x = (largest[0] + largest[2]) // 2
    center_y = (largest[1] + largest[3]) // 2
    return (center_x, center_y)

def interpolate_centers(detected_centers, detected_indices, total_frames):
    """Linearly interpolate centers for frames without detection."""
    if not detected_centers:
        return [(0,0)] * total_frames
    all_centers = []
    for i in range(total_frames):
        if i <= detected_indices[0]:
            all_centers.append(detected_centers[0])
        elif i >= detected_indices[-1]:
            all_centers.append(detected_centers[-1])
        else:
            for j in range(len(detected_indices)-1):
                if detected_indices[j] <= i <= detected_indices[j+1]:
                    ratio = (i - detected_indices[j]) / (detected_indices[j+1] - detected_indices[j])
                    cx = int(detected_centers[j][0] + ratio*(detected_centers[j+1][0] - detected_centers[j][0]))
                    cy = int(detected_centers[j][1] + ratio*(detected_centers[j+1][1] - detected_centers[j][1]))
                    all_centers.append((cx, cy))
                    break
    return all_centers

def process_video(input_path, output_path, sample_interval=15, target_size=(1080, 1920), progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    model = YOLO('yolov8n.pt')
    
    # Detect subject centers on sampled frames
    detected_centers = []
    detected_indices = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            center = detect_largest_person(frame, model)
            if center:
                detected_centers.append(center)
                detected_indices.append(frame_idx)
        frame_idx += 1
    cap.release()
    
    if not detected_centers:
        frame_center = (orig_w // 2, orig_h // 2)
        detected_centers = [frame_center]
        detected_indices = [0]
    
    all_centers = interpolate_centers(detected_centers, detected_indices, total_frames)
    all_centers = smooth_centers(all_centers, window=7)
    
    # Crop dimensions (9:16 aspect ratio)
    target_aspect = target_size[0] / target_size[1]
    crop_h = orig_h
    crop_w = int(crop_h * target_aspect)
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(crop_w / target_aspect)
    
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_path = temp_video.name
    temp_video.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, target_size)
    
    cap = cv2.VideoCapture(input_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        center_x, center_y = all_centers[frame_num]
        left = center_x - crop_w // 2
        top = center_y - crop_h // 2
        left = max(0, min(left, orig_w - crop_w))
        top = max(0, min(top, orig_h - crop_h))
        right = left + crop_w
        bottom = top + crop_h
        cropped = frame[top:bottom, left:right]
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        out.write(resized)
        frame_num += 1
        if progress_callback:
            progress_callback(frame_num / total_frames)
    cap.release()
    out.release()
    
    # Add audio
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = VideoFileClip(input_path).audio
    if audio_clip:
        final_clip = video_clip.set_audio(audio_clip)
    else:
        final_clip = video_clip
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    final_clip.close()
    os.unlink(temp_video_path)
    print(f"✅ Saved to {output_path}")
