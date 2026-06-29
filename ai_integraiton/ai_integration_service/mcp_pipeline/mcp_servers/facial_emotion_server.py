"""
MCP Server - Stream 4: Facial Expression Analysis (FER)
========================================================
Adapted from the project's original implementation.
Model: ImprovedHybridFER (EfficientNetV2-S + Transformer) - custom-trained
Face Detection: YOLOv11n-face
Output: Processed video (bounding box + emotion + FPS) + CSV + analysis results
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import deque, Counter
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FacialEmotionServer")

# Global model cache
_fer_model = None
_yolo_model = None

DEFAULT_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Visual settings
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 128)


# ═══════════════════════════════════════════════════════
#   MODEL TANIMI ()
# ═══════════════════════════════════════════════════════

class ImprovedHybridFER(nn.Module):
    """Custom-trained model — architecture preserved."""
    def __init__(self, num_classes=7, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.15, stochastic_depth=0.0):
        super().__init__()
        backbone = models.efficientnet_v2_s(weights=None)
        self.cnn = backbone.features
        cnn_out_channels = 1280

        self.token_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.num_tokens = 36
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, cnn_out_channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop_path = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_channels, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cnn_out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.token_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════
#   HELPER CLASSES AND FUNCTIONS (original implementation)
# ═══════════════════════════════════════════════════════

class EmotionSmoother:
    """Original implementation preserved."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.histories = {}

    def update(self, face_id, emotion, confidence):
        if face_id not in self.histories:
            self.histories[face_id] = deque(maxlen=self.window_size)
        self.histories[face_id].append((emotion, confidence))

    def get_smoothed(self, face_id):
        if face_id not in self.histories:
            return "neutral", 0.0
        history = self.histories[face_id]
        emotion_scores = {}
        for emo, conf in history:
            emotion_scores[emo] = emotion_scores.get(emo, 0) + conf
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion] / len(history)
        return best_emotion, best_score

    def cleanup(self, active_ids):
        dead_ids = [fid for fid in self.histories if fid not in active_ids]
        for fid in dead_ids:
            del self.histories[fid]


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def simple_face_tracker(prev_faces, curr_detections, iou_threshold=0.3):
    """Original implementation preserved."""
    if not hasattr(simple_face_tracker, 'next_id'):
        simple_face_tracker.next_id = 0

    matched = {}
    used_prev = set()
    used_curr = set()

    if prev_faces and curr_detections:
        for ci, cbox in enumerate(curr_detections):
            best_iou = 0
            best_pid = None
            for pid, pbox in prev_faces.items():
                if pid in used_prev:
                    continue
                iou = compute_iou(pbox, cbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pid = pid
            if best_iou >= iou_threshold and best_pid is not None:
                matched[best_pid] = cbox
                used_prev.add(best_pid)
                used_curr.add(ci)

    for ci, cbox in enumerate(curr_detections):
        if ci not in used_curr:
            matched[simple_face_tracker.next_id] = cbox
            simple_face_tracker.next_id += 1

    return matched


def draw_face_box(frame, x1, y1, x2, y2, emotion, confidence):
    """Original implementation preserved."""
    emotion_colors = {
        'happy':    (0, 255, 128),
        'sad':      (255, 128, 0),
        'angry':    (0, 0, 255),
        'surprise': (0, 255, 255),
        'fear':     (255, 0, 255),
        'disgust':  (0, 128, 128),
        'neutral':  (200, 200, 200)
    }
    color = emotion_colors.get(emotion, BOX_COLOR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{emotion} ({confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

    label_y = y2 + th + 10
    if label_y > frame.shape[0]:
        bg_y1 = y1 - th - 10
        bg_y2 = y1
    else:
        bg_y1 = y2
        bg_y2 = y2 + th + 10

    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, bg_y2), color, -1)
    cv2.putText(frame, label, (x1 + 5, bg_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)


def draw_fps(frame, fps):
    """Original implementation preserved."""
    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (15, 15 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def get_face_transform():
    """Matches the validation transform used during training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_device(device_str: str = "auto"):
    """Device selection logic."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def load_fer_model(model_path: str, num_classes: int, device):
    """Load FER model from checkpoint."""
    global _fer_model
    if _fer_model is not None:
        return _fer_model
    print(f"📦 FER model loading: {model_path}")
    _fer_model = ImprovedHybridFER(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    _fer_model.load_state_dict(state_dict)
    _fer_model.to(device)
    _fer_model.eval()
    print("✅ FER model ready!")
    return _fer_model


def load_yolo_model(model_path: str):
    """Load YOLO face detection model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    from ultralytics import YOLO
    print(f"📦 YOLO model loading: {model_path}")
    _yolo_model = YOLO(model_path)
    print("✅ YOLO model ready!")
    return _yolo_model


# ═══════════════════════════════════════════════════════
#   MAIN MCP TOOL - Video Processing + Output Video + CSV + JSON
# ═══════════════════════════════════════════════════════

@mcp.tool()
def analyze_facial_emotions(
    video_path: str,
    fer_model_path: str,
    yolo_model_path: str = "yolo11n-face.pt",
    output_video_path: str = "",
    output_csv_path: str = "",
    emotion_classes: list[str] | None = None,
    face_conf_threshold: float = 0.45,
    smoothing_window: int = 5,
    process_every_n: int = 1,
    device: str = "auto"
) -> dict:
    """
    Facial emotions in the video analyzes.
    Original implementation preserved.

    Processed video çıktısı: bounding rectangle + emotion label + FPS overlay.
    Output: frame-level CSV detail + minute-level summary.

    Args:
        video_path: Video file path
        fer_model_path: Trained FER model file (.pt)
        yolo_model_path: YOLO face detection model
        output_video_path: Processed video save path (auto-generated if empty)
        output_csv_path: CSV save path (auto-generated if empty)
        emotion_classes: Emotion class list
        face_conf_threshold: Face detection confidence threshold
        smoothing_window: Emotion smoothing window
        process_every_n: Process every Nth frame
        device: Cihaz

    Returns:
        Analysis results + output dosya yolları
    """
    if emotion_classes is None:
        emotion_classes = DEFAULT_EMOTIONS

    dev = get_device(device)

    # Load models
    fer_model = load_fer_model(fer_model_path, len(emotion_classes), dev)
    yolo_model = load_yolo_model(yolo_model_path)
    face_transform = get_face_transform()

    # GPU warmup: force CUDA kernel compilation before the main loop so the first
    # real frames run at full speed instead of spending ~267s on JIT compilation.
    if dev.type == "cuda":
        print("🔥 GPU warmup (CUDA kernel init)...")
        dummy_tensor = torch.zeros(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for _ in range(3):
                    fer_model(dummy_tensor)
        torch.cuda.synchronize()
        # Warmup YOLO with a small blank numpy image
        import numpy as np
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            yolo_model(dummy_img, verbose=False)
        print("✅ GPU warmup done")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30  # guard against bogus WebM FPS metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"\n🎬 Video Info: {width}x{height} @ {fps:.1f} FPS, {duration_sec:.1f}s")

    # Set output paths
    if not output_video_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(
            os.path.dirname(video_path) or ".", f"{base}_analyzed.mp4"
        )
    if not output_csv_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_csv_path = os.path.join(
            os.path.dirname(video_path) or ".", f"{base}_emotions.csv"
        )

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Original implementation preserved.
    smoother = EmotionSmoother(window_size=smoothing_window)
    emotion_records = []
    prev_faces = {}
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    display_fps = 0.0
    minute_emotion_buffer = {}
    all_emotions = []
    faces_detected_total = 0
    face_absence_frames = 0
    multi_face_frames_count = 0

    # Reset face tracker state
    simple_face_tracker.next_id = 0

    print(f"🔄 Processing video...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time_sec = frame_count / fps
            current_minute = current_time_sec / 60.0

            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()

            # Process every Nth frame
            if frame_count % process_every_n != 0:
                draw_fps(frame, display_fps)
                out.write(frame)
                continue

            # YOLO face detection
            results = yolo_model(frame, conf=face_conf_threshold, verbose=False)

            curr_detections = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        curr_detections.append((x1, y1, x2, y2))

            # Integrity metrics: count processed frames with no face or multiple faces
            n_curr = len(curr_detections)
            if n_curr == 0:
                face_absence_frames += 1
            elif n_curr > 1:
                multi_face_frames_count += 1

            # Simple tracking
            tracked_faces = simple_face_tracker(prev_faces, curr_detections)
            prev_faces = tracked_faces
            smoother.cleanup(set(tracked_faces.keys()))

            # Emotion prediction for each face
            for face_id, (x1, y1, x2, y2) in tracked_faces.items():
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_tensor = face_transform(face_rgb).unsqueeze(0).to(dev)

                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                        outputs = fer_model(face_tensor)
                        probs = torch.softmax(outputs, dim=1)[0]

                max_idx = probs.argmax().item()
                emotion = emotion_classes[max_idx]
                confidence = probs[max_idx].item()

                # Smoothing
                smoother.update(face_id, emotion, confidence)
                smoothed_emotion, smoothed_conf = smoother.get_smoothed(face_id)

                # Draw bounding box with emotion label
                draw_face_box(frame, x1, y1, x2, y2, smoothed_emotion, smoothed_conf)

                # Record
                emotion_records.append({
                    'frame': frame_count,
                    'time_sec': round(current_time_sec, 2),
                    'time_min': round(current_minute, 2),
                    'face_id': face_id,
                    'emotion': smoothed_emotion,
                    'confidence': round(smoothed_conf, 3),
                    'raw_emotion': emotion,
                    'raw_confidence': round(confidence, 3)
                })

                all_emotions.append(smoothed_emotion)
                faces_detected_total += 1

                # Minute-level buffer
                min_int = int(current_minute)
                if face_id not in minute_emotion_buffer:
                    minute_emotion_buffer[face_id] = {}
                if min_int not in minute_emotion_buffer[face_id]:
                    minute_emotion_buffer[face_id][min_int] = []
                minute_emotion_buffer[face_id][min_int].append(smoothed_emotion)

            # Draw FPS
            draw_fps(frame, display_fps)

            # Write frame
            out.write(frame)

            # Progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {display_fps:.1f}")

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user.")
    finally:
        cap.release()
        out.release()

    # ─── CSV Kaydet () ───
    _save_emotion_csv(output_csv_path, emotion_records, minute_emotion_buffer)

    # ─── Analysis results ───
    total = len(all_emotions) if all_emotions else 1
    distribution = {e: all_emotions.count(e) / total for e in emotion_classes}
    dominant = max(distribution, key=distribution.get) if all_emotions else "neutral"

    # Minute-level summary (for JSON)
    minute_summary = []
    all_minutes = set()
    for fid, mins in minute_emotion_buffer.items():
        all_minutes.update(mins.keys())
    for minute in sorted(all_minutes):
        all_emo_min = []
        for fid, mins in minute_emotion_buffer.items():
            if minute in mins:
                all_emo_min.extend(mins[minute])
        if all_emo_min:
            counts = Counter(all_emo_min)
            dom = counts.most_common(1)[0]
            minute_summary.append({
                "minute": minute,
                "time_range": f"{minute}:00 - {minute}:59",
                "dominant_emotion": dom[0],
                "count": dom[1],
                "total_detections": len(all_emo_min),
                "percentage": round(dom[1] / len(all_emo_min) * 100, 1),
                "all_emotions": dict(counts)
            })

    # Positivity score
    valence = {
        'happy': 1.0, 'surprise': 0.7, 'neutral': 0.5,
        'sad': 0.2, 'fear': 0.15, 'angry': 0.1, 'disgust': 0.05
    }
    positivity = sum(distribution.get(e, 0) * valence.get(e, 0.5) for e in emotion_classes)

    result = {
        "records_count": len(emotion_records),
        "minute_summary": minute_summary,
        "dominant_emotion": dominant,
        "emotion_distribution": {k: round(v, 3) for k, v in distribution.items()},
        "positivity_score": round(positivity, 3),
        "total_frames_analyzed": frame_count,
        "faces_detected": faces_detected_total,
        "face_absence_seconds": round(face_absence_frames * process_every_n / fps, 1) if fps > 0 else 0.0,
        "multi_face_frames": multi_face_frames_count,
        "output_video_path": output_video_path,
        "output_csv_path": output_csv_path,
        "output_csv_summary_path": output_csv_path.replace('.csv', '_summary.csv'),
    }

    print(f"\n✅ Facial emotion analysis completed!")
    print(f"   📹 Processed video: {output_video_path}")
    print(f"   📊 CSV log: {output_csv_path}")
    print(f"   Dominant emotion: {dominant} | Pozitiflik: {positivity:.2f}")
    print(f"   Total face detections: {faces_detected_total}")

    return result


def _save_emotion_csv(csv_path, records, minute_buffer):
    """Original implementation preserved."""
    # 1. Detaylı CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'time_sec', 'time_min', 'face_id',
            'emotion', 'confidence', 'raw_emotion', 'raw_confidence'
        ])
        writer.writeheader()
        writer.writerows(records)
    print(f"   📄 Detailed log: {csv_path} ({len(records)} records)")

    # 2. Dakika Bazlı Özet CSV
    summary_path = csv_path.replace('.csv', '_summary.csv')
    summary_rows = []

    all_minutes = set()
    for fid, mins in minute_buffer.items():
        all_minutes.update(mins.keys())

    for minute in sorted(all_minutes):
        all_emo = []
        for fid, mins in minute_buffer.items():
            if minute in mins:
                all_emo.extend(mins[minute])
        if all_emo:
            counts = Counter(all_emo)
            dominant = counts.most_common(1)[0]
            total = len(all_emo)
            summary_rows.append({
                'minute': minute,
                'time_range': f"{minute}:00 - {minute}:59",
                'dominant_emotion': dominant[0],
                'count': dominant[1],
                'total_detections': total,
                'percentage': round(dominant[1] / total * 100, 1),
                'all_emotions': dict(counts)
            })

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'minute', 'time_range', 'dominant_emotion',
            'count', 'total_detections', 'percentage', 'all_emotions'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"   📊 Minute summary: {summary_path} ({len(summary_rows)} minutes)")


if __name__ == "__main__":
    print("🚀 Facial Emotion MCP Server starting...")
    mcp.run(transport="stdio")
