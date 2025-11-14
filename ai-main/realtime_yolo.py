"""Real-time ISL Emergency Gesture Detection using native YOLOv5."""

import cv2
import numpy as np
import csv
import json
import os
import sys
import time
import torch
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    from playsound import playsound
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False

# Add YOLOv5 directory to path
PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Configuration
MODEL_PATH = "data/processed/artifacts/models/yolov5/best.pt"
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.45
IMG_SIZE = 416
SMOOTHING_WINDOW = 1
PERSISTENCE_THRESHOLD = 1
LOG_HISTORY_SIZE = 8
LOG_EVENT_COOLDOWN = 4.0
ALERT_CLASSES = {"accident"}
ALERT_THRESHOLD = 0.35
ALERT_REPEAT_COOLDOWN = 6.0
ALERT_SOUND_PATH = str(PROJECT_ROOT / "assets" / "alerts" / "accident_alert.mp3")
ALERT_DURATION = 3.0
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FRAMES_DIR = LOG_DIR / "frames"
LOG_CSV_PATH = LOG_DIR / "detections.csv"
LOG_JSON_PATH = LOG_DIR / "detections.jsonl"
COUNTER_RESET_KEY = "r"

# Class names for ISL emergency gestures
CLASS_NAMES = [
    "accident",
    "call",
    "doctor",
    "help",
    "hot",
    "lose",
    "pain",
    "thief",
]

# Color palette for bounding boxes (BGR format)
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 128, 0),
]


def play_alert_sound() -> None:
    """Play alert audio asynchronously if available."""

    if AUDIO_AVAILABLE and ALERT_SOUND_PATH and os.path.exists(ALERT_SOUND_PATH):
        threading.Thread(target=playsound, args=(ALERT_SOUND_PATH,), daemon=True).start()
        return

    if WINSOUND_AVAILABLE:
        def _beep_sequence():
            for _ in range(2):
                winsound.Beep(950, 220)
                time.sleep(0.1)
        threading.Thread(target=_beep_sequence, daemon=True).start()


def preprocess_frame(frame: np.ndarray, img_size: int, stride: int) -> torch.Tensor:
    """Preprocess an OpenCV frame for YOLO inference."""

    resized = letterbox(frame, img_size, stride=stride, auto=True)[0]
    resized = resized.transpose((2, 0, 1))[::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).float() / 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def annotate_detections(frame: np.ndarray, detections, fps: float) -> np.ndarray:
    """Overlay detections, FPS, and usage instructions on the frame."""

    if detections is not None and len(detections):
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(cls)
            label = CLASS_NAMES[cls_id]
            color = COLORS[cls_id % len(COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
            cv2.putText(
                frame,
                caption,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    status = f"FPS: {fps:.1f} | YOLOv5 Active"
    cv2.putText(
        frame,
        status,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Press 'q' quit | 's' save | 'c' clear log | 'r' reset counts",
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    return frame


def draw_persistence_overlay(
    frame: np.ndarray,
    detection_history: deque,
    aggregate_counts: dict,
    aggregate_conf: dict,
) -> None:
    """Draw status banner describing which gestures are stable."""

    if not detection_history:
        return

    persistent = [
        cls
        for cls, count in aggregate_counts.items()
        if count >= PERSISTENCE_THRESHOLD
    ]

    if persistent:
        message = (
            f"Stable detections ({PERSISTENCE_THRESHOLD}/{SMOOTHING_WINDOW}): "
            + ", ".join(
                f"{CLASS_NAMES[cls]} {aggregate_conf.get(cls, 0.0):.2f}"
                for cls in persistent
            )
        )
        cv2.putText(
            frame,
            message,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
    else:
        best_count = max(aggregate_counts.values()) if aggregate_counts else 0
        needed = max(PERSISTENCE_THRESHOLD - best_count, 1)
        message = (
            f"Stabilizing ({len(detection_history)}/{SMOOTHING_WINDOW}) – "
            f"need {needed} more hits"
        )
        cv2.putText(
            frame,
            message,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def draw_event_log(frame: np.ndarray, log_entries: deque) -> None:
    """Render recent detection events on the frame."""

    if not log_entries:
        return

    header_y = 90
    cv2.putText(
        frame,
        "Recent detections:",
        (10, header_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    for idx, entry in enumerate(log_entries):
        y = header_y + 22 * (idx + 1)
        if y > frame.shape[0] - 40:
            break
        cv2.putText(
            frame,
            entry,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def draw_class_counts(frame: np.ndarray, counts: dict) -> None:
    """Render per-class detection counts on the frame."""

    if not counts:
        return

    height, width = frame.shape[:2]
    base_x = max(width - 220, 10)
    base_y = 90

    cv2.putText(
        frame,
        "Detections:",
        (base_x, base_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    for idx, class_name in enumerate(CLASS_NAMES):
        count = counts.get(class_name, 0)
        y = base_y + 22 * (idx + 1)
        cv2.putText(
            frame,
            f"{class_name}: {count}",
            (base_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def draw_alert_banner(frame: np.ndarray, active: bool, message: str) -> None:
    """Render a prominent alert banner on the frame."""

    if not active or not message:
        return

    height, width = frame.shape[:2]
    banner_height = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        message,
        (20, int(banner_height * 0.65)),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (255, 255, 255),
        2,
    )


def log_detection_event(timestamp: str, class_name: str, confidence: float, frame: np.ndarray | None) -> str:
    """Persist detection metadata to CSV/JSON and optionally save a frame snapshot."""

    frame_path = ""
    if frame is not None:
        filename = f"{timestamp.replace(':', '').replace('-', '').replace(' ', '_')}_{class_name}.jpg"
        frame_path = str(LOG_FRAMES_DIR / filename)
        cv2.imwrite(frame_path, frame)

    with LOG_CSV_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, class_name, f"{confidence:.4f}", frame_path])

    record = {
        "timestamp": timestamp,
        "class": class_name,
        "confidence": round(confidence, 4),
        "frame_path": frame_path,
    }
    with LOG_JSON_PATH.open("a", encoding="utf-8") as jsonfile:
        jsonfile.write(json.dumps(record) + "\n")

    return frame_path


def main() -> None:
    print("=" * 60)
    print("ISL EMERGENCY GESTURE DETECTION - REAL-TIME")
    print("=" * 60)
    print("Using YOLOv5 for real-time hand gesture detection")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        return

    device = select_device("0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    try:
        model = DetectMultiBackend(MODEL_PATH, device=device, fp16=False)
        stride = model.stride
        imgsz = check_img_size(IMG_SIZE, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))
        print(f"[INFO] Model loaded successfully (image size: {imgsz})")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return

    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Starting real-time detection...")
    print("[CONTROLS]")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'c' to clear detection log")
    print(f"  - Press '{COUNTER_RESET_KEY}' to reset counters")
    print("=" * 60)

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    detection_history: deque = deque(maxlen=SMOOTHING_WINDOW)
    detection_log: deque[str] = deque(maxlen=LOG_HISTORY_SIZE)
    last_logged: dict[int, float] = {}
    last_alert: dict[str, float] = {}
    alert_active = False
    alert_message = ""
    alert_expire_time = 0.0
    class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV_PATH.exists():
        with LOG_CSV_PATH.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "class", "confidence", "frame_path"])
    if not LOG_JSON_PATH.exists():
        LOG_JSON_PATH.touch()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            tensor = preprocess_frame(frame, imgsz, stride).to(device)

            with torch.no_grad():
                predictions = model(tensor, augment=False, visualize=False)

            predictions = non_max_suppression(
                predictions,
                CONFIDENCE_THRESHOLD,
                IOU_THRESHOLD,
                classes=None,
                agnostic=False,
                max_det=1000,
            )

            detections = None
            summary = {}
            for det in predictions:
                if len(det):
                    det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], frame.shape).round()
                    detections = det
                    for *_, conf, cls in det:
                        conf_val = float(conf)
                        if conf_val < CONFIDENCE_THRESHOLD:
                            continue
                        cls_id = int(cls)
                        summary.setdefault(cls_id, []).append(conf_val)

            detection_history.append(summary)

            current_time = time.time()
            for cls_id, conf_list in summary.items():
                best_conf = max(conf_list)
                last_time = last_logged.get(cls_id, 0.0)
                if current_time - last_time >= LOG_EVENT_COOLDOWN:
                    last_logged[cls_id] = current_time
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry = f"{timestamp.split(' ')[1]} • {CLASS_NAMES[cls_id]} ({best_conf:.2f})"
                    detection_log.appendleft(entry)
                    class_name = CLASS_NAMES[cls_id]
                    class_counts[class_name] += 1
                    saved_frame = log_detection_event(timestamp, class_name, best_conf, frame if detections is not None else None)
                    if saved_frame:
                        print(f"[LOG] {entry} -> {saved_frame}")
                    else:
                        print(f"[LOG] {entry}")

                cls_name = CLASS_NAMES[cls_id]
                if cls_name in ALERT_CLASSES and best_conf >= ALERT_THRESHOLD:
                    previous = last_alert.get(cls_name, 0.0)
                    if current_time - previous >= ALERT_REPEAT_COOLDOWN:
                        last_alert[cls_name] = current_time
                        alert_active = True
                        alert_expire_time = current_time + ALERT_DURATION
                        alert_message = f"⚠ {cls_name.upper()} DETECTED ({best_conf:.2f})"
                        print(f"[ALERT] {alert_message}")
                        play_alert_sound()

            aggregate_counts: dict = {}
            aggregate_conf: dict = {}
            for hist in detection_history:
                for cls_id, conf_list in hist.items():
                    aggregate_counts[cls_id] = aggregate_counts.get(cls_id, 0) + 1
                    best_conf = max(conf_list)
                    aggregate_conf[cls_id] = max(aggregate_conf.get(cls_id, 0.0), best_conf)

            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                start_time = time.time()
                frame_count = 0

            if alert_active and current_time > alert_expire_time:
                alert_active = False
                alert_message = ""

            annotated = annotate_detections(frame.copy(), detections, fps)
            draw_persistence_overlay(annotated, detection_history, aggregate_counts, aggregate_conf)
            draw_event_log(annotated, detection_log)
            draw_class_counts(annotated, class_counts)
            draw_alert_banner(annotated, alert_active, alert_message)

            cv2.imshow("ISL Gesture Detection - YOLOv5", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit requested")
                break
            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output = f"isl_detection_{timestamp}.jpg"
                cv2.imwrite(output, annotated)
                print(f"[INFO] Screenshot saved: {output}")
            if key == ord("c"):
                detection_log.clear()
                last_logged.clear()
                print("[INFO] Detection log cleared")
            if key == ord(COUNTER_RESET_KEY):
                for name in class_counts:
                    class_counts[name] = 0
                print("[INFO] Detection counts reset")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully")


if __name__ == "__main__":
    main()