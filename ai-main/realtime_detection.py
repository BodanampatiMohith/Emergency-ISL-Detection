"""Real-time ISL detection using OpenCV DNN and an exported YOLOv5 ONNX model."""

import sys
import time
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Ensure we can reuse YOLOv5 utility functions such as letterbox
PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

from utils.augmentations import letterbox

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_MODEL_PATH = "data/processed/artifacts/models/yolov5/best.onnx"
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45
IMG_SIZE = 640

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


def load_onnx_model(model_path: str) -> cv2.dnn.Net:
    """Load the ONNX model with OpenCV's DNN module."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX weights not found at '{model_path}'.\n"
            "Export your PyTorch weights using:"
            "\n  python yolov5/export.py --weights data/processed/artifacts/models/yolov5/best.pt "
            "--include onnx --ops et 12 --simplify"
        )

    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def preprocess(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Letterbox the frame to the desired input size and return blob-ready RGB image."""

    img, ratio, dwdh = letterbox(frame, IMG_SIZE, auto=False)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
    return blob, ratio, dwdh


def postprocess(
    outputs: np.ndarray,
    ratio: Tuple[float, float],
    dwdh: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> List[Tuple[int, int, int, int, float, int]]:
    """Convert raw network output into bounding boxes mapped to the original frame."""

    if isinstance(outputs, list):
        outputs = outputs[0]

    predictions = outputs.squeeze(0)
    if predictions.ndim == 1:
        predictions = predictions[np.newaxis, :]

    boxes = []
    scores = []
    class_ids = []

    r_w, r_h = ratio
    dw, dh = dwdh
    orig_h, orig_w = original_shape

    for row in predictions:
        obj_conf = row[4]
        if obj_conf < CONFIDENCE_THRESHOLD:
            continue

        class_scores = row[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]
        confidence = obj_conf * class_conf

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        cx, cy, w, h = row[:4]

        x = (cx - w / 2 - dw) / r_w
        y = (cy - h / 2 - dh) / r_h
        w /= r_w
        h /= r_h

        x = max(int(x), 0)
        y = max(int(y), 0)
        w = int(w)
        h = int(h)

        if x + w > orig_w:
            w = orig_w - x
        if y + h > orig_h:
            h = orig_h - y

        boxes.append([x, y, w, h])
        scores.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    detections: List[Tuple[int, int, int, int, float, int]] = []

    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = boxes[idx]
            detections.append((x, y, x + w, y + h, scores[idx], class_ids[idx]))

    return detections


def draw_detections(frame: np.ndarray, detections: List[Tuple[int, int, int, int, float, int]]) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""

    for x1, y1, x2, y2, conf, cls in detections:
        color = COLORS[cls % len(COLORS)]
        label = CLASS_NAMES[cls]

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

    return frame


def main() -> None:
    print("=" * 60)
    print("ISL EMERGENCY GESTURE DETECTION - REAL-TIME (OpenCV DNN)")
    print("=" * 60)
    print("Using exported YOLOv5 ONNX weights for webcam inference")
    print("=" * 60)

    try:
        net = load_onnx_model(ONNX_MODEL_PATH)
        print(f"[INFO] Loaded ONNX model: {ONNX_MODEL_PATH}")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam")
                break

            blob, ratio, dwdh = preprocess(frame)
            net.setInput(blob)
            outputs = net.forward()

            detections = postprocess(outputs, ratio, dwdh, frame.shape[:2])

            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                start_time = time.time()
                frame_count = 0

            annotated = draw_detections(frame.copy(), detections)
            status = f"FPS: {fps:.2f} | Model: YOLOv5-ONNX"
            cv2.putText(
                annotated,
                status,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                "Press 'q' to quit, 's' to save screenshot",
                (10, annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("ISL Emergency Gesture Detection - OpenCV", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit requested")
                break
            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"[INFO] Screenshot saved: {filename}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed")


if __name__ == "__main__":
    main()
