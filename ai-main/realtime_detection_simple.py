"""
Simplified Real-time ISL Emergency Gesture Detection using YOLOv5
==================================================================
Alternative implementation using Ultralytics YOLO for easier deployment
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add YOLOv5 directory to path (needed for model loading)
PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

from ultralytics import YOLO
import time

# Configuration
MODEL_PATH = "data/processed/artifacts/models/yolov5/best.pt"
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

# Class names for ISL emergency gestures
CLASS_NAMES = ["accident", "call", "doctor", "help", "hot", "lose", "pain", "thief"]

# Color palette for bounding boxes (BGR format)
COLORS = {
    0: (255, 0, 0),    # Blue for accident
    1: (0, 255, 0),    # Green for call
    2: (0, 0, 255),    # Red for doctor
    3: (255, 255, 0),  # Cyan for help
    4: (255, 0, 255),  # Magenta for hot
    5: (0, 255, 255),  # Yellow for lose
    6: (128, 0, 128),  # Purple for pain
    7: (255, 128, 0),  # Orange for thief
}


def draw_boxes(frame, results):
    """Draw bounding boxes and labels on frame"""
    if results[0].boxes is None:
        return frame
    
    boxes = results[0].boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Get color and label
        color = COLORS.get(cls, (255, 255, 255))
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    """Main function for real-time detection"""
    print("="*60)
    print("ISL EMERGENCY GESTURE DETECTION - SIMPLIFIED VERSION")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        return
    
    # Load YOLOv5 model
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("[INFO] Model loaded successfully with Ultralytics")
    except Exception as e:
        print(f"[WARNING] Ultralytics loading failed: {str(e)}")
        print("[INFO] Trying alternative loading method...")
        
        # Use the direct YOLOv5 inference approach
        try:
            from models.common import DetectMultiBackend
            from utils.general import check_img_size, non_max_suppression, scale_boxes
            from utils.torch_utils import select_device
            from utils.augmentations import letterbox
            import torch
            
            # Load model using YOLOv5 native loader
            device = select_device('0' if torch.cuda.is_available() else 'cpu')
            model = DetectMultiBackend(MODEL_PATH, device=device, fp16=False)
            stride = model.stride
            imgsz = check_img_size(640, s=stride)
            model.warmup(imgsz=(1, 3, imgsz, imgsz))
            
            print("[INFO] Model loaded successfully with native YOLOv5 loader")
            
            # We'll need to handle inference differently
            return run_native_detection(model, stride, imgsz, device)
        except Exception as e2:
            print(f"[ERROR] Alternative loading also failed: {str(e2)}")
            return
    
    # Initialize webcam
    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        print("[TIP] Make sure no other application is using the webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INFO] Starting detection...")
    print("[CONTROLS]")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'r' to reset FPS counter")
    print("="*60)
    
    # FPS variables
    fps = 0
    prev_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        # Run YOLOv5 inference
        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        # Draw detections
        annotated_frame = draw_boxes(frame.copy(), results)
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time
        
        # Add status text
        status_text = f"FPS: {fps:.1f} | Gestures Detected: "
        if results[0].boxes is not None:
            num_detections = len(results[0].boxes)
            status_text += str(num_detections)
        else:
            status_text += "0"
        
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('ISL Gesture Detection (YOLOv5)', annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"[INFO] Screenshot saved: {filename}")
        elif key == ord('r'):
            frame_count = 0
            prev_time = time.time()
            print("[INFO] FPS counter reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed")


if __name__ == "__main__":
    main()
