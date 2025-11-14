"""
Debug script to understand why detections aren't showing
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

print("="*60)
print("DETECTION DEBUG MODE")
print("="*60)

# Test 1: Webcam capture
print("\n[TEST 1] Testing webcam capture...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✓ Webcam working - Frame size:", frame.shape)
        # Save a test frame
        cv2.imwrite("test_frame.jpg", frame)
        print("✓ Test frame saved as 'test_frame.jpg'")
    cap.release()
else:
    print("✗ Webcam not accessible")
    exit(1)

# Test 2: Model detection on saved frame
print("\n[TEST 2] Testing model on saved frame...")
try:
    from ultralytics import YOLO
    
    model_path = 'data/processed/artifacts/models/yolov5/best.pt'
    model = YOLO(model_path)
    
    # Test on the captured frame with VERY low threshold
    results = model.predict(
        source='test_frame.jpg',
        conf=0.1,  # Very low confidence to see ANY detection
        save=True,  # Save the annotated image
        save_txt=True,  # Save detection labels
        verbose=True  # Show what's happening
    )
    
    # Check results
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            print(f"✓ DETECTIONS FOUND: {len(r.boxes)} boxes")
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  - Class {cls} with confidence {conf:.3f}")
        else:
            print("✗ No detections in test frame")
            print("  This might mean:")
            print("  1. No clear hand gesture in frame")
            print("  2. Gesture not one of the 8 trained classes")
            print("  3. Model needs to see specific hand positions")
    
except Exception as e:
    print(f"✗ Error during detection: {e}")

# Test 3: Test on training data
print("\n[TEST 3] Testing on known good images from training...")
test_images = [
    "data/yolo/images/test/help_0000.jpg",
    "data/yolo/images/test/doctor_0000.jpg",
    "data/processed/frames/help/help_001_01_frame_0.jpg"
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\nTesting: {img_path}")
        try:
            results = model.predict(
                source=img_path,
                conf=0.25,
                save=True,
                verbose=False
            )
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    print(f"  ✓ Found {len(r.boxes)} detections")
                else:
                    print(f"  ✗ No detections")
        except:
            pass
    else:
        print(f"  - Image not found: {img_path}")

print("\n[INFO] Check 'runs/detect/' folder for annotated results")
print("[INFO] If detections work on training images but not webcam:")
print("  - Your hand position might be different from training data")
print("  - Try mimicking the exact gestures from the training videos")
print("  - Ensure good lighting and clear hand visibility")
