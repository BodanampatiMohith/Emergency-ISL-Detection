"""
Real-time ISL Detection using Ultralytics YOLO - Direct Implementation
======================================================================
Exactly as described in the requirements - using Ultralytics API directly
"""

import sys
import os
from pathlib import Path

# Add YOLOv5 to path
YOLOV5_DIR = Path(__file__).parent / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

# Set environment variable
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')};{YOLOV5_DIR}"

print("="*60)
print("ISL EMERGENCY GESTURE DETECTION")
print("Real-time Detection Using Your Webcam")
print("="*60)
print("\nINSTRUCTIONS:")
print("1. The webcam window will open shortly")
print("2. Perform one of the 8 emergency gestures:")
print("   - accident, call, doctor, help, hot, lose, pain, thief")
print("3. Keep your hand clearly visible and well-lit")
print("4. The model will draw boxes around detected gestures")
print("5. Press 'q' in the video window to quit")
print("\nTIPS FOR BETTER DETECTION:")
print("- Stand 1-2 meters from camera")
print("- Ensure good lighting on your hands")
print("- Perform gestures slowly and clearly")
print("- Try different angles if not detecting")
print("="*60)

# Import and run
from ultralytics import YOLO

# Load model and run prediction with lower confidence threshold for testing
model_path = 'data/processed/artifacts/models/yolov5/best.pt'
print(f"\n[INFO] Loading model from: {model_path}")

# Create YOLO model
model = YOLO(model_path)

# Run prediction with lower threshold to see more detections
print("[INFO] Starting webcam detection...")
print("[INFO] Opening webcam window...\n")

# Run with very low confidence to see ANY detections
results = model.predict(
    source=0,           # Use webcam
    show=True,         # Show live window
    conf=0.25,         # Lower confidence threshold
    iou=0.45,          # IOU threshold
    save=False,        # Don't save video
    stream=True,       # Stream mode for continuous detection
    verbose=False      # Less verbose output
)

# Process results (this will run continuously)
for result in results:
    # The display is handled by show=True
    # This loop keeps the detection running
    pass

print("\n[INFO] Detection stopped")
print("[INFO] Application closed")
