"""
Test script to verify webcam and model setup for real-time detection
"""

import cv2
import os
import sys
from pathlib import Path

# Add YOLOv5 directory to path
PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

def test_webcam():
    """Test if webcam is accessible"""
    print("[TEST] Checking webcam access...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("✓ Webcam is accessible")
            return True
        else:
            print("✗ Webcam is connected but cannot read frames")
            return False
    else:
        print("✗ Cannot access webcam")
        print("  - Make sure webcam is connected")
        print("  - Check if another application is using it")
        return False


def test_model():
    """Test if model file exists and can be loaded"""
    print("[TEST] Checking YOLOv5 model...")
    model_path = "data/processed/artifacts/models/yolov5/best.pt"
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found at: {model_path}")
        return False
    
    print(f"✓ Model file exists: {model_path}")
    
    # Try to load with ultralytics
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✓ Model loaded successfully with Ultralytics")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        
        # Try alternative loading method
        try:
            print("[TEST] Trying alternative loading method...")
            from models.common import DetectMultiBackend
            model = DetectMultiBackend(model_path, device='cpu')
            print("✓ Model loaded with alternative method")
            return True
        except Exception as e2:
            print(f"✗ Alternative method also failed: {str(e2)}")
            return False


def test_dependencies():
    """Test if all required packages are installed"""
    print("[TEST] Checking dependencies...")
    required = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n[ACTION REQUIRED] Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    print("="*60)
    print("WEBCAM DETECTION SETUP TEST")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model", test_model),
        ("Webcam", test_webcam)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{name} Test:")
        print("-"*30)
        results[name] = test_func()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:15} : {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Ready for real-time detection!")
        print("\nYou can now run:")
        print("  python realtime_detection_simple.py")
    else:
        print("\n✗ Some tests failed - Please fix the issues above")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
