"""
Complete System Test for ISL Detection
=======================================
Tests all features and reports what's working
"""

import os
import sys
import subprocess
from pathlib import Path
import cv2

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}{text:^60}{END}")
    print(f"{BLUE}{'='*60}{END}")

def test_status(name, status, details=""):
    symbol = f"{GREEN}✓{END}" if status else f"{RED}✗{END}"
    status_text = f"{GREEN}PASS{END}" if status else f"{RED}FAIL{END}"
    print(f"{symbol} {name:40} [{status_text}]")
    if details:
        print(f"  {details}")

def run_command(cmd, timeout=10):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

# Main tests
def main():
    print_header("ISL DETECTION SYSTEM TEST")
    
    # Test 1: Check model file
    print(f"\n{YELLOW}[1] Model File Check{END}")
    model_path = "data/processed/artifacts/models/yolov5/best.pt"
    model_exists = os.path.exists(model_path)
    test_status("YOLOv5 Model (best.pt)", model_exists, 
                f"Path: {model_path}" if model_exists else "Model not found")
    
    # Test 2: Check YOLOv5 directory
    print(f"\n{YELLOW}[2] YOLOv5 Repository Check{END}")
    yolo_dir = "yolov5"
    yolo_exists = os.path.isdir(yolo_dir)
    detect_script = os.path.exists(f"{yolo_dir}/detect.py")
    test_status("YOLOv5 Directory", yolo_exists)
    test_status("detect.py Script", detect_script)
    
    # Test 3: Check dataset
    print(f"\n{YELLOW}[3] Dataset Check{END}")
    dataset_dir = "Dataset/Raw_Data"
    gestures = ["accident", "call", "doctor", "help", "hot", "lose", "pain", "thief"]
    dataset_exists = os.path.isdir(dataset_dir)
    test_status("Dataset Directory", dataset_exists)
    
    if dataset_exists:
        for gesture in gestures:
            gesture_dir = f"{dataset_dir}/{gesture}_Raw"
            exists = os.path.isdir(gesture_dir)
            if exists:
                video_count = len([f for f in os.listdir(gesture_dir) if f.endswith('.AVI')])
                test_status(f"  {gesture} videos", exists, f"{video_count} videos found")
    
    # Test 4: Python packages
    print(f"\n{YELLOW}[4] Python Dependencies{END}")
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML'
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            test_status(name, True)
        except ImportError:
            test_status(name, False, "Not installed")
    
    # Test 5: Webcam
    print(f"\n{YELLOW}[5] Webcam Check{END}")
    try:
        cap = cv2.VideoCapture(0)
        webcam_ok = cap.isOpened()
        if webcam_ok:
            ret, frame = cap.read()
            webcam_ok = ret
        cap.release()
        test_status("Webcam Access", webcam_ok)
    except:
        test_status("Webcam Access", False, "Cannot access webcam")
    
    # Test 6: Model inference test
    print(f"\n{YELLOW}[6] Model Inference Test{END}")
    test_video = "Dataset/Raw_Data/help_Raw/help_001_01.AVI"
    if os.path.exists(test_video) and model_exists and detect_script:
        print("  Running detection on test video...")
        cmd = f'python yolov5/detect.py --weights {model_path} --source "{test_video}" --conf 0.25 --exist-ok --save-txt'
        success, stdout, stderr = run_command(cmd, timeout=30)
        
        if success and "help" in stdout:
            test_status("Model Inference", True, "Successfully detected 'help' gesture")
        else:
            test_status("Model Inference", False, "No detections or error")
    else:
        test_status("Model Inference", False, "Missing requirements")
    
    # Test 7: Available features
    print(f"\n{YELLOW}[7] Feature Availability{END}")
    features = {
        "Video File Detection": model_exists and detect_script,
        "Webcam Detection": model_exists and detect_script and webcam_ok,
        "Batch Processing": model_exists and detect_script,
        "Save Results": model_exists and detect_script,
        "Visualization": model_exists and detect_script
    }
    
    for feature, available in features.items():
        test_status(feature, available)
    
    # Summary
    print_header("TEST SUMMARY")
    
    if model_exists and detect_script:
        print(f"{GREEN}✓ SYSTEM READY FOR DETECTION!{END}")
        print(f"\n{YELLOW}Quick Start Commands:{END}")
        print(f"\n1. Test on training video:")
        print(f"   python yolov5\\detect.py --weights {model_path} --source Dataset/Raw_Data/help_Raw/help_001_01.AVI --view-img")
        print(f"\n2. Live webcam detection:")
        print(f"   python yolov5\\detect.py --weights {model_path} --source 0 --view-img --conf 0.2")
        print(f"\n3. Process all videos:")
        print(f"   python yolov5\\detect.py --weights {model_path} --source Dataset/Raw_Data --save-txt")
    else:
        print(f"{RED}✗ SYSTEM NOT READY - Fix issues above{END}")
    
    print(f"\n{BLUE}{'='*60}{END}")

if __name__ == "__main__":
    main()
