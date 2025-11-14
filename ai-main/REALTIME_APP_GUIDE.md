# Real-time ISL Emergency Gesture Detection Application

## Overview
This application uses the trained YOLOv5 model (92.5% mAP) to perform real-time detection of Indian Sign Language emergency gestures using your webcam. It displays live video with bounding boxes, class labels, and confidence scores for detected gestures.

## Prerequisites

### 1. Trained Model
Ensure you have the trained YOLOv5 model at:
```
data/processed/artifacts/models/yolov5/best.pt
```

### 2. Webcam
- A working webcam connected to your computer
- Ensure no other application is using the webcam

### 3. Dependencies
All required packages should already be installed from the main project setup:
- OpenCV (cv2)
- PyTorch
- NumPy
- YOLOv5 repository files

## Quick Start

### Step 1: Test Your Setup
First, verify everything is working:
```powershell
python test_webcam_setup.py
```

This will check:
- ✓ All dependencies are installed
- ✓ Model file exists and can be loaded
- ✓ Webcam is accessible

### Step 2: Run the Application
If all tests pass, run the real-time detection:
```powershell
python realtime_yolo.py
```

## Features

### Live Detection
- **Real-time Processing**: Processes webcam frames continuously
- **Bounding Boxes**: Draws colored rectangles around detected hand gestures
- **Class Labels**: Shows the gesture name (e.g., "help", "doctor", "pain")
- **Confidence Scores**: Displays model confidence (0.00 to 1.00)
- **FPS Counter**: Shows frames per second for performance monitoring

### Gesture Classes Detected
The application can detect 8 emergency gestures:
1. **accident** - Blue box
2. **call** - Green box
3. **doctor** - Red box
4. **help** - Cyan box
5. **hot** - Magenta box
6. **lose** - Yellow box
7. **pain** - Purple box
8. **thief** - Orange box

### Controls
- **'q'** - Quit the application
- **'s'** - Save a screenshot of current detection
- **ESC** - Alternative quit option

## Application Output

### On-screen Display
The application window shows:
```
┌─────────────────────────────────────┐
│ FPS: 25.3 | YOLOv5 Detection Active │
│                                     │
│     [Live video with overlays]      │
│     ┌──────────┐                   │
│     │ help 0.93│                   │
│     │          │ <- Detected hand   │
│     └──────────┘                   │
│                                     │
│ Press 'q' to quit, 's' to save     │
└─────────────────────────────────────┘
```

### Screenshot Files
Saved as: `isl_detection_YYYYMMDD_HHMMSS.jpg`

## Performance Notes

### Expected Performance
- **CPU**: 5-15 FPS (depending on processor)
- **GPU**: 20-30+ FPS (if CUDA is available)
- **Detection Range**: Best results at 0.5-2 meters from camera
- **Lighting**: Good lighting improves detection accuracy

### Optimization Tips
1. **Close other applications** to free up CPU/GPU resources
2. **Ensure good lighting** - avoid backlight or shadows on hands
3. **Keep hands clearly visible** in the camera frame
4. **Perform gestures slowly** for better detection

## Troubleshooting

### Issue: "Cannot access webcam"
**Solution**:
- Check if webcam is properly connected
- Close other applications using the webcam (Zoom, Teams, etc.)
- Try unplugging and reconnecting the webcam

### Issue: "Model not found"
**Solution**:
```powershell
# Check if model exists
dir data\processed\artifacts\models\yolov5\best.pt

# If missing, train the model
python -m src.train_yolov5
```

### Issue: Low FPS / Laggy detection
**Solution**:
- Close unnecessary applications
- Reduce webcam resolution in the code (line ~150 in realtime_yolo.py)
- Consider using GPU if available

### Issue: No detections appearing
**Solution**:
- Ensure hands are clearly visible
- Try moving closer to the camera
- Improve lighting conditions
- Perform gestures more distinctly

## Technical Details

### Model Architecture
- **Base**: YOLOv5s (small variant)
- **Input Size**: 640×640 pixels
- **Classes**: 8 ISL emergency gestures
- **Weights**: 7.03M parameters
- **Performance**: 92.5% mAP@0.5 on test set

### Processing Pipeline
1. **Frame Capture**: Read frame from webcam
2. **Preprocessing**: Resize and normalize to 640×640
3. **Inference**: Run YOLOv5 detection
4. **NMS**: Apply Non-Maximum Suppression
5. **Visualization**: Draw boxes and labels
6. **Display**: Show annotated frame

### Configuration
Key parameters in `realtime_yolo.py`:
```python
CONFIDENCE_THRESHOLD = 0.45  # Minimum detection confidence
IOU_THRESHOLD = 0.45        # NMS IoU threshold
IMG_SIZE = 640              # Model input size
```

## Alternative Versions

### 1. `realtime_detection.py`
- Full implementation with detailed error handling
- More verbose output

### 2. `realtime_detection_simple.py` 
- Simplified version using Ultralytics API
- Easier to modify but may have compatibility issues

### 3. `realtime_yolo.py` (Recommended)
- Robust implementation using native YOLOv5 loader
- Best compatibility and performance

## Integration with Main Pipeline

This real-time application is the final deployment stage of the ISL gesture recognition pipeline:

1. **Data Preparation** → 2. **Model Training** → 3. **Evaluation** → 4. **Real-time Deployment** (You are here)

The application uses Model III (YOLOv5) as specified in the research paper, which is designed for real-time detection tasks rather than video classification.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test output: `python test_webcam_setup.py`
3. Check the main project README for general setup issues

---

**Note**: This application fulfills the core objective of the Areeb et al. paper by providing real-time detection of emergency ISL gestures for practical deployment.
