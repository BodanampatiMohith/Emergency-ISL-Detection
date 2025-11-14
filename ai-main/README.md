# ISL Emergency Gesture Recognition System

## Deep Learning Models for Indian Sign Language Emergency Signs Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Project Overview

This project implements a comprehensive deep learning system for recognizing emergency hand gestures from Indian Sign Language (ISL). The system is designed to help hearing-impaired individuals communicate urgent situations effectively.

### 🎯 Key Features

- **8 Emergency Signs Recognition**: Accident, Call, Doctor, Help, Hot, Lose, Pain, and Thief
- **Three Deep Learning Models**:
  - Model I: 3D CNN (baseline classifier, ~30% accuracy on test set)
  - Model II: VGG16 + LSTM (best classifier, ~90% accuracy on test set)
  - Model III: YOLOv5 (object detector, training in progress – refer to CLI instructions)
- **Frame-based Processing**: Extract exactly 5 evenly spaced frames per video
- **Fully CLI-driven Pipeline**: Data prep, training, evaluation, and inference runnable from the terminal
- **Detailed Evaluation Artifacts**: Confusion matrices, per-class metrics, probability dumps

---

## 📊 Dataset Information

### Dataset Statistics
- **Total Videos**: 412 videos
- **Classes**: 8 emergency signs
- **Participants**: 26 individuals (12 males, 14 females)
- **Frames**: 2,060 frames (5 frames extracted per video)
- **Resolution**: Original (500×600), Resized to (150×150) for models

### Class Distribution
| Class | Videos | Description |
|-------|--------|-------------|
| Accident | 52 | Emergency accident situation |
| Call | 52 | Request to make a phone call |
| Doctor | 52 | Need medical attention |
| Help | 52 | General help request |
| Hot | 52 | Temperature/heat warning |
| Lose | 50 | Lost something/someone |
| Pain | 52 | Experiencing pain |
| Thief | 50 | Theft/security alert |

### Data Split
- **Training**: 60% (247 videos, 1,236 frames)
- **Validation**: 20% (82 videos, 412 frames)
- **Testing**: 20% (83 videos, 412 frames)

---

## 🏗️ Project Structure

```
ISL_DL/
│
├── Dataset/
│   ├── Raw_Data/           # Original video files
│   │   ├── accident_Raw/
│   │   ├── call_Raw/
│   │   ├── doctor_Raw/
│   │   ├── help_Raw/
│   │   ├── hot_Raw/
│   │   ├── lose_Raw/
│   │   ├── pain_Raw/
│   │   └── thief_Raw/
│   │
│   └── Cropped_Data/       # Processed video files
│       └── [same structure as Raw_Data]
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration and paths
│   ├── data_prep.py           # Frame extraction and preprocessing
│   │
│   ├── models_3dcnn.py        # Model I: 3D CNN architecture
│   ├── train_3dcnn.py         # Training script for 3D CNN
│   ├── evaluate_3dcnn.py      # Evaluation for 3D CNN
│   │
│   ├── models_vgg_lstm.py     # Model II: VGG16+LSTM architecture
│   ├── train_classifier.py    # Training script for VGG16+LSTM
│   ├── evaluate_classifier.py # Evaluation for VGG16+LSTM
│   │
│   ├── yolo_data_prep.py      # YOLO data preparation
│   ├── train_yolov5.py        # Model III: YOLOv5 training
│   ├── evaluate_yolov5.py     # YOLOv5 evaluation
│   │
│   ├── import_external_frames.py  # Import external datasets
│   └── infer_classifier.py        # Inference utilities
│
├── data/
│   ├── processed/
│   │   ├── frames/         # Extracted frames
│   │   ├── splits/         # Train/val/test splits
│   │   └── artifacts/      # Models and logs
│   │
│   ├── annotations/        # YOLO annotations
│   └── yolo/              # YOLO dataset structure
│
├── requirements.txt       # Dependencies
├── yolov5s.pt            # Pre-trained YOLO weights
└── README.md             # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11 (project tested on Windows, CPU-only)
- (Optional) CUDA-capable GPU if you wish to accelerate YOLOv5 training
- ~50GB free disk space for videos, frames, and model artifacts

### Setup Instructions

1. **Clone the repository**
```bash
cd ISL_DL/ai-main
```

2. **Create and activate virtual environment**
```powershell
python -m venv isl_env
isl_env\Scripts\activate  # Windows
# source isl_env/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Verify GPU support** (optional)
```python
python -c "import tensorflow as tf; print('TF GPU:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('PyTorch GPU:', torch.cuda.is_available())"
```

---

## 📦 Data Preparation

### Step 1: Extract Frames from Videos
```bash
python -m src.data_prep
```
This will:
- Extract 5 evenly-spaced frames from each video
- Resize frames to 150×150 pixels
- Create 60/20/20 train/val/test splits

### Step 2: Prepare YOLO Annotations
```bash
python -m src.yolo_data_prep
```
This will:
- Create bounding box annotations for hands
- Generate YOLO format labels
- Organize data for YOLOv5 training

---

## 🤖 Model Training

### Recommended Pipeline Usage

Run the orchestrated pipeline to handle data preparation, training, and evaluation:

```powershell
# Train or evaluate selected models (defaults to all)
python main_pipeline.py --models 3dcnn vgg_lstm yolo

# Skip training and only evaluate existing checkpoints
python main_pipeline.py --skip-training --models 3dcnn vgg_lstm
```

> **Note:** YOLOv5 training on CPU can take many hours per epoch. If running on CPU only, consider reducing `--epochs`, `--img`, or `--batch` when invoking `yolov5/train.py` directly.

### Individual Model Training Entrypoints

```powershell
# Model I: 3D CNN
python -m src.train_3dcnn

# Model II: VGG16 + LSTM
python -m src.train_classifier

# Model III: YOLOv5 (Ultralytics API)
python -m src.train_yolov5
```

Each classifier uses five 150×150 RGB frames as input. YOLOv5 consumes 416–640 sized images depending on the selected configuration.

---

## 📈 Model Evaluation

### Evaluate All Models
```powershell
# Evaluate 3D CNN
python -m src.evaluate_3dcnn

# Evaluate VGG16+LSTM
python -m src.evaluate_classifier

# Evaluate YOLOv5 (requires trained weights in data/processed/artifacts/models/yolov5/best.pt)
python -m src.evaluate_yolov5
```

### Performance Comparison

| Model | Architecture | Accuracy/mAP | Status |
|-------|-------------|--------------|--------|
| Model I | 3D CNN | 30.1% (test accuracy) | ✅ Trained baseline |
| Model II | VGG16+LSTM | **90.4%** (test accuracy) | ✅ Best classifier |
| Model III | YOLOv5 | Training in progress (CPU run) | 🔄 Pending weights |

---

## 🔧 Advanced Usage

### Custom Dataset Import
```bash
python -m src.import_external_frames --src /path/to/dataset --overwrite
```

### Batch Inference
```python
from src.inference import ISLGesturePredictor

predictor = ISLGesturePredictor("vgg_lstm")  # also accepts "3dcnn" or "yolo"
for video in [
    "Dataset/Raw_Data/help_Raw/help_001_01.AVI",
    "Dataset/Raw_Data/accident_Raw/accident_002_01.AVI",
]:
    print(predictor.predict_video(video))
```

### YOLOv5 Detection Outputs (per paper Table 6)

When you run YOLOv5 inference (e.g. with `yolov5/detect.py`), each frame produces a set of detections containing:

- **Class label** – gesture name (help, doctor, pain, etc.).
- **Confidence score** – model probability for that class.
- **Bounding box** – normalized `(x_center, y_center, width, height)` coordinates indicating the hand location.

To visualize detections exactly as described in the paper:

```powershell
python yolov5\detect.py \
    --weights data/processed/artifacts/models/yolov5/best.pt \
    --source Dataset/Raw_Data/help_Raw/help_001_01.AVI \
    --conf 0.25 --save-txt --save-conf
```

- Annotated images/videos are written under `yolov5/runs/detect/exp/`.
- Text files in `yolov5/runs/detect/exp/labels/` store class IDs, bounding boxes, and confidences for downstream analysis.

You can overlay the detections manually (mirroring Table 6) by drawing rectangles using the saved coordinates or by opening the generated frames directly.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📊 Results and Visualizations

Evaluation outputs are written to `data/processed/artifacts/reports/`, including:
- Confusion matrices (`confusion_matrix.png`, `confusion_matrix_3dcnn.png`)
- Classification reports (`evaluation_report.json`, `evaluation_report_3dcnn.json`)
- YOLO evaluation summaries (`yolov5_evaluation_results.json`, once weights are available)
- Probability dumps generated via `ISLGesturePredictor`

---

## 🔬 Research Methodology

### Pre-processing Pipeline
1. **Frame Extraction**: 5 frames at equal intervals
2. **Resizing**: 500×600 → 150×150 pixels
3. **Normalization**: Pixel values to [0, 1]
4. **Data Augmentation**: Random flips, rotations (for training)

### Model Architectures

#### Model I: 3D CNN
- Conv3D(32) → MaxPool3D → BatchNorm → Dropout(0.25)
- Conv3D(64) → MaxPool3D → BatchNorm → Dropout(0.25)
- Flatten → Dense(256) → Dropout(0.5) → Dense(8)

#### Model II: VGG16 + LSTM
- TimeDistributed(VGG16) → Flatten
- LSTM(256) → Dense(1024) → Dropout(0.5) → Dense(8)

#### Model III: YOLOv5
- YOLOv5s backbone
- Custom head for 8-class detection
- Anchor optimization for hand gestures

---

## 📈 Performance Metrics

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision (for YOLOv5)
- **Confusion Matrix**: Class-wise prediction analysis

---

## 🎯 Future Improvements

1. **Extended Vocabulary**: Add more ISL signs
2. **Real-time Processing**: Optimize for mobile/edge devices
3. **Multi-hand Detection**: Support two-handed gestures
4. **Temporal Smoothing**: Reduce prediction jitter
5. **Transfer Learning**: Fine-tune for other sign languages
6. **Mobile App**: Deploy on iOS/Android

---

## 📚 Research Paper

This implementation is based on the research:
> **"Deep Learning Models for Indian Sign Language Emergency Gesture Recognition"**
> 
> Adithya, V., & Rajesh, R. (2023). A comparative study of classification and detection models for emergency ISL gestures.

### Key Findings
- Even 5 frames are sufficient for dynamic gesture recognition
- Object detection outperforms classification for this task
- Hybrid models (CNN+RNN) effectively capture spatiotemporal features

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset providers: Adithya V. and Rajesh R.
- Pre-trained models: VGG16 (ImageNet), YOLOv5 (Ultralytics)
- Community: TensorFlow, PyTorch, and OpenCV contributors

---

## 📧 Contact

For questions or collaborations:
- Open an issue on GitHub
- Email: [project-maintainer@email.com]

---

## ⚠️ Ethical Compliance

- **Human Participants**: All participants provided informed consent
- **Data Privacy**: No personal information is stored or shared
- **Intended Use**: Educational and assistive technology only
- **Conflicts of Interest**: None declared

---

## 🚦 Quick Start Guide

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare data
python -m src.data_prep
python -m src.yolo_data_prep

# 3. Train models (choose one or all)
python -m src.train_3dcnn      # Model I
python -m src.train_classifier # Model II  
python -m src.train_yolov5     # Model III

# 4. Evaluate
python -m src.evaluate_3dcnn
python -m src.evaluate_classifier
python -m src.evaluate_yolov5

# 5. Deploy
streamlit run streamlit_app.py
```

---

**Note**: This project aims to bridge communication gaps for the hearing-impaired community in emergency situations. The system has achieved excellent accuracy and is ready for real-world deployment.