# 🏗️ System Architecture
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Model Architectures

### Model I: 3D CNN
- **Type**: Baseline Classifier
- **Accuracy**: ~30% (test set)
- **Input**: 5 RGB frames (150×150)
- **Architecture**:
  - 3D Convolutional layers for spatiotemporal feature extraction
  - MaxPooling layers for dimensionality reduction
  - Fully connected layers for classification
- **Use Case**: Baseline performance benchmark

### Model II: VGG16 + LSTM
- **Type**: Hybrid CNN-RNN
- **Accuracy**: ~90% (test set)
- **Input**: 5 RGB frames (150×150)
- **Architecture**:
  - **VGG16**: Pretrained on ImageNet for spatial feature extraction
  - **LSTM**: Processes temporal sequence of VGG16 features
  - Dense layers for final classification
- **Features**:
  - Transfer learning from ImageNet
  - Temporal modeling of sign language gestures
  - Better accuracy than 3D CNN

### Model III: YOLOv5
- **Type**: Object Detector
- **mAP@0.5**: 94.0%
- **Input**: 416-640px images
- **Architecture**:
  - Backbone: CSP (Cross Stage Partial) Network
  - Neck: PANet (Path Aggregation Network)
  - Head: Detection head with anchor boxes
- **Features**:
  - Real-time hand detection
  - High accuracy for emergency sign recognition
  - Efficient inference speed

## System Architecture

### Data Flow
1. **Input**: Video frames or real-time camera feed
2. **Preprocessing**:
   - Frame extraction (5 frames per video)
   - Resizing and normalization
   - Data augmentation (training only)
3. **Inference**:
   - Frame processing through the selected model
   - Feature extraction and classification
4. **Output**:
   - Predicted class label
   - Confidence score
   - Bounding box (YOLOv5 only)

### Project Structure
```
Emergency-ISL-Detection/
├── ai-main/                  # Main application code
│   ├── src/                  # Source code
│   │   ├── models_3dcnn.py   # 3D CNN implementation
│   │   ├── models_vgg_lstm.py # VGG16+LSTM implementation
│   │   ├── train_*.py        # Training scripts
│   │   └── evaluate_*.py     # Evaluation scripts
│   └── main_pipeline.py      # Main training pipeline
├── data/                     # Dataset and processed data
│   ├── raw/                  # Raw video files
│   └── processed/            # Processed frames and annotations
└── docs/                     # Documentation
```

## Performance Comparison

| Model | Architecture | Accuracy/mAP | Speed (FPS) | Parameters |
|-------|-------------|--------------|-------------|------------|
| 3D CNN | 3D Convolutional Network | 30.1% | 45 | ~1.2M |
| VGG16+LSTM | CNN-RNN Hybrid | 90.4% | 32 | ~15.3M |
| YOLOv5 | Object Detector | 94.0% mAP | 62 | ~7.2M |

## Model Selection Guide

### When to Use Which Model
- **For maximum accuracy**: Use YOLOv5 (94.0% mAP)
- **For real-time applications**: Use YOLOv5 (62 FPS)
- **For sequence understanding**: Use VGG16+LSTM
- **For baseline comparison**: Use 3D CNN

### Hardware Requirements
- **CPU**: All models can run on CPU
- **GPU**: Recommended for training (especially YOLOv5)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 50GB free space for dataset and models

## Future Improvements
1. Implement attention mechanisms
2. Add more sign language gestures
3. Optimize for mobile deployment
4. Add multi-person detection
5. Improve real-time performance
