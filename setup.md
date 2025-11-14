# 🛠️ Setup Guide
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Prerequisites
- Python 3.8+
- ~50GB free disk space
- CUDA-capable GPU (recommended for training)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Emergency-ISL-Detection.git
cd Emergency-ISL-Detection
```

### 2. Set Up Virtual Environment
```bash
# Create and activate virtual environment
python -m venv isl_env

# Windows
.\isl_env\Scripts\activate

# Linux/Mac
# source isl_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"

# Check PyTorch installation
python -c "import torch; print('PyTorch Version:', torch.__version__)"

# Check CUDA availability (if using GPU)
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## Dataset Setup

### 1. Download the Dataset
1. Download from Mendeley Data: [ISL Emergency Gestures Dataset](https://data.mendeley.com/datasets/2vfdm42337/1)
2. Click "Download" to get the ZIP file

### 2. Extract and Organize the Dataset
```bash
# Create data directory
mkdir -p data/raw

# Extract the downloaded ZIP
unzip ~/Downloads/ISL_Emergency_Gestures.zip -d data/raw/
```

### 3. Verify Dataset Structure
```bash
# Should show 8 class folders in both train and test
ls data/raw/train/
ls data/raw/test/
```

## Running the Application

### For Training
```bash
# Train all models
python main_pipeline.py

# Or train specific models
python -m src.train_3dcnn       # 3D CNN
python -m src.train_classifier  # VGG16+LSTM
python -m src.train_yolov5      # YOLOv5
```

### For Evaluation
```bash
# Evaluate all models
python -m src.evaluate_3dcnn
python -m src.evaluate_classifier
python -m src.evaluate_yolov5
```

### For Real-time Detection
```bash
python realtime_detection.py
```

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in the respective training scripts
2. **Missing Dependencies**: Ensure all packages in requirements.txt are installed
3. **Dataset Not Found**: Verify the dataset is extracted to `data/raw/` with correct folder structure

### Getting Help
For additional support, please open an issue on the GitHub repository.
