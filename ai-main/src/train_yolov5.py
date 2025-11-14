"""
Train YOLOv5 for ISL Emergency Gesture Detection (Model III)
============================================================
This script trains a YOLOv5 model to detect and classify hand gestures
"""

import os
import sys
import shutil
import torch
import yaml
import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODEL_DIR,
    LOGS_DIR,
    YOLO_IMG_SIZE,
    YOLO_BATCH_SIZE,
    YOLO_EPOCHS,
    CLASSES
)

# Ensure local YOLOv5 repository (for legacy module imports in checkpoints) is importable
YOLOV5_DIR = os.path.join(PROJECT_ROOT, "yolov5")
if os.path.isdir(YOLOV5_DIR) and YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)


def train_yolov5():
    """
    Train YOLOv5 model for hand gesture detection and classification
    """
    print("[INFO] Starting YOLOv5 training for ISL Emergency Gesture Detection...")
    
    # Paths
    yolo_data_dir = os.path.join(DATA_DIR, "yolo")
    dataset_yaml = os.path.join(yolo_data_dir, "dataset.yaml")
    
    # Check if dataset exists
    if not os.path.exists(dataset_yaml):
        print("[ERROR] YOLOv5 dataset not found. Please run yolo_data_prep.py first.")
        from src.yolo_data_prep import prepare_yolo_dataset
        prepare_yolo_dataset()
    
    # Initialize YOLOv5 model using yolov5n which is more compatible
    model = YOLO('yolov5n.pt')  # Use nano model 
    print("[INFO] Using YOLOv5n pre-trained weights")
    
    # Training parameters
    run_name = f"isl_gesture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_params = {
        'data': dataset_yaml,
        'epochs': YOLO_EPOCHS,
        'imgsz': YOLO_IMG_SIZE,
        'batch': YOLO_BATCH_SIZE,
        'project': os.path.join(MODEL_DIR, 'yolov5'),
        'name': run_name,
        'exist_ok': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 2,
        'verbose': True,
        'seed': 42,
    }
    
    print(f"[INFO] Training on device: {train_params['device']}")
    print(f"[INFO] Training parameters:")
    for key, value in train_params.items():
        print(f"  - {key}: {value}")
    
    # Train the model
    try:
        results = model.train(**train_params)
        
        # Find the best weights from the training run
        train_dir = os.path.join(MODEL_DIR, 'yolov5', run_name)
        best_src = os.path.join(train_dir, 'weights', 'best.pt')
        
        # Wait a moment for file to be written
        import time
        time.sleep(2)
        
        if not os.path.exists(best_src):
            # Try to find any best.pt in the yolov5 directory
            for root, dirs, files in os.walk(os.path.join(MODEL_DIR, 'yolov5')):
                if 'best.pt' in files:
                    best_src = os.path.join(root, 'best.pt')
                    break
        
        if os.path.exists(best_src):
            best_model_path = os.path.join(MODEL_DIR, 'yolov5', 'best.pt')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            shutil.copy2(best_src, best_model_path)
            print(f"[INFO] Best model saved to: {best_model_path}")
        else:
            print(f"[WARNING] Could not find best.pt after training")
        
        # Save training results
        results_dict = {
            'model': 'YOLOv5s',
            'classes': CLASSES,
            'num_classes': len(CLASSES),
            'image_size': YOLO_IMG_SIZE,
            'epochs': YOLO_EPOCHS,
            'batch_size': YOLO_BATCH_SIZE,
            'best_model_path': best_model_path,
            'training_completed': datetime.now().isoformat()
        }
        
        results_path = os.path.join(LOGS_DIR, 'yolov5_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"[INFO] Training completed successfully!")
        print(f"[INFO] Results saved to: {results_path}")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        raise


def validate_yolov5(model=None):
    """
    Validate the trained YOLOv5 model
    """
    if model is None:
        # Load the best model
        best_model_path = os.path.join(MODEL_DIR, 'yolov5', 'best.pt')
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(MODEL_DIR, 'yolov5_isl_gesture.pt')
        
        if not os.path.exists(best_model_path):
            print("[ERROR] No trained model found. Please train first.")
            return
        
        model = YOLO(best_model_path)
    
    # Validation data
    yolo_data_dir = os.path.join(DATA_DIR, "yolo")
    dataset_yaml = os.path.join(yolo_data_dir, "dataset.yaml")
    
    # Validate
    print("[INFO] Running validation...")
    metrics = model.val(data=dataset_yaml, imgsz=YOLO_IMG_SIZE, batch=YOLO_BATCH_SIZE)
    
    # Extract and save metrics
    val_results = {
        'box_mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
        'box_mAP50-95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
        'validation_completed': datetime.now().isoformat()
    }
    
    # Save validation results
    val_results_path = os.path.join(LOGS_DIR, 'yolov5_validation_results.json')
    with open(val_results_path, 'w') as f:
        json.dump(val_results, f, indent=2)
    
    print(f"[INFO] Validation completed!")
    print(f"[INFO] mAP@50: {val_results['box_mAP50']:.4f}")
    print(f"[INFO] mAP@50-95: {val_results['box_mAP50-95']:.4f}")
    print(f"[INFO] Results saved to: {val_results_path}")
    
    return val_results


if __name__ == "__main__":
    # Check if data preparation is needed
    yolo_data_dir = os.path.join(DATA_DIR, "yolo")
    if not os.path.exists(yolo_data_dir):
        print("[INFO] YOLOv5 dataset not found. Preparing data...")
        from src.yolo_data_prep import prepare_yolo_dataset
        prepare_yolo_dataset()
    
    # Train the model
    model = train_yolov5()
    
    # Validate the model
    if model:
        validate_yolov5(model)
