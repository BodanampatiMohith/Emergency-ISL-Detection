"""
Main Pipeline for ISL Emergency Gesture Recognition
=====================================================
Complete pipeline to train and evaluate all three models
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    RAW_VIDEOS_DIR,
    FRAMES_DIR
)


def check_dataset():
    """Check if dataset exists and is properly structured"""
    print("\n" + "="*60)
    print("CHECKING DATASET")
    print("="*60)
    
    # Check for raw videos
    raw_video_count = 0
    for class_name in ['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief']:
        class_dir = os.path.join(RAW_VIDEOS_DIR, f"{class_name}_Raw")
        if os.path.exists(class_dir):
            videos = [f for f in os.listdir(class_dir) if f.endswith(('.AVI', '.avi', '.mp4', '.mov'))]
            raw_video_count += len(videos)
            print(f"  {class_name:10s}: {len(videos):3d} videos")
    
    if raw_video_count == 0:
        print("\n[ERROR] No videos found in Dataset/Raw_Data/")
        print("Please ensure videos are placed in the correct directories.")
        return False
    
    print(f"\n[INFO] Total videos found: {raw_video_count}")
    return True


def prepare_data():
    """Extract frames and create splits"""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Check if frames already exist
    if os.path.exists(FRAMES_DIR) and os.listdir(FRAMES_DIR):
        print("[INFO] Frames already extracted. Skipping...")
        return True
    
    print("[INFO] Extracting frames from videos...")
    try:
        from src.data_prep import prepare_frames, make_splits
        prepare_frames()
        make_splits()
        print("[SUCCESS] Data preparation completed!")
        return True
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {str(e)}")
        return False


def prepare_yolo_data():
    """Prepare YOLO annotations and dataset"""
    print("\n" + "="*60)
    print("YOLO DATA PREPARATION")
    print("="*60)
    
    yolo_dir = os.path.join(DATA_DIR, "yolo")
    if os.path.exists(yolo_dir) and os.listdir(yolo_dir):
        print("[INFO] YOLO data already prepared. Skipping...")
        return True
    
    print("[INFO] Preparing YOLO dataset...")
    try:
        from src.yolo_data_prep import prepare_yolo_dataset
        prepare_yolo_dataset()
        print("[SUCCESS] YOLO data preparation completed!")
        return True
    except Exception as e:
        print(f"[ERROR] YOLO data preparation failed: {str(e)}")
        return False


def train_model_1():
    """Train 3D CNN model"""
    print("\n" + "="*60)
    print("TRAINING MODEL I: 3D CNN")
    print("="*60)
    
    model_path = os.path.join(MODEL_DIR, "3dcnn.h5")
    if os.path.exists(model_path):
        print(f"[INFO] Model already exists at {model_path}")
        response = input("Retrain model? (y/n): ").lower()
        if response != 'y':
            return True
    
    print("[INFO] Starting 3D CNN training...")
    start_time = time.time()
    
    try:
        from src.train_3dcnn import main as train_3dcnn
        train_3dcnn()
        elapsed = time.time() - start_time
        print(f"[SUCCESS] 3D CNN training completed in {elapsed/60:.1f} minutes!")
        return True
    except Exception as e:
        print(f"[ERROR] 3D CNN training failed: {str(e)}")
        return False


def train_model_2():
    """Train VGG16+LSTM model"""
    print("\n" + "="*60)
    print("TRAINING MODEL II: VGG16+LSTM")
    print("="*60)
    
    model_path = os.path.join(MODEL_DIR, "vgg16_lstm.h5")
    if os.path.exists(model_path):
        print(f"[INFO] Model already exists at {model_path}")
        response = input("Retrain model? (y/n): ").lower()
        if response != 'y':
            return True
    
    print("[INFO] Starting VGG16+LSTM training...")
    start_time = time.time()
    
    try:
        from src.train_classifier import main as train_classifier
        train_classifier()
        elapsed = time.time() - start_time
        print(f"[SUCCESS] VGG16+LSTM training completed in {elapsed/60:.1f} minutes!")
        return True
    except Exception as e:
        print(f"[ERROR] VGG16+LSTM training failed: {str(e)}")
        return False


def train_model_3():
    """Train YOLOv5 model"""
    print("\n" + "="*60)
    print("TRAINING MODEL III: YOLOv5")
    print("="*60)
    
    model_dir = os.path.join(MODEL_DIR, "yolov5")
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"[INFO] YOLOv5 model already exists")
        response = input("Retrain model? (y/n): ").lower()
        if response != 'y':
            return True
    
    print("[INFO] Starting YOLOv5 training...")
    start_time = time.time()
    
    try:
        from src.train_yolov5 import train_yolov5
        train_yolov5()
        elapsed = time.time() - start_time
        print(f"[SUCCESS] YOLOv5 training completed in {elapsed/60:.1f} minutes!")
        return True
    except Exception as e:
        print(f"[ERROR] YOLOv5 training failed: {str(e)}")
        return False


def evaluate_models():
    """Evaluate all trained models"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = {}
    
    # Evaluate 3D CNN
    print("\n[INFO] Evaluating 3D CNN...")
    try:
        from src.evaluate_3dcnn import main as eval_3dcnn
        eval_3dcnn()
        
        # Read results
        report_path = os.path.join(REPORTS_DIR, "evaluation_report_3dcnn.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                results['3dcnn'] = json.load(f)
                print(f"  Accuracy: {results['3dcnn']['overall_accuracy']:.2%}")
    except Exception as e:
        print(f"  [WARNING] 3D CNN evaluation failed: {str(e)}")
    
    # Evaluate VGG16+LSTM
    print("\n[INFO] Evaluating VGG16+LSTM...")
    try:
        from src.evaluate_classifier import main as eval_classifier
        eval_classifier()
        
        # Read results
        report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                results['vgg_lstm'] = json.load(f)
                print(f"  Accuracy: {results['vgg_lstm']['overall_accuracy']:.2%}")
    except Exception as e:
        print(f"  [WARNING] VGG16+LSTM evaluation failed: {str(e)}")
    
    # Evaluate YOLOv5
    print("\n[INFO] Evaluating YOLOv5...")
    try:
        from src.evaluate_yolov5 import main as eval_yolo
        eval_yolo()
        
        # Read results
        report_path = os.path.join(REPORTS_DIR, "yolov5_evaluation_results.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                results['yolov5'] = json.load(f)
                print(f"  mAP: {results['yolov5'].get('accuracy', 0):.2%}")
    except Exception as e:
        print(f"  [WARNING] YOLOv5 evaluation failed: {str(e)}")
    
    return results


def print_summary(results):
    """Print final summary"""
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    print("\n== Model Performance Comparison ==")
    print("-" * 40)
    print(f"{'Model':<20} {'Accuracy/mAP':<15} {'Status'}")
    print("-" * 40)
    
    models = [
        ('3D CNN (Model I)', results.get('3dcnn', {}).get('overall_accuracy', 0)),
        ('VGG16+LSTM (Model II)', results.get('vgg_lstm', {}).get('overall_accuracy', 0)),
        ('YOLOv5 (Model III)', results.get('yolov5', {}).get('accuracy', 0))
    ]
    
    for model_name, accuracy in models:
        status = "OK" if accuracy > 0 else "FAIL"
        print(f"{model_name:<20} {accuracy:>6.1%}         {status}")
    
    print("-" * 40)
    
    # Best model
    if models:
        best_model = max(models, key=lambda x: x[1])
        print(f"\n** Best Performing Model: {best_model[0]} ({best_model[1]:.1%}) **")
    
    print("\n== Output Locations ==")
    print(f"  - Models: {MODEL_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    print(f"  - Frames: {FRAMES_DIR}")
    
    print("\n** Pipeline execution completed successfully! **")
    print("Use the CLI utilities for inference, e.g.:")
    print("  - python -m src.infer_classifier <model_path> <video_path>")
    print("  - python -c \"from ultralytics import YOLO; YOLO('<yolov5_best>').predict(source='<media>', save=True)\"")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="ISL Gesture Recognition Pipeline")
    parser.add_argument('--skip-data-prep', action='store_true', 
                       help='Skip data preparation if already done')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only evaluate existing models')
    parser.add_argument('--models', nargs='+', choices=['3dcnn', 'vgg_lstm', 'yolo', 'all'],
                       default=['all'], help='Which models to train')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ISL EMERGENCY GESTURE RECOGNITION PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dataset
    if not check_dataset():
        print("\n[ERROR] Dataset check failed. Exiting...")
        return
    
    # Data preparation
    if not args.skip_data_prep:
        if not prepare_data():
            print("\n[ERROR] Data preparation failed. Exiting...")
            return
        
        if 'yolo' in args.models or 'all' in args.models:
            if not prepare_yolo_data():
                print("\n[WARNING] YOLO data preparation failed. Continuing...")
    
    # Model training
    if not args.skip_training:
        models_to_train = args.models if 'all' not in args.models else ['3dcnn', 'vgg_lstm', 'yolo']
        
        if '3dcnn' in models_to_train:
            train_model_1()
        
        if 'vgg_lstm' in models_to_train:
            train_model_2()
        
        if 'yolo' in models_to_train:
            train_model_3()
    
    # Evaluation
    results = evaluate_models()
    
    # Summary
    print_summary(results)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
