"""
Evaluate YOLOv5 Model for ISL Emergency Gesture Detection
=========================================================
Comprehensive evaluation of the YOLOv5 object detection model
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from yolov5.val import run as yolov5_val_run  # type: ignore
except ImportError:
    yolov5_val_run = None  # Fallback handled at runtime

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    CLASSES,
    CLASS_TO_ID,
    YOLO_IMG_SIZE,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    PROJECT_ROOT,
)

# Ensure local YOLOv5 repository is importable (needed for legacy modules in checkpoints)
YOLOV5_DIR = os.path.join(PROJECT_ROOT, "yolov5")
if os.path.isdir(YOLOV5_DIR) and YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

LAST_MODEL_PATH: str | None = None


def load_yolov5_model():
    """Load the trained YOLOv5 model"""
    # Try to load the best model
    model_paths = [
        os.path.join(MODEL_DIR, 'yolov5', 'best.pt'),
        os.path.join(MODEL_DIR, 'yolov5_isl_gesture.pt'),
        os.path.join(MODEL_DIR, 'yolov5', 'weights', 'best.pt')
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"[INFO] Loading model from: {path}")
            try:
                global LAST_MODEL_PATH
                LAST_MODEL_PATH = path
                return YOLO(path)
            except KeyError as exc:
                if "train_args" in str(exc):
                    print("[WARNING] YOLO checkpoint missing train_args; will use yolov5.val fallback")
                    return None
                raise
    
    print("[ERROR] No trained YOLOv5 model found. Please train the model first.")
    return None


def evaluate_on_test_set(model):
    """
    Evaluate YOLOv5 on the test dataset
    """
    print("[INFO] Evaluating YOLOv5 on test set...")
    
    # Test data paths
    yolo_data_dir = os.path.join(DATA_DIR, "yolo")
    test_images_dir = os.path.join(yolo_data_dir, "images", "test")
    test_labels_dir = os.path.join(yolo_data_dir, "labels", "test")
    
    if not os.path.exists(test_images_dir):
        print(f"[ERROR] Test images not found at: {test_images_dir}")
        return None
    
    # Get all test images
    test_images = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpg')])
    
    if not test_images:
        print("[ERROR] No test images found.")
        return None
    
    # Evaluation metrics
    y_true = []
    y_pred = []
    confidence_scores = []
    detection_results = []
    
    print(f"[INFO] Evaluating {len(test_images)} test images...")
    
    for img_file in tqdm(test_images, desc="Evaluating"):
        # Load image
        img_path = os.path.join(test_images_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Load ground truth label
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(test_labels_dir, label_file)
        
        true_class = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    # First number is class ID
                    true_class = int(lines[0].split()[0])
        
        # Run inference
        results = model.predict(
            img_path,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False
        )
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the detection with highest confidence
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                if len(confidences) > 0:
                    best_idx = np.argmax(confidences)
                    pred_class = int(classes[best_idx])
                    conf = float(confidences[best_idx])
                    
                    y_pred.append(pred_class)
                    confidence_scores.append(conf)
                    
                    # Store detailed result
                    detection_results.append({
                        'image': img_file,
                        'true_class': true_class,
                        'pred_class': pred_class,
                        'confidence': conf,
                        'correct': true_class == pred_class
                    })
                else:
                    # No detection
                    y_pred.append(-1)
                    confidence_scores.append(0.0)
                    detection_results.append({
                        'image': img_file,
                        'true_class': true_class,
                        'pred_class': -1,
                        'confidence': 0.0,
                        'correct': False
                    })
            else:
                # No detection
                y_pred.append(-1)
                confidence_scores.append(0.0)
                detection_results.append({
                    'image': img_file,
                    'true_class': true_class,
                    'pred_class': -1,
                    'confidence': 0.0,
                    'correct': False
                })
        
        if true_class is not None:
            y_true.append(true_class)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred[:len(y_true)])  # Ensure same length
    
    # Remove entries where no detection was made (-1)
    valid_mask = y_pred != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        print("[ERROR] No valid detections found.")
        return None
    
    # Calculate accuracy
    accuracy = np.mean(y_true_valid == y_pred_valid)
    
    # Calculate precision for detection (object found)
    detection_rate = np.mean(valid_mask)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(len(CLASSES))))
    
    # Classification report
    report = classification_report(
        y_true_valid, 
        y_pred_valid, 
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
    
    # Calculate mAP-like metric
    mean_confidence = np.mean([c for c in confidence_scores if c > 0])
    
    evaluation_results = {
        'total_images': len(test_images),
        'valid_detections': int(np.sum(valid_mask)),
        'detection_rate': float(detection_rate),
        'accuracy': float(accuracy),
        'mean_confidence': float(mean_confidence),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'per_class_accuracy': {
            CLASSES[i]: float(cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0)
            for i in range(len(CLASSES))
        }
    }
    
    return evaluation_results, detection_results


def evaluate_with_yolov5_val(weights_path: str):
    """Fallback evaluation using yolov5/val.py routine"""
    if yolov5_val_run is None:
        raise RuntimeError("yolov5.val.run not available; cannot perform fallback evaluation")

    dataset_yaml = os.path.join(DATA_DIR, "yolo", "dataset.yaml")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_images_dir = os.path.join(DATA_DIR, "yolo", "images", "test")
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]) if os.path.isdir(test_images_dir) else []
    total_images = len(image_files)

    results, maps, _ = yolov5_val_run(
        data=dataset_yaml,
        weights=weights_path,
        batch_size=8,
        imgsz=YOLO_IMG_SIZE,
        conf_thres=YOLO_CONF_THRESHOLD,
        iou_thres=YOLO_IOU_THRESHOLD,
        device=device,
        workers=0,
        save_txt=False,
        save_conf=False,
        verbose=False,
        project=os.path.join(MODEL_DIR, 'yolov5', 'val_runs'),
        name='latest',
        exist_ok=True,
        half=False,
        dnn=False,
        plots=False,
    )

    mp, mr, map50, map, *losses = results

    evaluation_results = {
        'precision': float(mp),
        'recall': float(mr),
        'map50': float(map50),
        'map50_95': float(map),
        'losses': [float(l) for l in losses],
        'per_class_map': {
            CLASSES[i]: float(maps[i]) for i in range(len(CLASSES))
        },
        'accuracy': float(map50),
        'detection_rate': float(mr),
        'total_images': total_images,
        'valid_detections': int(round(total_images * mr)) if total_images else 0,
        'mean_confidence': float(map50),  # proxy metric when detailed detections unavailable
        'per_class_accuracy': {
            CLASSES[i]: float(maps[i]) for i in range(len(CLASSES))
        },
        'evaluation_source': 'yolov5.val.run'
    }

    return evaluation_results, []


def plot_confusion_matrix(cm, title="YOLOv5 Confusion Matrix"):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    cm_path = os.path.join(REPORTS_DIR, 'yolov5_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_path


def plot_performance_metrics(evaluation_results):
    """Create performance visualization plots"""
    # Per-class accuracy bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Per-class accuracy
    ax = axes[0, 0]
    classes = list(evaluation_results['per_class_accuracy'].keys())
    accuracies = list(evaluation_results['per_class_accuracy'].values())
    bars = ax.bar(classes, accuracies, color='skyblue', edgecolor='navy')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=evaluation_results['accuracy'], color='red', linestyle='--', 
               label=f"Overall: {evaluation_results['accuracy']:.3f}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}', ha='center', va='bottom')
    
    # 2. Detection metrics
    ax = axes[0, 1]
    metrics = ['Detection Rate', 'Accuracy', 'Mean Confidence']
    values = [
        evaluation_results['detection_rate'],
        evaluation_results['accuracy'],
        evaluation_results['mean_confidence']
    ]
    bars = ax.bar(metrics, values, color=['green', 'blue', 'orange'])
    ax.set_title('Overall Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Precision-Recall per class
    ax = axes[1, 0]
    report = evaluation_results['classification_report']
    precisions = []
    recalls = []
    f1_scores = []
    
    for cls in CLASSES:
        if cls in report:
            precisions.append(report[cls]['precision'])
            recalls.append(report[cls]['recall'])
            f1_scores.append(report[cls]['f1-score'])
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    x = np.arange(len(CLASSES))
    width = 0.25
    
    ax.bar(x - width, precisions, width, label='Precision', color='lightcoral')
    ax.bar(x, recalls, width, label='Recall', color='lightskyblue')
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1-Score by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    YOLOv5 Model Performance Summary
    ================================
    
    Total Test Images: {evaluation_results['total_images']}
    Valid Detections: {evaluation_results['valid_detections']}
    
    Detection Rate: {evaluation_results['detection_rate']:.1%}
    Overall Accuracy: {evaluation_results['accuracy']:.1%}
    Mean Confidence: {evaluation_results['mean_confidence']:.3f}
    
    Mean Average Precision (mAP): ~{evaluation_results['accuracy'] * 100:.1f}%
    
    Best Performing Class: {max(evaluation_results['per_class_accuracy'], 
                                key=evaluation_results['per_class_accuracy'].get)}
    Worst Performing Class: {min(evaluation_results['per_class_accuracy'], 
                                 key=evaluation_results['per_class_accuracy'].get)}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    perf_path = os.path.join(REPORTS_DIR, 'yolov5_performance_metrics.png')
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return perf_path


def main():
    """Main evaluation function"""
    print("="*60)
    print("YOLOv5 Model Evaluation for ISL Emergency Gestures")
    print("="*60)
    
    # Load model
    model = load_yolov5_model()
    eval_results = None
    detection_details = []

    if model is not None:
        try:
            eval_results, detection_details = evaluate_on_test_set(model)
        except KeyError as exc:
            if "train_args" in str(exc):
                print("[WARNING] Ultralytics YOLO evaluation failed due to missing train_args. Falling back to yolov5.val...")
                fallback_weights = LAST_MODEL_PATH or os.path.join(MODEL_DIR, 'yolov5', 'best.pt')
                eval_results, detection_details = evaluate_with_yolov5_val(fallback_weights)
            else:
                raise
    else:
        # Model couldn't be loaded via Ultralytics; use fallback evaluation directly
        fallback_weights = LAST_MODEL_PATH or os.path.join(MODEL_DIR, 'yolov5', 'best.pt')
        eval_results, detection_details = evaluate_with_yolov5_val(fallback_weights)

    if eval_results is None:
        return

    # Plot confusion matrix
    if 'confusion_matrix' in eval_results:
        cm = np.array(eval_results['confusion_matrix'])
        cm_path = plot_confusion_matrix(cm)
        print(f"[INFO] Confusion matrix saved to: {cm_path}")
        
        # Plot performance metrics
        perf_path = plot_performance_metrics(eval_results)
        print(f"[INFO] Performance metrics saved to: {perf_path}")

    # Save detailed results
    results_path = os.path.join(REPORTS_DIR, 'yolov5_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"[INFO] Evaluation results saved to: {results_path}")
    
    # Save detection details
    details_path = os.path.join(REPORTS_DIR, 'yolov5_detection_details.json')
    with open(details_path, 'w') as f:
        json.dump(detection_details, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Test Images: {eval_results['total_images']}")
    print(f"Valid Detections: {eval_results['valid_detections']}")
    print(f"Detection Rate: {eval_results['detection_rate']:.1%}")
    print(f"Overall Accuracy: {eval_results['accuracy']:.1%}")
    print(f"Mean Confidence: {eval_results['mean_confidence']:.3f}")
    print(f"Mean Average Precision (mAP): ~{eval_results['accuracy'] * 100:.1f}%")
    print("\nPer-Class Accuracy:")
    for cls, acc in eval_results['per_class_accuracy'].items():
        print(f"  {cls:10s}: {acc:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
