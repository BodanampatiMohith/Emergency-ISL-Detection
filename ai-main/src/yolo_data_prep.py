"""
YOLO Data Preparation for ISL Emergency Gesture Detection
==========================================================
Prepares data for YOLOv5 object detection model (Model III)

This script:
1. Extracts frames from videos
2. Creates YOLO format annotations (bounding boxes for hands)
3. Organizes data for YOLOv5 training
"""

import os
import cv2
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm
import random

from src.config import (
    RAW_VIDEOS_DIR,
    FRAMES_DIR,
    ANNOTATIONS_DIR,
    DATA_DIR,
    CLASSES,
    CLASS_TO_ID,
    FRAMES_PER_VIDEO,
    IMG_H, IMG_W,
    YOLO_IMG_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)


def create_yolo_directories():
    """Create necessary directories for YOLO training"""
    yolo_dir = os.path.join(DATA_DIR, "yolo")
    
    dirs = [
        os.path.join(yolo_dir, "images", "train"),
        os.path.join(yolo_dir, "images", "val"),
        os.path.join(yolo_dir, "images", "test"),
        os.path.join(yolo_dir, "labels", "train"),
        os.path.join(yolo_dir, "labels", "val"),
        os.path.join(yolo_dir, "labels", "test"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return yolo_dir


def detect_hand_region(frame: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Detect hand region using skin detection and contours
    Returns normalized YOLO format: [x_center, y_center, width, height]
    """
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create skin mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Default to center region if no hand detected
        return 0.5, 0.5, 0.4, 0.4
    
    # Find largest contour (assumed to be hand)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Convert to normalized YOLO format
    img_h, img_w = frame.shape[:2]
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    width = w / img_w
    height = h / img_h
    
    # Ensure values are in [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def create_yolo_annotations(frames_dir: str, output_dir: str):
    """
    Create YOLO format annotations for all frames
    Format: <class_id> <x_center> <y_center> <width> <height>
    """
    print("[INFO] Creating YOLO annotations...")
    
    for cls in CLASSES:
        cls_dir = os.path.join(frames_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        class_id = CLASS_TO_ID[cls]
        frame_files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
        
        for frame_file in tqdm(frame_files, desc=f"Annotating {cls}"):
            frame_path = os.path.join(cls_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
            
            # Detect hand region
            x_center, y_center, width, height = detect_hand_region(frame)
            
            # Create annotation file
            ann_filename = frame_file.replace('.jpg', '.txt')
            ann_path = os.path.join(output_dir, cls, ann_filename)
            os.makedirs(os.path.dirname(ann_path), exist_ok=True)
            
            # Write YOLO format annotation
            with open(ann_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def prepare_yolo_dataset():
    """
    Prepare complete YOLOv5 dataset structure
    """
    print("[INFO] Preparing YOLOv5 dataset...")
    
    # Create directories
    yolo_dir = create_yolo_directories()
    
    # Create annotations
    create_yolo_annotations(FRAMES_DIR, ANNOTATIONS_DIR)
    
    # Collect all frame-annotation pairs
    data_pairs = []
    for cls in CLASSES:
        cls_frames_dir = os.path.join(FRAMES_DIR, cls)
        cls_ann_dir = os.path.join(ANNOTATIONS_DIR, cls)
        
        if not os.path.exists(cls_frames_dir):
            continue
        
        frame_files = [f for f in os.listdir(cls_frames_dir) if f.endswith('.jpg')]
        for frame_file in frame_files:
            frame_path = os.path.join(cls_frames_dir, frame_file)
            ann_file = frame_file.replace('.jpg', '.txt')
            ann_path = os.path.join(cls_ann_dir, ann_file)
            
            if os.path.exists(ann_path):
                data_pairs.append((frame_path, ann_path, cls))
    
    # Shuffle and split data
    random.shuffle(data_pairs)
    total = len(data_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_pairs = data_pairs[:train_end]
    val_pairs = data_pairs[train_end:val_end]
    test_pairs = data_pairs[val_end:]
    
    print(f"[INFO] Dataset split - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Copy files to YOLO structure
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        for idx, (img_path, ann_path, cls) in enumerate(tqdm(pairs, desc=f"Copying {split_name}")):
            # Create unique filename
            new_filename = f"{cls}_{idx:04d}"
            
            # Copy image
            img_dst = os.path.join(yolo_dir, "images", split_name, f"{new_filename}.jpg")
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            cv2.imwrite(img_dst, img_resized)
            
            # Copy annotation
            ann_dst = os.path.join(yolo_dir, "labels", split_name, f"{new_filename}.txt")
            shutil.copy2(ann_path, ann_dst)
    
    # Create YOLO configuration file
    create_yolo_config(yolo_dir)
    
    print(f"[INFO] YOLOv5 dataset prepared at: {yolo_dir}")
    return yolo_dir


def create_yolo_config(yolo_dir: str):
    """
    Create YOLOv5 dataset configuration YAML file
    """
    config = {
        'path': os.path.abspath(yolo_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: cls for i, cls in enumerate(CLASSES)},
        'nc': len(CLASSES)
    }
    
    config_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[INFO] Created YOLOv5 config: {config_path}")
    return config_path


if __name__ == "__main__":
    # First extract frames from videos if not done
    from src.data_prep import prepare_frames, make_splits
    
    if not os.path.exists(FRAMES_DIR) or not os.listdir(FRAMES_DIR):
        print("[INFO] Extracting frames from videos...")
        prepare_frames()
        make_splits()
    
    # Prepare YOLO dataset
    prepare_yolo_dataset()
