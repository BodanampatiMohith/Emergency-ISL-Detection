"""
Inference Module for ISL Emergency Gesture Recognition
=======================================================
Unified inference interface for all three models
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Union, Tuple
import torch
from ultralytics import YOLO

from src.config import (
    MODEL_DIR,
    CLASSES,
    IMG_H, IMG_W,
    FRAMES_PER_VIDEO,
    YOLO_IMG_SIZE,
    YOLO_CONF_THRESHOLD
)


class ISLGesturePredictor:
    """
    Unified predictor for ISL emergency gestures
    Supports all three models: 3D CNN, VGG16+LSTM, YOLOv5
    """
    
    def __init__(self, model_type: str = 'vgg_lstm'):
        """
        Initialize predictor with specified model type
        
        Args:
            model_type: '3dcnn', 'vgg_lstm', or 'yolo'
        """
        self.model_type = model_type.lower()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the specified model"""
        if self.model_type == '3dcnn':
            model_path = os.path.join(MODEL_DIR, '3dcnn.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"[INFO] Loaded 3D CNN model from {model_path}")
            else:
                raise FileNotFoundError(f"3D CNN model not found at {model_path}")
        
        elif self.model_type == 'vgg_lstm':
            model_path = os.path.join(MODEL_DIR, 'vgg16_lstm.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"[INFO] Loaded VGG16+LSTM model from {model_path}")
            else:
                raise FileNotFoundError(f"VGG16+LSTM model not found at {model_path}")
        
        elif self.model_type == 'yolo':
            model_paths = [
                os.path.join(MODEL_DIR, 'yolov5', 'best.pt'),
                os.path.join(MODEL_DIR, 'yolov5_isl_gesture.pt')
            ]
            for path in model_paths:
                if os.path.exists(path):
                    self.model = YOLO(path)
                    print(f"[INFO] Loaded YOLOv5 model from {path}")
                    break
            if self.model is None:
                raise FileNotFoundError("YOLOv5 model not found")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices
        indices = self._evenly_spaced_indices(total_frames, FRAMES_PER_VIDEO)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMG_W, IMG_H))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) != FRAMES_PER_VIDEO:
            raise ValueError(f"Could not extract {FRAMES_PER_VIDEO} frames")
        
        return np.array(frames, dtype=np.float32) / 255.0
    
    def _evenly_spaced_indices(self, n: int, k: int) -> List[int]:
        """Get evenly spaced indices"""
        if n <= 0 or k <= 0:
            return []
        return sorted({int(round(i)) for i in 
                      [j * (n - 1) / max(k - 1, 1) for j in range(k)]})
    
    def predict_video(self, video_path: str) -> Dict:
        """
        Predict gesture from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with prediction results
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if self.model_type in ['3dcnn', 'vgg_lstm']:
            # Add batch dimension
            frames_batch = frames[np.newaxis, ...]
            
            # Get predictions
            predictions = self.model.predict(frames_batch, verbose=0)[0]
            
            # Get top prediction
            class_id = np.argmax(predictions)
            confidence = float(predictions[class_id])
            class_name = CLASSES[class_id]
            
            # Get top 3 predictions
            top3_indices = np.argsort(predictions)[-3:][::-1]
            top3_predictions = [
                {
                    'class': CLASSES[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in top3_indices
            ]
            
            return {
                'model': self.model_type,
                'prediction': class_name,
                'confidence': confidence,
                'all_predictions': {
                    CLASSES[i]: float(predictions[i]) 
                    for i in range(len(CLASSES))
                },
                'top3': top3_predictions
            }
        
        elif self.model_type == 'yolo':
            # For YOLO, process middle frame
            middle_frame = frames[FRAMES_PER_VIDEO // 2]
            middle_frame = (middle_frame * 255).astype(np.uint8)
            middle_frame = cv2.resize(middle_frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            
            # Run detection
            results = self.model.predict(
                middle_frame,
                conf=YOLO_CONF_THRESHOLD,
                verbose=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get best detection
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    best_idx = np.argmax(confidences)
                    class_id = int(classes[best_idx])
                    confidence = float(confidences[best_idx])
                    class_name = CLASSES[class_id]
                    
                    return {
                        'model': 'yolo',
                        'prediction': class_name,
                        'confidence': confidence,
                        'detections': len(result.boxes),
                        'all_predictions': {
                            CLASSES[int(cls)]: float(conf)
                            for cls, conf in zip(classes, confidences)
                        }
                    }
            
            return {
                'model': 'yolo',
                'prediction': 'No detection',
                'confidence': 0.0,
                'detections': 0
            }
    
    def predict_frame(self, frame: np.ndarray) -> Dict:
        """
        Predict gesture from single frame (only for YOLO)
        
        Args:
            frame: Single image frame
            
        Returns:
            Dictionary with prediction results
        """
        if self.model_type != 'yolo':
            raise ValueError("Single frame prediction only supported for YOLO model")
        
        # Preprocess frame
        if frame.max() > 1:
            frame = frame / 255.0
        
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
        
        # Run detection
        results = self.model.predict(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                detections = []
                for i in range(len(confidences)):
                    detections.append({
                        'class': CLASSES[int(classes[i])],
                        'confidence': float(confidences[i]),
                        'bbox': boxes[i].tolist()
                    })
                
                # Sort by confidence
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                return {
                    'model': 'yolo',
                    'detections': detections,
                    'num_detections': len(detections),
                    'best_prediction': detections[0]['class'] if detections else None
                }
        
        return {
            'model': 'yolo',
            'detections': [],
            'num_detections': 0,
            'best_prediction': None
        }
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict]:
        """
        Predict gestures from multiple videos
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for path in video_paths:
            try:
                result = self.predict_video(path)
                result['video_path'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'video_path': path,
                    'error': str(e)
                })
        
        return results


class EnsemblePredictor:
    """
    Ensemble predictor combining all three models
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize ensemble predictor
        
        Args:
            weights: Dictionary of model weights for ensemble
        """
        if weights is None:
            # Default weights based on individual model performance
            self.weights = {
                '3dcnn': 0.2,
                'vgg_lstm': 0.4,
                'yolo': 0.4
            }
        else:
            self.weights = weights
        
        # Load all models
        self.models = {}
        for model_type in self.weights.keys():
            try:
                self.models[model_type] = ISLGesturePredictor(model_type)
                print(f"[INFO] Loaded {model_type} for ensemble")
            except Exception as e:
                print(f"[WARNING] Could not load {model_type}: {str(e)}")
                self.weights[model_type] = 0
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def predict_video(self, video_path: str) -> Dict:
        """
        Ensemble prediction from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with ensemble prediction results
        """
        individual_predictions = {}
        ensemble_scores = np.zeros(len(CLASSES))
        
        for model_type, predictor in self.models.items():
            if self.weights[model_type] > 0:
                try:
                    result = predictor.predict_video(video_path)
                    individual_predictions[model_type] = result
                    
                    # Add weighted predictions
                    if 'all_predictions' in result:
                        for i, cls in enumerate(CLASSES):
                            if cls in result['all_predictions']:
                                ensemble_scores[i] += (
                                    result['all_predictions'][cls] * 
                                    self.weights[model_type]
                                )
                except Exception as e:
                    print(f"[WARNING] {model_type} prediction failed: {str(e)}")
        
        # Get ensemble prediction
        best_class_id = np.argmax(ensemble_scores)
        best_class = CLASSES[best_class_id]
        best_confidence = float(ensemble_scores[best_class_id])
        
        # Get top 3
        top3_indices = np.argsort(ensemble_scores)[-3:][::-1]
        top3_predictions = [
            {
                'class': CLASSES[idx],
                'confidence': float(ensemble_scores[idx])
            }
            for idx in top3_indices
        ]
        
        return {
            'model': 'ensemble',
            'prediction': best_class,
            'confidence': best_confidence,
            'individual_predictions': individual_predictions,
            'ensemble_scores': {
                CLASSES[i]: float(ensemble_scores[i])
                for i in range(len(CLASSES))
            },
            'top3': top3_predictions,
            'weights': self.weights
        }


def demo_inference():
    """Demo function to test inference"""
    print("\n" + "="*60)
    print("ISL GESTURE INFERENCE DEMO")
    print("="*60)
    
    # Test with sample video
    import glob
    test_videos = glob.glob(os.path.join(DATA_DIR, "*.mp4"))[:1]
    
    if not test_videos:
        print("[WARNING] No test videos found")
        return
    
    test_video = test_videos[0]
    print(f"\nTesting with video: {test_video}")
    
    # Test individual models
    for model_type in ['3dcnn', 'vgg_lstm', 'yolo']:
        try:
            print(f"\n--- {model_type.upper()} Model ---")
            predictor = ISLGesturePredictor(model_type)
            result = predictor.predict_video(test_video)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Test ensemble
    try:
        print("\n--- ENSEMBLE Model ---")
        ensemble = EnsemblePredictor()
        result = ensemble.predict_video(test_video)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Weights: {result['weights']}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    demo_inference()
