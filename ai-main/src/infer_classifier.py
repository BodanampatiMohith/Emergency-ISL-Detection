"""Inference utility for a single video
- Extract 5 frames, resize, predict top-1 class and probability
"""
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    FRAMES_PER_VIDEO,
    IMG_H,
    IMG_W,
    CLASSES,
)


def _evenly_spaced_indices(num_frames: int, k: int):
    if num_frames <= 0:
        return []
    if k <= 0:
        return []
    return sorted({int(round(i)) for i in [j * (num_frames - 1) / max(k - 1, 1) for j in range(k)]})


def extract_5_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = _evenly_spaced_indices(total, FRAMES_PER_VIDEO)
    imgs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        imgs.append(frame.astype(np.float32) / 255.0)
    cap.release()
    if len(imgs) != FRAMES_PER_VIDEO:
        raise RuntimeError("Could not extract 5 frames from video")
    return np.stack(imgs, axis=0)


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.infer_classifier <model_path> <video_path>")
        return
    model_path = sys.argv[1]
    video_path = sys.argv[2]

    model = tf.keras.models.load_model(model_path)
    seq = extract_5_frames(video_path)[None, ...]
    probs = model.predict(seq, verbose=0)[0]
    top_id = int(np.argmax(probs))
    print(f"Prediction: {CLASSES[top_id]} | prob={probs[top_id]:.4f}")


if __name__ == "__main__":
    main()
