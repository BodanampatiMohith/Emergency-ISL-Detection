"""Evaluate trained classifier
- Loads best model
- Evaluates on test split
- Outputs confusion matrix, per-class accuracy, overall accuracy
- Saves confusion matrix plot and a report json
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    FRAMES_DIR,
    SPLIT_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    CLASSES,
    CLASS_TO_ID,
    FRAMES_PER_VIDEO,
    IMG_H,
    IMG_W,
)
from src.train_classifier import _read_split, _video_to_frame_paths


def _load_sequence_np(paths):
    imgs = []
    for p in paths:
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (IMG_H, IMG_W))
        imgs.append(img.numpy())
    seq = np.stack(imgs, axis=0)
    return seq


def main():
    model_path = os.path.join(MODEL_DIR, "vgg16_lstm.h5")
    model = tf.keras.models.load_model(model_path)

    items = _read_split("test")

    y_true, y_pred = [], []
    for rec in items:
        cls = rec["class"]
        vid = rec["video_id"]
        fps = _video_to_frame_paths(cls, vid)
        if not all(os.path.exists(p) for p in fps):
            continue
        seq = _load_sequence_np(fps)[None, ...]  # (1,T,H,W,C)
        probs = model.predict(seq, verbose=0)[0]
        pred_id = int(np.argmax(probs))
        y_pred.append(pred_id)
        y_true.append(CLASS_TO_ID[cls])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    acc = (y_true == y_pred).mean().item() if len(y_true) else 0.0

    # Per-class report
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix - VGG16+LSTM")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Save JSON report
    out = {
        "overall_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    rep_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Overall accuracy: {acc:.4f}")
    print(f"[INFO] Saved confusion matrix -> {cm_path}")
    print(f"[INFO] Saved evaluation report -> {rep_path}")


if __name__ == "__main__":
    main()
