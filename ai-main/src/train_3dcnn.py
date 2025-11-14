"""Train 3D CNN classifier (Model I)
- Loads 60/20/20 splits from `data/processed/splits/*.txt`
- Builds tf.data.Dataset from frame folders into 5x150x150x3 sequences
- Trains with Adam(lr=1e-3), batch=32
- Saves best model to `data/processed/artifacts/models/3dcnn.h5`
"""
import os
import json
from typing import Dict, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from src.config import (
    FRAMES_DIR,
    SPLIT_DIR,
    MODEL_DIR,
    LOGS_DIR,
    CLASSES,
    CLASS_TO_ID,
    FRAMES_PER_VIDEO,
    IMG_H,
    IMG_W,
    BATCH_SIZE,
    EPOCHS,
    LR,
)
from src.models_3dcnn import build_3d_cnn


def _read_split(name: str) -> List[Dict]:
    path = os.path.join(SPLIT_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().strip().splitlines()]
    return lines


def _video_to_frame_paths(cls: str, vid: str) -> List[str]:
    cdir = os.path.join(FRAMES_DIR, cls)
    fps = [os.path.join(cdir, f"{vid}_f{i}.jpg") for i in range(FRAMES_PER_VIDEO)]
    return fps


def make_tf_dataset(split_name: str, shuffle: bool = True) -> tf.data.Dataset:
    items = _read_split(split_name)
    X, y = [], []
    for rec in items:
        cls = rec["class"]
        vid = rec["video_id"]
        fps = _video_to_frame_paths(cls, vid)
        if not all(os.path.exists(p) for p in fps):
            continue
        X.append(fps)
        y.append(CLASS_TO_ID[cls])

    X = np.array(X)
    y = np.array(y)

    def _load_sequence(paths):
        imgs = []
        for p in paths:
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (IMG_H, IMG_W))
            imgs.append(img)
        seq = tf.stack(imgs, axis=0)  # (T,H,W,C)
        return seq

    def _gen():
        for i in range(len(X)):
            yield X[i], y[i]

    def _map(paths, label):
        seq = tf.py_function(func=_load_sequence, inp=[paths], Tout=tf.float32)
        seq.set_shape((FRAMES_PER_VIDEO, IMG_H, IMG_W, 3))
        label = tf.one_hot(label, depth=len(CLASSES))
        return seq, label

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(FRAMES_PER_VIDEO,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    train_ds = make_tf_dataset("train", shuffle=True)
    val_ds = make_tf_dataset("val", shuffle=False)

    model = build_3d_cnn()
    opt = optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    ckpt_path = os.path.join(MODEL_DIR, "3dcnn.h5")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        callbacks.CSVLogger(os.path.join(LOGS_DIR, "training_log_3dcnn.csv")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

    # Save final as well
    model.save(ckpt_path)
    print(f"[INFO] Saved best model to: {ckpt_path}")


if __name__ == "__main__":
    main()
