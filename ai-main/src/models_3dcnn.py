"""3D CNN model builder (Model I)
- Two Conv3D blocks (32, 64)
- MaxPooling3D, BatchNorm, Dropout
- Dense + Softmax(8)
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMG_H, IMG_W, IMG_C, NUM_CLASSES, FRAMES_PER_VIDEO


def build_3d_cnn() -> tf.keras.Model:
    input_shape = (FRAMES_PER_VIDEO, IMG_H, IMG_W, IMG_C)

    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # preserve time dimension
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    return model


if __name__ == "__main__":
    m = build_3d_cnn()
    m.summary()
