"""VGG16 + LSTM model builder
- TimeDistributed VGG16 conv-base over 5 frames (150x150x3)
- LSTM(256) + Dense(1024) + Softmax(8)
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMG_H, IMG_W, IMG_C, NUM_CLASSES, FRAMES_PER_VIDEO


def build_vgg16_lstm(trainable_backbone: bool = False) -> tf.keras.Model:
    time_steps = FRAMES_PER_VIDEO
    input_shape = (time_steps, IMG_H, IMG_W, IMG_C)

    # VGG16 conv base
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_H, IMG_W, IMG_C),
    )
    base.trainable = trainable_backbone

    inp = layers.Input(shape=input_shape)
    x = layers.TimeDistributed(base)(inp)
    x = layers.TimeDistributed(layers.Flatten())(x)
    # Alternatively use GlobalAveragePooling2D for lighter features
    # x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    return model


if __name__ == "__main__":
    m = build_vgg16_lstm()
    m.summary()
