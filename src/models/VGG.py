import tensorflow as tf
import tensorflow.keras.layers as tfl
import numpy as np
from tensorflow.keras.applications import VGG16


def build_clothes_model(
    input_shape=(128, 128, 3),
    num_classes=4,
    weights="imagenet",
    train_base=False,
):
    """Build a VGG16 transfer-learning classifier.

    The existing data pipeline returns RGB images scaled to 0..1. VGG16's
    ImageNet weights expect VGG preprocessing on 0..255 images, so the model
    does that conversion internally to keep training and prediction consistent.
    """
    base_model = VGG16(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape)
    x = _vgg16_preprocess(inputs)
    x = base_model(x)
    x = tfl.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tfl.Dense(256, activation="relu")(x)
    x = tfl.Dropout(0.5)(x)
    outputs = tfl.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="clothes_vgg16")


def _vgg16_preprocess(inputs):
    preprocess = tfl.Conv2D(
        3,
        kernel_size=1,
        padding="same",
        use_bias=True,
        trainable=False,
        name="vgg16_preprocess",
    )
    outputs = preprocess(inputs)

    kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kernel[0, 0, 2, 0] = 255.0  # RGB input -> BGR output
    kernel[0, 0, 1, 1] = 255.0
    kernel[0, 0, 0, 2] = 255.0
    bias = np.array([-103.939, -116.779, -123.68], dtype=np.float32)
    preprocess.set_weights([kernel, bias])

    return outputs


def unlock_last_block(model):
    """Freeze VGG16 except block5 for fine-tuning after warmup training."""
    for layer in model.layers:
        layer.trainable = False

    vgg16 = model.get_layer("vgg16")
    vgg16.trainable = True

    for layer in vgg16.layers:
        layer.trainable = layer.name.startswith("block5")

    for layer in model.layers:
        if not layer.name.startswith("vgg16"):
            layer.trainable = True
