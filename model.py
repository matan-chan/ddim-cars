from keras.layers import Conv2D, BatchNormalization, Add, AveragePooling2D, UpSampling2D, Concatenate, Lambda
from config import img_size, embedding_max_frequency, embedding_dims
from tensorflow import keras
import tensorflow as tf
import math


def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = Conv2D(width, kernel_size=1)(x)
        x = BatchNormalization(center=False, scale=False)(x)
        x = Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = Conv2D(width, kernel_size=3, padding="same")(x)
        x = Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(widths, block_depth):
    noisy_images = keras.Input(shape=(img_size, img_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = Lambda(sinusoidal_embedding)(noise_variances)
    e = UpSampling2D(size=(img_size, img_size), interpolation="nearest")(e)

    x = Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")
