from config import batch_size, img_size, dataset_repetitions
import tensorflow as tf
import numpy as np


def preprocess_image(data):
    img = tf.image.decode_jpeg(tf.io.read_file(data))
    height, width, z = tf.shape(img)
    if z != 3:
        img = tf.image.grayscale_to_rgb(img)
    crop_size = tf.minimum(height, width)
    img = tf.image.crop_to_bounding_box(img, (height - crop_size) // 2, (width - crop_size) // 2, crop_size, crop_size)
    img = tf.image.resize(img, size=[img_size, img_size], antialias=True)
    return tf.clip_by_value(img / 255.0, 0.0, 1.0)


def prepare_dataset():
    list_ds = tf.data.Dataset.list_files(r"data/*", shuffle=True). \
        shuffle(10 * batch_size).repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(
        buffer_size=tf.data.AUTOTUNE)  # cache().

    return map(lambda x: np.array(list(map(preprocess_image, x))), list_ds)