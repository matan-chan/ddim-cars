from config import batch_size, img_size, dataset_repetitions
import tensorflow as tf
from os import listdir
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import requests
import torch
import clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, jit=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_text_emb(text):
    # len(text.numpy())
    text = text.numpy()[:77]
    text = clip.tokenize([text.decode('utf-8')]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        return text_features


def read_img(image_url):
    try:
        img_bytes = requests.get(image_url.numpy().decode('utf-8')).content
        image = np.array(Image.open(BytesIO(img_bytes)))
    except:
        return np.zeros((img_size, img_size, 3))
    return image


def preprocess_image(data):
    image_url, description = data
    image = tf.py_function(read_img, [image_url], tf.float64)
    text_emb = tf.py_function(get_text_emb, [description], tf.float64)
    if len(tf.shape(image)) != 3:
        image = image[..., np.newaxis]
    height, width, depth = tf.shape(image)
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(image, (height - crop_size) // 2, (width - crop_size) // 2, crop_size,
                                          crop_size)
    image = tf.image.resize(image, size=[img_size, img_size], antialias=True)
    return np.array(tf.clip_by_value(image / 255.0, 0.0, 1.0)), text_emb


def extract_relevant_data_from_files():
    only_files = ['data/' + file for file in listdir('data/')]
    engine = ['pyarrow'] * len(only_files)
    data_frame = pd.concat(map(pd.read_parquet, only_files, engine))
    urls_and_captions = np.stack((data_frame['url'], data_frame['caption']), axis=-1)
    return urls_and_captions


def prepare_dataset():
    urls_and_captions = extract_relevant_data_from_files()
    list_ds = tf.data.Dataset.from_tensor_slices(urls_and_captions). \
        shuffle(10 * batch_size).repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(
        buffer_size=tf.data.AUTOTUNE)  # cache().
    return map(lambda x: np.transpose(np.array(list(map(preprocess_image, x)))), list_ds)


images = np.stack(images)