from config import img_size, dataset_repetitions, batch_size, min_signal_rate, max_signal_rate, plot_diffusion_steps, \
    widths, block_depth
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from data import prepare_dataset, preprocess_image
import matplotlib.pyplot as plt
from model import get_network
from config import save_every
import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

file_count = sum(len(files) for _, _, files in os.walk(r'data'))


class Diffusion():

    def __init__(self):
        self.normalizer = tf.keras.layers.Normalization()
        self.network = get_network(widths, block_depth)

    @tf.function
    def init_model(self):
        self.network([tf.zeros((64, 96, 96, 3)), tf.zeros((64, 1, 1))])

    def loss_fn(self, real, generated):
        loss = tf.math.reduce_mean((real - generated) ** 2)
        return loss

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # predict noise component and calculate the image component using it
        pred_noises = self.network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, False)
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, img_size, img_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(num_rows * num_cols, plot_diffusion_steps)[0]

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"output_images/generated_plot_epoch-{epoch}.png")

    def train_step(self, optimizer, batch):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(batch, training=True)[0]
        noises = tf.random.normal(shape=(batch_size, img_size, img_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, True)

            noise_loss = self.loss_fn(noises, pred_noises)  # used for training
            image_loss = self.loss_fn(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        return image_loss

    def train(self):
        start_time = time.time()
        dataset = prepare_dataset()

        # self.normalizer.adapt(np.array([next(dataset) for _ in range(250)]))  # 250

        checkpoint = tf.train.Checkpoint(optimizer=Adam(learning_rate=1e-4), unet=self.network)
        manager = tf.train.CheckpointManager(checkpoint, directory='models/', max_to_keep=3)

        epoch_start = int(
            manager.latest_checkpoint.split(sep='ckpt-')[-1]) * save_every if manager.latest_checkpoint else 0
        print('starting at:', epoch_start)
        checkpoint.restore(manager.latest_checkpoint)
        bar = tf.keras.utils.Progbar(file_count * dataset_repetitions / batch_size - 1)
        for epoch, batch in enumerate(dataset):
            loss = self.train_step(checkpoint.optimizer, batch)
            bar.update(epoch + epoch_start, values=[("loss", loss)])  # Open a file with access mode 'a'
            if (epoch + epoch_start) % save_every == 0:
                with open("logs.txt", "a") as file_object:
                    file_object.write(
                        "\n" + f'epoch: {epoch + epoch_start} time: {time.time() - start_time} loss: {loss} ')
                self.plot_images(epoch + epoch_start)
                manager.save()


d = Diffusion()
d.train()
