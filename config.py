import numpy as np

width = 180
height = 100
timesteps = 300
time_emb_dim = (int(width / 2), int(height / 2))

channels = (64, 128, 256, 512, 1024)

beta = np.linspace(0.0001, 0.02, timesteps)

alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]))
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1 - alpha_bar)
save_every = 250
batch_size = 32

dataset_repetitions = 100

img_size = 128
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 50

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128, 256]
block_depth = 2

# optimization
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4
