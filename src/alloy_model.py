import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# Sample data (replace with actual alloy compositions and properties dataset)
compositions = [
    [10, 20, 30, 40],
    [15, 25, 35, 25],
    [20, 30, 25, 25],
    [25, 15, 30, 30]
]
properties = [
    [500, 200],
    [550, 210],
    [520, 205],
    [530, 215]
]

# Normalize compositions data
compositions_normalized = np.array(compositions, dtype=np.float32) / 100.0
properties_normalized = np.array(properties, dtype=np.float32) / 1000.0

# VAE model parameters
input_dim = compositions_normalized.shape[1]
latent_dim = 2

# Encoder model
composition_inputs = Input(shape=(input_dim,), name="composition_inputs")
h = layers.Dense(16, activation="relu")(composition_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(h)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

# Decoder model
decoder_h = layers.Dense(16, activation="relu")
decoder_mean = layers.Dense(input_dim, activation="sigmoid")

h_decoded = decoder_h(z)
reconstructed = decoder_mean(h_decoded)

# VAE model
vae = Model(composition_inputs, reconstructed, name="vae")

# VAE Loss Function
class VAELossLayer(layers.Layer):
    def call(self, inputs):
        composition_inputs, reconstructed, z_mean, z_log_var = inputs
        reconstruction_loss = MeanSquaredError()(composition_inputs, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss

# Add loss to the VAE model
loss_layer = VAELossLayer()([composition_inputs, reconstructed, z_mean, z_log_var])
vae.add_loss(loss_layer)

# Compile the VAE model
vae.compile(optimizer=Adam())
vae.summary()

# Train the model
vae.fit(compositions_normalized, compositions_normalized, epochs=50, batch_size=2)

# Testing the model (Example usage)
sample_composition = np.array([[12, 22, 32, 34]], dtype=np.float32) / 100.0
predicted = vae.predict(sample_composition)
print("Input Composition:", sample_composition * 100)
print("Reconstructed Composition:", predicted * 100)
