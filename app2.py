import os
import numpy as np
import random
import warnings
import streamlit as st
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
ds = tfp.distributions
from PIL import Image

warnings.simplefilter("ignore")

# Load Data: Fashion MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

TRAIN_BUF = 60000
BATCH_SIZE = 512
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)

# split dataset
train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Variational Autoencoder Model
class VAE(tf.keras.Model):

    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def reconstruct(self, x):
        mu, _ = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return self.decode(mu)

    def decode(self, z):
        return self.dec(z)

    def compute_loss(self, x):
        q_z = self.encode(x)
        z = q_z.sample()
        x_recon = self.decode(z)
        p_z = ds.MultivariateNormalDiag(
          loc=[0.] * z.shape[-1], scale_diag=[1.] * z.shape[-1]
          )
        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0))
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - x_recon), axis=0))

        return recon_loss, latent_loss

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

N_Z = 2
encoder = [
    tf.keras.layers.InputLayer(input_shape=DIMS),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=N_Z*2),
]

decoder = [
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]

st.sidebar.header("Search Options")

# Allow the user to specify the number of similar images to retrieve
num_similar_images = st.sidebar.slider("Number of similar images to retrieve:", 1, 20, 6)

# Allow the user to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read the uploaded image
    image = Image.open(uploaded_image)
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    image = np.array(image)
    image = 1 - image.astype("float32") / 255.0

    # Show the uploaded and resized image
    #st.image(1 - image, caption="Uploaded and Resized Image (28x28 pixels)", use_column_width=True)

    model = VAE(
        enc=encoder,
        dec=decoder,
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
    )

    # Execute the query when the user clicks the button
    if st.button("Search"):
        st.write("Running, Please wait...")
        for epoch in range(5):
            print(f'Epoch {epoch}...')
            for batch, train_x in zip(range(N_TRAIN_BATCHES), train_dataset):
                model.train(train_x)

        # Define a function to query for nearest neighbors
        def query(image_embedding, k):
            distances = np.zeros(len(train_images))
            for i, e in enumerate(train_images):
                distances[i] = np.linalg.norm(image_embedding - model.enc(e[tf.newaxis, ...]))
            return np.argpartition(distances, k)[:k]

        # Get the embedding of the uploaded image
        image_embedding = model.enc(image[tf.newaxis, ...])
        idx = query(image_embedding, k=num_similar_images)
        st.write("Query Image:")
        st.image(1 - image, caption="Query Image", use_column_width=True)
        st.write("Similar Images:")
        for i in range(num_similar_images):
            st.image(1 - train_images[idx[i]], caption=f"Image {i+1}", use_column_width=True)
