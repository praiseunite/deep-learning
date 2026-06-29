# Session 22 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Compression Game"
### Duration: 45-55 minutes

---

> **Instructor Note:** Students will first build intuition for autoencoders with a pen-and-paper exercise, then run a complete VAE on Kaggle step-by-step. Ensure all students have their Kaggle GPU enabled before starting Part 2.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Explain the role of the Encoder, Bottleneck, and Decoder in an autoencoder.
- [ ] Calculate the compression ratio of a bottleneck layer.
- [ ] Successfully run a VAE on Kaggle and observe reconstructed images.
- [ ] Sample from the latent space to generate new images.

---

## PART 1 -- Pen-and-Paper: Understanding the Bottleneck (15 minutes)

You are an AI engineer at a medical imaging startup. Your boss wants you to build an autoencoder that compresses brain scan images for fast transmission over slow internet connections.

### Question 1: Architecture Design
Your input images are `64 x 64` pixels in grayscale (1 channel).

a) How many total pixel values does each input image contain?
*Answer:* __________________________________________________________________

b) Your Encoder has the following layers: `Dense(1024)` -> `Dense(256)` -> `Dense(16)`. What is the size of the latent code (bottleneck)?
*Answer:* __________________________________________________________________

c) Calculate the **compression ratio**: divide the input size by the bottleneck size.
*Answer:* __________________________________________________________________

d) Your Decoder must reconstruct the original image. Write out the layer sizes the Decoder needs (hint: it mirrors the Encoder in reverse):
*Answer:* Dense(___) -> Dense(___) -> Dense(___)

### Question 2: VAE vs AE
In a regular Autoencoder, the encoder outputs a single latent vector of size 16.

In a **Variational Autoencoder**, the encoder outputs TWO vectors instead. What are they called, and what size would each be?

*Answer:* __________________________________________________________________

### Question 3: Why Does it Matter?
In one sentence, explain why outputting a distribution (mean + variance) instead of a single point makes the VAE better at *generating* new images.

*Answer:* __________________________________________________________________

---

## PART 2 -- Kaggle Lab: Build a VAE on MNIST (30 minutes)

Open your Kaggle Notebook. Make sure the GPU is turned on (Settings -> Accelerator -> GPU T4 x2).

### Cell 1: Import Libraries
Create a new code cell, paste this code, and hit **Play**.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("All libraries loaded!")
```

### Cell 2: Load and Prepare the Data
Create a new cell. This downloads 60,000 handwritten digit images.

```python
print("Downloading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixels to range [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape: add a channel dimension (28, 28) -> (28, 28, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"Training images shape: {X_train.shape}")
print(f"Test images shape: {X_test.shape}")
print("Data ready!")
```

### Cell 3: Visualize Some Training Images
Let's see what our data looks like before we train.

```python
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.show()
print("These are the ORIGINAL images. Let's see how well our VAE can reconstruct them!")
```

### Cell 4: Define the VAE Model
This is the heart of the session. Read the comments carefully.

```python
# --- HYPERPARAMETERS ---
LATENT_DIM = 2  # We use just 2 dimensions so we can visualize the latent space!

# --- ENCODER ---
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)

# The VAE's special sauce: TWO output heads instead of one
z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)

# --- REPARAMETERIZATION TRICK ---
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# --- DECODER ---
decoder_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(256, activation='relu')(decoder_inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_outputs = layers.Reshape((28, 28, 1))(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

print("Encoder Summary:")
encoder.summary()
print("\nDecoder Summary:")
decoder.summary()
print("\nModel architecture built!")
```

### Cell 5: Define the VAE Loss and Training Step
This cell defines how the VAE learns.

```python
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Force 1: Reconstruction Loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )

            # Force 2: KL Divergence Loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
print("VAE compiled and ready to train!")
```

### Cell 6: Train the VAE!
This will take about 2-3 minutes. Watch the loss values decrease.

```python
print("Training started! Watch the loss values go down...")
history = vae.fit(X_train, epochs=15, batch_size=128, verbose=1)
print("\nTraining complete!")
```

### Cell 7: Compare Original vs Reconstructed Images
Now let's see how well our VAE learned to compress and reconstruct.

```python
# Encode and then decode the test images
z_mean_test, z_log_var_test, z_test = vae.encoder.predict(X_test[:10])
reconstructed = vae.decoder.predict(z_test)

# Plot side by side
plt.figure(figsize=(14, 3))
for i in range(10):
    # Original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title("ORIGINAL", fontsize=8)

    # Reconstructed
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title("RECONSTRUCTED", fontsize=8)

plt.suptitle("Top Row: Original | Bottom Row: VAE Reconstruction", fontsize=12)
plt.tight_layout()
plt.show()
print("Notice: reconstructions are slightly blurry -- that's normal for VAEs!")
```

### Cell 8: Generate BRAND NEW Digits!
This is the magic. We sample random points from the latent space and decode them into images that never existed.

```python
# Sample 20 random points from a standard normal distribution
random_latent_vectors = np.random.normal(size=(20, LATENT_DIM))

# Decode them into images
generated_images = vae.decoder.predict(random_latent_vectors)

# Display
plt.figure(figsize=(14, 2))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle("GENERATED Digits -- These Never Existed in the Training Data!", fontsize=12)
plt.tight_layout()
plt.show()
print("You just created brand new images using AI!")
```

---

## PART 3 -- Observation Questions (5 minutes)

Answer these based on what you observed in your Kaggle notebook:

1. Look at the reconstructed images from Cell 7. Are they perfect copies of the originals, or are they slightly blurry? Why do you think that is?
*Answer:* __________________________________________________________________

2. Look at the generated images from Cell 8. Do they look like real handwritten digits? Are some more convincing than others?
*Answer:* __________________________________________________________________

3. We used `LATENT_DIM = 2`. If we increased this to `LATENT_DIM = 32`, do you think the reconstructions would be sharper or blurrier? Why?
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: All pen-and-paper questions answered
- [ ] Part 2: All 8 code cells executed successfully on Kaggle
- [ ] Part 2: Cell 7 plot shows original vs reconstructed comparison
- [ ] Part 2: Cell 8 plot shows generated images
- [ ] Part 3: All 3 observation questions answered

---
*Session 22 | Deep Learning Using Neural Networks | Aptech*
