# Session 22 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Generate New Fashion"
### Due: Before Session 23 begins
### Estimated Time: 45 minutes

---

> **Professor's Note:** *"In class, we trained a VAE on handwritten digits. For homework, you will train one on clothing items. By the end, your AI will be designing T-shirts and sneakers that have never existed."*

---

## Assignment Objectives

By completing this assignment, you will:
- Demonstrate that you can adapt a VAE to a different dataset independently.
- Explore the latent space and observe how it organizes different clothing categories.
- Generate new fashion items by sampling from the latent space.

---

## THE SCENARIO

You have been hired by a fashion startup. They want an AI that can generate new clothing designs. Your job is to train a Variational Autoencoder on the **Fashion-MNIST** dataset (10 categories of clothing) and show that the AI can create new items.

---

## Step-by-Step Instructions (Kaggle)

Open a **new** Kaggle Notebook. Turn on the GPU.

### Cell 1: Import and Load Data
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Fashion-MNIST has 10 clothing categories
CATEGORY_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Downloading Fashion-MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"Training samples: {X_train.shape[0]}")
print(f"Categories: {CATEGORY_NAMES}")
```

### Cell 2: Visualize Fashion Items
```python
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title(CATEGORY_NAMES[y_train[i]], fontsize=7)
    plt.axis('off')
plt.suptitle("Sample Fashion Items", fontsize=13)
plt.tight_layout()
plt.show()
```

### Cell 3: Build the VAE
Copy and adapt the VAE architecture from the in-class task. Use `LATENT_DIM = 2` so you can visualize the latent space later.

```python
LATENT_DIM = 2

# --- ENCODER ---
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)

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

print("Encoder and Decoder built!")
```

### Cell 4: Define the VAE Class and Train
```python
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"total_loss": self.total_loss_tracker.result()}

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

print("Training on Fashion-MNIST...")
history = vae.fit(X_train, epochs=20, batch_size=128, verbose=1)
print("Training complete!")
```

### Cell 5: Visualize the Latent Space
This is the most revealing plot. Each dot is a clothing item, colored by category.

```python
z_mean_test, _, _ = vae.encoder.predict(X_test)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], 
                      c=y_test, cmap='tab10', alpha=0.5, s=2)
plt.colorbar(scatter, ticks=range(10), label='Category')
plt.clim(-0.5, 9.5)
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("VAE Latent Space: Each Color is a Different Clothing Category")
plt.tight_layout()
plt.show()
print("Notice how similar items cluster together!")
```

### Cell 6: Generate New Fashion Items
```python
random_latent_vectors = np.random.normal(size=(20, LATENT_DIM))
generated = vae.decoder.predict(random_latent_vectors)

plt.figure(figsize=(14, 2))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(generated[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle("AI-Generated Fashion Items -- These Never Existed!", fontsize=12)
plt.tight_layout()
plt.show()
```

---

## Written Questions

Answer these in your submission document:

1. Look at the latent space plot (Cell 5). Which two clothing categories seem to overlap the most? Why do you think the VAE places them close together?

2. Compare the quality of generated fashion items (Cell 6) to the generated digits from class. Which dataset seems harder for the VAE to reconstruct? Why?

3. If a fashion company wanted to use this VAE to generate only sneakers, how could they sample from a specific region of the latent space instead of randomly?

---

## Submission Requirements

Submit the following to your instructor:
1. A screenshot of your latent space plot (Cell 5).
2. A screenshot of your generated fashion items (Cell 6).
3. Written answers to the 3 questions above.

---
*Session 22 | Deep Learning Using Neural Networks | Aptech*
