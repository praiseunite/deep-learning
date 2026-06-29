# Session 26 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Generate On Command"
### Duration: 45-55 minutes

---

> **Instructor Note:** Students will build a CVAE on MNIST and demonstrate controlled digit generation. The key "aha moment" is Cell 6 where they specify a digit and the CVAE generates it.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Modify a VAE architecture to accept conditional class labels.
- [ ] Train a CVAE that generates specific digit classes on demand.
- [ ] Compare CVAE output quality to the cGAN from Session 24.

---

## PART 1 -- Concept Check (5 minutes)

1. In a standard VAE, the Encoder input is just the image (784 numbers). In a CVAE, what is the Encoder input?
*Answer:* __________________________________________________________________

2. Why does the Decoder also receive the class label?
*Answer:* __________________________________________________________________

3. The CVAE latent space encodes __________ (class identity / within-class variation / random noise). Choose one and explain why.
*Answer:* __________________________________________________________________

---

## PART 2 -- Kaggle Lab: Build a CVAE (40 minutes)

Open your Kaggle Notebook with the GPU enabled.

### Cell 1: Import and Load Data
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("Loading MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten images
X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)

# One-hot encode labels
NUM_CLASSES = 10
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Images: {X_train_flat.shape}, Labels: {y_train_oh.shape}")
print("Ready!")
```

### Cell 2: Build the CVAE Encoder
Notice how the image and label are concatenated before being processed.

```python
LATENT_DIM = 2

# Image input
image_input = keras.Input(shape=(784,), name='image')
# Label input
label_input = keras.Input(shape=(NUM_CLASSES,), name='label')

# Concatenate image + label
x = layers.Concatenate()([image_input, label_input])  # Shape: (794,)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)

# VAE heads: mean and log_variance
z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = keras.Model([image_input, label_input], [z_mean, z_log_var, z], name='cvae_encoder')
print("CVAE Encoder built!")
print(f"Input: image(784) + label({NUM_CLASSES}) -> latent({LATENT_DIM})")
```

### Cell 3: Build the CVAE Decoder
The Decoder also receives the label, concatenated with the latent code.

```python
# Latent input
latent_input = keras.Input(shape=(LATENT_DIM,), name='z')
# Label input for decoder
label_input_dec = keras.Input(shape=(NUM_CLASSES,), name='label_dec')

# Concatenate latent + label
x = layers.Concatenate()([latent_input, label_input_dec])  # Shape: (12,)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
decoder_output = layers.Dense(784, activation='sigmoid')(x)

decoder = keras.Model([latent_input, label_input_dec], decoder_output, name='cvae_decoder')
print("CVAE Decoder built!")
print(f"Input: latent({LATENT_DIM}) + label({NUM_CLASSES}) -> image(784)")
```

### Cell 4: Define the CVAE Training Loop
```python
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            # Encode (with label)
            z_mean, z_log_var, z = self.encoder([images, labels])
            # Decode (with label)
            reconstruction = self.decoder([z, labels])
            
            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(
                        tf.reshape(images, (-1, 784)),
                        reconstruction
                    ), axis=1
                )
            )
            
            # KL Divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

cvae = CVAE(encoder, decoder)
cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
print("CVAE compiled!")
```

### Cell 5: Train the CVAE
```python
# Create a tf.data.Dataset that pairs images with labels
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train_flat, y_train_oh)
).shuffle(60000).batch(128)

print("Training CVAE...")
history = cvae.fit(train_dataset, epochs=20, verbose=1)
print("Training complete!")
```

### Cell 6: Generate Specific Digits On Command!
This is the payoff. We tell the CVAE which digit to generate.

```python
fig, axes = plt.subplots(10, 10, figsize=(12, 12))

for digit in range(10):
    # Create the condition: "generate digit [digit]"
    label = keras.utils.to_categorical([digit] * 10, NUM_CLASSES)
    # Sample random latent codes
    z_sample = np.random.normal(0, 1, (10, LATENT_DIM))
    # Decode with condition
    generated = decoder.predict([z_sample, label], verbose=0)
    
    for j in range(10):
        axes[digit, j].imshow(generated[j].reshape(28, 28), cmap='gray')
        axes[digit, j].axis('off')
    axes[digit, 0].set_ylabel(f"Digit {digit}", fontsize=10, rotation=0, labelpad=40)

plt.suptitle("CVAE: Each Row is a Requested Digit", fontsize=14)
plt.tight_layout()
plt.show()
print("Each row shows 10 variations of the SAME digit -- generated on command!")
```

### Cell 7: Visualize the CVAE Latent Space
```python
z_means, _, _ = encoder.predict([X_test_flat, y_test_oh])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_means[:, 0], z_means[:, 1], c=y_test, cmap='tab10', alpha=0.5, s=2)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("CVAE Latent Space (Each Color = Different Digit)")
plt.tight_layout()
plt.show()
print("Compare this to the VAE latent space from Session 22!")
print("In the CVAE, the classes may overlap MORE because the decoder")
print("already knows the class -- the latent space only encodes style/variation.")
```

---

## PART 3 -- Comparison (5 minutes)

1. Compare the CVAE's 10x10 grid (Cell 6) with the cGAN's grid from Session 24. Which approach seems to produce more consistent digits? Which produces sharper digits?
*Answer:* __________________________________________________________________

2. Look at the CVAE latent space (Cell 7). How does it differ from the standard VAE latent space in Session 22? Why?
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: All 3 concept check questions answered
- [ ] Part 2: All 7 cells executed successfully
- [ ] Part 2: 10x10 conditional generation grid visible
- [ ] Part 2: Latent space plot visible
- [ ] Part 3: Both comparison questions answered

---
*Session 26 | Deep Learning Using Neural Networks | Aptech*
