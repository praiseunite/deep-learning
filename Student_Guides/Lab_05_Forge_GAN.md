# Lab Guide 05: "The Forge" -- Build an AI Image Generator with GANs
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Sessions 23-25 (GANs, DCGAN, Conditional GAN)
### Estimated Time: 2-3 hours (including deployment)
---

> *"By the end of this guide, you will have built two neural networks that fight each other -- a Generator that creates fake handwritten digits and a Discriminator that tries to catch the fakes. Through this competition, the Generator learns to create convincing images. You will deploy it so anyone can generate digits with one click."*

---

## What You Will Build

A **Deep Convolutional GAN (DCGAN)** that generates realistic handwritten digits. Click a button, and the AI creates a batch of brand new handwritten numbers that look real. The Generator starts by producing random noise and gradually learns to make convincing digits.

---

## What You Need Before Starting

- [ ] A **Kaggle** account with phone verification (for GPU)
- [ ] A **Hugging Face** account (free)
- [ ] No software to install

---

# PHASE 1: Get the Data

### Cell 1: Load MNIST

```python
# ============================================================
# CELL 1: LOAD AND PREPARE MNIST
# What this cell does: Downloads 60,000 handwritten digit images
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow: {tf.__version__}")

# Load MNIST -- 60,000 training images of handwritten digits (0-9).
(X_train, _), (_, _) = keras.datasets.mnist.load_data()

# We don't need labels for a standard GAN.
# The GAN learns to generate images without knowing what digit they are.

print(f"Training images: {X_train.shape}")  # (60000, 28, 28)
```

---

# PHASE 2: Explore the Data

### Cell 2: Visualize Real Digits

```python
# ============================================================
# CELL 2: SHOW REAL DIGITS
# What this cell does: Displays actual handwritten digits from the dataset
# ============================================================

fig, axes = plt.subplots(4, 8, figsize=(14, 7))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.axis('off')

plt.suptitle("Real Handwritten Digits (from the MNIST Dataset)", fontsize=14)
plt.tight_layout()
plt.show()

print("These are REAL images. Our GAN will learn to create fakes that look like these.")
```

---

# PHASE 3: Prepare the Data

### Cell 3: Normalize and Reshape

```python
# ============================================================
# CELL 3: PREPARE THE DATA FOR THE GAN
# What this cell does: Scales pixels to [-1, 1] and adds channel dimension
# ============================================================

# For GANs, we normalize pixels to [-1, 1] instead of [0, 1].
# Why? Because the Generator's final layer uses tanh activation,
# which outputs values in the range [-1, 1].
X_train = X_train.astype('float32')
X_train = (X_train - 127.5) / 127.5  # Scale from [0, 255] to [-1, 1]

# Add a channel dimension: (60000, 28, 28) -> (60000, 28, 28, 1)
# The "1" means grayscale (1 color channel). Color images would have 3.
X_train = X_train.reshape(-1, 28, 28, 1)

print(f"Shape: {X_train.shape}")
print(f"Pixel range: [{X_train.min():.1f}, {X_train.max():.1f}]")

# Create a TensorFlow dataset for efficient batching.
BATCH_SIZE = 256
BUFFER_SIZE = 60000

dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Batch size: {BATCH_SIZE}")
print("Dataset ready!")
```

---

# PHASE 4: Build the Model

A GAN has TWO models that compete:
1. **Generator** -- creates fake images from random noise
2. **Discriminator** -- tries to tell real images from fakes

### Cell 4: Build the Generator

```python
# ============================================================
# CELL 4: BUILD THE GENERATOR
# What this cell does: Creates the network that generates fake images
# ============================================================

# The Generator takes a vector of 100 random numbers ("noise")
# and transforms it into a 28x28 image.
#
# It works in reverse compared to a CNN:
# - CNN: big image -> small feature map (shrinking)
# - Generator: small noise -> big image (growing)
#
# It uses Conv2DTranspose (the opposite of Conv2D) to "upscale" the noise.

NOISE_DIM = 100  # The Generator's input: 100 random numbers.

def build_generator():
    model = keras.Sequential(name='Generator')
    
    # Layer 1: Dense layer to create initial feature map.
    # Takes 100 numbers and expands to 7*7*256 = 12,544 numbers.
    # Why 7x7? Because 7 -> 14 -> 28 (we will double the size twice).
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    
    # BatchNormalization stabilizes training by normalizing layer outputs.
    model.add(layers.BatchNormalization())
    
    # LeakyReLU: like ReLU but allows small negative values.
    # This prevents "dead neurons" in the Generator.
    model.add(layers.LeakyReLU(0.2))
    
    # Reshape the flat vector into a 7x7 grid with 256 feature maps.
    model.add(layers.Reshape((7, 7, 256)))
    
    # Layer 2: First upsampling. 7x7 -> 7x7 (same size, fewer filters).
    # Conv2DTranspose is the "reverse convolution" -- it expands the image.
    # 128 = number of feature maps (filters).
    # (5, 5) = filter size.
    # strides=(1, 1) = do not change spatial size yet.
    # padding='same' = keep the same dimensions.
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    # Layer 3: Second upsampling. 7x7 -> 14x14 (doubled!).
    # strides=(2, 2) = double both width and height.
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    # Layer 4: Final upsampling. 14x14 -> 28x28 (doubled again!).
    # 1 filter = 1 output channel (grayscale image).
    # tanh activation outputs values in [-1, 1], matching our data range.
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', 
                                      use_bias=False, activation='tanh'))
    
    return model

generator = build_generator()
generator.summary()
```

### Cell 5: Build the Discriminator

```python
# ============================================================
# CELL 5: BUILD THE DISCRIMINATOR
# What this cell does: Creates the network that detects fakes
# ============================================================

# The Discriminator is a standard CNN classifier.
# It takes a 28x28 image and outputs a single number:
#   Close to 1 = "I think this image is REAL"
#   Close to 0 = "I think this image is FAKE"

def build_discriminator():
    model = keras.Sequential(name='Discriminator')
    
    # Layer 1: First convolution. 28x28 -> 14x14.
    # 64 filters detect basic patterns (edges, curves).
    # strides=(2, 2) = halve the image size.
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(0.2))
    
    # Dropout prevents the Discriminator from becoming too powerful.
    # If it's too good, the Generator never learns.
    model.add(layers.Dropout(0.3))
    
    # Layer 2: Second convolution. 14x14 -> 7x7.
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    # Layer 3: Flatten and classify.
    model.add(layers.Flatten())
    
    # Single output neuron. No activation (we use logits for the loss function).
    model.add(layers.Dense(1))
    
    return model

discriminator = build_discriminator()
discriminator.summary()
```

### Cell 6: Define Loss Functions and Optimizers

```python
# ============================================================
# CELL 6: LOSS FUNCTIONS AND OPTIMIZERS
# What this cell does: Defines how each network measures success
# ============================================================

# Binary Cross Entropy from logits.
# "from_logits=True" means the output has no sigmoid activation --
# the loss function handles the conversion internally (more numerically stable).
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """The Discriminator wants: real -> 1, fake -> 0."""
    # Loss for classifying real images as 1 (real).
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Loss for classifying fake images as 0 (fake).
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """The Generator wants: fake -> 1 (fool the Discriminator)."""
    # The Generator succeeds when the Discriminator thinks fakes are real.
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Separate optimizers for each network.
# They learn independently -- each adjusts its own weights.
gen_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
disc_optimizer = keras.optimizers.Adam(learning_rate=1e-4)

print("Loss functions and optimizers ready!")
```

---

# PHASE 5: Train the Model

### Cell 7: Custom Training Loop

```python
# ============================================================
# CELL 7: TRAINING LOOP
# What this cell does: Alternately trains Generator and Discriminator
# ============================================================

EPOCHS = 50

# We save a fixed noise vector to track progress.
# By generating images from the SAME noise each epoch,
# we can see the Generator improving over time.
SEED_NOISE = tf.random.normal([16, NOISE_DIM])

# Store losses for plotting.
gen_losses = []
disc_losses = []

@tf.function  # This decorator compiles the function for faster execution.
def train_step(real_images):
    # Generate random noise.
    noise = tf.random.normal([tf.shape(real_images)[0], NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator creates fake images from noise.
        fake_images = generator(noise, training=True)
        
        # Discriminator judges both real and fake images.
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        
        # Calculate losses.
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)
    
    # Update Generator weights.
    gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    
    # Update Discriminator weights.
    disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    return g_loss, d_loss

print("Training loop defined. Starting training...")
```

### Cell 8: Run Training

```python
# ============================================================
# CELL 8: RUN THE TRAINING
# What this cell does: Trains the GAN for 50 epochs
# ============================================================

import time

for epoch in range(EPOCHS):
    start = time.time()
    
    epoch_gen_loss = []
    epoch_disc_loss = []
    
    for batch in dataset:
        g_loss, d_loss = train_step(batch)
        epoch_gen_loss.append(g_loss.numpy())
        epoch_disc_loss.append(d_loss.numpy())
    
    avg_g = np.mean(epoch_gen_loss)
    avg_d = np.mean(epoch_disc_loss)
    gen_losses.append(avg_g)
    disc_losses.append(avg_d)
    
    elapsed = time.time() - start
    
    # Print progress every 5 epochs.
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:>3d}/{EPOCHS} | "
              f"Gen Loss: {avg_g:.4f} | Disc Loss: {avg_d:.4f} | "
              f"Time: {elapsed:.1f}s")

print("\nTraining complete!")
```

### Cell 9: Plot Training Progress

```python
# ============================================================
# CELL 9: PLOT LOSS CURVES
# What this cell does: Shows how Generator and Discriminator losses evolved
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label='Generator Loss', linewidth=2)
plt.plot(disc_losses, label='Discriminator Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training: Generator vs Discriminator')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Ideally, both losses stabilize (they bounce around, that's normal for GANs).")
print("If Gen loss keeps rising and Disc loss drops to 0, the Discriminator won.")
```

---

# PHASE 6: Test the Model

### Cell 10: Generate Fake Digits

```python
# ============================================================
# CELL 10: GENERATE FAKE DIGITS
# What this cell does: Creates brand new handwritten digits
# ============================================================

# Generate 32 fake images.
noise = tf.random.normal([32, NOISE_DIM])
generated_images = generator(noise, training=False)

fig, axes = plt.subplots(4, 8, figsize=(14, 7))

for i, ax in enumerate(axes.flat):
    # Convert from [-1, 1] back to [0, 1] for display.
    img = (generated_images[i, :, :, 0].numpy() + 1) / 2
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.suptitle("AI-Generated Handwritten Digits (FAKE!)", fontsize=14)
plt.tight_layout()
plt.show()

print("Every image above was generated by the AI -- none are from the dataset!")
```

### Cell 11: Compare Real vs Fake

```python
# ============================================================
# CELL 11: REAL VS FAKE COMPARISON
# What this cell does: Shows real and fake images side by side
# ============================================================

fig, axes = plt.subplots(2, 8, figsize=(14, 4))

# Top row: Real images.
for i in range(8):
    img = (X_train[i, :, :, 0] + 1) / 2
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title("REAL", fontsize=12, color='green')

# Bottom row: Fake images.
noise = tf.random.normal([8, NOISE_DIM])
fakes = generator(noise, training=False)
for i in range(8):
    img = (fakes[i, :, :, 0].numpy() + 1) / 2
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title("FAKE (AI)", fontsize=12, color='red')

plt.suptitle("Can You Tell Which Row Is Real?", fontsize=14)
plt.tight_layout()
plt.show()
```

---

# PHASE 7: Save the Model

### Cell 12: Save the Generator

```python
# ============================================================
# CELL 12: SAVE THE GENERATOR
# What this cell does: Saves only the Generator (for deployment)
# ============================================================

import json

# We only need the Generator for deployment.
# The Discriminator was just a training tool.
generator.save('digit_generator.keras')

config = {
    'noise_dim': NOISE_DIM,
    'image_size': 28
}

with open('model_config.json', 'w') as f:
    json.dump(config, f)

print("Saved:")
print("  1. digit_generator.keras")
print("  2. model_config.json")
print("\nDownload BOTH from the Output panel.")
```

---

# PHASE 8: Deploy to Hugging Face

## Step 8.1: Download from Kaggle

Download `digit_generator.keras` and `model_config.json`.

## Step 8.2: Create Hugging Face Space

Name: `digit-forge` | SDK: **Gradio** | Hardware: **CPU Basic**

## Step 8.3: Upload Files

Upload both files.

## Step 8.4: Create `requirements.txt`

```
tensorflow==2.15.0
gradio==4.44.0
numpy
Pillow
```

## Step 8.5: Create `app.py`

```python
# ============================================================
# app.py -- Digit Generator Web App
# Click "Generate" and the AI creates new handwritten digits.
# ============================================================

import gradio as gr
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------------------------
# STEP 1: Load model and config
# -----------------------------------------------
generator = load_model('digit_generator.keras')

with open('model_config.json', 'r') as f:
    config = json.load(f)

NOISE_DIM = config['noise_dim']

# -----------------------------------------------
# STEP 2: Generation function
# -----------------------------------------------
def generate_digits(num_digits):
    """Generate a grid of fake handwritten digits."""
    
    num_digits = int(num_digits)
    
    noise = np.random.normal(size=(num_digits, NOISE_DIM)).astype('float32')
    generated = generator.predict(noise, verbose=0)
    
    images = []
    for i in range(num_digits):
        img_array = ((generated[i, :, :, 0] + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img = img.resize((112, 112), Image.NEAREST)
        images.append(img)
    
    return images

# -----------------------------------------------
# STEP 3: Gradio interface
# -----------------------------------------------
demo = gr.Interface(
    fn=generate_digits,
    inputs=gr.Slider(minimum=1, maximum=16, value=9, step=1,
                     label="Number of Digits to Generate"),
    outputs=gr.Gallery(label="Generated Digits", columns=4, height=500),
    title="The Forge: AI Digit Generator (DCGAN)",
    description="Click Submit and the AI will generate brand new handwritten digits "
                "that have never existed. Powered by a Deep Convolutional GAN trained on MNIST.",
)

demo.launch()
```

Commit and wait for the build.

---

# PHASE 9: Test It Live

## Test 1: Generate Batches

Use the slider to generate different amounts (1, 4, 9, 16). Each click produces completely new, unique digits.

## Test 2: Repeated Generation

Click "Submit" several times. Notice how every batch is different -- the AI creates new images from random noise each time.

## Test 3: Share

Send the URL to classmates. Challenge them: "Can you tell these are AI-generated?"

---

# Troubleshooting

### Problem: Generated images are just noise (after 50 epochs)
**Solution:** GANs are fragile. Try reducing the Discriminator's learning rate to `5e-5` while keeping the Generator at `1e-4`. This gives the Generator a head start.

### Problem: All generated images look the same (mode collapse)
**Solution:** Increase Dropout in the Discriminator to 0.4. Also try adding noise to the Discriminator's input: `real_images + tf.random.normal(shape=real_images.shape) * 0.1`.

### Problem: Training is very slow
**Solution:** Make sure GPU is enabled in Kaggle settings. DCGAN training on CPU is extremely slow.

### Problem: "NaN" in losses
**Solution:** Reduce the learning rate for both optimizers to `5e-5`. NaN usually means the gradients exploded.

---

## What You Learned

| Concept | Where You Used It |
|---------|------------------|
| GANs (Session 23) | The Generator vs Discriminator framework |
| DCGAN (Session 24) | Conv2DTranspose layers in the Generator |
| Mode Collapse (Session 24) | Dropout as a mitigation strategy |
| Custom Training Loop | The `train_step` function with dual gradient tapes |
| BatchNormalization | Stabilizing GAN training |
| LeakyReLU | Preventing dead neurons |

---
*Lab Guide 05 | Deep Learning Using Neural Networks | Aptech*
