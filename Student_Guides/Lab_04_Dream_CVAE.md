# Lab Guide 04: "The Dream Machine" -- Build an AI Fashion Designer
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Sessions 22, 26-27 (VAE, CVAE, Conditional Generation)
### Estimated Time: 2-3 hours (including deployment)
---

> *"By the end of this guide, you will have built an AI that creates brand new clothing designs that have never existed. Select 'Dress' from a dropdown and the AI generates a completely new dress. Select 'Sneaker' and it generates a sneaker. You will deploy it so anyone can generate fashion designs from their browser."*

---

## What You Will Build

A **Conditional Variational Autoencoder (CVAE)** trained on Fashion-MNIST. The user picks a clothing category (T-shirt, Trouser, Dress, Sneaker, etc.) and the AI generates a new item from that category. Think of it as an AI sketchpad for fashion designers.

---

## What You Need Before Starting

- [ ] A **Kaggle** account with phone verification (for GPU)
- [ ] A **Hugging Face** account (free)
- [ ] No software to install

---

# PHASE 1: Get the Data

## Step 1.1: Create a Notebook

1. Go to [kaggle.com](https://www.kaggle.com) -> **Create** -> **New Notebook**.
2. Turn on GPU: Right panel -> **Settings** -> **Accelerator** -> **GPU T4 x2**.

Fashion-MNIST is built into Keras, so no dataset download is needed.

### Cell 1: Load Fashion-MNIST

```python
# ============================================================
# CELL 1: LOAD THE DATA
# What this cell does: Downloads 70,000 tiny clothing images
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Fashion-MNIST is a dataset of 70,000 grayscale images (28x28 pixels).
# Each image is a piece of clothing from one of 10 categories.
# It's built into Keras -- no downloading needed.
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# The 10 clothing categories.
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
NUM_CLASSES = len(CLASS_NAMES)

print(f"Training images: {X_train.shape}")   # (60000, 28, 28)
print(f"Training labels: {y_train.shape}")    # (60000,)
print(f"Test images: {X_test.shape}")          # (10000, 28, 28)
print(f"Categories: {CLASS_NAMES}")
```

---

# PHASE 2: Explore the Data

### Cell 2: Visualize Samples

```python
# ============================================================
# CELL 2: VISUALIZE SAMPLE IMAGES
# What this cell does: Shows one example from each category
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for i, ax in enumerate(axes.flat):
    # Find the first image with label i.
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(CLASS_NAMES[i], fontsize=10)
    ax.axis('off')

plt.suptitle("One Example from Each Category", fontsize=14)
plt.tight_layout()
plt.show()
```

---

# PHASE 3: Prepare the Data

### Cell 3: Normalize and Reshape

```python
# ============================================================
# CELL 3: PREPARE THE DATA
# What this cell does: Normalizes pixels and flattens images
# ============================================================

# Normalize: scale pixels from [0, 255] to [0.0, 1.0].
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten: reshape from 28x28 grid to a flat list of 784 numbers.
# The CVAE processes images as flat vectors.
IMG_SIZE = 28 * 28  # 784

X_train_flat = X_train.reshape(-1, IMG_SIZE)
X_test_flat = X_test.reshape(-1, IMG_SIZE)

# One-hot encode the labels.
# Instead of label=3 (Dress), we use [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
# This is needed because we CONCATENATE the label with the image data.
y_train_onehot = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_onehot = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Flattened image shape: {X_train_flat.shape}")   # (60000, 784)
print(f"One-hot label shape: {y_train_onehot.shape}")    # (60000, 10)
print(f"Example label 'Dress': {y_train_onehot[np.where(y_train == 3)[0][0]]}")
```

### Cell 4: Concatenate Images with Labels

```python
# ============================================================
# CELL 4: CONCATENATE IMAGES WITH LABELS
# What this cell does: Glues the label information onto each image
# ============================================================

# The "Conditional" in CVAE means we TELL the network what category
# to generate. We do this by sticking the one-hot label vector
# onto the end of the image vector.
#
# Before: image = [pixel1, pixel2, ..., pixel784]  (784 numbers)
# After:  image + label = [pixel1, ..., pixel784, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (794 numbers)
#
# Now the network sees BOTH the image AND which category it is.

X_train_cond = np.concatenate([X_train_flat, y_train_onehot], axis=1)
X_test_cond = np.concatenate([X_test_flat, y_test_onehot], axis=1)

INPUT_DIM = X_train_cond.shape[1]  # 784 + 10 = 794

print(f"Conditioned input shape: {X_train_cond.shape}")  # (60000, 794)
print(f"Input dimension: {INPUT_DIM}")
```

---

# PHASE 4: Build the Model

A CVAE has three parts:
1. **Encoder** -- compresses an image into a tiny "recipe" (latent vector)
2. **Sampling** -- adds controlled randomness to the recipe
3. **Decoder** -- takes the recipe and generates a new image

### Cell 5: Define the Sampling Layer

```python
# ============================================================
# CELL 5: THE SAMPLING LAYER
# What this cell does: Implements the "reparameterization trick"
# ============================================================

from tensorflow.keras import backend as K

# The CVAE learns TWO things about each image:
# 1. z_mean: the "average recipe" for this type of image
# 2. z_log_var: how much variation is allowed
#
# The Sampling layer combines these with random noise to create
# a unique "recipe" each time. This is the reparameterization trick.
#
# Formula: z = z_mean + exp(0.5 * z_log_var) * random_noise
#
# Plain English: "Take the average recipe, and add a little random twist."

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        # Get the shape of z_mean (batch_size, latent_dim).
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Generate random noise from a normal distribution (mean=0, std=1).
        epsilon = tf.random.normal(shape=(batch, dim))
        
        # Apply the reparameterization trick.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

print("Sampling layer defined!")
```

### Cell 6: Build Encoder, Decoder, and Full CVAE

```python
# ============================================================
# CELL 6: BUILD THE CVAE
# What this cell does: Creates the encoder, decoder, and training model
# ============================================================

LATENT_DIM = 8  # The "recipe" is just 8 numbers (very compressed!)

# ---- ENCODER ----
# Takes a conditioned image (794 numbers) and compresses it to 8 numbers.

encoder_input = keras.Input(shape=(INPUT_DIM,), name='encoder_input')
x = keras.layers.Dense(256, activation='relu')(encoder_input)
x = keras.layers.Dense(128, activation='relu')(x)

# Two outputs: the mean and variance of the latent space.
z_mean = keras.layers.Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = keras.layers.Dense(LATENT_DIM, name='z_log_var')(x)

# Sample from the latent space using the reparameterization trick.
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
print("Encoder built!")
encoder.summary()

# ---- DECODER ----
# Takes 8 latent numbers + 10 label numbers = 18 numbers, and generates 784 pixels.

decoder_input = keras.Input(shape=(LATENT_DIM + NUM_CLASSES,), name='decoder_input')
x = keras.layers.Dense(128, activation='relu')(decoder_input)
x = keras.layers.Dense(256, activation='relu')(x)
decoder_output = keras.layers.Dense(IMG_SIZE, activation='sigmoid')(x)

decoder = keras.Model(decoder_input, decoder_output, name='decoder')
print("\nDecoder built!")
decoder.summary()
```

### Cell 7: Custom Training Step

```python
# ============================================================
# CELL 7: CREATE THE CVAE MODEL WITH CUSTOM LOSS
# What this cell does: Combines encoder + decoder + special loss function
# ============================================================

class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]
    
    def train_step(self, data):
        # data = the conditioned input (image + label, 794 numbers)
        with tf.GradientTape() as tape:
            # Encode: compress the conditioned image.
            z_mean, z_log_var, z = self.encoder(data)
            
            # Extract the label from the input (last 10 numbers).
            labels = data[:, IMG_SIZE:]
            
            # Concatenate z (8 numbers) with label (10 numbers) = 18 numbers.
            z_cond = tf.concat([z, labels], axis=1)
            
            # Decode: generate the image from the recipe + label.
            reconstruction = self.decoder(z_cond)
            
            # Extract just the image pixels from the input (first 784 numbers).
            original_image = data[:, :IMG_SIZE]
            
            # RECONSTRUCTION LOSS: How different is the generated image from the original?
            # Binary crossentropy measures pixel-by-pixel difference.
            recon_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(original_image, reconstruction)
            ) * IMG_SIZE
            
            # KL DIVERGENCE LOSS: How far is the latent space from a normal distribution?
            # This keeps the latent space organized and smooth.
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            total_loss = recon_loss + kl_loss
        
        # Calculate gradients and update weights.
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Track losses.
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'total_loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }

# Build the CVAE.
cvae = CVAE(encoder, decoder)
cvae.compile(optimizer='adam')

print("CVAE model ready!")
```

---

# PHASE 5: Train the Model

### Cell 8: Train

```python
# ============================================================
# CELL 8: TRAIN THE CVAE
# What this cell does: The model learns to compress and regenerate images
# ============================================================

history = cvae.fit(
    X_train_cond,
    epochs=30,
    batch_size=128,
    verbose=1
)

print("\nTraining complete!")
```

### Cell 9: Plot Training Loss

```python
# ============================================================
# CELL 9: PLOT TRAINING LOSS
# What this cell does: Shows how the loss decreased
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history.history['total_loss'], linewidth=2)
axes[0].set_title('Total Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['recon_loss'], linewidth=2, color='orange')
axes[1].set_title('Reconstruction Loss')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

axes[2].plot(history.history['kl_loss'], linewidth=2, color='green')
axes[2].set_title('KL Divergence Loss')
axes[2].set_xlabel('Epoch')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

# PHASE 6: Test the Model

### Cell 10: Generate Specific Clothing Items

```python
# ============================================================
# CELL 10: GENERATE CLOTHING BY CATEGORY
# What this cell does: Creates new clothing items on command
# ============================================================

def generate_items(category_idx, num_items=10):
    """Generate new items from a specific category."""
    
    # Create a one-hot label for the requested category.
    label = np.zeros((num_items, NUM_CLASSES))
    label[:, category_idx] = 1.0
    
    # Sample random "recipes" from a normal distribution.
    z_random = np.random.normal(size=(num_items, LATENT_DIM))
    
    # Concatenate: recipe (8) + label (10) = 18 numbers.
    decoder_input = np.concatenate([z_random, label], axis=1).astype('float32')
    
    # Generate images.
    generated = decoder.predict(decoder_input, verbose=0)
    
    return generated.reshape(num_items, 28, 28)

# Generate and display 10 items from each category.
fig, axes = plt.subplots(10, 10, figsize=(14, 14))

for row in range(10):
    items = generate_items(row, 10)
    for col in range(10):
        axes[row, col].imshow(items[col], cmap='gray')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(CLASS_NAMES[row], fontsize=8, rotation=0, 
                                       labelpad=60)

plt.suptitle("Generated Fashion Items (Each Row = One Category)", fontsize=14)
plt.tight_layout()
plt.show()

print("Each image is a BRAND NEW design that never existed in the training data!")
```

### Cell 11: Latent Space Exploration

```python
# ============================================================
# CELL 11: EXPLORE THE LATENT SPACE
# What this cell does: Shows smooth transitions between designs
# ============================================================

# Generate a row of sneakers where we smoothly change one latent dimension.
fig, axes = plt.subplots(2, 10, figsize=(14, 3))

for row, category in enumerate([7, 3]):  # Sneaker, Dress
    label = np.zeros((1, NUM_CLASSES))
    label[0, category] = 1.0
    
    for col, val in enumerate(np.linspace(-3, 3, 10)):
        z = np.zeros((1, LATENT_DIM))
        z[0, 0] = val  # Vary the first latent dimension.
        
        decoder_input = np.concatenate([z, label], axis=1).astype('float32')
        img = decoder.predict(decoder_input, verbose=0).reshape(28, 28)
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(CLASS_NAMES[category], fontsize=9)

plt.suptitle("Smooth Transitions (Varying Latent Dimension 1)", fontsize=12)
plt.tight_layout()
plt.show()

print("The latent space is smooth -- nearby points produce similar designs!")
```

---

# PHASE 7: Save the Model

### Cell 12: Save Encoder and Decoder Separately

```python
# ============================================================
# CELL 12: SAVE THE MODELS
# What this cell does: Saves the decoder (we only need it for generation)
# ============================================================

import json

# For deployment, we only need the DECODER.
# The decoder takes a latent vector + label and generates an image.
decoder.save('fashion_decoder.keras')

# Save config.
config = {
    'class_names': CLASS_NAMES,
    'num_classes': NUM_CLASSES,
    'latent_dim': LATENT_DIM,
    'img_size': IMG_SIZE
}

with open('model_config.json', 'w') as f:
    json.dump(config, f)

print("Saved:")
print("  1. fashion_decoder.keras")
print("  2. model_config.json")
print("\nDownload BOTH files from the Output panel.")
```

---

# PHASE 8: Deploy to Hugging Face

## Step 8.1: Download from Kaggle

Download `fashion_decoder.keras` and `model_config.json`.

## Step 8.2: Create Hugging Face Space

1. [huggingface.co](https://huggingface.co) -> **New Space**.
2. Name: `ai-fashion-designer`
3. SDK: **Gradio**
4. Hardware: **CPU Basic**
5. **Create Space**.

## Step 8.3: Upload Files

Upload `fashion_decoder.keras` and `model_config.json`.

## Step 8.4: Create `requirements.txt`

```
tensorflow
gradio
numpy
Pillow
spaces
```

## Step 8.5: Create `app.py`

```python
# ============================================================
# app.py -- AI Fashion Designer Web App
# Select a clothing category and the AI generates new designs.
# ============================================================

import os
# CRITICAL: Force TensorFlow to CPU-only BEFORE importing it.
# Hugging Face's ZeroGPU injects CUDA libraries that conflict with TF.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gradio as gr
import numpy as np
import json
from PIL import Image
import spaces

# Import TensorFlow AFTER disabling CUDA.
import tensorflow as tf

# -----------------------------------------------
# STEP 1: Load model and config
# -----------------------------------------------
decoder = tf.keras.models.load_model('fashion_decoder.keras')

# Dummy function to satisfy Hugging Face's ZeroGPU requirement.
# DO NOT put @spaces.GPU on the actual predict function.
@spaces.GPU
def dummy_gpu():
    pass

with open('model_config.json', 'r') as f:
    config = json.load(f)

CLASS_NAMES = config['class_names']
NUM_CLASSES = config['num_classes']
LATENT_DIM = config['latent_dim']

# -----------------------------------------------
# STEP 2: Generation function
# -----------------------------------------------
def generate_fashion(category, num_designs):
    """Generate new clothing designs for the selected category."""
    
    num_designs = int(num_designs)
    category_idx = CLASS_NAMES.index(category)
    
    # Create one-hot label.
    label = np.zeros((num_designs, NUM_CLASSES))
    label[:, category_idx] = 1.0
    
    # Random latent vectors.
    z = np.random.normal(size=(num_designs, LATENT_DIM))
    
    # Concatenate and generate.
    decoder_input = np.concatenate([z, label], axis=1).astype('float32')
    generated = decoder.predict(decoder_input, verbose=0)
    
    # Convert to images.
    images = []
    for i in range(num_designs):
        img_array = (generated[i].reshape(28, 28) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img = img.resize((112, 112), Image.NEAREST)
        images.append(img)
    
    return images

# -----------------------------------------------
# STEP 3: Gradio interface
# -----------------------------------------------
demo = gr.Interface(
    fn=generate_fashion,
    inputs=[
        gr.Dropdown(choices=CLASS_NAMES, value="Sneaker",
                    label="Select Clothing Category"),
        gr.Slider(minimum=1, maximum=9, value=4, step=1,
                  label="Number of Designs")
    ],
    outputs=gr.Gallery(label="Generated Designs", columns=3, height=400),
    title="AI Fashion Designer (CVAE)",
    description="Select a clothing category and the AI will generate brand new designs "
                "that have never existed before. Powered by a Conditional Variational Autoencoder "
                "trained on Fashion-MNIST.",
)

demo.launch()
```

Commit and wait for the build.

---

# PHASE 9: Test It Live

## Test 1: Generate Every Category

Select each clothing type from the dropdown and generate 4 designs. See how the AI creates different variations each time.

## Test 2: Generate Multiple Times

Click "Submit" multiple times with the same category. Each time you get different designs because the random latent vector changes.

## Test 3: Share

Send the URL to friends and let them generate their own fashion designs!

---

# Troubleshooting

### Problem: Generated images are blurry
**Solution:** This is normal for VAEs. They produce smooth, sometimes blurry images. For sharper images, GANs are better (see Lab 05). Training for more epochs (50+) can help slightly.

### Problem: All generated images look the same
**Solution:** The model may not have trained enough. Increase epochs to 50. Also check that the KL loss is not too high (it should decrease during training).

### Problem: "ValueError: shapes not compatible"
**Solution:** Check that LATENT_DIM and NUM_CLASSES in `app.py` match the values from training. They should be 8 and 10 respectively.

---

## What You Learned

| Concept | Where You Used It |
|---------|------------------|
| VAE (Session 22) | Encoder-decoder architecture with latent space |
| Reparameterization Trick (Session 22) | The Sampling layer |
| KL Divergence (Session 22) | Part of the loss function |
| Conditional Generation (Session 26) | Concatenating one-hot labels |
| CVAE (Session 26-27) | The complete conditional generation pipeline |
| Custom Training (Session 9) | The `train_step` override |

---
*Lab Guide 04 | Deep Learning Using Neural Networks | Aptech*
