# Session 24 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "DCGAN on Fashion"
### Due: Before Session 25 begins
### Estimated Time: 50 minutes

---

> **Professor's Note:** *"In class, we added conditional labels to a Dense-layer GAN. For homework, you will take a different upgrade path: replacing Dense layers with Convolutional layers. This is the DCGAN architecture, and it produces dramatically sharper images."*

---

## Assignment Objectives

By completing this assignment, you will:
- Build a DCGAN with Conv2DTranspose (Generator) and Conv2D (Discriminator) layers.
- Train it on Fashion-MNIST and compare the quality to your Session 23 Dense-layer GAN.
- Understand why convolutions produce better image generation results.

---

## THE SCENARIO

The fashion startup from Session 23 was impressed by your GAN prototype but said the generated images were "too blurry." Your boss wants you to upgrade to a DCGAN architecture to produce sharper outputs.

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

print("Loading Fashion-MNIST...")
(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (60000, 28, 28, 1)

NOISE_DIM = 100
print(f"Data shape: {X_train.shape}")
print("Ready!")
```

### Cell 2: Build the DCGAN Generator
This uses Conv2DTranspose layers to "grow" the image from a tiny 7x7 seed.

```python
def build_dcgan_generator():
    model = keras.Sequential(name='dcgan_generator')
    
    # Start: map noise to a small spatial tensor
    model.add(layers.Dense(7 * 7 * 256, input_dim=NOISE_DIM))
    model.add(layers.Reshape((7, 7, 256)))
    
    # Upsample: 7x7 -> 14x14
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Upsample: 14x14 -> 28x28
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Output: 28x28x1
    model.add(layers.Conv2DTranspose(1, kernel_size=5, strides=1, padding='same', activation='tanh'))
    
    return model

dc_generator = build_dcgan_generator()
print("DCGAN Generator built!")
dc_generator.summary()
```

### Cell 3: Build the DCGAN Discriminator
This uses regular Conv2D layers to shrink the image and classify it.

```python
def build_dcgan_discriminator():
    model = keras.Sequential(name='dcgan_discriminator')
    
    # Downsample: 28x28 -> 14x14
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    # Downsample: 14x14 -> 7x7
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    # Classify
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

dc_discriminator = build_dcgan_discriminator()
dc_discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("DCGAN Discriminator built!")
dc_discriminator.summary()
```

### Cell 4: Assemble and Train
```python
dc_discriminator.trainable = False
gan_input = keras.Input(shape=(NOISE_DIM,))
dcgan = keras.Model(gan_input, dc_discriminator(dc_generator(gan_input)))
dcgan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

EPOCHS = 200
BATCH_SIZE = 128
snapshots = []

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real = X_train[idx]
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    fake = dc_generator.predict(noise, verbose=0)
    
    d_real = dc_discriminator.train_on_batch(real, np.ones((BATCH_SIZE, 1)))
    d_fake = dc_discriminator.train_on_batch(fake, np.zeros((BATCH_SIZE, 1)))
    d_loss = 0.5 * np.add(d_real, d_fake)
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    g_loss = dcgan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:>4d} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")
        sample = dc_generator.predict(np.random.normal(0, 1, (10, NOISE_DIM)), verbose=0)
        snapshots.append((epoch, sample))

print("Training complete!")
```

### Cell 5: Training Progress
```python
n = len(snapshots)
fig, axes = plt.subplots(n, 10, figsize=(14, n * 1.5))
for row, (ep, imgs) in enumerate(snapshots):
    for col in range(10):
        axes[row, col].imshow(imgs[col].squeeze(), cmap='gray')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"Ep {ep}", fontsize=8, rotation=0, labelpad=30)
plt.suptitle("DCGAN Training Progress on Fashion-MNIST", fontsize=13)
plt.tight_layout()
plt.show()
```

### Cell 6: Final Generated Images
```python
noise = np.random.normal(0, 1, (30, NOISE_DIM))
final = dc_generator.predict(noise, verbose=0)

plt.figure(figsize=(15, 3))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(final[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle("DCGAN-Generated Fashion Items", fontsize=13)
plt.tight_layout()
plt.show()
```

---

## Written Questions

1. **Compare:** Look at your DCGAN-generated fashion items (Cell 6) and your Session 23 Dense-GAN results. Which produces sharper, more recognizable clothing? Why do convolutions help?

2. **Architecture:** The Generator uses `Conv2DTranspose` while the Discriminator uses `Conv2D`. In one sentence each, explain what each layer type does to the spatial dimensions of its input.

3. **Parameter Count:** Look at the `.summary()` output. How many total trainable parameters does your DCGAN Generator have? Compare this to the Dense Generator from Session 23 (which had roughly 1 million parameters). Which has more?

---

## Submission Requirements

1. Screenshot of training progress (Cell 5).
2. Screenshot of final generated items (Cell 6).
3. Written answers to the 3 questions.

---
*Session 24 | Deep Learning Using Neural Networks | Aptech*
