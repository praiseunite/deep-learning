# Session 23 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Art Forger vs The Detective"
### Duration: 45-55 minutes

---

> **Instructor Note:** This task has two parts: a conceptual exercise to cement the GAN architecture, followed by a hands-on Kaggle lab where students build and train their first GAN. The code is intentionally simple (Dense layers, not CNNs) so students can focus on the adversarial training loop.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Draw and label the complete GAN data flow.
- [ ] Explain the role of the Generator and Discriminator.
- [ ] Train a GAN on MNIST and observe generated images improving over epochs.
- [ ] Identify signs of training progress in the loss values.

---

## PART 1 -- Conceptual Exercise: Draw the GAN (10 minutes)

On a piece of paper, draw the complete GAN architecture and label these components:

1. **Random Noise Input** (vector of 100 random numbers)
2. **Generator Network** (with at least 2 Dense layers)
3. **Fake Image Output** (28x28 pixels)
4. **Real Images** (from dataset)
5. **Discriminator Network** (with at least 2 Dense layers)
6. **Final Output** (single number: 0 = fake, 1 = real)
7. Draw arrows showing:
   - Where the fake images go
   - Where the real images go
   - Which network gets updated during "Phase 1"
   - Which network gets updated during "Phase 2"

### Discussion Question:
The Generator never sees real images directly. How does it learn what real images look like?

*Answer:* __________________________________________________________________

---

## PART 2 -- Kaggle Lab: Build Your First GAN (35 minutes)

Open your Kaggle Notebook with the GPU enabled.

### Cell 1: Imports and Data
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("Loading MNIST...")
(X_train, _), (_, _) = keras.datasets.mnist.load_data()

# Normalize to [-1, 1] range (important for GANs!)
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = X_train.reshape(-1, 784)  # Flatten images

NOISE_DIM = 100  # Size of the random noise vector
print(f"Training data shape: {X_train.shape}")
print(f"Pixel range: [{X_train.min():.1f}, {X_train.max():.1f}]")
print("Data ready!")
```

### Cell 2: Build the Generator
The Generator turns random noise into fake images.

```python
def build_generator():
    model = keras.Sequential([
        layers.Dense(256, input_dim=NOISE_DIM),
        layers.LeakyReLU(0.2),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.Dense(784, activation='tanh')  # tanh outputs values in [-1, 1]
    ])
    return model

generator = build_generator()
print("Generator built!")
print(f"Input: {NOISE_DIM} random numbers -> Output: 784 pixel values (28x28 image)")
generator.summary()
```

### Cell 3: Build the Discriminator
The Discriminator judges whether an image is real or fake.

```python
def build_discriminator():
    model = keras.Sequential([
        layers.Dense(1024, input_dim=784),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation='sigmoid')  # sigmoid outputs 0 to 1
    ])
    return model

discriminator = build_discriminator()
discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Discriminator built and compiled!")
print(f"Input: 784 pixel values -> Output: 1 number (0=fake, 1=real)")
```

### Cell 4: Build the Combined GAN Model
We connect Generator -> Discriminator. When training the Generator, we freeze the Discriminator.

```python
# When training the Generator, the Discriminator should NOT update
discriminator.trainable = False

gan_input = keras.Input(shape=(NOISE_DIM,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = keras.Model(gan_input, gan_output)
gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)
print("GAN assembled: Generator -> Discriminator (frozen)")
print("When we train this model, ONLY the Generator's weights update!")
```

### Cell 5: The Training Loop
This is where the magic happens. We alternate between training the Discriminator and the Generator.

```python
EPOCHS = 100
BATCH_SIZE = 256
SAMPLE_INTERVAL = 10  # Save generated images every 10 epochs

# Storage for generated image snapshots
generated_snapshots = []

for epoch in range(EPOCHS):
    # ----- Phase 1: Train the Discriminator -----
    
    # Get a batch of real images
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]
    
    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    fake_images = generator.predict(noise, verbose=0)
    
    # Labels: real = 1, fake = 0
    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))
    
    # Train Discriminator on real and fake separately
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # ----- Phase 2: Train the Generator -----
    
    # New random noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    
    # We want the Generator to fool the Discriminator (label = 1)
    misleading_labels = np.ones((BATCH_SIZE, 1))
    
    # Train Generator (Discriminator weights are frozen)
    g_loss = gan.train_on_batch(noise, misleading_labels)
    
    # ----- Logging -----
    if epoch % SAMPLE_INTERVAL == 0:
        print(f"Epoch {epoch:>4d} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.1f}% | G Loss: {g_loss:.4f}")
        
        # Save snapshot of generated images
        sample_noise = np.random.normal(0, 1, (10, NOISE_DIM))
        sample_images = generator.predict(sample_noise, verbose=0)
        generated_snapshots.append((epoch, sample_images))

print("\nTraining complete!")
```

### Cell 6: Watch the Generator Improve Over Time
This shows how the generated images got better as training progressed.

```python
n_snapshots = len(generated_snapshots)
fig, axes = plt.subplots(n_snapshots, 10, figsize=(14, n_snapshots * 1.5))

for row, (epoch, images) in enumerate(generated_snapshots):
    for col in range(10):
        img = images[col].reshape(28, 28)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"Ep {epoch}", fontsize=8, rotation=0, labelpad=30)

plt.suptitle("GAN Training Progress: Generated Images Over Time", fontsize=13)
plt.tight_layout()
plt.show()
print("Top row = early training (random noise). Bottom row = later training (recognizable digits)!")
```

### Cell 7: Generate a Final Batch of New Digits
```python
noise = np.random.normal(0, 1, (30, NOISE_DIM))
final_generated = generator.predict(noise, verbose=0)

plt.figure(figsize=(15, 3))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(final_generated[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Final Generated Digits -- Created by YOUR GAN!", fontsize=13)
plt.tight_layout()
plt.show()
print("These digits were generated from pure random noise!")
```

---

## PART 3 -- Observation Questions (5 minutes)

1. Look at the training progress plot (Cell 6). At which epoch did the generated images start looking like recognizable digits?
*Answer:* __________________________________________________________________

2. The Discriminator accuracy started high and then dropped. Why does this happen during successful GAN training?
*Answer:* __________________________________________________________________

3. Compare these GAN-generated digits to the VAE-generated digits from Session 22. Which look sharper? Why do you think GANs produce crisper images?
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: GAN architecture drawn with all 7 components labeled
- [ ] Part 2: All 7 code cells executed successfully
- [ ] Part 2: Training progress plot shows improvement over epochs
- [ ] Part 2: Final generated digits are visible
- [ ] Part 3: All 3 observation questions answered

---
*Session 23 | Deep Learning Using Neural Networks | Aptech*
