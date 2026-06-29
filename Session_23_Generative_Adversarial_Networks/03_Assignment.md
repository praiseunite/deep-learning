# Session 23 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "My First GAN -- Fashion Edition"
### Due: Before Session 24 begins
### Estimated Time: 45 minutes

---

> **Professor's Note:** *"In class, you trained a GAN to generate handwritten digits. Now, adapt that same code to generate clothing items. This exercise proves that the GAN architecture is not tied to one specific dataset -- it can learn to generate anything."*

---

## Assignment Objectives

By completing this assignment, you will:
- Adapt a GAN from one dataset (MNIST digits) to another (Fashion-MNIST clothing).
- Compare the quality of GAN-generated clothing vs GAN-generated digits.
- Observe which fashion categories are easier or harder for the GAN to generate.

---

## THE SCENARIO

A fashion tech startup has hired you to prototype an AI that generates new clothing designs. You will adapt the MNIST GAN from class to work on Fashion-MNIST (T-shirts, trousers, sneakers, bags, etc.).

---

## Step-by-Step Instructions (Kaggle)

Open a **new** Kaggle Notebook with the GPU enabled.

### Cell 1: Load Fashion-MNIST
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

CATEGORY_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Loading Fashion-MNIST...")
(X_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = X_train.reshape(-1, 784)

NOISE_DIM = 100
print(f"Data shape: {X_train.shape}")
print("Ready!")
```

### Cell 2: Build Generator and Discriminator
Use the exact same architecture from the in-class task.

```python
def build_generator():
    model = keras.Sequential([
        layers.Dense(256, input_dim=NOISE_DIM),
        layers.LeakyReLU(0.2),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.Dense(784, activation='tanh')
    ])
    return model

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
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
gan_input = keras.Input(shape=(NOISE_DIM,))
gan = keras.Model(gan_input, discriminator(generator(gan_input)))
gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

print("Generator and Discriminator built!")
```

### Cell 3: Train the GAN
```python
EPOCHS = 200
BATCH_SIZE = 256
SAMPLE_INTERVAL = 25
generated_snapshots = []

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    fake_images = generator.predict(noise, verbose=0)
    
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((BATCH_SIZE, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((BATCH_SIZE, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))
    
    if epoch % SAMPLE_INTERVAL == 0:
        print(f"Epoch {epoch:>4d} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.1f}% | G Loss: {g_loss:.4f}")
        sample_noise = np.random.normal(0, 1, (10, NOISE_DIM))
        generated_snapshots.append((epoch, generator.predict(sample_noise, verbose=0)))

print("\nTraining complete!")
```

### Cell 4: Visualize Training Progress
```python
n = len(generated_snapshots)
fig, axes = plt.subplots(n, 10, figsize=(14, n * 1.5))
for row, (epoch, images) in enumerate(generated_snapshots):
    for col in range(10):
        axes[row, col].imshow(images[col].reshape(28, 28), cmap='gray')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"Ep {epoch}", fontsize=8, rotation=0, labelpad=30)
plt.suptitle("Fashion GAN: Training Progress", fontsize=13)
plt.tight_layout()
plt.show()
```

### Cell 5: Final Generated Fashion Items
```python
noise = np.random.normal(0, 1, (30, NOISE_DIM))
final = generator.predict(noise, verbose=0)

plt.figure(figsize=(15, 3))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(final[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("GAN-Generated Fashion Items", fontsize=13)
plt.tight_layout()
plt.show()
print("Can you identify what types of clothing the GAN learned to create?")
```

---

## Written Questions

1. Compare the GAN-generated fashion items to the GAN-generated digits from class. Which dataset seems harder for the GAN? Explain why in 2-3 sentences.

2. Look at your final generated images. Can you identify any specific clothing types (T-shirt, trouser, shoe, bag)? List which ones you can recognize.

3. Fashion has more complex shapes and textures than simple digits. What do you think would happen if we trained for 1000 epochs instead of 200? Would the images keep improving forever, or would we hit a limit?

---

## Submission Requirements

1. Screenshot of your training progress plot (Cell 4).
2. Screenshot of your final generated fashion items (Cell 5).
3. Written answers to the 3 questions above.

---
*Session 23 | Deep Learning Using Neural Networks | Aptech*
