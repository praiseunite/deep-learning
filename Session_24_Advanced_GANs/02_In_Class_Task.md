# Session 24 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Controlling the Generator"
### Duration: 45-55 minutes

---

> **Instructor Note:** Students will build a Conditional GAN (cGAN) that generates specific digits on command. This is a significant step up from Session 23's basic GAN -- the Generator now takes both noise AND a label as input.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Modify a GAN architecture to accept conditional labels.
- [ ] Train a cGAN that generates specific digit classes on demand.
- [ ] Demonstrate controlled generation by requesting specific digits.

---

## PART 1 -- Conceptual Warm-Up (5 minutes)

Answer these questions before coding:

1. In a basic GAN, the Generator's input is just random noise. In a cGAN, what *additional* input does the Generator receive?
*Answer:* __________________________________________________________________

2. If we are generating MNIST digits (10 classes), and our noise vector is 100 numbers, what is the total input size to the cGAN Generator?
*Answer:* __________________________________________________________________

3. Why does the Discriminator also need to receive the label? (Hint: Think about what happens if the Generator produces a perfect "3" but the label said "7.")
*Answer:* __________________________________________________________________

---

## PART 2 -- Kaggle Lab: Build a Conditional GAN (40 minutes)

Open your Kaggle Notebook with the GPU enabled.

### Cell 1: Imports and Data
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("Loading MNIST...")
(X_train, y_train), (_, _) = keras.datasets.mnist.load_data()
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = X_train.reshape(-1, 784)

NUM_CLASSES = 10
NOISE_DIM = 100
print(f"Data: {X_train.shape[0]} images, {NUM_CLASSES} classes")
print("Ready!")
```

### Cell 2: Build the Conditional Generator
Notice how we concatenate the label to the noise input.

```python
def build_cgan_generator():
    # Noise input
    noise_input = keras.Input(shape=(NOISE_DIM,), name='noise')
    # Label input (one-hot encoded)
    label_input = keras.Input(shape=(NUM_CLASSES,), name='label')
    
    # Concatenate noise + label
    combined = layers.Concatenate()([noise_input, label_input])
    
    x = layers.Dense(256)(combined)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    output = layers.Dense(784, activation='tanh')(x)
    
    model = keras.Model([noise_input, label_input], output, name='cgan_generator')
    return model

cgan_generator = build_cgan_generator()
print("Conditional Generator built!")
print(f"Inputs: noise({NOISE_DIM}) + label({NUM_CLASSES}) = {NOISE_DIM + NUM_CLASSES} total")
```

### Cell 3: Build the Conditional Discriminator
The Discriminator also receives the label so it can judge "is this a convincing '7'?"

```python
def build_cgan_discriminator():
    # Image input
    image_input = keras.Input(shape=(784,), name='image')
    # Label input
    label_input = keras.Input(shape=(NUM_CLASSES,), name='label')
    
    # Concatenate image + label
    combined = layers.Concatenate()([image_input, label_input])
    
    x = layers.Dense(1024)(combined)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([image_input, label_input], output, name='cgan_discriminator')
    return model

cgan_discriminator = build_cgan_discriminator()
cgan_discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Conditional Discriminator built!")
```

### Cell 4: Assemble the Combined cGAN
```python
cgan_discriminator.trainable = False

noise_in = keras.Input(shape=(NOISE_DIM,))
label_in = keras.Input(shape=(NUM_CLASSES,))
fake_image = cgan_generator([noise_in, label_in])
validity = cgan_discriminator([fake_image, label_in])

cgan = keras.Model([noise_in, label_in], validity, name='cgan')
cgan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')
print("cGAN assembled!")
```

### Cell 5: Train the cGAN
```python
EPOCHS = 150
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    # --- Phase 1: Train Discriminator ---
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_images = X_train[idx]
    real_labels = keras.utils.to_categorical(y_train[idx], NUM_CLASSES)
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    fake_labels = keras.utils.to_categorical(
        np.random.randint(0, NUM_CLASSES, BATCH_SIZE), NUM_CLASSES
    )
    fake_images = cgan_generator.predict([noise, fake_labels], verbose=0)
    
    d_loss_real = cgan_discriminator.train_on_batch(
        [real_images, real_labels], np.ones((BATCH_SIZE, 1))
    )
    d_loss_fake = cgan_discriminator.train_on_batch(
        [fake_images, fake_labels], np.zeros((BATCH_SIZE, 1))
    )
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # --- Phase 2: Train Generator ---
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    target_labels = keras.utils.to_categorical(
        np.random.randint(0, NUM_CLASSES, BATCH_SIZE), NUM_CLASSES
    )
    g_loss = cgan.train_on_batch([noise, target_labels], np.ones((BATCH_SIZE, 1)))
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:>4d} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.1f}% | G Loss: {g_loss:.4f}")

print("\nTraining complete!")
```

### Cell 6: Generate Specific Digits On Command!
This is the payoff -- we tell the Generator EXACTLY which digit to produce.

```python
fig, axes = plt.subplots(10, 10, figsize=(12, 12))

for digit in range(10):
    noise = np.random.normal(0, 1, (10, NOISE_DIM))
    label = keras.utils.to_categorical([digit] * 10, NUM_CLASSES)
    generated = cgan_generator.predict([noise, label], verbose=0)
    
    for j in range(10):
        axes[digit, j].imshow(generated[j].reshape(28, 28), cmap='gray')
        axes[digit, j].axis('off')
    axes[digit, 0].set_ylabel(f"Digit {digit}", fontsize=10, rotation=0, labelpad=40)

plt.suptitle("Conditional GAN: Each Row is a Specific Requested Digit", fontsize=14)
plt.tight_layout()
plt.show()
print("Each row shows 10 variations of the SAME digit -- all generated on command!")
```

---

## PART 3 -- Reflection (5 minutes)

1. Look at the 10x10 grid. Does each row consistently show the correct digit? Are some digits better than others?
*Answer:* __________________________________________________________________

2. Within a single row (e.g., all 7s), the images look different from each other even though they are all "7". What causes this variation?
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: All 3 warm-up questions answered
- [ ] Part 2: All 6 code cells executed successfully
- [ ] Part 2: Cell 6 shows a 10x10 grid of conditionally generated digits
- [ ] Part 3: Both reflection questions answered

---
*Session 24 | Deep Learning Using Neural Networks | Aptech*
