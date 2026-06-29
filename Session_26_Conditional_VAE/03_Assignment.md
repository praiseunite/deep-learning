# Session 26 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Fashion Designer AI"
### Due: Before Session 27 begins
### Estimated Time: 45 minutes

---

> **Professor's Note:** *"In class, you built a CVAE that generates digits on command. Now, adapt it to Fashion-MNIST. Your AI will become a fashion designer that creates T-shirts, sneakers, or bags whenever you ask."*

---

## Assignment Objectives

By completing this assignment, you will:
- Adapt the CVAE from digits to clothing items.
- Generate specific clothing categories on command.
- Explore within-class variation by sampling different latent codes for the same category.

---

## THE SCENARIO

You are building a "Fashion Idea Generator" app. A user selects a clothing type (T-shirt, Sneaker, Bag, etc.), and the AI instantly generates 10 different design variations. Your job is to build the CVAE that powers this app.

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

CATEGORIES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
NUM_CLASSES = 10
LATENT_DIM = 2

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Categories: {CATEGORIES}")
print("Data loaded!")
```

### Cell 2: Build and Train the Fashion CVAE
Adapt the CVAE code from the in-class task.

```python
# Encoder
img_in = keras.Input(shape=(784,))
lbl_in = keras.Input(shape=(NUM_CLASSES,))
x = layers.Concatenate()([img_in, lbl_in])
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)

def sampling(args):
    mu, lv = args
    return mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))

z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model([img_in, lbl_in], [z_mean, z_log_var, z])

# Decoder
z_in = keras.Input(shape=(LATENT_DIM,))
lbl_in_dec = keras.Input(shape=(NUM_CLASSES,))
x = layers.Concatenate()([z_in, lbl_in_dec])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
dec_out = layers.Dense(784, activation='sigmoid')(x)
decoder = keras.Model([z_in, lbl_in_dec], dec_out)

# CVAE class
class CVAE(keras.Model):
    def __init__(self, enc, dec, **kw):
        super().__init__(**kw)
        self.enc, self.dec = enc, dec
        self.loss_tracker = keras.metrics.Mean(name="loss")
    @property
    def metrics(self):
        return [self.loss_tracker]
    def train_step(self, data):
        imgs, labels = data
        with tf.GradientTape() as tape:
            zm, zlv, z = self.enc([imgs, labels])
            recon = self.dec([z, labels])
            r_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(tf.reshape(imgs, (-1, 784)), recon), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + zlv - tf.square(zm) - tf.exp(zlv), axis=1))
            loss = r_loss + kl
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_weights), self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

cvae = CVAE(encoder, decoder)
cvae.compile(optimizer='adam')

ds = tf.data.Dataset.from_tensor_slices((X_train_flat, y_train_oh)).shuffle(60000).batch(128)
print("Training Fashion CVAE...")
cvae.fit(ds, epochs=25, verbose=1)
print("Done!")
```

### Cell 3: Generate Specific Clothing On Command
```python
fig, axes = plt.subplots(10, 10, figsize=(14, 14))

for cat in range(10):
    label = keras.utils.to_categorical([cat] * 10, NUM_CLASSES)
    z_sample = np.random.normal(0, 1, (10, LATENT_DIM))
    generated = decoder.predict([z_sample, label], verbose=0)
    
    for j in range(10):
        axes[cat, j].imshow(generated[j].reshape(28, 28), cmap='gray')
        axes[cat, j].axis('off')
    axes[cat, 0].set_ylabel(CATEGORIES[cat], fontsize=9, rotation=0, labelpad=55)

plt.suptitle("Fashion CVAE: Each Row = A Different Clothing Type", fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 4: Within-Class Exploration
Generate 20 variations of a single category to see the diversity.

```python
# Generate 20 sneakers
target_class = 7  # Sneaker
label = keras.utils.to_categorical([target_class] * 20, NUM_CLASSES)
z_sample = np.random.normal(0, 1, (20, LATENT_DIM))
sneakers = decoder.predict([z_sample, label], verbose=0)

plt.figure(figsize=(14, 3))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(sneakers[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle(f"20 AI-Generated {CATEGORIES[target_class]}s -- All Different!", fontsize=13)
plt.tight_layout()
plt.show()
```

### Cell 5: Latent Space Visualization
```python
z_means, _, _ = encoder.predict([X_test_flat, y_test_oh])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_means[:, 0], z_means[:, 1], c=y_test, cmap='tab10', alpha=0.4, s=2)
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.ax.set_yticklabels(CATEGORIES, fontsize=7)
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Fashion CVAE Latent Space")
plt.tight_layout()
plt.show()
```

---

## Written Questions

1. Look at the 10x10 grid (Cell 3). Which clothing categories look the most realistic? Which look the most blurry or confused? Why do you think some categories are harder?

2. Look at the 20 sneakers in Cell 4. Describe the variation you see. Do they all look like sneakers, or are some ambiguous?

3. If you were building this for a real fashion app, what would you change to improve the quality? (Hint: think about latent dimensions, model size, and dataset.)

---

## Submission Requirements

1. Screenshot of the 10x10 clothing grid (Cell 3).
2. Screenshot of the 20 sneakers (Cell 4).
3. Screenshot of the latent space (Cell 5).
4. Written answers to the 3 questions.

---
*Session 26 | Deep Learning Using Neural Networks | Aptech*
