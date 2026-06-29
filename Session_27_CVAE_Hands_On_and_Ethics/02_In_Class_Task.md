# Session 27 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Ethics Debate + Multi-Condition Build"
### Duration: 45-55 minutes

---

> **Instructor Note:** This task is split evenly between an ethics discussion (no code) and a coding exercise. The ethics portion is critical -- students must internalize these concepts before they have the skills to potentially cause harm.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Articulate ethical risks of generative AI in their own words.
- [ ] Propose responsible safeguards for a generative AI product.
- [ ] Build a multi-condition CVAE with digit class AND thickness control.

---

## PART 1 -- Ethics Debate (20 minutes)

### Instructions
Split into groups of 3-4. Each group will discuss ONE scenario for 10 minutes, then present their conclusion to the class in 2 minutes.

### Scenario A: The Portrait Studio
A photo studio wants to use your CVAE to generate "family photos" for customers who cannot physically gather (e.g., a grandparent in another country). They will combine real faces of family members into a single generated image.

**Discuss:**
1. What are the benefits of this application?
2. What could go wrong? List at least 3 potential harms.
3. What safeguards would you require before agreeing to build this?

*Group Notes:* __________________________________________________________________
__________________________________________________________________
__________________________________________________________________

### Scenario B: The Medical Dataset
A hospital has only 50 brain scan images of a rare tumor type. They want you to use a CVAE to generate 5,000 synthetic brain scans to train a diagnostic AI.

**Discuss:**
1. Why is the hospital doing this? What problem does it solve?
2. What happens if the synthetic scans are subtly unrealistic in ways that humans cannot detect, but that train the diagnostic AI to make wrong predictions?
3. How would you validate that the synthetic data is safe to use?

*Group Notes:* __________________________________________________________________
__________________________________________________________________
__________________________________________________________________

### Scenario C: The Fashion Startup
A startup wants to use a GAN trained on images scraped from Instagram to generate fashion designs and sell them as "AI-designed" clothing.

**Discuss:**
1. Whose creative work was used to train the model? Were they credited or compensated?
2. If the GAN produces a design that closely resembles an existing designer's work, who owns it?
3. What ethical framework would you apply? (Refer to the DARE framework from the lecture.)

*Group Notes:* __________________________________________________________________
__________________________________________________________________
__________________________________________________________________

---

## PART 2 -- Kaggle Lab: Multi-Condition CVAE (25 minutes)

Open your Kaggle Notebook with the GPU enabled. We will extend the Session 26 CVAE to condition on BOTH digit class AND thickness.

### Cell 1: Load Data and Compute Thickness
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)

NUM_CLASSES = 10
LATENT_DIM = 2

# One-hot labels
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Compute "thickness" as average pixel intensity per image
thickness_train = X_train_flat.mean(axis=1, keepdims=True)  # Shape: (60000, 1)
thickness_test = X_test_flat.mean(axis=1, keepdims=True)

# Combine into full condition: [one-hot label (10) + thickness (1)] = 11
cond_train = np.concatenate([y_train_oh, thickness_train], axis=1)
cond_test = np.concatenate([y_test_oh, thickness_test], axis=1)
COND_DIM = cond_train.shape[1]  # 11

print(f"Condition vector size: {COND_DIM} (10 class + 1 thickness)")
print(f"Example thickness values: {thickness_train[:5].flatten()}")
print("Ready!")
```

### Cell 2: Build Multi-Condition CVAE
```python
# Encoder
img_in = keras.Input(shape=(784,))
cond_in = keras.Input(shape=(COND_DIM,))
x = layers.Concatenate()([img_in, cond_in])
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)

def sampling(args):
    mu, lv = args
    return mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))

z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model([img_in, cond_in], [z_mean, z_log_var, z])

# Decoder
z_in = keras.Input(shape=(LATENT_DIM,))
cond_in_dec = keras.Input(shape=(COND_DIM,))
x = layers.Concatenate()([z_in, cond_in_dec])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
dec_out = layers.Dense(784, activation='sigmoid')(x)
decoder = keras.Model([z_in, cond_in_dec], dec_out)

# CVAE
class CVAE(keras.Model):
    def __init__(self, enc, dec, **kw):
        super().__init__(**kw)
        self.enc, self.dec = enc, dec
        self.loss_tracker = keras.metrics.Mean(name="loss")
    @property
    def metrics(self):
        return [self.loss_tracker]
    def train_step(self, data):
        imgs, conds = data
        with tf.GradientTape() as tape:
            zm, zlv, z = self.enc([imgs, conds])
            recon = self.dec([z, conds])
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
ds = tf.data.Dataset.from_tensor_slices((X_train_flat, cond_train)).shuffle(60000).batch(128)

print("Training Multi-Condition CVAE...")
cvae.fit(ds, epochs=20, verbose=1)
print("Done!")
```

### Cell 3: Generate Digits with Controlled Thickness!
```python
fig, axes = plt.subplots(3, 10, figsize=(14, 5))
thickness_levels = [0.05, 0.15, 0.30]  # thin, medium, thick
thickness_names = ['Thin', 'Medium', 'Thick']

for row, (thick, name) in enumerate(zip(thickness_levels, thickness_names)):
    for digit in range(10):
        # Build condition: digit class + thickness
        label_oh = np.zeros(NUM_CLASSES)
        label_oh[digit] = 1.0
        condition = np.concatenate([label_oh, [thick]])
        condition = condition.reshape(1, -1)
        
        z_sample = np.random.normal(0, 1, (1, LATENT_DIM))
        generated = decoder.predict([z_sample, condition], verbose=0)
        
        axes[row, digit].imshow(generated[0].reshape(28, 28), cmap='gray')
        axes[row, digit].axis('off')
        if digit == 0:
            axes[row, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=45)
        if row == 0:
            axes[0, digit].set_title(str(digit), fontsize=12)

plt.suptitle("Multi-Condition CVAE: Each Column = Digit, Each Row = Thickness", fontsize=13)
plt.tight_layout()
plt.show()
print("You are now controlling TWO attributes simultaneously!")
```

---

## PART 3 -- Reflection (5 minutes)

1. Look at the thickness-controlled grid. Can you see a difference between "Thin" and "Thick" rows? Is the effect subtle or dramatic?
*Answer:* __________________________________________________________________

2. After today's ethics discussion, describe one safeguard you would add to a generative AI product before releasing it publicly.
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: Group ethics discussion completed, notes recorded
- [ ] Part 2: All 3 code cells executed successfully
- [ ] Part 2: Multi-condition grid shows thickness variation across rows
- [ ] Part 3: Both reflection questions answered

---
*Session 27 | Deep Learning Using Neural Networks | Aptech*
