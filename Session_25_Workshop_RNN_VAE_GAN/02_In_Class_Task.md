# Session 25 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Triple Challenge"
### Duration: 1 hour 30 minutes

---

> **Instructor Note:** Students work through three independent challenges. Each one applies a different architecture from Sessions 20-24. Students who finish early can help classmates or attempt the bonus sections.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Generate text using a character-level RNN on custom data.
- [ ] Interpolate between points in a VAE's latent space.
- [ ] Analyze GAN training dynamics by studying loss curves.

---

## Challenge 1: RNN Text Generation on Custom Data (30 minutes)

In Session 21, we trained an RNN on a short string like "APTECH." Now, let's train on something more interesting.

### Your Task: Train an RNN on a Famous Quote

Open a new Kaggle Notebook (GPU not required for this one -- CPU is fine).

### Cell 1: Setup
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Our training text -- a famous quote repeated to give the network more data
TEXT = "to be or not to be that is the question " * 20

# Build vocabulary
chars = sorted(list(set(TEXT)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
VOCAB_SIZE = len(chars)

print(f"Text length: {len(TEXT)} characters")
print(f"Vocabulary: {chars}")
print(f"Vocabulary size: {VOCAB_SIZE}")
```

### Cell 2: Prepare Training Sequences
```python
SEQ_LENGTH = 10  # The RNN looks at 10 characters to predict the 11th
sequences = []
targets = []

for i in range(len(TEXT) - SEQ_LENGTH):
    seq = TEXT[i:i + SEQ_LENGTH]
    target = TEXT[i + SEQ_LENGTH]
    sequences.append([char_to_idx[c] for c in seq])
    targets.append(char_to_idx[target])

X = np.array(sequences)
y = np.array(targets)

# One-hot encode
X = tf.keras.utils.to_categorical(X, VOCAB_SIZE)
y = tf.keras.utils.to_categorical(y, VOCAB_SIZE)

print(f"Training sequences: {X.shape[0]}")
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Cell 3: Build and Train the RNN
```python
model = keras.Sequential([
    layers.SimpleRNN(128, input_shape=(SEQ_LENGTH, VOCAB_SIZE)),
    layers.Dense(VOCAB_SIZE, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training...")
model.fit(X, y, epochs=50, batch_size=64, verbose=1)
print("Training complete!")
```

### Cell 4: Generate Text!
```python
def generate_text(model, seed_text, length=100):
    result = seed_text
    current_seq = seed_text
    
    for _ in range(length):
        # Encode the current sequence
        x = np.array([[char_to_idx[c] for c in current_seq[-SEQ_LENGTH:]]])
        x = tf.keras.utils.to_categorical(x, VOCAB_SIZE)
        
        # Predict next character
        prediction = model.predict(x, verbose=0)[0]
        next_idx = np.random.choice(len(prediction), p=prediction)
        next_char = idx_to_char[next_idx]
        
        result += next_char
        current_seq += next_char
    
    return result

# Generate!
seed = "to be or n"
generated = generate_text(model, seed, length=80)
print(f"Seed: '{seed}'")
print(f"Generated: '{generated}'")
print("\nDoes it look like Shakespeare? Try running this cell again for different results!")
```

### Challenge 1 Questions:
1. Run Cell 4 three times. Write down the three different outputs. Why are they different each time?
*Answer:* __________________________________________________________________

2. What would happen if we changed `SEQ_LENGTH` from 10 to 3? Would the generated text be better or worse?
*Answer:* __________________________________________________________________

---

## Challenge 2: VAE Latent Space Interpolation (30 minutes)

One of the most fascinating things about VAEs is that you can "walk" between two points in the latent space and watch one image smoothly transform into another.

### Your Task: Interpolate Between Two Digits

Open a new Kaggle Notebook with the GPU enabled.

### Cell 1: Build and Train the VAE (Quick Version)
Copy this complete VAE (condensed from Session 22).

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_train = np.expand_dims(X_train, axis=-1)

LATENT_DIM = 2

# Encoder
enc_in = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(enc_in)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)

def sampling(args):
    mu, lv = args
    return mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))

z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(enc_in, [z_mean, z_log_var, z])

# Decoder
dec_in = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(256, activation='relu')(dec_in)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(784, activation='sigmoid')(x)
dec_out = layers.Reshape((28, 28, 1))(x)
decoder = keras.Model(dec_in, dec_out)

# VAE
class VAE(keras.Model):
    def __init__(self, enc, dec, **kw):
        super().__init__(**kw)
        self.enc = enc
        self.dec = dec
        self.loss_tracker = keras.metrics.Mean(name="loss")
    @property
    def metrics(self):
        return [self.loss_tracker]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            zm, zlv, z = self.enc(data)
            recon = self.dec(z)
            r_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(data, recon), axis=(1,2)))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + zlv - tf.square(zm) - tf.exp(zlv), axis=1))
            loss = r_loss + kl
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_weights), self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')
print("Training VAE...")
vae.fit(X_train, epochs=15, batch_size=128, verbose=1)
print("Done!")
```

### Cell 2: Find Two Digits in Latent Space
```python
# Encode all test images
z_means, _, _ = encoder.predict(X_test)

# Find a "3" and a "7"
idx_3 = np.where(y_test == 3)[0][0]
idx_7 = np.where(y_test == 7)[0][0]

point_a = z_means[idx_3]  # Latent coordinates of a "3"
point_b = z_means[idx_7]  # Latent coordinates of a "7"

print(f"Digit 3 is at latent coordinates: {point_a}")
print(f"Digit 7 is at latent coordinates: {point_b}")
```

### Cell 3: Interpolate!
```python
# Create 12 evenly spaced points between A and B
num_steps = 12
ratios = np.linspace(0, 1, num_steps)
interpolated_points = np.array([point_a * (1 - r) + point_b * r for r in ratios])

# Decode each interpolated point
interpolated_images = decoder.predict(interpolated_points)

# Display
plt.figure(figsize=(16, 2))
for i in range(num_steps):
    plt.subplot(1, num_steps, i + 1)
    plt.imshow(interpolated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title("3", fontsize=12, color='blue')
    elif i == num_steps - 1:
        plt.title("7", fontsize=12, color='red')
plt.suptitle("Smooth Interpolation: Digit 3 -> Digit 7", fontsize=14)
plt.tight_layout()
plt.show()
print("Watch the 3 smoothly transform into a 7!")
```

### Challenge 2 Questions:
1. Describe what happens in the middle images. Does the transition look smooth or abrupt?
*Answer:* __________________________________________________________________

2. Try changing the digits (e.g., interpolate between 1 and 8). Which pair produces the most interesting transition?
*Answer:* __________________________________________________________________

---

## Challenge 3: GAN Training Analysis (30 minutes)

Understanding GAN training dynamics is crucial. In this challenge, you will train a GAN and carefully study what the loss values tell you.

### Your Task: Train and Analyze a GAN

Open a new Kaggle Notebook with the GPU enabled.

### Cell 1: Train a GAN with Detailed Logging
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train, _), (_, _) = keras.datasets.mnist.load_data()
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = X_train.reshape(-1, 784)
NOISE_DIM = 100

# Generator
gen = keras.Sequential([
    layers.Dense(256, input_dim=NOISE_DIM), layers.LeakyReLU(0.2),
    layers.Dense(512), layers.LeakyReLU(0.2),
    layers.Dense(784, activation='tanh')
])

# Discriminator
disc = keras.Sequential([
    layers.Dense(512, input_dim=784), layers.LeakyReLU(0.2), layers.Dropout(0.3),
    layers.Dense(256), layers.LeakyReLU(0.2),
    layers.Dense(1, activation='sigmoid')
])
disc.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5),
             loss='binary_crossentropy', metrics=['accuracy'])

disc.trainable = False
gan_in = keras.Input(shape=(NOISE_DIM,))
gan = keras.Model(gan_in, disc(gen(gan_in)))
gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Training with FULL logging
EPOCHS = 200
d_losses, g_losses, d_accs = [], [], []

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_train.shape[0], 256)
    noise = np.random.normal(0, 1, (256, NOISE_DIM))
    fake = gen.predict(noise, verbose=0)
    
    dr = disc.train_on_batch(X_train[idx], np.ones((256, 1)))
    df = disc.train_on_batch(fake, np.zeros((256, 1)))
    d_loss = 0.5 * np.add(dr, df)
    
    noise = np.random.normal(0, 1, (256, NOISE_DIM))
    g_loss = gan.train_on_batch(noise, np.ones((256, 1)))
    
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    d_accs.append(d_loss[1] * 100)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss[0]:.3f} | D Acc: {d_loss[1]*100:.0f}% | G Loss: {g_loss:.3f}")

print("Training complete!")
```

### Cell 2: Plot the Loss Curves
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(d_losses, label='Discriminator Loss', alpha=0.7)
ax1.plot(g_losses, label='Generator Loss', alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GAN Training Losses')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(d_accs, label='Discriminator Accuracy', color='green', alpha=0.7)
ax2.axhline(y=50, color='red', linestyle='--', label='50% (ideal)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Discriminator Accuracy Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Cell 3: Generate Final Images
```python
noise = np.random.normal(0, 1, (20, NOISE_DIM))
generated = gen.predict(noise, verbose=0)
plt.figure(figsize=(14, 2))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(generated[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Final Generated Digits", fontsize=13)
plt.tight_layout()
plt.show()
```

### Challenge 3 Questions:
1. Look at the Discriminator Accuracy plot. Did it trend toward 50%? What does it mean when the accuracy is at 50%?
*Answer:* __________________________________________________________________

2. Look at the loss curves. Unlike a classifier (where loss steadily decreases), GAN losses oscillate. Why is oscillation normal for GANs?
*Answer:* __________________________________________________________________

3. If the Discriminator accuracy stayed at 100% throughout training, what would that indicate about the Generator?
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Challenge 1: Text generation working, 3 different outputs recorded
- [ ] Challenge 2: Interpolation plot showing smooth transition between two digits
- [ ] Challenge 3: Loss curves plotted and all analysis questions answered

---
*Session 25 | Deep Learning Using Neural Networks | Aptech*
