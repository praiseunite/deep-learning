# Session 31 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Boot Up All 5 Deep Learning Machines"
### Duration: 1 hour 45 minutes

---

> **Instructor Note:** This is the grand finale practical. Students will run ONE mega Kaggle notebook that touches every major era of the course. Each "machine" is intentionally kept brief — the goal is recognition and confidence, not depth. Every cell should produce output that the student has seen before in a previous session. Encourage students to run each machine and say OUT LOUD what it is doing. GPU is required for Machines 3, 4, and 5.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Run a complete ANN classifier from scratch in under 10 lines of Keras.
- [ ] Load a pre-trained CNN and make a real image prediction in seconds.
- [ ] Generate new text character-by-character using an RNN.
- [ ] Generate a class-conditioned image using a trained CVAE.
- [ ] Apply Arbitrary Style Transfer to any photo in under 1 second.
- [ ] Explain in plain English what each machine is doing and why.

---

## Before You Start

Open a **new Kaggle Notebook** and enable **GPU accelerator**.
Name your notebook: `"Deep Learning - Final Masterclass - [Your Name]"`

In the first cell, add this Markdown header:
```markdown
# 🏆 My Deep Learning Masterclass
### Student: [Your Full Name] | Aptech Deep Learning Course
---
This notebook boots up all 5 Deep Learning Machines from Sessions 1–30.
```

---

## 🏷️ MACHINE 1 — The Classifier Machine (10 minutes)
*Sessions 1–13 | Feedforward Neural Networks*

> **What is this machine doing?** Taking numbers as input, learning patterns, and predicting a category. This is the original deep learning machine — every other machine builds on this one.

### Cell 1A: Import and Load Data
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print(f"TensorFlow {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Load the classic dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Flatten 28x28 images into 784 numbers
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Each sample is: {X_train.shape[1]} numbers (28x28 pixels flattened)")
```

### Cell 1B: Build and Train the Classifier
```python
# Build the ANN Classifier in 5 lines!
model_classifier = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer
    layers.Dropout(0.2),                                        # Regularization (Session 8)
    layers.Dense(64, activation='relu'),                        # Hidden layer 2
    layers.Dense(10, activation='softmax')                      # Output: 10 digit classes
], name="The_Classifier_Machine")

model_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_classifier.summary()

print("\nTraining the Classifier Machine...")
history = model_classifier.fit(X_train, y_train, epochs=5, batch_size=128,
                                validation_split=0.1, verbose=1)
test_loss, test_acc = model_classifier.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Classifier Machine ready! Test Accuracy: {test_acc*100:.1f}%")
```

### Cell 1C: Make Real Predictions
```python
# Show 10 predictions vs reality
predictions = model_classifier.predict(X_test[:10], verbose=0)
fig, axes = plt.subplots(2, 5, figsize=(14, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    pred = np.argmax(predictions[i])
    true = y_test[i]
    color = 'green' if pred == true else 'red'
    ax.set_title(f"Pred: {pred}\nTrue: {true}", color=color, fontsize=10)
    ax.axis('off')
plt.suptitle("🏷️ Classifier Machine — Predicting Handwritten Digits", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Quick Check ✍️:** What layer makes this an "ANN" and not a "CNN"? ______________________

---

## 👁️ MACHINE 2 — The Vision Machine (10 minutes)
*Sessions 14–19 | Convolutional Neural Networks*

> **What is this machine doing?** Looking at an image through layers of filters — each layer detecting more complex visual patterns. It was trained on 1.3 million photos and can recognise 1000 different objects.

### Cell 2A: Load Pre-Trained VGG19 (The Vision Machine)
```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions

# Load the Vision Machine (pre-trained on ImageNet — 1000 object classes)
vision_machine = VGG19(weights='imagenet')
print(f"✅ Vision Machine loaded!")
print(f"This machine has {vision_machine.count_params():,} parameters trained on 1.3 million images")
```

### Cell 2B: Make a Real-World Prediction
```python
# Download a test image
img_path = keras.utils.get_file(
    'elephant.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/800px-African_Bush_Elephant.jpg'
)

from tensorflow.keras.preprocessing import image as keras_image
img = keras_image.load_img(img_path, target_size=(224, 224))
img_array = keras_image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict!
predictions_vgg = vision_machine.predict(img_array, verbose=0)
top5 = decode_predictions(predictions_vgg, top=5)[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(keras_image.load_img(img_path, target_size=(224, 224)))
ax1.set_title("Input Image", fontsize=12); ax1.axis('off')

labels = [p[1].replace('_', ' ').title() for p in top5]
scores = [p[2] * 100 for p in top5]
bars = ax2.barh(labels[::-1], scores[::-1], color=['#4CAF50' if i == 4 else '#2196F3' for i in range(5)])
ax2.set_xlabel("Confidence (%)")
ax2.set_title("👁️ Vision Machine — Top 5 Predictions", fontsize=12)
for bar, score in zip(bars, scores[::-1]):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{score:.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.show()
print(f"\n🏆 Top Prediction: {top5[0][1].replace('_', ' ').title()} ({top5[0][2]*100:.1f}% confident)")
```

**Quick Check ✍️:** What is the name of the pre-trained model we used? Why don't we need to train it? ______________________

---

## 📝 MACHINE 3 — The Text Machine (15 minutes)
*Sessions 20–21 | Recurrent Neural Networks*

> **What is this machine doing?** Reading a sequence of characters one at a time, building up a "memory" of what came before, then predicting the next character. This is exactly how autocomplete works at a basic level.

### Cell 3A: Quick Character-Level RNN
```python
# Our training text — a famous excerpt repeated many times for enough data
TEXT = ("to be or not to be that is the question whether tis nobler in the mind "
        "to suffer the slings and arrows of outrageous fortune ") * 15

chars    = sorted(set(TEXT))
c2i      = {c: i for i, c in enumerate(chars)}
i2c      = {i: c for i, c in enumerate(chars)}
VOCAB    = len(chars)
SEQ_LEN  = 20

# Build training sequences
seqs, tgts = [], []
for i in range(len(TEXT) - SEQ_LEN):
    seqs.append([c2i[c] for c in TEXT[i:i+SEQ_LEN]])
    tgts.append(c2i[TEXT[i+SEQ_LEN]])

X_rnn = tf.keras.utils.to_categorical(np.array(seqs), VOCAB)
y_rnn = tf.keras.utils.to_categorical(np.array(tgts), VOCAB)
print(f"Sequences: {X_rnn.shape[0]:,} | Vocabulary: {VOCAB} characters")
```

### Cell 3B: Build and Train the Text Machine
```python
text_machine = keras.Sequential([
    layers.SimpleRNN(256, input_shape=(SEQ_LEN, VOCAB), return_sequences=False),
    layers.Dense(VOCAB, activation='softmax')
], name="The_Text_Machine")

text_machine.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training the Text Machine (2 minutes)...")
text_machine.fit(X_rnn, y_rnn, epochs=40, batch_size=128, verbose=0,
                 callbacks=[keras.callbacks.LambdaCallback(
                     on_epoch_end=lambda e, l: print(f"  Epoch {e+1:>3d}/40 | acc: {l['accuracy']:.3f}") if (e+1) % 10 == 0 else None
                 )])
print("✅ Text Machine trained!")
```

### Cell 3C: Generate Shakespeare-Style Text
```python
def generate(model, seed, length=120, temperature=0.8):
    result = seed
    for _ in range(length):
        x = np.array([[c2i.get(c, 0) for c in result[-SEQ_LEN:]]])
        x = tf.keras.utils.to_categorical(x, VOCAB)
        probs = model.predict(x, verbose=0)[0]
        # Temperature scaling: higher = more creative, lower = more conservative
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        next_c = i2c[np.random.choice(len(probs), p=probs)]
        result += next_c
    return result

seed_text = "to be or not to be "
print("📝 Text Machine Output:")
print("=" * 60)
print(generate(text_machine, seed_text))
print("=" * 60)
print("\nRun this cell again for a different output (temperature adds randomness)!")
```

**Quick Check ✍️:** What does the RNN "remember" as it reads character by character? ______________________

---

## 🌙 MACHINE 4 — The Dream Machine (20 minutes)
*Sessions 22–27 | Variational Autoencoder*

> **What is this machine doing?** Learning the "recipe" for each digit class, then using that recipe + random variations to bake a brand new digit that never existed before. This is controlled generation.

### Cell 4A: Build a Quick CVAE
```python
LATENT_DIM = 8
NUM_CLASSES = 10

# Reload data for the Dream Machine
(X_train_d, y_train_d), _ = keras.datasets.mnist.load_data()
X_train_d = X_train_d.astype("float32").reshape(-1, 784) / 255.0
y_ohe = tf.keras.utils.to_categorical(y_train_d, NUM_CLASSES)  # One-hot labels

# Encoder: image + label → latent code
img_in   = keras.Input(shape=(784,), name="image")
lbl_in   = keras.Input(shape=(NUM_CLASSES,), name="label")
x        = layers.Concatenate()([img_in, lbl_in])
x        = layers.Dense(512, activation='relu')(x)
x        = layers.Dense(256, activation='relu')(x)
z_mean   = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_v  = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z        = layers.Lambda(lambda args: args[0] + tf.exp(0.5*args[1]) * tf.random.normal(tf.shape(args[0])),
                         name="z")([z_mean, z_log_v])
encoder  = keras.Model([img_in, lbl_in], [z_mean, z_log_v, z], name="Encoder")

# Decoder: latent code + label → reconstructed image
z_in     = keras.Input(shape=(LATENT_DIM,), name="z")
lbl_in2  = keras.Input(shape=(NUM_CLASSES,), name="label_dec")
x        = layers.Concatenate()([z_in, lbl_in2])
x        = layers.Dense(256, activation='relu')(x)
x        = layers.Dense(512, activation='relu')(x)
out      = layers.Dense(784, activation='sigmoid')(x)
decoder  = keras.Model([z_in, lbl_in2], out, name="Decoder")

print("✅ Dream Machine (CVAE) built!")
print(f"   Encoder: image(784) + label(10) → latent code({LATENT_DIM} numbers)")
print(f"   Decoder: latent code({LATENT_DIM}) + label(10) → image(784)")
```

### Cell 4B: Train the Dream Machine
```python
class DreamMachine(keras.Model):
    def __init__(self, enc, dec):
        super().__init__(name="The_Dream_Machine")
        self.enc, self.dec = enc, dec
        self.loss_tracker = keras.metrics.Mean(name="loss")
    @property
    def metrics(self): return [self.loss_tracker]
    def train_step(self, data):
        imgs, lbls = data
        with tf.GradientTape() as tape:
            zm, zlv, z = self.enc([imgs, lbls])
            recon = self.dec([z, lbls])
            r_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(imgs, recon)))
            kl     = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + zlv - tf.square(zm) - tf.exp(zlv), axis=1))
            loss   = r_loss + kl
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_weights), self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

dream_machine = DreamMachine(encoder, decoder)
dream_machine.compile(optimizer=keras.optimizers.Adam(1e-3))
print("Training the Dream Machine (takes 2–3 minutes)...")
dream_machine.fit([X_train_d, y_ohe], epochs=20, batch_size=256, verbose=0,
                  callbacks=[keras.callbacks.LambdaCallback(
                      on_epoch_end=lambda e, l: print(f"  Epoch {e+1:>3d}/20 | loss: {l['loss']:.1f}") if (e+1) % 5 == 0 else None
                  )])
print("✅ Dream Machine ready!")
```

### Cell 4C: Dream Up New Digits On Command!
```python
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for digit in range(10):
    # Create the condition: "Generate digit [digit]"
    condition = tf.keras.utils.to_categorical([digit], NUM_CLASSES)
    # Sample a random latent code (the "coin" for the vending machine!)
    z_sample = np.random.normal(0, 1, (1, LATENT_DIM))
    # Dream it up!
    generated = decoder.predict([z_sample, condition], verbose=0)
    axes[0][digit].imshow(generated[0].reshape(28, 28), cmap='gray')
    axes[0][digit].set_title(f"Dream '{digit}'", fontsize=9)
    axes[0][digit].axis('off')
    # Try a different random variation of the same digit
    z_sample2 = np.random.normal(0, 1, (1, LATENT_DIM))
    generated2 = decoder.predict([z_sample2, condition], verbose=0)
    axes[1][digit].imshow(generated2[0].reshape(28, 28), cmap='gray')
    axes[1][digit].set_title(f"Another '{digit}'", fontsize=9)
    axes[1][digit].axis('off')
plt.suptitle("🌙 Dream Machine — Generating All 10 Digits On Command (Two Variations Each)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("Every image above never existed before — the Dream Machine created them all!")
```

**Quick Check ✍️:** What is the difference between the two rows of digits? What changed between them? ______________________

---

## 🎨 MACHINE 5 — The Art Machine (15 minutes)
*Sessions 28–30 | Neural Style Transfer*

> **What is this machine doing?** Using a frozen VGG19 as a "Team of Art Critics". The early critics look at brushstrokes (Style). The deep critics look at subject matter (Content). In under 1 second, it blends any two images into art.

### Cell 5A: Load the Art Machine
```python
import tensorflow_hub as hub

print("Loading the Art Machine (Arbitrary Style Transfer)...")
art_machine = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("✅ Art Machine loaded! One model — infinite styles — instant results.")
```

### Cell 5B: Apply 3 Different Styles to 1 Photo
```python
def load_art_img(url_or_path, max_dim=512):
    if url_or_path.startswith('http'):
        url_or_path = keras.utils.get_file('art_img.jpg', url_or_path)
    img = tf.io.read_file(url_or_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    s = tf.cast(tf.shape(img)[:-1], tf.float32)
    img = tf.image.resize(img, tf.cast(s * (max_dim / max(s)), tf.int32))
    return img[tf.newaxis, :]

# Content: A dog photo
content_url = keras.utils.get_file('dog.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
content = load_art_img(content_url)

# 3 Famous Styles
styles = {
    "🌊 Great Wave (Hokusai)": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/800px-Tsunami_by_hokusai_19th_century.jpg",
    "⭐ Starry Night (Van Gogh)": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "🟥 Composition (Kandinsky)": "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
}

import time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: Content + 3 style paintings
axes[0][0].imshow(content[0]); axes[0][0].set_title("📷 Your Photo", fontsize=11, fontweight='bold'); axes[0][0].axis('off')
results = {}
for i, (name, url) in enumerate(styles.items()):
    style_img = load_art_img(url)
    style_resized = tf.image.resize(style_img, [256, 256])
    axes[0][i+1].imshow(style_img[0]); axes[0][i+1].set_title(name, fontsize=10); axes[0][i+1].axis('off')
    t = time.time()
    output = art_machine(tf.constant(content), tf.constant(style_resized))[0]
    elapsed = time.time() - t
    results[name] = (output, elapsed)

# Row 2: Results
axes[1][0].text(0.5, 0.5, '🎨\nArt Machine\nOutput', ha='center', va='center',
                fontsize=14, fontweight='bold', transform=axes[1][0].transAxes); axes[1][0].axis('off')
for i, (name, (img, elapsed)) in enumerate(results.items()):
    axes[1][i+1].imshow(img[0])
    axes[1][i+1].set_title(f"⚡ {elapsed:.2f}s", fontsize=11, fontweight='bold', color='green')
    axes[1][i+1].axis('off')

plt.suptitle("🎨 Art Machine — Your Photo Painted in 3 Styles Instantly!", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n✅ All 5 Deep Learning Machines have been booted up successfully!")
print("You have just re-ran the entire course in one notebook!")
```

**Quick Check ✍️:** The Art Machine ran in under 1 second per style. Why was Session 28's Classic NST much slower? ______________________

---

## 🏢 MACHINE 6 — The Fraud Detector (Real-World Deployment)
*The Capstone Bonus | Business Use Case*

> **What is this machine doing?** Solving a real business problem. We simulate Nomentral (a PropTech company) needing to catch forged land documents. We train a CNN to spot fake seals, then save it so it can be deployed to a real web app.

### Cell 6A: Generate Dummy Documents & Train the Fraud AI
```python
import os
os.makedirs("nomentral/authentic", exist_ok=True)
os.makedirs("nomentral/fraudulent", exist_ok=True)

# Generate dummy images
for i in range(50):
    auth = np.ones((64, 64, 3)) * 255; auth[10:20, 10:20] = [0, 0, 0] # Real seal is black
    tf.keras.preprocessing.image.save_img(f"nomentral/authentic/a_{i}.jpg", auth)
    fake = np.ones((64, 64, 3)) * 255; fake[10:20, 10:20] = [100, 100, 100] # Fake seal is grey
    tf.keras.preprocessing.image.save_img(f"nomentral/fraudulent/f_{i}.jpg", fake)

fraud_data = keras.preprocessing.image_dataset_from_directory("nomentral", image_size=(64, 64), batch_size=16)

fraud_ai = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(8, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid') # 0 = Authentic, 1 = Fraudulent
])

fraud_ai.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining Nomentral Fraud Detector...")
fraud_ai.fit(fraud_data, epochs=3, verbose=0)
print("✅ Fraud Detector Trained!")

# SAVE IT FOR DEPLOYMENT!
fraud_ai.save("nomentral_fraud_ai.keras")
print("💾 Model saved as 'nomentral_fraud_ai.keras'. You can now download it and upload to Hugging Face!")
```

---

## 🏆 Final Reflection (5 minutes)

In a Markdown cell in your notebook, answer these in your own words:

1. Which of the 5 machines do you find most exciting, and why?
2. Name one real-world product or service you use daily that likely uses ONE of these 5 machines.
3. What is the very next deep learning topic you want to learn after this course?

---

## Task Completion Criteria

- [ ] Notebook named correctly with your name
- [ ] Machine 1: Classifier trained, predictions plot visible, ≥95% accuracy
- [ ] Machine 2: VGG19 loaded, elephant prediction plot visible
- [ ] Machine 3: RNN trained, Shakespeare text generated
- [ ] Machine 4: CVAE trained, all 10 digits generated in 2 variations each
- [ ] Machine 5: Art Machine applied 3 styles with timing shown
- [ ] Final Reflection: All 3 questions answered in a Markdown cell

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 31 — The Final Practical*
