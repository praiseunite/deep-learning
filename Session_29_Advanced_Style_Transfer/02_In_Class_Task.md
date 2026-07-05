# Session 29 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Fast Style Transfer & Texture Synthesis"
### Duration: 50 minutes

---

> **Instructor Note:** This task has two parts. Part A demonstrates how to build a fast style transfer pipeline using a pre-trained fast NST model. Part B explores texture synthesis from pure noise. Both use Kaggle with GPU enabled. Each cell should be typed and run one at a time.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Load and use a pre-trained Fast Style Transfer model from TensorFlow Hub.
- [ ] Compare the speed of Fast NST vs Classic NST.
- [ ] Generate a new texture from random noise using only style loss.
- [ ] Explain why Fast NST cannot easily switch between styles.

---

## PART 1 — Concept Check (5 minutes)

Answer these before you open Kaggle:

1. Classic NST updates the **network weights / image pixels** (circle one). Fast Style Transfer trains a network that updates the **network weights / image pixels** (circle one).

2. If you want your app to support 10 different art styles using Fast Style Transfer, how many separate networks would you need to train?
   *Answer:* __________________________________________________________________

3. What does "temporal inconsistency" mean when applying style transfer to a video?
   *Answer:* __________________________________________________________________

---

## PART 2A — Kaggle Lab: Fast Style Transfer with TF Hub (25 minutes)

Open your Kaggle notebook with the **GPU enabled**.

### Cell 1: Setup and Install
```python
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time

print(f"TensorFlow version: {tf.__version__}")
print("TensorFlow Hub loaded!")
```

**What is TensorFlow Hub?**
> TF Hub is a library of pre-trained models that researchers share with the world. Instead of training for weeks, we can download a model that is already trained!

---

### Cell 2: Load the Pre-Trained Fast Style Transfer Model
```python
# This model was trained using Fast Neural Style Transfer
# It can apply style in ONE forward pass — no 500 iterations needed!
print("Downloading pre-trained model... (may take 1-2 minutes)")

fast_nst_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)

print("✅ Model loaded! This model supports ARBITRARY styles — any content + any style!")
```

> **💡 This is actually Arbitrary Style Transfer (AdaIN)!** The TF Hub model we are loading is the Huang & Belongie model from Session 29 notes. It handles any style in milliseconds.

---

### Cell 3: Load Content and Style Images
```python
def load_image_for_hub(path_or_url, max_dim=512):
    """Load an image and normalize pixels to [0, 1]."""
    img = tf.io.read_file(path_or_url) if not path_or_url.startswith('http') else \
          tf.keras.utils.get_file('img.jpg', path_or_url)
    
    img = tf.io.read_file(img if isinstance(img, str) else path_or_url)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Resize keeping aspect ratio
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Load sample images
content_url = tf.keras.utils.get_file(
    'content_dog.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)
style_url = tf.keras.utils.get_file(
    'style_kandinsky.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)

content_image = load_image_for_hub(content_url)
style_image   = load_image_for_hub(style_url)

# Display both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(content_image[0])
ax1.set_title("Content Image (Dog)", fontsize=13)
ax1.axis('off')
ax2.imshow(style_image[0])
ax2.set_title("Style Image (Kandinsky)", fontsize=13)
ax2.axis('off')
plt.tight_layout()
plt.show()
print("Images loaded!")
```

---

### Cell 4: Run Fast Style Transfer and Time It!
```python
# Resize style image to 256x256 (recommended for this model)
style_image_resized = tf.image.resize(style_image, [256, 256])

print("⏱️ Timing Fast Style Transfer...")
start_time = time.time()

# This is ONE forward pass — not 500 iterations!
stylized_image = fast_nst_model(
    tf.constant(content_image),
    tf.constant(style_image_resized)
)[0]

end_time = time.time()
elapsed = end_time - start_time

print(f"✅ Style transfer complete!")
print(f"⚡ Time taken: {elapsed:.3f} seconds")
print(f"💡 Classic NST would take ~30-60 seconds for the same result!")
print(f"🚀 Fast NST speedup: ~{60/elapsed:.0f}x faster!")
```

---

### Cell 5: Display the Result
```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(content_image[0])
ax1.set_title("Content Image", fontsize=13)
ax1.axis('off')

ax2.imshow(style_image[0])
ax2.set_title("Style Image", fontsize=13)
ax2.axis('off')

ax3.imshow(stylized_image[0])
ax3.set_title(f"Fast Style Transfer\n({elapsed:.2f} seconds!)", fontsize=13)
ax3.axis('off')

plt.suptitle("Arbitrary Style Transfer Result", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

### Cell 6: Try Different Styles (Experiment!)
```python
# Let's try another style image and see how the model adapts instantly!
style_url_2 = tf.keras.utils.get_file(
    'style_wave.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1280px-Tsunami_by_hokusai_19th_century.jpg'
)
style_image_2 = load_image_for_hub(style_url_2)
style_image_2_resized = tf.image.resize(style_image_2, [256, 256])

# Apply the SAME model with a DIFFERENT style — no retraining!
start_2 = time.time()
stylized_2 = fast_nst_model(
    tf.constant(content_image),
    tf.constant(style_image_2_resized)
)[0]
end_2 = time.time()

print(f"Second style applied in: {end_2 - start_2:.3f} seconds")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0][0].imshow(content_image[0]);  axes[0][0].set_title("Content Image"); axes[0][0].axis('off')
axes[0][1].imshow(style_image[0]);    axes[0][1].set_title("Style 1: Kandinsky"); axes[0][1].axis('off')
axes[0][2].imshow(stylized_image[0]); axes[0][2].set_title("Result 1"); axes[0][2].axis('off')
axes[1][0].imshow(content_image[0]);  axes[1][0].set_title("Content Image"); axes[1][0].axis('off')
axes[1][1].imshow(style_image_2[0]);  axes[1][1].set_title("Style 2: The Great Wave"); axes[1][1].axis('off')
axes[1][2].imshow(stylized_2[0]);     axes[1][2].set_title("Result 2"); axes[1][2].axis('off')

plt.suptitle("One Model — Two Different Styles — Instantly!", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## PART 2B — Texture Synthesis from Noise (10 minutes)

Now let's create a texture from pure random noise — no content image at all!

### Cell 7: Texture Synthesis Setup
```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

IMG_SIZE = 128  # Small size for speed

def load_for_texture(path, size=IMG_SIZE):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(size, size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def deprocess(img):
    img = img.copy().reshape((IMG_SIZE, IMG_SIZE, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')

def gram_matrix(feat):
    channels = int(feat.shape[-1])
    a = tf.reshape(feat, [-1, channels])
    return tf.matmul(a, a, transpose_a=True) / tf.cast(tf.shape(a)[0], tf.float32)

# Build a mini feature extractor using only style layers
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1']
vgg = VGG19(weights='imagenet', include_top=False)
vgg.trainable = False
style_outputs = [vgg.get_layer(n).output for n in STYLE_LAYERS]
texture_extractor = tf.keras.Model(inputs=vgg.input, outputs=style_outputs)

print("Texture extractor ready!")
```

---

### Cell 8: Run Texture Synthesis
```python
# Load the style texture (we use the style image as our "fabric swatch")
target_texture = load_for_texture(style_url)
target_features = texture_extractor(target_texture)

# START FROM PURE RANDOM NOISE — no content image!
np.random.seed(42)
random_noise = np.random.uniform(0, 128, (1, IMG_SIZE, IMG_SIZE, 3)).astype('float32')
generated_texture = tf.Variable(random_noise, dtype=tf.float32)

optimizer_tex = tf.keras.optimizers.Adam(learning_rate=5.0)

print("Starting texture synthesis from random noise...")
for i in range(500):
    with tf.GradientTape() as tape:
        gen_features = texture_extractor(generated_texture)
        style_loss = sum(
            tf.reduce_mean(tf.square(gram_matrix(gf) - gram_matrix(tf_)))
            for gf, tf_ in zip(gen_features, target_features)
        )
    grads = tape.gradient(style_loss, generated_texture)
    optimizer_tex.apply_gradients([(grads, generated_texture)])
    if i % 100 == 0:
        print(f"  Iter {i:>4d} | Style loss: {style_loss:.2f}")

print("✅ Texture synthesis complete!")
```

---

### Cell 9: Compare Noise to Synthesized Texture
```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.imshow(np.clip(random_noise[0] + 128, 0, 255).astype('uint8'))
ax1.set_title("Starting Point\n(Random Noise)", fontsize=12)
ax1.axis('off')

ax2.imshow(deprocess(target_texture[0].numpy()))
ax2.set_title("Target Style\n(Kandinsky Painting)", fontsize=12)
ax2.axis('off')

ax3.imshow(deprocess(generated_texture.numpy()))
ax3.set_title("Synthesized Texture\n(No content image used!)", fontsize=12)
ax3.axis('off')

plt.suptitle("Texture Synthesis: From Noise → Art!", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("The random noise has become a texture that matches Kandinsky's style!")
```

---

## PART 3 — Observation Questions (5 minutes)

1. In Cell 4, how many seconds did Fast Style Transfer take? Compare this to Classic NST which takes ~30 seconds. How many times faster is it?
   *Answer:* __________________________________________________________________

2. In Cell 6, we applied TWO different styles using the SAME model. Could we do this with a Fast Style Transfer model trained on ONE style? Why or why not?
   *Answer:* __________________________________________________________________

3. In Cell 8, we started from RANDOM NOISE with NO content image. What would happen to the output if we set the style loss to zero and the content loss to zero? (Think carefully!)
   *Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: All 3 concept check questions answered
- [ ] Part 2A: All 6 code cells (Cells 1–6) executed successfully  
- [ ] Part 2A: Two style transfer outputs visible on screen
- [ ] Part 2B: All 3 code cells (Cells 7–9) executed successfully
- [ ] Part 2B: Synthesized texture image visible on screen
- [ ] Part 3: All 3 observation questions answered

---
*Session 29 | Deep Learning Using Neural Networks | Aptech*
