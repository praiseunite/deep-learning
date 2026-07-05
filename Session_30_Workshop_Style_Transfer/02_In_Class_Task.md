# Session 30 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Style Transfer Challenge: Create Your Own AI Art"
### Duration: 1 hour 35 minutes

---

> **Instructor Note:** This is a fully student-driven workshop. Students work through three independent challenges. Each challenge applies a different style transfer technique from Sessions 28–29. Encourage students to experiment with their own images and styles! Those who finish early should help classmates and explore the bonus sections. Share results on the projector at the end!

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Tune alpha and beta weights in Classic NST to control the content-style balance.
- [ ] Apply Arbitrary Style Transfer to their own personal photo using TF Hub.
- [ ] Generate a texture from random noise using only style loss.
- [ ] Explain why each method has different trade-offs in speed, quality, and flexibility.

---

## Challenge 1: Classic NST — Tune Your Masterpiece (35 minutes)

In Session 28, we ran Classic NST with fixed alpha and beta values. Now you will tune those values yourself and discover how they control the final artwork.

> **GPU Required:** Enable GPU in your Kaggle settings before starting.

### Cell 1: Setup — Paste the Full Classic NST Pipeline
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import matplotlib.pyplot as plt
import time

IMG_HEIGHT = 192
IMG_WIDTH  = 192

def load_and_process(path):
    img = keras.preprocessing.image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess(img):
    img = img.copy().reshape((IMG_HEIGHT, IMG_WIDTH, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')

def gram_matrix(feat):
    channels = int(feat.shape[-1])
    a = tf.reshape(feat, [-1, channels])
    return tf.matmul(a, a, transpose_a=True) / tf.cast(tf.shape(a)[0], tf.float32)

def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def compute_style_loss(style, generated):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_matrix(generated)))

CONTENT_LAYERS = ['block4_conv2']
STYLE_LAYERS   = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def build_extractor():
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False
    outputs = [vgg.get_layer(n).output for n in CONTENT_LAYERS + STYLE_LAYERS]
    return keras.Model(inputs=vgg.input, outputs=outputs)

extractor = build_extractor()
print("✅ Setup complete!")
```

---

### Cell 2: Load Content and Style Images
```python
content_path = keras.utils.get_file(
    'content.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)
style_path = keras.utils.get_file(
    'style.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)

content_img = load_and_process(content_path)
style_img   = load_and_process(style_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(deprocess(content_img)); ax1.set_title("Content Image"); ax1.axis('off')
ax2.imshow(deprocess(style_img));   ax2.set_title("Style Image");   ax2.axis('off')
plt.tight_layout(); plt.show()
print("Images loaded!")
```

---

### Cell 3: Run the Tuning Experiment
```python
# ========================================================
# 🎛️  STUDENT EXPERIMENT:
# Change these three settings and run to see the effect!
# ========================================================
ALPHA      = 1e3    # Content weight — try: 1e2, 1e3, 1e4
BETA       = 1e-2   # Style weight  — try: 1e-3, 1e-2, 1e-1
ITERATIONS = 300    # More = higher quality but slower

def run_nst(content, style, alpha, beta, iterations):
    all_feats = extractor(content)
    target_content = all_feats[:len(CONTENT_LAYERS)]
    all_feats = extractor(style)
    target_style = all_feats[len(CONTENT_LAYERS):]

    generated = tf.Variable(content, dtype=tf.float32)
    optimizer = keras.optimizers.Adam(learning_rate=5.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            gen_feats = extractor(generated)
            gen_content = gen_feats[:len(CONTENT_LAYERS)]
            gen_style   = gen_feats[len(CONTENT_LAYERS):]

            c_loss = sum(compute_content_loss(tc, gc)
                         for tc, gc in zip(target_content, gen_content))
            s_loss = sum(compute_style_loss(ts, gs)
                         for ts, gs in zip(target_style, gen_style))
            total  = alpha * c_loss + beta * s_loss

        grads = tape.gradient(total, generated)
        optimizer.apply_gradients([(grads, generated)])
        if i % 100 == 0:
            print(f"  Iter {i:>4d} | Content: {c_loss:.1f} | Style: {s_loss:.1f}")

    return deprocess(generated.numpy())

print(f"Running NST with Alpha={ALPHA}, Beta={BETA}, Iterations={ITERATIONS}...")
t0 = time.time()
result = run_nst(content_img, style_img, ALPHA, BETA, ITERATIONS)
print(f"✅ Done in {time.time()-t0:.1f}s")
plt.figure(figsize=(8, 6))
plt.imshow(result); plt.axis('off')
plt.title(f"Alpha={ALPHA} | Beta={BETA}", fontsize=13)
plt.show()
```

---

### Cell 4: Side-by-Side Comparison of Three Settings
Run the previous cell three times with different alpha/beta values, then paste the results here to compare:

```python
# Run all three presets and compare them in one figure
results = {}
presets = [
    {"alpha": 1e4, "beta": 1e-2, "label": "High Content\n(α=1e4, β=1e-2)"},
    {"alpha": 1e3, "beta": 1e-2, "label": "Balanced\n(α=1e3, β=1e-2)"},
    {"alpha": 1e2, "beta": 1e-2, "label": "High Style\n(α=1e2, β=1e-2)"},
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, preset in enumerate(presets):
    print(f"\n🎨 Experiment {i+1}: {preset['label'].replace(chr(10), ' ')}")
    r = run_nst(content_img, style_img, preset["alpha"], preset["beta"], 200)
    axes[i].imshow(r)
    axes[i].set_title(preset["label"], fontsize=12)
    axes[i].axis('off')

plt.suptitle("Alpha-Beta Tuning Comparison", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n✅ All three experiments complete! Which do you prefer?")
```

### Challenge 1 Questions:
1. In which experiment did the output look most like the original photo? Why?
   *Answer:* __________________________________________________________________

2. In which experiment did the output look most like the painting style? Why?
   *Answer:* __________________________________________________________________

3. If BETA was 1,000,000 times larger than ALPHA, what do you think the output would look like?
   *Answer:* __________________________________________________________________

---

## Challenge 2: Fast NST — Build a Multi-Style Photo Gallery (35 minutes)

Now apply Arbitrary Style Transfer to **your own personal photo** using the pre-trained TF Hub model.

> **💡 How to add your own photo to Kaggle:**
> 1. In your Kaggle Notebook, click the **"Add Data"** button on the right panel.
> 2. Choose **"Upload"** and upload your photo (any `.jpg` or `.png`).
> 3. Your file will appear at `/kaggle/input/your-filename.jpg`.
> 
> *Alternatively, use the sample images from Cell 2 if you don't have a photo handy.*

### Cell 1: Load TF Hub Model
```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import time

print("Loading Arbitrary Style Transfer model...")
nst_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("✅ Model ready!")
```

---

### Cell 2: Load Your Personal Photo + Three Style Images
```python
def load_hub_image(path_or_url, max_dim=512):
    if path_or_url.startswith('http'):
        path_or_url = tf.keras.utils.get_file('img.jpg', path_or_url)
    img = tf.io.read_file(path_or_url)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    img = tf.image.resize(img, tf.cast(shape * scale, tf.int32))
    return img[tf.newaxis, :]

# ====================================================
# 📷 STUDENT ACTION: Replace the URL below with the
# path to YOUR own photo!  e.g. '/kaggle/input/myface.jpg'
# ====================================================
MY_PHOTO_PATH = tf.keras.utils.get_file(
    'my_photo.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)

# Three different artistic styles
STYLE_URLS = {
    "Kandinsky": "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    "The Great Wave": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1280px-Tsunami_by_hokusai_19th_century.jpg",
    "Starry Night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
}

content = load_hub_image(MY_PHOTO_PATH)
styles  = {name: load_hub_image(url) for name, url in STYLE_URLS.items()}

print(f"✅ Your photo shape: {content.shape}")
print(f"✅ Styles loaded: {list(styles.keys())}")
```

---

### Cell 3: Generate Your Multi-Style Gallery!
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Top row: content + 3 style images
axes[0][0].imshow(content[0]); axes[0][0].set_title("Your Photo", fontsize=12, fontweight='bold'); axes[0][0].axis('off')
for i, (name, style_img) in enumerate(styles.items()):
    axes[0][i+1].imshow(style_img[0]); axes[0][i+1].set_title(f"Style: {name}", fontsize=11); axes[0][i+1].axis('off')

# Bottom row: label + 3 styled outputs
axes[1][0].text(0.5, 0.5, 'AI-Generated\nArtwork', ha='center', va='center',
                fontsize=14, fontweight='bold', transform=axes[1][0].transAxes)
axes[1][0].axis('off')

for i, (name, style_img) in enumerate(styles.items()):
    s_resized = tf.image.resize(style_img, [256, 256])
    t0 = time.time()
    stylized = nst_model(tf.constant(content), tf.constant(s_resized))[0]
    elapsed = time.time() - t0

    axes[1][i+1].imshow(stylized[0])
    axes[1][i+1].set_title(f"{name} Style\n({elapsed:.2f}s)", fontsize=11)
    axes[1][i+1].axis('off')

plt.suptitle("🎨 My Personal Style Transfer Gallery", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
print("✅ Gallery complete! All three styles applied in under 1 second each!")
```

### Challenge 2 Questions:
1. Which of the three styles did you prefer on your photo? What made it look good?
   *Answer:* __________________________________________________________________

2. How long did each style transfer take (look at the seconds shown under each image)?
   *Answer:* __________________________________________________________________

3. Could you get these results with Classic NST in the same amount of time? Why / why not?
   *Answer:* __________________________________________________________________

---

## Challenge 3: Texture Synthesis — Design from Noise (15 minutes)

Create a brand new texture from pure random static noise — no content image needed!

> **Quick Challenge:** Can you recognize what the texture looks like before and after optimization?

### Cell 1: Quick Texture Synthesis
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt

IMG_S = 128

# Download the style texture
style_path = tf.keras.utils.get_file(
    'texture_style.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)

def load_tex(path, s=IMG_S):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(s, s))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def deprocess_tex(img, s=IMG_S):
    img = img.copy().reshape((s, s, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')

def gram(feat):
    c = int(feat.shape[-1])
    a = tf.reshape(feat, [-1, c])
    return tf.matmul(a, a, transpose_a=True) / tf.cast(tf.shape(a)[0], tf.float32)

# Build mini extractor
vgg_tex = VGG19(weights='imagenet', include_top=False)
vgg_tex.trainable = False
tex_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
tex_extractor = tf.keras.Model(
    inputs=vgg_tex.input,
    outputs=[vgg_tex.get_layer(n).output for n in tex_layers]
)

target_tex = load_tex(style_path)
target_feats = tex_extractor(target_tex)

# Start from PURE RANDOM NOISE
np.random.seed(0)
noise = np.random.uniform(0, 128, (1, IMG_S, IMG_S, 3)).astype('float32')
generated_tex = tf.Variable(noise, dtype=tf.float32)
opt_tex = tf.keras.optimizers.Adam(learning_rate=5.0)

print("🎲 Starting from random noise...")
print(f"The input right now is just {IMG_S}x{IMG_S} pixels of random static.")
print("Optimizing texture (this takes ~1 minute)...")

for i in range(500):
    with tf.GradientTape() as tape:
        gen_feats = tex_extractor(generated_tex)
        loss = sum(tf.reduce_mean(tf.square(gram(gf) - gram(tf_)))
                   for gf, tf_ in zip(gen_feats, target_feats))
    grads = tape.gradient(loss, generated_tex)
    opt_tex.apply_gradients([(grads, generated_tex)])
    if i % 100 == 0:
        print(f"  Iter {i:>4d} | Loss: {loss:.1f}")

print("\n✅ Texture synthesis complete!")
```

---

### Cell 2: Show the Magic Transformation
```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.imshow(np.clip(noise[0] + 128, 0, 255).astype('uint8'))
ax1.set_title("🎲 Starting Point\n(Random Noise)", fontsize=12, fontweight='bold')
ax1.axis('off')

ax2.imshow(deprocess_tex(target_tex[0].numpy()))
ax2.set_title("🎨 Target Texture\n(Kandinsky Style)", fontsize=12, fontweight='bold')
ax2.axis('off')

ax3.imshow(deprocess_tex(generated_tex.numpy()))
ax3.set_title("✨ Synthesized Texture\n(No content image!)", fontsize=12, fontweight='bold')
ax3.axis('off')

plt.suptitle("Texture Synthesis: Random Noise → Art", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
print("The Gram Matrix pulled the style pattern OUT of the painting and burned it INTO the noise!")
```

### Challenge 3 Questions:
1. Describe what the random noise looks like before optimization (Cell 2, left image). Describe what it looks like after (right image).
   *Answer:* __________________________________________________________________

2. We used ZERO content loss here — only style loss. What would the output look like if we also added a content image (e.g., a photo of a building)? How would it differ from Classic NST?
   *Answer:* __________________________________________________________________

---

## 🏆 Gallery Showcase (Final 10 minutes)

At the end of the session, every student will **screen-share their best styled image** from Challenge 2. The class votes on:
- 🥇 Most Creative Style Choice
- 🖼️ Best Looking Output
- 💡 Most Interesting Observation

---

## Task Completion Criteria

- [ ] Challenge 1: All 4 code cells run, three alpha-beta presets compared, questions answered
- [ ] Challenge 2: Personal photo uploaded, all 3 styles applied, gallery plot visible
- [ ] Challenge 3: Both cells run, noise-to-texture transformation visible, questions answered

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 30*
