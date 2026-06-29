# Session 28 -- In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Your First Style Transfer"
### Duration: 45-55 minutes

---

> **Instructor Note:** Students will run a complete neural style transfer pipeline on Kaggle. This task uses TensorFlow's pre-trained VGG19. The process takes 3-5 minutes to run with GPU, so students should use that time to study the code.

---

## Learning Objectives

By the end of this task, students will be able to:
- [ ] Load a pre-trained VGG19 model and extract features from specific layers.
- [ ] Compute content loss and style loss (Gram Matrix).
- [ ] Run the NST optimization loop and produce a styled image.
- [ ] Experiment with alpha/beta weights to control the content-style balance.

---

## PART 1 -- Concept Check (5 minutes)

1. In NST, what stays frozen and what gets updated?
   - Frozen: __________
   - Updated: __________

2. Content features are extracted from __________ (early/deep) layers of VGG19 because they capture __________.

3. The Gram Matrix captures "which features appear together." Why is this better than directly comparing feature maps for style?
*Answer:* __________________________________________________________________

---

## PART 2 -- Kaggle Lab: Neural Style Transfer (40 minutes)

Open your Kaggle Notebook with the **GPU enabled**.

### Cell 1: Import Libraries and Load VGG19
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import matplotlib.pyplot as plt

# We will use small images (192x192) to keep things fast
IMG_HEIGHT = 192
IMG_WIDTH = 192

print(f"TensorFlow: {tf.__version__}")
print("Libraries loaded!")
```

### Cell 2: Create Content and Style Images
Since loading external images can be tricky on Kaggle, we will use built-in sample images.

```python
# Download sample images using keras
content_path = keras.utils.get_file(
    'content.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)
style_path = keras.utils.get_file(
    'style.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)

def load_and_process_image(path):
    img = keras.preprocessing.image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(processed_img):
    img = processed_img.copy().reshape((IMG_HEIGHT, IMG_WIDTH, 3))
    # Reverse VGG19 preprocessing
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

content_image = load_and_process_image(content_path)
style_image = load_and_process_image(style_path)

# Display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(deprocess_image(content_image))
ax1.set_title("Content Image", fontsize=13)
ax1.axis('off')
ax2.imshow(deprocess_image(style_image))
ax2.set_title("Style Image (Kandinsky)", fontsize=13)
ax2.axis('off')
plt.tight_layout()
plt.show()
print("Images loaded!")
```

### Cell 3: Build the Feature Extractor
We extract features from specific VGG19 layers.

```python
# Which layers to use for content and style
CONTENT_LAYERS = ['block4_conv2']
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def build_feature_extractor():
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False  # FROZEN! We never update VGG19.
    
    # Get outputs from specific layers
    content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    
    model = keras.Model(inputs=vgg.input, outputs=content_outputs + style_outputs)
    return model

feature_extractor = build_feature_extractor()
print(f"Feature extractor built!")
print(f"Content layers: {CONTENT_LAYERS}")
print(f"Style layers: {STYLE_LAYERS}")
```

### Cell 4: Define Loss Functions
```python
def gram_matrix(feature_map):
    """Compute the Gram Matrix for a feature map."""
    # feature_map shape: (1, height, width, channels)
    channels = int(feature_map.shape[-1])
    a = tf.reshape(feature_map, [-1, channels])  # (height*width, channels)
    gram = tf.matmul(a, a, transpose_a=True)      # (channels, channels)
    n = tf.cast(tf.shape(a)[0], tf.float32)
    return gram / n

def compute_content_loss(content_features, generated_features):
    return tf.reduce_mean(tf.square(content_features - generated_features))

def compute_style_loss(style_features, generated_features):
    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)
    return tf.reduce_mean(tf.square(style_gram - generated_gram))

print("Loss functions defined!")
print("gram_matrix: captures which features appear together")
print("content_loss: how different is the content?")
print("style_loss: how different is the style?")
```

### Cell 5: Run the Style Transfer Optimization
```python
# Weights
ALPHA = 1e3      # Content weight
BETA = 1e-2      # Style weight
LEARNING_RATE = 5.0
ITERATIONS = 500

# Extract target features (these stay fixed)
all_features = feature_extractor(content_image)
target_content = all_features[:len(CONTENT_LAYERS)]

all_features = feature_extractor(style_image)
target_style = all_features[len(CONTENT_LAYERS):]

# Start with a copy of the content image
generated = tf.Variable(content_image, dtype=tf.float32)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Snapshots for visualization
snapshots = []

print("Starting style transfer optimization...")
for i in range(ITERATIONS):
    with tf.GradientTape() as tape:
        # Extract features from generated image
        gen_features = feature_extractor(generated)
        gen_content = gen_features[:len(CONTENT_LAYERS)]
        gen_style = gen_features[len(CONTENT_LAYERS):]
        
        # Content loss
        content_loss = sum(compute_content_loss(tc, gc) 
                         for tc, gc in zip(target_content, gen_content))
        
        # Style loss
        style_loss = sum(compute_style_loss(ts, gs)
                        for ts, gs in zip(target_style, gen_style))
        
        # Total loss
        total_loss = ALPHA * content_loss + BETA * style_loss
    
    # Update the image pixels (NOT the network!)
    grads = tape.gradient(total_loss, generated)
    optimizer.apply_gradients([(grads, generated)])
    
    if i % 100 == 0:
        print(f"  Iteration {i:>4d} | Content: {content_loss:.2f} | Style: {style_loss:.2f} | Total: {total_loss:.2f}")
        snapshots.append((i, deprocess_image(generated.numpy())))

# Final snapshot
snapshots.append((ITERATIONS, deprocess_image(generated.numpy())))
print("Style transfer complete!")
```

### Cell 6: View the Progress
```python
fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
for idx, (iteration, img) in enumerate(snapshots):
    axes[idx].imshow(img)
    axes[idx].set_title(f"Iter {iteration}", fontsize=10)
    axes[idx].axis('off')
plt.suptitle("Style Transfer Progress", fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 7: Final Comparison
```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.imshow(deprocess_image(content_image))
ax1.set_title("Content Image", fontsize=13)
ax1.axis('off')

ax2.imshow(deprocess_image(style_image))
ax2.set_title("Style Image", fontsize=13)
ax2.axis('off')

ax3.imshow(snapshots[-1][1])
ax3.set_title("Generated Image!", fontsize=13)
ax3.axis('off')

plt.suptitle("Neural Style Transfer Result", fontsize=15)
plt.tight_layout()
plt.show()
print("The generated image has the CONTENT of the photo but the STYLE of the painting!")
```

---

## PART 3 -- Observation Questions (5 minutes)

1. Look at the progress images (Cell 6). At which iteration did the style start becoming visible?
*Answer:* __________________________________________________________________

2. What happens if you increase BETA (style weight) to 1.0? Would the output look more like the content image or more like the style image?
*Answer:* __________________________________________________________________

3. Why does NST take hundreds of iterations while training a classifier takes a few epochs?
*Answer:* __________________________________________________________________

---

## Task Completion Criteria

- [ ] Part 1: All 3 concept check questions answered
- [ ] Part 2: All 7 code cells executed successfully
- [ ] Part 2: Final comparison plot shows content, style, and generated images
- [ ] Part 3: All 3 observation questions answered

---
*Session 28 | Deep Learning Using Neural Networks | Aptech*
