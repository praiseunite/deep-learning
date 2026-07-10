# Lab Guide 06: "The Art Machine" -- Build an AI Art Filter
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Sessions 28-30 (Neural Style Transfer, VGG19, AdaIN)
### Estimated Time: 1.5-2 hours (including deployment)
---

> *"By the end of this guide, you will have built an AI art filter app. Upload any photo and any painting, and the AI mashes them together -- your photo in the style of Van Gogh, Picasso, or any artwork you choose. You will deploy it as a web app that anyone can use from their phone."*

---

## What You Will Build

An **Arbitrary Style Transfer** app. Upload a photo (your selfie, a landscape, anything) and a painting (Van Gogh's Starry Night, a Picasso, any artwork), and the AI transfers the painting's style onto your photo -- instantly. It uses a pre-trained model from TensorFlow Hub that handles any style without retraining.

---

## What You Need Before Starting

- [ ] A **Kaggle** account with phone verification (for GPU)
- [ ] A **Hugging Face** account (free)
- [ ] 2-3 photos on your computer or phone (any photos)
- [ ] 1-2 paintings saved as images (search Google for famous paintings)

---

# PHASE 1: Get the Data

For style transfer, the "data" is just the images you want to combine. No dataset download needed.

### Cell 1: Set Up the Notebook

```python
# ============================================================
# CELL 1: IMPORT LIBRARIES
# What this cell does: Loads all tools and the pre-trained style model
# ============================================================

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print(f"TensorFlow: {tf.__version__}")

# Load the Arbitrary Style Transfer model from TensorFlow Hub.
# This model was trained by Google's Magenta team.
# It can apply ANY painting's style to ANY photo in about 1 second.
# We do NOT need to train anything -- it's already fully trained.
style_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)

print("Style Transfer model loaded!")
print("This model can apply any painting's style to any photo.")
```

---

# PHASE 2: Get Sample Images

### Cell 2: Download Sample Images

```python
# ============================================================
# CELL 2: DOWNLOAD SAMPLE IMAGES
# What this cell does: Gets a photo and a painting from the internet
# ============================================================

# Download a content image (the photo to be styled).
content_path = tf.keras.utils.get_file(
    'content.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/'
    'Golden_Gate_Bridge_%28cropped%29.jpg/800px-Golden_Gate_Bridge_%28cropped%29.jpg'
)

# Download a style image (the painting whose style we want to copy).
style_path = tf.keras.utils.get_file(
    'style.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/'
    'Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/'
    '800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
)

print("Downloaded:")
print(f"  Content: Golden Gate Bridge")
print(f"  Style: Van Gogh's Starry Night")
```

### Cell 3: Create the Image Loading Function

```python
# ============================================================
# CELL 3: IMAGE LOADING FUNCTION
# What this cell does: Reads and prepares images for the model
# ============================================================

def load_image(path, max_dim=512):
    """
    Load an image from disk and prepare it for the style model.
    
    Parameters:
    - path: file path to the image
    - max_dim: resize the longest side to this size (saves memory)
    """
    
    # Read the raw bytes from disk.
    img = tf.io.read_file(path)
    
    # Decode the bytes into pixel values (3 channels = RGB).
    img = tf.image.decode_image(img, channels=3)
    
    # Convert to float32 and scale to [0, 1].
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Resize so the longest side is max_dim pixels.
    # This prevents memory issues with very large photos.
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    
    # Add batch dimension: (H, W, 3) -> (1, H, W, 3).
    # The model expects a batch even for single images.
    img = img[tf.newaxis, :]
    
    return img

# Load both images.
content_image = load_image(content_path)
style_image = load_image(style_path)

print(f"Content shape: {content_image.shape}")
print(f"Style shape: {style_image.shape}")
```

### Cell 4: Display the Input Images

```python
# ============================================================
# CELL 4: DISPLAY INPUT IMAGES
# What this cell does: Shows the content photo and style painting
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.imshow(content_image[0])
ax1.set_title("Content Image\n(Your Photo)", fontsize=13)
ax1.axis('off')

ax2.imshow(style_image[0])
ax2.set_title("Style Image\n(The Painting)", fontsize=13)
ax2.axis('off')

plt.tight_layout()
plt.show()

print("The AI will transfer the painting's style onto the photo.")
```

---

# PHASE 3-4: Apply Style Transfer (No Training Needed!)

Unlike Labs 01-05, we don't build or train a model here. The TF Hub model is already trained. We just USE it.

### Cell 5: Apply Style Transfer

```python
# ============================================================
# CELL 5: APPLY STYLE TRANSFER
# What this cell does: Combines the photo and painting into styled art
# ============================================================

# The style model takes two inputs:
# 1. content_image = the photo to be styled
# 2. style_image = the painting whose style to apply
#
# It returns one output: the styled image.
# This happens in about 1 second on GPU!

stylized_image = style_model(
    tf.constant(content_image),
    tf.constant(style_image)
)[0]  # [0] gets the first (and only) output tensor.

print(f"Styled image shape: {stylized_image.shape}")
print("Style transfer complete!")
```

### Cell 6: Display the Result

```python
# ============================================================
# CELL 6: DISPLAY ALL THREE IMAGES
# What this cell does: Shows content, style, and result side by side
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(content_image[0])
axes[0].set_title("Content (Your Photo)", fontsize=13)
axes[0].axis('off')

axes[1].imshow(style_image[0])
axes[1].set_title("Style (The Painting)", fontsize=13)
axes[1].axis('off')

axes[2].imshow(stylized_image[0])
axes[2].set_title("Result (AI-Styled Art!)", fontsize=13)
axes[2].axis('off')

plt.suptitle("Neural Style Transfer: Photo + Painting = Art", fontsize=15)
plt.tight_layout()
plt.show()
```

---

# PHASE 5-6: Experiment

### Cell 7: Try Multiple Styles

```python
# ============================================================
# CELL 7: TRY DIFFERENT STYLES
# What this cell does: Applies multiple painting styles to the same photo
# ============================================================

# Download additional style images.
styles = {
    'Starry Night': style_path,
    'The Great Wave': tf.keras.utils.get_file(
        'wave.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/'
        'Tsunami_by_hokusai_19th_century.jpg/800px-Tsunami_by_hokusai_19th_century.jpg'
    ),
    'Kandinsky': tf.keras.utils.get_file(
        'kandinsky.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/'
        'Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/'
        '800px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
    )
}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Show original.
axes[0].imshow(content_image[0])
axes[0].set_title("Original", fontsize=11)
axes[0].axis('off')

# Apply each style.
for idx, (name, path) in enumerate(styles.items()):
    s_img = load_image(path)
    result = style_model(tf.constant(content_image), tf.constant(s_img))[0]
    axes[idx + 1].imshow(result[0])
    axes[idx + 1].set_title(f"Style: {name}", fontsize=11)
    axes[idx + 1].axis('off')

plt.suptitle("Same Photo, Different Styles", fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 8: Try Your Own Photos (Upload Method)

```python
# ============================================================
# CELL 8: USE YOUR OWN PHOTOS
# What this cell does: Explains how to upload and use your own images
# ============================================================

# TO USE YOUR OWN PHOTOS:
# 1. Click "Add Data" (top right in Kaggle)
# 2. Click "Upload" and drag your photo/painting
# 3. Give it a name and click "Create"
# 4. Use the path: /kaggle/input/YOUR-DATASET-NAME/your_file.jpg

# Example (uncomment and modify the paths):
# my_photo = load_image('/kaggle/input/my-photos/selfie.jpg')
# my_painting = load_image('/kaggle/input/my-photos/painting.jpg')
# my_art = style_model(tf.constant(my_photo), tf.constant(my_painting))[0]
# plt.imshow(my_art[0])
# plt.axis('off')
# plt.show()

print("Upload your own photos using Kaggle's 'Add Data' button!")
print("Then modify the paths in this cell to use them.")
```

---

# PHASE 7: Save for Deployment

For this lab, we don't save a custom model because we use the TF Hub model directly. The Hugging Face app will download the TF Hub model itself.

### Cell 9: Save a Test Image

```python
# ============================================================
# CELL 9: SAVE A TEST RESULT
# What this cell does: Saves the styled image for reference
# ============================================================

# Save the styled image so you can use it as an example.
result_img = (stylized_image[0].numpy() * 255).astype(np.uint8)
pil_img = Image.fromarray(result_img)
pil_img.save('styled_result.jpg')

print("Saved styled_result.jpg")
print("You can download this from the Output panel as a sample.")
```

---

# PHASE 8: Deploy to Hugging Face

## Step 8.1: Create Hugging Face Space

1. [huggingface.co](https://huggingface.co) -> **New Space**.
2. Name: `ai-art-filter`
3. SDK: **Gradio**
4. Hardware: **CPU Basic** (free)
5. **Create Space**.

## Step 8.2: Create `requirements.txt`

```
tensorflow==2.15.0
tensorflow-hub==0.16.1
gradio==4.44.0
numpy
Pillow
```

## Step 8.3: Create `app.py`

```python
# ============================================================
# app.py -- AI Art Filter Web App
# Upload a photo and a painting, get styled art instantly.
# ============================================================

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# -----------------------------------------------
# STEP 1: Load the pre-trained style model
# -----------------------------------------------
# This downloads the TF Hub model on first run (~100MB).
# After that, it's cached.
style_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)

# -----------------------------------------------
# STEP 2: Style transfer function
# -----------------------------------------------
def apply_style(content_img, style_img):
    """Apply the style of one image onto another."""
    
    if content_img is None or style_img is None:
        return None
    
    # Convert to float32 and scale to [0, 1].
    content = np.array(content_img).astype('float32') / 255.0
    style = np.array(style_img).astype('float32') / 255.0
    
    # Resize content to manageable size.
    content_pil = Image.fromarray(content_img).convert('RGB')
    max_dim = 512
    ratio = max_dim / max(content_pil.size)
    new_size = (int(content_pil.size[0] * ratio), int(content_pil.size[1] * ratio))
    content_pil = content_pil.resize(new_size)
    content = np.array(content_pil).astype('float32') / 255.0
    
    # Resize style image.
    style_pil = Image.fromarray(style_img).convert('RGB')
    style_pil = style_pil.resize((256, 256))
    style = np.array(style_pil).astype('float32') / 255.0
    
    # Add batch dimensions.
    content = content[np.newaxis, :]
    style = style[np.newaxis, :]
    
    # Apply style transfer.
    result = style_model(tf.constant(content), tf.constant(style))[0]
    
    # Convert back to uint8 image.
    result_img = (result[0].numpy() * 255).astype(np.uint8)
    
    return result_img

# -----------------------------------------------
# STEP 3: Gradio interface
# -----------------------------------------------
demo = gr.Interface(
    fn=apply_style,
    inputs=[
        gr.Image(label="Your Photo (Content)"),
        gr.Image(label="Painting (Style)")
    ],
    outputs=gr.Image(label="AI-Styled Result"),
    title="The Art Machine: AI Style Transfer",
    description="Upload any photo and any painting. The AI will transfer the painting's "
                "artistic style onto your photo. Try famous paintings like Van Gogh's "
                "Starry Night, Monet's Water Lilies, or Picasso's cubist works!",
)

demo.launch()
```

Commit and wait for the build (may take 3-5 minutes as it downloads the TF Hub model).

---

# PHASE 9: Test It Live

## Test 1: Use Saved Test Images

Download famous paintings from Google Images and upload them as the style image. Use any personal photo as the content.

## Test 2: Try Different Combinations

- Your selfie + Van Gogh's Starry Night
- A city photo + Picasso's cubism
- A nature photo + Monet's Water Lilies
- A pet photo + Japanese woodblock print

## Test 3: Share

Send the URL to friends. Let them upload their own photos and paintings!

## Test 4: Phone Test

Open the Hugging Face URL on your phone. Take a selfie and upload it directly. Find a painting on Google Images, screenshot it, and upload as the style.

---

# Troubleshooting

### Problem: App takes 2+ minutes to load initially
**Solution:** Normal. The TF Hub model (~100MB) downloads on first run. After caching, it loads in seconds.

### Problem: Result image looks nothing like the style
**Solution:** Make sure the style image is an actual painting/artwork, not a photo. The model works best with images that have strong artistic textures.

### Problem: "Out of memory" error
**Solution:** The `max_dim = 512` in `app.py` should prevent this. If it still happens, reduce to `max_dim = 384`.

### Problem: Colors look wrong
**Solution:** Make sure both images are RGB (color), not grayscale. The model expects 3-channel color images.

---

## What You Learned

| Concept | Where You Used It |
|---------|------------------|
| Neural Style Transfer (Session 28) | The entire lab is style transfer |
| VGG19 Features (Session 28) | The TF Hub model uses VGG-style feature extraction |
| Content vs Style Loss (Session 28) | The model was trained with these losses |
| Gram Matrix (Session 28) | Used internally for style representation |
| Transfer Learning (Session 17) | Using a pre-trained model without retraining |
| TF Hub (Session 11) | Loading a pre-trained model from the hub |

---
*Lab Guide 06 | Deep Learning Using Neural Networks | Aptech*
