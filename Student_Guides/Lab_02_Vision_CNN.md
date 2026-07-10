# Lab Guide 02: "The Vision Machine" -- Build a Scene Classifier
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Sessions 14-19 (CNNs, VGG, ResNet, Transfer Learning, Fine-tuning)
### Estimated Time: 2-3 hours (including deployment)
---

> *"By the end of this guide, you will have a working AI that looks at a photo and tells you what kind of scene it is -- a forest, a glacier, a city street, or a mountain. You will deploy it to the internet so anyone can upload photos and test it."*

---

## What You Will Build

A **Scene Classifier**. Upload any photo and the AI tells you if it's a Building, Forest, Glacier, Mountain, Sea, or Street. It uses a technique called **Transfer Learning** -- borrowing the "eyes" of a model that Google already trained on millions of images, and teaching it your specific task.

---

## What You Need Before Starting

- [ ] A **Kaggle** account with phone verification (for GPU)
- [ ] A **Hugging Face** account (free)
- [ ] No software to install

---

# PHASE 1: Get the Data

## Step 1.1: Find the Dataset

1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Search for **"Intel Image Classification"** in the search bar.
3. Click on the dataset by **puneet6060** (or the most popular one with ~150MB).
4. Click **"New Notebook"** to create a notebook with data attached.

## Step 1.2: Turn on the GPU

1. Right panel -> **Settings** -> **Accelerator** -> **GPU T4 x2**.
2. Wait for restart.

## Step 1.3: Understand the Data Structure

The Intel Image Classification dataset is organized in folders:

```
seg_train/seg_train/
├── buildings/    (~2,191 images)
├── forest/       (~2,271 images)
├── glacier/      (~2,404 images)
├── mountain/     (~2,512 images)
├── sea/          (~2,274 images)
└── street/       (~2,382 images)

seg_test/seg_test/
├── buildings/
├── forest/
...
```

Each folder name IS the label. The images are 150x150 color photos of real-world scenes.

---

# PHASE 2: Explore the Data

### Cell 1: Import Libraries and Set Paths

```python
# ============================================================
# CELL 1: IMPORT LIBRARIES
# What this cell does: Loads all the tools we need
# ============================================================

# TensorFlow is the engine that runs neural networks.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# NumPy for math operations on arrays of numbers.
import numpy as np

# Matplotlib for drawing charts and displaying images.
import matplotlib.pyplot as plt

# os is a built-in Python library for working with file paths and directories.
import os

print(f"TensorFlow version: {tf.__version__}")

# Define where the data lives.
# Kaggle puts the dataset files in /kaggle/input/[dataset-name].
TRAIN_DIR = '/kaggle/input/intel-image-classification/seg_train/seg_train'
TEST_DIR = '/kaggle/input/intel-image-classification/seg_test/seg_test'

# List the class names (folder names).
CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes: {CLASS_NAMES}")
print(f"Number of classes: {NUM_CLASSES}")
```

### Cell 2: Count Images Per Class

```python
# ============================================================
# CELL 2: COUNT IMAGES
# What this cell does: Shows how many images are in each class
# ============================================================

print("Training data:")
total_train = 0
for class_name in CLASS_NAMES:
    # os.path.join combines folder paths correctly.
    class_path = os.path.join(TRAIN_DIR, class_name)
    
    # os.listdir() lists all files in a folder.
    count = len(os.listdir(class_path))
    total_train += count
    print(f"  {class_name:>12s}: {count:,} images")

print(f"  {'TOTAL':>12s}: {total_train:,} images")

print("\nTest data:")
total_test = 0
for class_name in CLASS_NAMES:
    class_path = os.path.join(TEST_DIR, class_name)
    count = len(os.listdir(class_path))
    total_test += count
    print(f"  {class_name:>12s}: {count:,} images")

print(f"  {'TOTAL':>12s}: {total_test:,} images")
```

### Cell 3: Display Sample Images

```python
# ============================================================
# CELL 3: VISUALIZE SAMPLE IMAGES
# What this cell does: Shows 2 example images from each class
# ============================================================

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for col, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(TRAIN_DIR, class_name)
    image_files = os.listdir(class_path)
    
    for row in range(2):
        # Read one image file.
        img_path = os.path.join(class_path, image_files[row])
        
        # tf.io.read_file reads raw bytes from disk.
        img = tf.io.read_file(img_path)
        
        # tf.image.decode_jpeg converts those bytes into a 3D array of pixel values.
        img = tf.image.decode_jpeg(img, channels=3)
        
        axes[row, col].imshow(img.numpy())
        axes[row, col].set_title(class_name, fontsize=11)
        axes[row, col].axis('off')

plt.suptitle("Sample Images from Each Class", fontsize=14)
plt.tight_layout()
plt.show()
```

---

# PHASE 3: Prepare the Data

### Cell 4: Create Data Pipelines

```python
# ============================================================
# CELL 4: CREATE DATA PIPELINES
# What this cell does: Loads images in batches, resizes, and normalizes
# ============================================================

# We define the image size that our model expects.
# MobileNetV2 was originally trained on 224x224 images, but it can accept
# 150x150 as well. We use 150x150 since our images are already that size.
IMG_SIZE = 150
BATCH_SIZE = 32

# keras.utils.image_dataset_from_directory does 3 things automatically:
# 1. Reads images from folders
# 2. Uses the folder names as labels
# 3. Splits them into batches
#
# It is the easiest way to load image data in Keras.

train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,              # Path to the training folder
    image_size=(IMG_SIZE, IMG_SIZE),  # Resize all images to 150x150
    batch_size=BATCH_SIZE,  # Load 32 images at a time
    label_mode='int',       # Labels as integers (0, 1, 2, 3, 4, 5)
    shuffle=True            # Randomize the order (prevents learning order-based patterns)
)

test_dataset = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False           # Don't shuffle test data (we want consistent evaluation)
)

# Verify the class names match what we expect.
print(f"Class names: {train_dataset.class_names}")
```

### Cell 5: Normalize and Optimize

```python
# ============================================================
# CELL 5: NORMALIZE AND OPTIMIZE THE DATA PIPELINE
# What this cell does: Scales pixels to [0,1] and speeds up loading
# ============================================================

# MobileNetV2 expects pixel values between 0 and 1.
# Raw images have pixels from 0 to 255.
# This function divides every pixel by 255.
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply normalization to both datasets.
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# .cache() keeps the dataset in memory after the first load (faster epochs).
# .prefetch() loads the next batch while the GPU processes the current batch.
# AUTOTUNE lets TensorFlow automatically choose the number of parallel calls.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Verify: grab one batch and check the values.
for images, labels in train_dataset.take(1):
    print(f"Batch shape: {images.shape}")    # Should be (32, 150, 150, 3)
    print(f"Labels shape: {labels.shape}")   # Should be (32,)
    print(f"Pixel range: {images.numpy().min():.2f} to {images.numpy().max():.2f}")
    # Should be 0.00 to 1.00
```

---

# PHASE 4: Build the Model

This is where **Transfer Learning** comes in. Instead of training a CNN from scratch (which would need millions of images), we borrow the "eyes" of MobileNetV2 -- a model Google already trained on 1.4 million images.

### Cell 6: Load the Pre-trained MobileNetV2

```python
# ============================================================
# CELL 6: LOAD PRE-TRAINED MOBILENETV2
# What this cell does: Downloads a model Google already trained
# ============================================================

# MobileNetV2 is a CNN that Google trained on ImageNet (1.4 million images,
# 1,000 categories). It already knows how to "see" edges, textures,
# shapes, and objects. We borrow that knowledge.

# weights='imagenet' = use the pre-trained weights (the learned knowledge).
# include_top=False = remove the final classification layer.
#   Why? Google trained it to classify 1,000 categories (dogs, cars, etc.).
#   We only need 6 categories. So we remove their "head" and add our own.
# input_shape=(150, 150, 3) = our images are 150x150 pixels, 3 color channels (RGB).

base_model = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FREEZE the base model. This means we will NOT change its weights during training.
# Why? It already knows how to see. If we change its weights, we might destroy
# that knowledge. We only want to train the new head we add on top.
base_model.trainable = False

print(f"MobileNetV2 loaded!")
print(f"Layers in base model: {len(base_model.layers)}")
print(f"Trainable parameters: {sum(p.numpy().size for p in base_model.trainable_weights):,}")
print("(Should be 0 because we froze the base)")
```

### Cell 7: Add Our Classification Head

```python
# ============================================================
# CELL 7: BUILD THE COMPLETE MODEL
# What this cell does: Adds our custom layers on top of MobileNetV2
# ============================================================

# We build the model as a chain:
# Input -> MobileNetV2 (frozen) -> Our Layers -> Output

model = keras.Sequential([
    # Layer 1: The pre-trained base.
    # This takes a 150x150x3 image and outputs a 5x5x1280 feature map.
    # Think of it as: "I see 1,280 different visual patterns in a 5x5 grid."
    base_model,
    
    # Layer 2: Global Average Pooling.
    # Converts the 5x5x1280 feature map into a single vector of 1,280 numbers.
    # It takes the average of each 5x5 grid, producing one number per pattern.
    # Why not Flatten? Flatten would create 5*5*1280 = 32,000 numbers (too many).
    layers.GlobalAveragePooling2D(),
    
    # Layer 3: Dropout for regularization.
    # Randomly turns off 30% of neurons during training to prevent overfitting.
    layers.Dropout(0.3),
    
    # Layer 4: A Dense hidden layer.
    # 128 neurons that learn to combine MobileNetV2's visual patterns
    # into features specific to our 6 scene categories.
    layers.Dense(128, activation='relu'),
    
    # Layer 5: Output layer.
    # 6 neurons, one per class.
    # Softmax converts raw numbers into probabilities that sum to 1.0.
    layers.Dense(NUM_CLASSES, activation='softmax')
], name="Scene_Classifier")

model.summary()

# Count only OUR trainable parameters (the head we added).
trainable = sum(p.numpy().size for p in model.trainable_weights)
non_trainable = sum(p.numpy().size for p in model.non_trainable_weights)
print(f"\nTrainable parameters (our head): {trainable:,}")
print(f"Non-trainable parameters (frozen MobileNetV2): {non_trainable:,}")
print("We only train ~0.5% of the total parameters!")
```

### Cell 8: Compile

```python
# ============================================================
# CELL 8: COMPILE THE MODEL
# What this cell does: Configures how the model learns
# ============================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled!")
print("Ready to train.")
```

---

# PHASE 5: Train the Model

### Cell 9: Train

```python
# ============================================================
# CELL 9: TRAIN THE MODEL
# What this cell does: Shows the training data to the model repeatedly
# ============================================================

# We only train for 10 epochs because transfer learning converges fast.
# The model already "sees" -- we just need to teach it our 6 categories.
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    verbose=1
)

print("\nTraining complete!")
```

### Cell 10: Plot Training History

```python
# ============================================================
# CELL 10: PLOT TRAINING CURVES
# What this cell does: Visualizes accuracy and loss over time
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

final_acc = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {final_acc * 100:.1f}%")
```

---

# PHASE 6: Test the Model

### Cell 11: Evaluate on Test Set

```python
# ============================================================
# CELL 11: EVALUATE ON TEST DATA
# What this cell does: Measures accuracy on unseen images
# ============================================================

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)

print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Loss: {test_loss:.4f}")
```

### Cell 12: Visualize Predictions

```python
# ============================================================
# CELL 12: SHOW PREDICTIONS ON REAL IMAGES
# What this cell does: Displays model guesses next to true labels
# ============================================================

# Grab one batch of test images.
for images, labels in test_dataset.take(1):
    predictions = model.predict(images, verbose=0)
    break

fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for i, ax in enumerate(axes.flat):
    if i >= len(images):
        break
    
    # Display the image.
    ax.imshow(images[i].numpy())
    
    # Get prediction.
    pred_class = np.argmax(predictions[i])
    pred_name = CLASS_NAMES[pred_class]
    confidence = predictions[i][pred_class] * 100
    
    # Get true label.
    true_name = CLASS_NAMES[labels[i].numpy()]
    
    color = 'green' if pred_name == true_name else 'red'
    ax.set_title(f"Pred: {pred_name} ({confidence:.0f}%)\nTrue: {true_name}",
                 color=color, fontsize=9)
    ax.axis('off')

plt.suptitle("Predictions (Green = Correct, Red = Wrong)", fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 13: Confusion Matrix

```python
# ============================================================
# CELL 13: CONFUSION MATRIX
# What this cell does: Shows which scenes the model confuses
# ============================================================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Collect all predictions and true labels.
all_preds = []
all_labels = []

for images, labels in test_dataset:
    preds = np.argmax(model.predict(images, verbose=0), axis=1)
    all_preds.extend(preds)
    all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
```

---

# PHASE 7: Save the Model

### Cell 14: Save

```python
# ============================================================
# CELL 14: SAVE THE MODEL
# What this cell does: Saves the trained model to a file
# ============================================================

model.save('scene_classifier.keras')

print("Model saved as 'scene_classifier.keras'")
print("To download: Right panel -> Output -> click the file -> Download")
```

### Cell 15: Verify

```python
# ============================================================
# CELL 15: VERIFY THE SAVED MODEL
# What this cell does: Loads the saved model and checks it works
# ============================================================

loaded_model = keras.models.load_model('scene_classifier.keras')

# Test on one batch.
for images, labels in test_dataset.take(1):
    preds = loaded_model.predict(images[:3], verbose=0)
    for i in range(3):
        pred_name = CLASS_NAMES[np.argmax(preds[i])]
        true_name = CLASS_NAMES[labels[i].numpy()]
        print(f"  Predicted: {pred_name:>10s} | True: {true_name:>10s} | {'CORRECT' if pred_name == true_name else 'WRONG'}")

print("\nSaved model works correctly!")
```

---

# PHASE 8: Deploy to Hugging Face

## Step 8.1: Download the Model

1. In Kaggle, click **Output** on the right panel.
2. Find `scene_classifier.keras` and click **Download**.

## Step 8.2: Create a Hugging Face Space

1. Go to [huggingface.co](https://huggingface.co) -> log in.
2. Profile picture -> **New Space**.
3. Fill in:
   - **Space name:** `scene-classifier`
   - **SDK:** **Gradio**
   - **Hardware:** **CPU Basic** (free)
4. Click **Create Space**.

## Step 8.3: Upload Your Model

1. **Files** tab -> **Add file** -> **Upload files**.
2. Upload `scene_classifier.keras`.
3. Click **Commit changes**.

## Step 8.4: Create `requirements.txt`

1. **Add file** -> **Create a new file** -> name it `requirements.txt`.
2. Paste:

```
tensorflow==2.15.0
gradio==4.44.0
numpy
Pillow
```

3. Click **Commit new file**.

## Step 8.5: Create `app.py`

1. **Add file** -> **Create a new file** -> name it `app.py`.
2. Paste this complete code:

```python
# ============================================================
# app.py -- Scene Classifier Web App
# Upload a photo and the AI tells you what scene it is.
# ============================================================

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------------------------
# STEP 1: Load the trained model
# -----------------------------------------------
model = load_model('scene_classifier.keras')

# The 6 scene classes (must match the training order).
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# -----------------------------------------------
# STEP 2: Define the prediction function
# -----------------------------------------------
def classify_scene(image):
    """Takes an uploaded image and returns scene predictions."""
    
    # Convert to PIL Image and resize to 150x150 (what the model expects).
    image = Image.fromarray(image).resize((150, 150))
    
    # Convert to NumPy array and normalize pixels to [0, 1].
    img_array = np.array(image).astype('float32') / 255.0
    
    # Add a batch dimension: (150, 150, 3) -> (1, 150, 150, 3).
    # The model expects a "batch" even if it's just one image.
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run prediction.
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Return a dictionary: {class_name: confidence_score}.
    return {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

# -----------------------------------------------
# STEP 3: Build the Gradio interface
# -----------------------------------------------
demo = gr.Interface(
    fn=classify_scene,
    inputs=gr.Image(label="Upload a Scene Photo"),
    outputs=gr.Label(num_top_classes=6, label="Scene Classification"),
    title="Scene Classifier (CNN + Transfer Learning)",
    description="Upload a photo of a building, forest, glacier, mountain, sea, or street. "
                "The AI uses MobileNetV2 with transfer learning to classify the scene. "
                "Trained on the Intel Image Classification dataset.",
)

demo.launch()
```

3. Click **Commit new file**.

## Step 8.6: Wait for Build

Go to **App** tab. Wait 2-5 minutes for it to build and show "Running".

---

# PHASE 9: Test It Live

## Test 1: Save Test Images from Kaggle

Add this cell to your Kaggle notebook:

```python
# ============================================================
# SAVE TEST IMAGES FOR HUGGING FACE
# ============================================================
from PIL import Image as PILImage

for class_name in CLASS_NAMES:
    class_path = os.path.join(TEST_DIR, class_name)
    img_file = os.listdir(class_path)[0]
    img_path = os.path.join(class_path, img_file)
    
    img = PILImage.open(img_path)
    img.save(f'test_{class_name}.jpg')
    print(f"Saved test_{class_name}.jpg")

print("\nDownload from Output panel and upload to Hugging Face!")
```

## Test 2: Upload to Your App

1. Download the test images from Kaggle.
2. Go to your Hugging Face Space.
3. Upload each image and check if the prediction matches.

## Test 3: Try Your Own Photos

1. Take a photo of a street scene or a forest with your phone.
2. Upload it to the app.
3. See if the AI gets it right!

## Test 4: Share the URL

Copy the Hugging Face Space URL and send it to friends. They can test it from any device.

---

# Troubleshooting

### Problem: Low accuracy (below 80%)
**Solution:** Increase epochs to 15 or 20. Transfer learning usually reaches 85%+ on this dataset.

### Problem: "Invalid image" error on Hugging Face
**Solution:** Make sure your uploaded image is a standard JPEG or PNG. Very large images might need to be resized first.

### Problem: Model file too large to upload to Hugging Face
**Solution:** MobileNetV2 is ~14MB, which is well within Hugging Face's free limit. If it fails, try uploading via Git instead of the web interface.

### Problem: App crashes with "OOM" (Out of Memory)
**Solution:** CPU Basic should handle this model fine. If it still crashes, try using `tf.lite` (TensorFlow Lite) to convert the model to a smaller format.

---

## What You Learned

| Concept | Where You Used It |
|---------|------------------|
| CNNs (Sessions 14-16) | MobileNetV2 is a deep CNN architecture |
| Transfer Learning (Session 17) | Borrowing Google's pre-trained weights |
| Fine-tuning (Session 18) | Freezing base, training only our head |
| Data Pipelines (Session 19) | `image_dataset_from_directory`, caching, prefetching |
| Regularization (Session 8) | Dropout layer in our classification head |
| Deployment | Saving, Hugging Face, Gradio |

---

## Challenge: Fine-Tune the Base Model

Want even higher accuracy? Unfreeze the last 30 layers of MobileNetV2 and train again with a very small learning rate:

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=5, validation_data=test_dataset)
```

This typically pushes accuracy above 90%.

---
*Lab Guide 02 | Deep Learning Using Neural Networks | Aptech*
