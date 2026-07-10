# Lab Guide 01: "The Classifier" -- Build a Sign Language Translator
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Sessions 1-13 (ANN, Activation Functions, Backpropagation, Regularization, Keras)
### Estimated Time: 2-3 hours (including deployment)
---

> *"By the end of this guide, you will have a working AI that recognizes hand signs for letters of the alphabet. You will train it on Kaggle, deploy it to the internet, and send a link to your friends so they can test it on their phones."*

---

## What You Will Build

A **Sign Language Letter Classifier**. A user shows a photo of a hand sign (like the letter "A" in American Sign Language), and your AI instantly predicts which letter it is. The final product will be a public web app hosted on Hugging Face that anyone can use.

---

## What You Need Before Starting

- [ ] A **Kaggle** account (free) with phone verification completed (for GPU access)
- [ ] A **Hugging Face** account (free) -- sign up at [huggingface.co](https://huggingface.co)
- [ ] **No software to install** -- everything runs in your web browser

---

# PHASE 1: Get the Data

## Step 1.1: Find the Dataset on Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. In the search bar at the top, type: **"Sign Language MNIST"**
3. Click on the dataset called **"Sign Language MNIST"** by datamunge.
4. You will see a page describing the dataset. It contains 27,455 training images and 7,172 test images of hand signs representing letters A-Z (excluding J and Z because they require motion).
5. Click the **"New Notebook"** button on the dataset page. This creates a new notebook with the data already attached.

## Step 1.2: Verify the Data is Attached

In your new notebook, the data files are located at:
- `/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv`
- `/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv`

## Step 1.3: Turn on the GPU

1. On the right panel, click **Settings**.
2. Under **Accelerator**, select **GPU T4 x2** (or GPU P100).
3. The notebook will restart. Wait for it.

---

# PHASE 2: Explore the Data

### Cell 1: Load the Data

Create your first code cell and paste this:

```python
# ============================================================
# CELL 1: LOAD THE DATA
# What this cell does: Reads the CSV files into memory
# ============================================================

# 'pandas' is a library for reading spreadsheet-like data (CSV files).
# We give it the nickname 'pd' so we don't have to type 'pandas' every time.
import pandas as pd

# 'numpy' is a library for doing math on large arrays of numbers.
# We give it the nickname 'np'.
import numpy as np

# 'matplotlib.pyplot' is a library for drawing charts and showing images.
# We give it the nickname 'plt'.
import matplotlib.pyplot as plt

# Read the training data CSV file.
# Each row in this file is one image: the first column is the label (which letter),
# and the remaining 784 columns are the pixel values (28x28 = 784 pixels).
train_df = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

# Read the test data CSV file (same format).
test_df = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')

# Show the first 5 rows so we can see what the data looks like.
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nFirst 3 rows:")
train_df.head(3)
```

**What you should see:** A table with 785 columns. The first column is `label` (a number 0-24 representing a letter), and columns `pixel1` through `pixel784` are the grayscale pixel values.

### Cell 2: Understand the Labels

```python
# ============================================================
# CELL 2: UNDERSTAND THE LABELS
# What this cell does: Shows which numbers map to which letters
# ============================================================

# The dataset uses numbers 0-24 to represent letters A-Z.
# But J (9) and Z (25) are excluded because they require motion.
# So label 0 = A, label 1 = B, ..., label 8 = I, label 9 = K, etc.

# Create a mapping from number to letter.
# We skip J and Z in the alphabet.
alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# alphabet[0] = 'A', alphabet[1] = 'B', etc.
print("Label mapping:")
for i, letter in enumerate(alphabet):
    print(f"  Label {i:>2d} = Letter '{letter}'")

print(f"\nTotal classes: {len(alphabet)}")
```

### Cell 3: Visualize Some Hand Signs

```python
# ============================================================
# CELL 3: VISUALIZE HAND SIGNS
# What this cell does: Shows actual images from the dataset
# ============================================================

# Create a grid of 24 images (one for each letter).
fig, axes = plt.subplots(4, 6, figsize=(14, 10))

# For each of the 24 classes, find one example image and display it.
for label_num in range(24):
    # Find the first image in the training data with this label.
    row = train_df[train_df['label'] == label_num].iloc[0]
    
    # Extract the 784 pixel values (everything except the 'label' column).
    pixels = row.drop('label').values
    
    # Reshape the flat list of 784 numbers into a 28x28 grid (an image).
    image = pixels.reshape(28, 28)
    
    # Calculate which subplot position this image goes in.
    ax = axes[label_num // 6, label_num % 6]
    
    # Display the image in grayscale.
    ax.imshow(image, cmap='gray')
    
    # Put the letter as the title above the image.
    ax.set_title(f"'{alphabet[label_num]}'", fontsize=12)
    
    # Remove the axis ticks (they add clutter).
    ax.axis('off')

plt.suptitle("Sample Hand Signs from the Dataset", fontsize=15)
plt.tight_layout()
plt.show()

print("Each image is 28x28 pixels in grayscale.")
print(f"Training examples: {len(train_df):,}")
print(f"Test examples: {len(test_df):,}")
```

---

# PHASE 3: Prepare the Data

Neural networks cannot work directly with raw CSV data. We need to:
1. Separate the labels (answers) from the pixels (inputs)
2. Normalize the pixel values
3. Reshape into the correct format

### Cell 4: Separate Labels and Pixels

```python
# ============================================================
# CELL 4: SEPARATE LABELS AND PIXELS
# What this cell does: Splits the data into inputs (X) and answers (y)
# ============================================================

# TRAINING DATA:
# y_train = the labels (the correct answers). Just the 'label' column.
y_train = train_df['label'].values

# X_train = the pixel values (the inputs). Everything EXCEPT the 'label' column.
X_train = train_df.drop('label', axis=1).values

# TEST DATA (same process):
y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

print(f"X_train shape: {X_train.shape}")
# Should show (27455, 784) -- 27,455 images, each with 784 pixel values.

print(f"y_train shape: {y_train.shape}")
# Should show (27455,) -- 27,455 labels (one per image).

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
```

### Cell 5: Normalize the Pixels

```python
# ============================================================
# CELL 5: NORMALIZE THE PIXELS
# What this cell does: Scales pixel values from 0-255 to 0.0-1.0
# ============================================================

# Pixel values in images range from 0 (pure black) to 255 (pure white).
# Neural networks learn much better when inputs are small numbers (0 to 1).
# Why? Large numbers cause large weight updates, making training unstable.
# Dividing by 255.0 scales everything to the range [0.0, 1.0].

# .astype("float32") converts integers to decimal numbers (required for division).
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print(f"Before normalization: pixels ranged from 0 to 255")
print(f"After normalization: pixels range from {X_train.min():.1f} to {X_train.max():.1f}")
print("Normalization complete!")
```

### Cell 6: Count the Classes

```python
# ============================================================
# CELL 6: COUNT THE CLASSES
# What this cell does: Verifies how many unique classes (letters) exist
# ============================================================

NUM_CLASSES = len(np.unique(y_train))
INPUT_SIZE = X_train.shape[1]

print(f"Number of unique classes: {NUM_CLASSES}")
print(f"Input size (pixels per image): {INPUT_SIZE}")
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
```

---

# PHASE 4: Build the Model

Now we build the neural network. This is a Dense (fully connected) feedforward network -- the type you learned in Sessions 1-5.

### Cell 7: Import TensorFlow and Build the Architecture

```python
# ============================================================
# CELL 7: BUILD THE NEURAL NETWORK
# What this cell does: Creates the brain architecture
# ============================================================

# TensorFlow is the engine that runs neural networks.
# Keras is the user-friendly interface on top of TensorFlow.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")

# keras.Sequential() creates a model where layers are stacked one after another.
# Like building a tower of LEGO blocks -- data flows from bottom to top.
model = keras.Sequential(name="Sign_Language_Classifier")

# LAYER 1: First Hidden Layer
# Dense(256) = 256 neurons. Each neuron receives ALL 784 input pixels.
# activation='relu' = ReLU activation function.
#   What ReLU does: if the output is negative, it becomes 0. If positive, it stays.
#   Why ReLU? It prevents the "vanishing gradient" problem (Session 7).
# input_shape=(784,) = tells the network to expect 784 numbers as input.
model.add(layers.Dense(256, activation='relu', input_shape=(INPUT_SIZE,)))

# LAYER 2: Dropout (Regularization)
# Dropout(0.3) = during training, randomly turn off 30% of neurons each step.
# Why? This prevents overfitting (Session 8). The network cannot memorize --
#   it must learn GENERAL patterns because different neurons are off each time.
model.add(layers.Dropout(0.3))

# LAYER 3: Second Hidden Layer
# 128 neurons. Fewer than Layer 1 because we are "funneling" the information
# down from 784 -> 256 -> 128, extracting increasingly abstract features.
model.add(layers.Dense(128, activation='relu'))

# LAYER 4: Another Dropout
model.add(layers.Dropout(0.3))

# LAYER 5: Third Hidden Layer
model.add(layers.Dense(64, activation='relu'))

# LAYER 6: Output Layer
# Dense(24) = 24 neurons, one for each letter class.
# activation='softmax' = converts raw outputs into probabilities that sum to 1.0.
#   Example output: [0.01, 0.02, 0.90, 0.01, ...] means "90% sure it's letter C."
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Show a summary of the architecture.
model.summary()

# Count parameters: these are the "knobs" the network will learn to tune.
total_params = model.count_params()
print(f"\nTotal trainable parameters: {total_params:,}")
print("Each parameter is a number the AI will adjust during training.")
```

### Cell 8: Compile the Model

```python
# ============================================================
# CELL 8: COMPILE THE MODEL
# What this cell does: Tells the network HOW to learn
# ============================================================

# Compiling configures three things:
# 1. optimizer = the algorithm that adjusts weights (Adam is the best default)
# 2. loss = the function that measures "how wrong" the network is
# 3. metrics = what we want to see during training (accuracy)

model.compile(
    # 'adam' = Adaptive Moment Estimation optimizer.
    # It automatically adjusts the learning rate for each weight.
    # Think of it as a smart GPS that adjusts speed based on road conditions.
    optimizer='adam',
    
    # 'sparse_categorical_crossentropy' = the loss function for multi-class classification.
    # "sparse" = our labels are integers (0, 1, 2, ...) not one-hot vectors.
    # "categorical" = we are classifying into categories (letters).
    # "crossentropy" = the math formula that measures how wrong the probabilities are.
    loss='sparse_categorical_crossentropy',
    
    # metrics = what to display during training. 'accuracy' shows the % of correct predictions.
    metrics=['accuracy']
)

print("Model compiled!")
print("  Optimizer: Adam")
print("  Loss: Sparse Categorical Crossentropy")
print("  Metrics: Accuracy")
```

---

# PHASE 5: Train the Model

### Cell 9: Train!

```python
# ============================================================
# CELL 9: TRAIN THE MODEL
# What this cell does: Shows the data to the network so it can learn
# ============================================================

# model.fit() is the training command. It shows the training data to the network
# over and over, and the network adjusts its weights to get better.

history = model.fit(
    X_train,           # The input images (27,455 images, each 784 pixels)
    y_train,           # The correct labels (27,455 letters)
    
    epochs=20,         # How many times to go through the ENTIRE dataset.
                       # 1 epoch = the network sees all 27,455 images once.
                       # 20 epochs = it sees them 20 times total.
    
    batch_size=128,    # Process 128 images at a time, then update weights.
                       # Why not 1 at a time? Too slow and noisy.
                       # Why not all 27,455? Too much memory needed.
                       # 128 is a common sweet spot.
    
    validation_split=0.15,  # Hold back 15% of training data for validation.
                            # The network NEVER trains on this 15%.
                            # We use it to check if the network is overfitting.
    
    verbose=1          # Show a progress bar for each epoch.
)

print("\nTraining complete!")
```

### Cell 10: Plot the Training History

```python
# ============================================================
# CELL 10: PLOT TRAINING CURVES
# What this cell does: Visualizes how the model improved over time
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy over epochs
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss over epochs
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# What to look for:
# GOOD: Both training and validation curves move in the same direction.
# BAD: Training accuracy keeps rising but validation accuracy stops or drops.
#      This means OVERFITTING -- the model is memorizing, not learning.
print("If the lines diverge heavily, the model is overfitting.")
print("Our Dropout layers help prevent this!")
```

---

# PHASE 6: Test the Model

### Cell 11: Evaluate on Test Data

```python
# ============================================================
# CELL 11: EVALUATE ON TEST DATA
# What this cell does: Checks accuracy on images the model has NEVER seen
# ============================================================

# model.evaluate() runs the model on the test set and returns the loss and accuracy.
# The test set was completely separate -- the model never trained on these images.
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"\nThe model correctly classifies {test_accuracy * 100:.1f}% of hand signs it has NEVER seen before.")
```

### Cell 12: Show Predictions vs Reality

```python
# ============================================================
# CELL 12: VISUALIZE PREDICTIONS
# What this cell does: Shows the model's predictions next to the true answers
# ============================================================

# Get predictions for the first 20 test images.
predictions = model.predict(X_test[:20], verbose=0)

fig, axes = plt.subplots(4, 5, figsize=(14, 10))

for i, ax in enumerate(axes.flat):
    # Get the image and reshape it to 28x28.
    image = X_test[i].reshape(28, 28)
    
    # The model outputs 24 probabilities. np.argmax finds which has the highest.
    predicted_label = np.argmax(predictions[i])
    predicted_letter = alphabet[predicted_label]
    confidence = predictions[i][predicted_label] * 100
    
    # The true answer.
    true_label = y_test[i]
    true_letter = alphabet[true_label]
    
    # Color the title green if correct, red if wrong.
    color = 'green' if predicted_label == true_label else 'red'
    
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Pred: {predicted_letter} ({confidence:.0f}%)\nTrue: {true_letter}", 
                 color=color, fontsize=9)
    ax.axis('off')

plt.suptitle("Model Predictions vs True Labels (Green = Correct, Red = Wrong)", fontsize=13)
plt.tight_layout()
plt.show()
```

### Cell 13: Confusion Matrix

```python
# ============================================================
# CELL 13: CONFUSION MATRIX
# What this cell does: Shows which letters the model confuses with each other
# ============================================================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get predictions for ALL test images.
all_predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

# Build the confusion matrix.
# Each cell (i, j) = how many times the true label was i but the model predicted j.
cm = confusion_matrix(y_test, all_predictions)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(alphabet),
            yticklabels=list(alphabet))
plt.xlabel('Predicted Letter')
plt.ylabel('True Letter')
plt.title('Confusion Matrix: Which Letters Does the AI Mix Up?')
plt.tight_layout()
plt.show()

# Print a text report with precision and recall per class.
print("\nDetailed Report:")
print(classification_report(y_test, all_predictions, target_names=list(alphabet)))
```

---

# PHASE 7: Save the Model

### Cell 14: Save for Deployment

```python
# ============================================================
# CELL 14: SAVE THE MODEL
# What this cell does: Saves the trained brain to a file
# ============================================================

# model.save() writes the entire model (architecture + trained weights) to a file.
# The '.keras' format is the modern standard for TensorFlow/Keras models.
model.save('sign_language_model.keras')

print("Model saved as 'sign_language_model.keras'")
print("File location: /kaggle/working/sign_language_model.keras")
print("\nTo download: Click on 'Output' in the right panel -> click the file -> Download")
```

### Cell 15: Verify the Saved Model Works

```python
# ============================================================
# CELL 15: VERIFY THE SAVED MODEL
# What this cell does: Loads the saved model and confirms it still works
# ============================================================

# Load the model from the file we just saved.
loaded_model = keras.models.load_model('sign_language_model.keras')

# Test it on one image to make sure it works.
test_image = X_test[0:1]  # Take the first test image (keep the batch dimension)
prediction = loaded_model.predict(test_image, verbose=0)
predicted_letter = alphabet[np.argmax(prediction)]
true_letter = alphabet[y_test[0]]

print(f"Loaded model prediction: '{predicted_letter}'")
print(f"True answer: '{true_letter}'")
print(f"Match: {'YES' if predicted_letter == true_letter else 'NO'}")
print("\nThe saved model works correctly!")
```

---

# PHASE 8: Deploy to Hugging Face

Now we take the trained model out of Kaggle and put it on the internet where anyone can use it.

## Step 8.1: Download the Model from Kaggle

1. In your Kaggle notebook, look at the **right panel**.
2. Click on **Output** (or "Data" in some layouts).
3. You should see `sign_language_model.keras` listed.
4. Click the **Download** button (arrow icon) next to the file.
5. Save it to your computer (e.g., your Desktop or Downloads folder).

## Step 8.2: Create a Hugging Face Space

1. Go to [huggingface.co](https://huggingface.co) and log in.
2. Click your **profile picture** (top right) -> **New Space**.
3. Fill in:
   - **Space name:** `sign-language-classifier` (or any name you like)
   - **License:** Apache 2.0
   - **SDK:** Select **Gradio**
   - **Hardware:** Select **CPU Basic** (free tier -- our model is small enough)
4. Click **Create Space**.

## Step 8.3: Upload Your Model File

1. On your new Space page, click **Files** tab.
2. Click **Add file** -> **Upload files**.
3. Drag and drop your `sign_language_model.keras` file.
4. Click **Commit changes**.

## Step 8.4: Create the `requirements.txt` File

1. Click **Add file** -> **Create a new file**.
2. Name it: `requirements.txt`
3. Paste this content:

```
tensorflow==2.15.0
gradio==4.44.0
numpy
Pillow
```

4. Click **Commit new file**.

## Step 8.5: Create the `app.py` File

1. Click **Add file** -> **Create a new file**.
2. Name it: `app.py`
3. Paste this complete code:

```python
# ============================================================
# app.py -- Sign Language Classifier Web App
# This file creates a web page where users can upload hand sign
# images and the AI predicts which letter it is.
# ============================================================

# Gradio is the library that creates the web interface.
import gradio as gr

# NumPy for number operations.
import numpy as np

# PIL (Pillow) for image processing.
from PIL import Image

# TensorFlow/Keras to load and run our trained model.
from tensorflow.keras.models import load_model

# -----------------------------------------------
# STEP 1: Load the trained model
# -----------------------------------------------

# This loads the model file we uploaded from Kaggle.
model = load_model('sign_language_model.keras')

# The alphabet mapping (same as in training -- no J or Z).
ALPHABET = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# -----------------------------------------------
# STEP 2: Define the prediction function
# -----------------------------------------------

def predict_sign_language(image):
    """
    Takes an image uploaded by the user, processes it,
    and returns the predicted letter with confidence scores.
    """
    
    # Convert the image to grayscale (our model was trained on grayscale images).
    image = Image.fromarray(image).convert('L')
    
    # Resize to 28x28 pixels (the same size our model expects).
    image = image.resize((28, 28))
    
    # Convert to a NumPy array of numbers.
    img_array = np.array(image).astype('float32')
    
    # Normalize pixel values from 0-255 to 0-1 (same as training).
    img_array = img_array / 255.0
    
    # Flatten from 28x28 grid to a flat list of 784 numbers.
    img_array = img_array.reshape(1, 784)
    
    # Run the model to get predictions (24 probabilities).
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Create a dictionary mapping each letter to its confidence score.
    # Gradio's gr.Label component expects this format.
    result = {ALPHABET[i]: float(predictions[i]) for i in range(len(ALPHABET))}
    
    return result

# -----------------------------------------------
# STEP 3: Build the Gradio web interface
# -----------------------------------------------

demo = gr.Interface(
    # The function to call when a user uploads an image.
    fn=predict_sign_language,
    
    # The input component: an image upload box.
    inputs=gr.Image(label="Upload a Hand Sign Image"),
    
    # The output component: shows the top predictions with confidence bars.
    outputs=gr.Label(num_top_classes=5, label="Predicted Letter"),
    
    # The title displayed at the top of the web page.
    title="Sign Language Letter Classifier",
    
    # A description shown below the title.
    description="Upload an image of an American Sign Language (ASL) hand sign, "
                "and the AI will predict which letter (A-Y, excluding J and Z) it represents. "
                "Trained on the Sign Language MNIST dataset using a Dense Neural Network.",
)

# Launch the web app. Hugging Face will provide the public URL.
demo.launch()
```

4. Click **Commit new file**.

## Step 8.6: Wait for the Build

1. Go to the **App** tab of your Space.
2. Hugging Face will install the dependencies and start the app. This takes 2-5 minutes.
3. You will see a **Building** status, then **Running**.
4. Once running, you will see your web app with an image upload box!

---

# PHASE 9: Test It Live

## Test 1: Use Test Images from the Dataset

1. Go back to your Kaggle notebook.
2. Add this cell to save some test images as actual image files:

```python
# ============================================================
# SAVE TEST IMAGES FOR HUGGING FACE TESTING
# ============================================================
from PIL import Image as PILImage

for i in range(5):
    img = (X_test[i].reshape(28, 28) * 255).astype(np.uint8)
    pil_img = PILImage.fromarray(img, mode='L')
    pil_img.save(f'test_sign_{alphabet[y_test[i]]}.png')
    print(f"Saved test_sign_{alphabet[y_test[i]]}.png (True label: {alphabet[y_test[i]]})")

print("\nDownload these files from the Output panel and upload them to your Hugging Face app!")
```

3. Download these test images from Kaggle's Output panel.
4. Go to your Hugging Face Space.
5. Drag and drop each test image into the upload box.
6. The AI should predict the correct letter!

## Test 2: Share with Friends

1. Copy the URL of your Hugging Face Space (it looks like `https://huggingface.co/spaces/YOUR-USERNAME/sign-language-classifier`).
2. Send this link to a friend or classmate.
3. They can open it on their phone or laptop and test it without installing anything!

## Test 3: Try Real Hand Signs

1. Use your phone to take a photo of your own hand making a sign language letter.
2. Upload it to the app.
3. Note: Real photos may not work as well as the dataset images because:
   - The model was trained on 28x28 grayscale crops
   - Real photos have backgrounds, lighting differences, and different hand sizes
   - This is a learning moment: the model works well on similar data, but real-world deployment requires more training data diversity

---

# Troubleshooting

### Problem: "ModuleNotFoundError" on Hugging Face
**Solution:** Check your `requirements.txt`. Make sure `tensorflow` and `gradio` are listed. Commit and wait for rebuild.

### Problem: App shows "Error" when uploading an image
**Solution:** The image might be too large or in an unexpected format. Try uploading a simple 28x28 PNG. Check the Hugging Face Logs tab for the specific error message.

### Problem: Model accuracy is low (below 70%)
**Solution:** Try increasing epochs to 30, or increase the network size (e.g., 512 -> 256 -> 128 neurons). Make sure you normalized the pixels to [0, 1].

### Problem: "Out of memory" on Kaggle
**Solution:** Make sure GPU is enabled. Reduce batch_size from 128 to 64.

### Problem: Cannot download the model file from Kaggle
**Solution:** Make sure you ran the `model.save()` cell. Click on the **Output** tab on the right panel. If you don't see the file, click "Save Version" (top right), wait for it to complete, then check Output again.

---

## What You Learned

In this lab, you practiced every major concept from Sessions 1-13:

| Concept | Where You Used It |
|---------|------------------|
| Neurons & Layers (Sessions 1-3) | Building the Sequential model with Dense layers |
| Activation Functions (Session 6) | ReLU in hidden layers, Softmax in output |
| Backpropagation (Session 7) | Happens inside `model.fit()` -- the optimizer adjusts weights |
| Regularization (Session 8) | Dropout layers prevent overfitting |
| TensorFlow & Keras (Session 9) | The entire codebase uses Keras |
| Model Evaluation (Session 10) | `model.evaluate()`, confusion matrix |
| Data Preparation | Normalization, train/test split |
| Deployment | Saving model, Hugging Face, Gradio |

---

## Challenge: Extend This Project

If you want to push further, try these:

1. **Add BatchNormalization** layers after each Dense layer (Session 8 concept). Does accuracy improve?
2. **Try different optimizers:** Replace `'adam'` with `'sgd'` or `'rmsprop'`. Compare the training curves.
3. **Increase the dataset:** The Sign Language MNIST also has a "train" set with augmented images. Can you use data augmentation to generate even more?

---
*Lab Guide 01 | Deep Learning Using Neural Networks | Aptech*
