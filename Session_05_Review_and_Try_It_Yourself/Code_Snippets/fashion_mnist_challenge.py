"""
============================================================
  FILE: fashion_mnist_challenge.py
  SESSION: 05 — Review & Try It Yourself
  PURPOSE: Solution script for the Fashion MNIST assignment.
           Demonstrates adapting an FNN to a new dataset.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("  Aptech AI Lab: Fashion MNIST Classifier")
print("=" * 50)

# ─── 1. LOAD AND PREPARE DATA ─────────────────────────────────────────────────
print("Loading Fashion MNIST data...")
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Category names for Fashion MNIST (0-9 mapping)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize pixel values to 0.0 - 1.0
print("Normalizing data...")
X_train = X_train / 255.0
X_test = X_test / 255.0


# ─── 2. BUILD THE NEURAL NETWORK ──────────────────────────────────────────────
print("Building the model architecture (2 Hidden Layers)...")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),    # Flatten 2D to 1D
    keras.layers.Dense(128, activation='relu'),    # Hidden Layer 1
    keras.layers.Dense(64, activation='relu'),     # Hidden Layer 2
    keras.layers.Dense(10, activation='softmax')   # Output Layer (10 classes)
])


# ─── 3. COMPILE THE MODEL ─────────────────────────────────────────────────────
print("Compiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ─── 4. TRAIN THE MODEL ───────────────────────────────────────────────────────
print("\n--- STARTING TRAINING (10 Epochs) ---")
# Training for 10 epochs as per assignment requirements
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)


# ─── 5. EVALUATE THE MODEL ────────────────────────────────────────────────────
print("\n--- EVALUATING ON TEST DATA ---")
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print(f"\n✅ FINAL TEST ACCURACY: {test_acc*100:.2f}%")


# ─── 6. VISUALIZE A PREDICTION ────────────────────────────────────────────────
# Let's predict a random image to see the result visually
image_idx = 12  # Change this to see different images
image = X_test[image_idx]
actual_label = y_test[image_idx]

# Get the network's prediction
prediction = model.predict(image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

actual_name = class_names[actual_label]
predicted_name = class_names[predicted_label]

# Plot it
plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray')
color = 'green' if predicted_label == actual_label else 'red'
plt.title(f"Predicted: {predicted_name}\nActual: {actual_name}", color=color)
plt.axis('off')
print("\nOpening image visualization window...")
plt.show()
