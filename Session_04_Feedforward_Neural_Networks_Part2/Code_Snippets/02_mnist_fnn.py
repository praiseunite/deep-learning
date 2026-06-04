"""
============================================================
  FILE: 02_mnist_fnn.py
  SESSION: 04 — Feedforward Neural Networks
  PURPOSE: Build, compile, and train a complete Feedforward
           Neural Network to classify handwritten digits using 
           the Keras Sequential API.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

print("=" * 50)
print("  Aptech AI Lab: MNIST Digit Classifier")
print("=" * 50)

# ─── 1. LOAD AND PREPARE DATA ─────────────────────────────────────────────────
print("\n[1/4] Loading MNIST dataset...")
mnist = keras.datasets.mnist

# load_data() automatically splits the 70k images into Training (60k) and Testing (10k)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training images: {X_train.shape[0]} | Image size: {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"Testing images: {X_test.shape[0]}")

# Normalize the data: Neural networks learn best when numbers are small.
# Pixel values range from 0 to 255. Dividing by 255.0 squishes them between 0.0 and 1.0.
print("Normalizing pixel values (0-255 -> 0.0-1.0)...")
X_train = X_train / 255.0
X_test = X_test / 255.0


# ─── 2. BUILD THE NEURAL NETWORK ──────────────────────────────────────────────
print("\n[2/4] Building the Feedforward Neural Network architecture...")

model = keras.Sequential([
    # Input Layer: 
    # An image is a 28x28 grid (2D). We must "flatten" it into a single line
    # of 784 pixels (1D) because a standard FNN takes a 1D vector as input.
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # Hidden Layer:
    # 128 neurons. 'Dense' means every one of the 784 inputs connects to all 128 neurons.
    # We use 'relu' to introduce non-linearity so it can learn complex patterns.
    keras.layers.Dense(128, activation='relu'),
    
    # Output Layer:
    # 10 neurons, one for each possible digit (0, 1, 2, ..., 9).
    # 'softmax' converts the raw outputs into probabilities that sum to 100%.
    keras.layers.Dense(10, activation='softmax')
])

print("Model architecture built successfully!")
# This prints a table showing the layers and how many parameters (weights+biases) exist.
model.summary() 


# ─── 3. COMPILE THE MODEL ─────────────────────────────────────────────────────
print("\n[3/4] Compiling the model (attaching Loss and Optimizer)...")

model.compile(
    # The Optimizer is the specific Gradient Descent algorithm used to update weights.
    # 'adam' is currently the most popular and efficient choice for most tasks.
    optimizer='adam',
    
    # The Loss Function measures how wrong the model is. 
    # Sparse Categorical Crossentropy is the standard for multi-class classification.
    loss='sparse_categorical_crossentropy',
    
    # Metrics are human-readable scores we want to monitor during training.
    metrics=['accuracy']
)


# ─── 4. TRAIN AND EVALUATE ────────────────────────────────────────────────────
print("\n[4/4] Commencing Training (The Learning Phase)...")

# epochs=5 means the model will look at all 60,000 images 5 times over.
history = model.fit(X_train, y_train, epochs=5)

print("\n--- Training Complete! ---")
print("Let's test the model on the 10,000 images it has NEVER seen before.")

# evaluate() calculates the loss and accuracy on the test set without updating weights.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(f"\n✅ FINAL TEST ACCURACY: {test_acc*100:.2f}%")

if test_acc > 0.95:
    print("Excellent! Your AI is over 95% accurate at reading human handwriting.")
