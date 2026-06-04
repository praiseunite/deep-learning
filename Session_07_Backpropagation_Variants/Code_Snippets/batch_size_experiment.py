"""
============================================================
  FILE: batch_size_experiment.py
  SESSION: 07 — Backpropagation Variants
  PURPOSE: Test the difference in training speed and accuracy
           when using SGD, Mini-Batch, and Full Batch GD.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import time

print("=" * 50)
print("  Aptech AI Lab: The Batch Size Experiment")
print("=" * 50)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("Loading Fashion MNIST data...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# ─── 2. BUILD MODEL FUNCTION ──────────────────────────────────────────────────
def build_compiled_model():
    """Returns a fresh, un-trained neural network."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# ─── 3. EXPERIMENT SETUP ──────────────────────────────────────────────────────
# IN-CLASS TASK: Change this number!
# Try: 1 (SGD), 32 (Mini-batch), 1024 (Large Mini-batch), 60000 (Full Batch)

CURRENT_BATCH_SIZE = 32  # <--- CHANGE THIS VALUE

print(f"\n🚀 Running Training with Batch Size: {CURRENT_BATCH_SIZE}")

# Get a fresh model
model = build_compiled_model()

# Start stopwatch
start_time = time.time()

# Train for exactly 3 epochs
# Notice the batch_size parameter!
history = model.fit(X_train, y_train, epochs=3, batch_size=CURRENT_BATCH_SIZE)

# Stop stopwatch
end_time = time.time()
time_taken = end_time - start_time

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Print Report
print("\n" + "=" * 50)
print("  EXPERIMENT RESULTS")
print("=" * 50)
print(f"Batch Size Used : {CURRENT_BATCH_SIZE}")
print(f"Total Time      : {time_taken:.2f} seconds")
print(f"Final Accuracy  : {accuracy*100:.2f}%")
print("=" * 50)
