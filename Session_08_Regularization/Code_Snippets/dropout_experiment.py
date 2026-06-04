"""
============================================================
  FILE: dropout_experiment.py
  SESSION: 08 — Regularization
  PURPOSE: Intentionally overfit a model to demonstrate
           the rescue power of the Keras Dropout layer.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

print("=" * 50)
print("  Aptech AI Lab: The Dropout Experiment")
print("=" * 50)

# ─── 1. LOAD AND SABOTAGE THE DATA ────────────────────────────────────────────
print("Loading Fashion MNIST data...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# SABOTAGE: We only use the first 5,000 images for training instead of 60,000.
# Small data + Large Network = Guaranteed Overfitting.
X_train_small = X_train[:5000]
y_train_small = y_train[:5000]

# We will use the next 5,000 images as our Validation set to check for overfitting.
X_val = X_train[5000:10000]
y_val = y_train[5000:10000]


# ─── 2. THE BASELINE MODEL (NO DROPOUT) ───────────────────────────────────────
def build_baseline_model():
    """Builds a massive network with no regularization."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'), # Massive capacity
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ─── 3. THE REGULARIZED MODEL (WITH DROPOUT) ──────────────────────────────────
def build_dropout_model():
    """Builds the same network, but adds a Dropout layer."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'),
        
        # ---> IN-CLASS TASK: UN-COMMENT THE LINE BELOW <---
        # keras.layers.Dropout(0.4), 
        
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ─── 4. RUN THE EXPERIMENT ────────────────────────────────────────────────────
# Switch this to True when you are ready for Step 3!
run_dropout_experiment = False

if not run_dropout_experiment:
    print("\n🚀 Running BASELINE Model (No Dropout)")
    model = build_baseline_model()
else:
    print("\n🛡️ Running REGULARIZED Model (With Dropout)")
    model = build_dropout_model()

# Train for 20 epochs, monitoring validation data after every epoch
history = model.fit(X_train_small, y_train_small, 
                    epochs=20, 
                    validation_data=(X_val, y_val))

print("\n" + "=" * 50)
print("  FINAL RESULTS AT EPOCH 20")
print("=" * 50)
print(f"Training Accuracy   : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy : {history.history['val_accuracy'][-1]*100:.2f}%")
print("=" * 50)
