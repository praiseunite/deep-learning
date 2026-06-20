"""
============================================================
  FILE: depth_vs_width.py
  SESSION: 14 — Deep vs Shallow Networks
  PURPOSE: Compare the training efficiency of a single, 
           massive wide layer vs multiple narrow layers.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: Width vs Depth Showdown")
print("=" * 50)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
# We use Fashion MNIST (clothing items) as it requires learning complex shapes
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0


# ─── 2. BUILD THE SHALLOW (WIDE) MODEL ────────────────────────────────────────
print("\n[1/2] Building and Training the SHALLOW Model...")
print("Architecture: 1 Hidden Layer (512 neurons)")

shallow_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'), # Massive single layer
    keras.layers.Dense(10, activation='softmax')
])

shallow_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
shallow_history = shallow_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)


# ─── 3. BUILD THE DEEP (NARROW) MODEL ─────────────────────────────────────────
print("\n" + "="*50)
print("[2/2] Building and Training the DEEP Model...")
print("Architecture: 4 Hidden Layers (128 -> 64 -> 32 -> 16)")

deep_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

deep_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
deep_history = deep_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)


# ─── 4. COMPARE RESULTS ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  SHOWDOWN RESULTS (FINAL VALIDATION ACCURACY)")
print("=" * 50)
print(f"SHALLOW MODEL ACCURACY: {shallow_history.history['val_accuracy'][-1]:.4f}")
print(f"DEEP MODEL ACCURACY:    {deep_history.history['val_accuracy'][-1]:.4f}")
print("=" * 50)
