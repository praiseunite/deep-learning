"""
============================================================
  FILE: cnn_basics.py
  SESSION: 16 — Convolutional Neural Networks
  PURPOSE: Build a classic CNN architecture and analyze 
           how the shapes of the data change at each layer.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: The CNN Architect")
print("=" * 50)

# ─── 1. BUILD THE CNN ARCHITECTURE ────────────────────────────────────────────

model = keras.Sequential()

# Block 1: The first Convolution & Pooling
# Input: A 32x32 color image (3 color channels: RGB)
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Block 2: The second Convolution & Pooling
# Notice we increase the filters (from 32 to 64) as we go deeper!
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Block 3: The third Convolution
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Transition: Flatten the 3D features into a 1D array
model.add(keras.layers.Flatten())

# The "Brain" (Dense Classification Layers)
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax')) # 10 Output classes


# ─── 2. PRINT THE SUMMARY ─────────────────────────────────────────────────────
print("\nLook closely at the 'Output Shape' column to see how")
print("MaxPooling shrinks the spatial dimensions!\n")
model.summary()
