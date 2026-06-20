"""
============================================================
  FILE: end_to_end_training.py
  SESSION: 13 — Workshop: End-to-End Training
  PURPOSE: Starter code for the Try It Yourself Lab.
           Students must fill in the missing pieces to
           successfully train the model.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import os

# Suppress TF logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: CIFAR-10 Training Run")
print("=" * 50)

# ─── STEP 1: LOAD DATA ────────────────────────────────────────────────────────
print("Loading CIFAR-10 dataset...")
# CIFAR-10 contains 60,000 images of 10 different categories (dogs, cats, cars, etc.)
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0


# ─── STEP 2 & 3: BUILD ARCHITECTURE & ADD REGULARIZATION ──────────────────────
print("Building Model Architecture...")
model = keras.Sequential()

# Flatten the 32x32 color images into a 1D array
model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))

# FIXME 1: Add the standard activation function to these hidden layers!
# FIXME 3: Add a keras.layers.Dropout(0.3) layer between the first and second hidden layer!
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(64))

# FIXME 2: This is the output layer for 10 categories. Add the correct activation function!
model.add(keras.layers.Dense(10))


# ─── STEP 4: COMPILE THE MODEL ────────────────────────────────────────────────
# FIXME 4: Compile the model using 'adam', 'sparse_categorical_crossentropy', and ['accuracy']
print("Compiling Model...")
# model.compile(...)


# ─── STEP 5: TRAIN THE MODEL ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STARTING TRAINING LOOP")
print("=" * 50 + "\n")

# FIXME 5: Call model.fit() with X_train, y_train, epochs=10, and validation_data=(X_test, y_test)
history = None # REPLACE THIS LINE WITH YOUR model.fit() COMMAND

print("\nTraining Complete!")
