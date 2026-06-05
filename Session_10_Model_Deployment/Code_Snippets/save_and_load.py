"""
============================================================
  FILE: save_and_load.py
  SESSION: 10 — Model Deployment
  PURPOSE: Demonstrate how to save a trained model to the 
           hard drive and resurrect it from memory.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: Model Cryogenics")
print("=" * 50)

# ─── 1. CREATE DUMMY DATA ─────────────────────────────────────────────────────
# We are creating a dataset that teaches the AI that:
# [0, 0] = 0
# [1, 1] = 1
X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2], [0.8, 0.8]])
y_train = np.array([0, 1, 0.5, 0.2, 0.8])

# ─── 2. BUILD AND TRAIN A TINY MODEL ──────────────────────────────────────────
print("\nBuilding and training a tiny model...")
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, verbose=0)
print("Training Complete!")

# Let's test it while it's still alive in memory
test_data = np.array([[0.9, 0.9]])
print(f"Original Model Prediction for [0.9, 0.9]: {model.predict(test_data, verbose=0)}")


# ─── 3. SAVE THE MODEL TO THE HARD DRIVE ──────────────────────────────────────
print("\nSaving models to disk...")

# Method A: The older H5 format (1 file)
model.save('my_legacy_model.h5')

# Method B: The modern TensorFlow SavedModel format (A folder)
model.save('my_modern_model')

print("Models saved successfully!")
print("Please minimize your code editor and check your file explorer.")

# ==========================================
# IN-CLASS TASK: ADD STEP 3 (RESURRECTION) CODE BELOW THIS LINE
# ==========================================
