"""
============================================================
  FILE: rnn_toy_sequence.py
  SESSION: 20 — Recurrent Neural Networks (RNNs)
  PURPOSE: Demonstrate a SimpleRNN learning a basic
           time-series sequence.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("===========================================")
print("  Aptech AI Lab: Sequence Prediction (RNN)")
print("===========================================")

# ─── 1. CREATE A TOY TIME-SERIES DATASET ──────────────────────────────────────
# We will teach the AI that [10, 20, 30] -> 40
# And [20, 30, 40] -> 50

# X represents the input sequences (Time Steps = 3, Features = 1)
X = np.array([
    [[10], [20], [30]],
    [[20], [30], [40]],
    [[30], [40], [50]],
    [[40], [50], [60]]
], dtype=float)

# y represents the target answer that comes next
y = np.array([40, 50, 60, 70], dtype=float)


# ─── 2. BUILD THE RNN MODEL ───────────────────────────────────────────────────
print("\nBuilding the SimpleRNN Model...")

model = keras.Sequential([
    # We use a SimpleRNN layer with 50 neurons.
    # input_shape=(3, 1) means we look at 3 time steps at a time, with 1 feature per step.
    keras.layers.SimpleRNN(50, activation='relu', input_shape=(3, 1)),
    
    # The output is a single number (the prediction)
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')


# ─── 3. TRAIN THE MODEL ───────────────────────────────────────────────────────
print("\nTraining the RNN to learn the pattern (100 Epochs)...")
# We use verbose=0 to hide the output of the 100 epochs so it doesn't flood the screen
model.fit(X, y, epochs=100, verbose=0)
print("Training Complete!")


# ─── 4. TEST THE MEMORY ───────────────────────────────────────────────────────
print("\nTesting the RNN on a sequence it has never seen...")

# We give it [70, 80, 90]. It should predict roughly 100.
test_sequence = np.array([[[70], [80], [90]]], dtype=float)

prediction = model.predict(test_sequence, verbose=0)

print("\n" + "="*50)
print(f"  Input Sequence: [70, 80, 90]")
print(f"  RNN Prediction: {prediction[0][0]:.2f}")
print("="*50)
