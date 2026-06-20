"""
============================================================
  FILE: my_first_model.py
  SESSION: 19 — Zero-to-Hero Workshop
  PURPOSE: The simplest possible MNIST neural network.
           Designed to be copy-pasted into Kaggle and run.
============================================================
"""

import tensorflow as tf
from tensorflow import keras

print("===========================================")
print("  Aptech AI Lab: Training My First Model")
print("===========================================")

# 1. DOWNLOAD THE DATA
print("Downloading MNIST Dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. PREPARE THE DATA
# The images are grids of pixels (0 to 255). We divide by 255 to make them (0 to 1).
# Neural networks prefer small numbers!
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. BUILD THE BRAIN
print("Building the Neural Network...")
model = keras.Sequential([
    # Flatten turns the 28x28 square image into a single straight line of 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # A hidden layer with 128 neurons
    keras.layers.Dense(128, activation='relu'),
    
    # The output layer with 10 neurons (because there are 10 digits: 0 through 9)
    keras.layers.Dense(10, activation='softmax')
])

# 4. COMPILE THE BRAIN
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. TRAIN THE BRAIN (The exciting part!)
print("\n--- STARTING TRAINING ---")
# This tells the model to look at the data 5 times (epochs=5)
model.fit(X_train, y_train, epochs=5)

print("\n--- TRAINING COMPLETE ---")
