"""
============================================================
  FILE: activation_playground.py
  SESSION: 06 — Activation Functions Deep Dive
  PURPOSE: Loops through different advanced activation 
           functions, trains a model for each, and logs
           the time and accuracy to compare trade-offs.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import time

print("=" * 50)
print("  Aptech AI Lab: Activation Function Race")
print("=" * 50)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("Loading Fashion MNIST data...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# ─── 2. MODEL BUILDER FUNCTION ────────────────────────────────────────────────
def build_and_compile_model(activation_function):
    """
    Builds a standard 2-hidden-layer network.
    The activation function is passed in as a variable so we can change it dynamically.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=activation_function),
        keras.layers.Dense(64, activation=activation_function),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ─── 3. THE EXPERIMENT LOOP ───────────────────────────────────────────────────
# We test Standard ReLU, Leaky ReLU, ELU, and Swish
activations_to_test = ['relu', 'leaky_relu', 'elu', 'swish']

print("\n--- STARTING THE ACTIVATION RACE ---")
print("Training each model for 5 epochs. Please wait...\n")

for act in activations_to_test:
    print(f"Training model with [ {act.upper()} ]...")
    
    # Build a fresh model for this activation
    model = build_and_compile_model(act)
    
    # Start the stopwatch
    start_time = time.time()
    
    # Train the model (verbose=0 hides the progress bar to keep output clean)
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Stop the stopwatch
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Evaluate final accuracy on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Print results
    print(f"  -> Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"  -> Time Taken:          {time_taken:.2f} seconds\n")

print("=" * 50)
print("Race Complete! Analyze your results.")
