"""
============================================================
  FILE: manual_rnn_forward_pass.py
  SESSION: 20 — Recurrent Neural Networks (RNNs)
  PURPOSE: Manually code the Hidden State mathematics of an
           RNN without using pre-built Keras layers.
============================================================
"""

import numpy as np

print("==================================================")
print("  Aptech AI Lab: The Math of the Hidden State     ")
print("==================================================")

# ─── 1. INITIALIZE WEIGHTS AND INPUTS ────────────────────────────────────────
# Imagine a sequence of 3 time steps (e.g., 3 words in a sentence)
# Each input x is a single number for simplicity.
sequence = [0.5, 0.2, 0.8]  # Time steps t=1, t=2, t=3

# The weights the network has learned. 
# W_x = Weight applied to the input
# W_h = Weight applied to the previous hidden state (memory)
# b = Bias
W_x = 0.6
W_h = 0.9
b = 0.1

# Initialize the Hidden State at t=0 (blank memory before the sequence starts)
h_prev = 0.0

print(f"Input Sequence: {sequence}")
print(f"Weights -> W_x: {W_x}, W_h: {W_h}, Bias: {b}\n")
print("--- Starting Manual Forward Pass Through Time ---")

# ─── 2. THE MANUAL FORWARD PASS ──────────────────────────────────────────────
# We will loop through time, applying the RNN equation:
# h_t = tanh( (W_h * h_prev) + (W_x * x_t) + b )

for t, x_t in enumerate(sequence, start=1):
    
    # Calculate the inner linear combination
    linear_comb = (W_h * h_prev) + (W_x * x_t) + b
    
    # Apply the tanh activation function to keep the memory bounded between -1 and 1
    h_current = np.tanh(linear_comb)
    
    print(f"Time Step {t}:")
    print(f"  Input (x_{t}): {x_t}")
    print(f"  Previous Memory (h_{t-1}): {h_prev:.4f}")
    print(f"  Math: tanh( ({W_h} * {h_prev:.4f}) + ({W_x} * {x_t}) + {b} )")
    print(f"  New Memory (h_{t}): {h_current:.4f}\n")
    
    # The current memory becomes the 'previous' memory for the next time step!
    h_prev = h_current

print("==================================================")
print(f"FINAL HIDDEN STATE (Memory of entire sequence): {h_prev:.4f}")
print("==================================================")

# ─── 3. THE VANISHING GRADIENT DEMONSTRATION ─────────────────────────────────
# Why does the gradient vanish? Look at W_h (0.9). 
# During Backpropagation Through Time, the error signal multiplies by W_h repeatedly.
print("\n--- Why Gradients Vanish ---")
gradient = 1.0 # Starting error signal
print(f"Initial Error Signal at end of sequence: {gradient}")
for t in range(50):
    gradient = gradient * W_h

print(f"Error Signal after traveling backward 50 time steps: {gradient:.6f}")
print("Notice how close to zero it is! The network forgot the beginning.")
