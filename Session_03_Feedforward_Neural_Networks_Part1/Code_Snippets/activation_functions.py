"""
============================================================
  FILE: activation_functions.py
  SESSION: 03 — Feedforward Neural Networks
  PURPOSE: Visualize how different activation functions
           shape the data passing through a neuron.
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── 1. Define the Activation Functions ───────────────────────────────────────
def step_function(x):
    """Outputs 1 if x > 0, else 0."""
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    """Outputs a probability between 0 and 1."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Outputs x if x > 0, else 0."""
    return np.maximum(0, x)

def tanh(x):
    """Outputs a value between -1 and 1."""
    return np.tanh(x)

# ─── 2. Generate Data ─────────────────────────────────────────────────────────
# Create an array of 100 values evenly spaced between -10 and 10
z = np.linspace(-10, 10, 100)

# Calculate outputs
y_step = step_function(z)
y_sigmoid = sigmoid(z)
y_relu = relu(z)
y_tanh = tanh(z)

# ─── 3. Plotting ──────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Activation Functions Comparison", fontsize=16, fontweight='bold', color='white')

# Top Left: Step
axs[0, 0].plot(z, y_step, color='#00FFFF', linewidth=2)
axs[0, 0].set_title("Step Function", color='white')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_ylim(-0.2, 1.2)

# Top Right: Sigmoid
axs[0, 1].plot(z, y_sigmoid, color='#FFA500', linewidth=2)
axs[0, 1].set_title("Sigmoid", color='white')
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].set_ylim(-0.2, 1.2)

# Bottom Left: Tanh
axs[1, 0].plot(z, y_tanh, color='#FF00FF', linewidth=2)
axs[1, 0].set_title("Tanh", color='white')
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_ylim(-1.2, 1.2)

# Bottom Right: ReLU
axs[1, 1].plot(z, y_relu, color='#32CD32', linewidth=2)
axs[1, 1].set_title("ReLU", color='white')
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].set_ylim(-1, 10)

plt.tight_layout()
print("Showing plot... Close the window to exit.")
plt.show()
