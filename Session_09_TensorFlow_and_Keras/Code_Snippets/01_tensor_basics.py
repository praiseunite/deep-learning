"""
============================================================
  FILE: 01_tensor_basics.py
  SESSION: 09 — TensorFlow and Keras
  PURPOSE: Demonstrate raw TensorFlow operations without Keras.
============================================================
"""

import tensorflow as tf
import os

# Suppress some TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print("=" * 50)
print("  Aptech AI Lab: Raw TensorFlow Basics")
print("=" * 50)

# ─── 1. CREATING TENSORS ──────────────────────────────────────────────────────

# A constant tensor cannot be changed once created. (e.g., input data)
t_constant = tf.constant([[1, 2], [3, 4]])

# A variable tensor can have its values updated. (e.g., neural network weights)
t_variable = tf.Variable([[5, 6], [7, 8]])

print("\n--- Tensor Types ---")
print("Constant Tensor:\n", t_constant.numpy())
print("\nVariable Tensor:\n", t_variable.numpy())


# ─── 2. TENSOR MATH (THE COMPUTATIONAL GRAPH) ─────────────────────────────────

# TensorFlow overloads standard python operators to build graph nodes
addition_result = t_constant + t_variable

print("\n--- Tensor Math ---")
print("Addition Result:\n", addition_result.numpy())

# Matrix Multiplication (Dot Product) - The core of Neural Networks!
dot_product = tf.matmul(t_constant, t_variable)
print("\nMatrix Multiplication Result:\n", dot_product.numpy())


# ─── 3. BROADCASTING ──────────────────────────────────────────────────────────
# Applying a smaller tensor (or scalar) to a larger tensor automatically
vector = tf.constant([1, 2, 3])
scalar = tf.constant(5)

broadcast_result = vector * scalar

print("\n--- Broadcasting ---")
print(f"Multiplying {vector.numpy()} by {scalar.numpy()}...")
print("Result:", broadcast_result.numpy())

print("\n" + "=" * 50)
