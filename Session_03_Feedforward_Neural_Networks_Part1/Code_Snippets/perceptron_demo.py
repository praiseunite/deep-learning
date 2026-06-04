"""
============================================================
  FILE: perceptron_demo.py
  SESSION: 03 — Feedforward Neural Networks
  PURPOSE: Demonstrate a single Perceptron solving an AND gate,
           and failing to solve an XOR gate.
============================================================
"""

import numpy as np

# ─── 1. The Perceptron Class ──────────────────────────────────────────────────
class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def step_function(self, x):
        """Step activation function: 1 if x > 0 else 0"""
        return 1 if x > 0 else 0

    def predict(self, inputs):
        """Calculate weighted sum and apply step function"""
        inputs = np.array(inputs)
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.step_function(weighted_sum)

# ─── 2. Solving the AND Gate ──────────────────────────────────────────────────
print("=" * 40)
print("  TEST 1: The AND Gate (Linearly Separable)")
print("=" * 40)

# Weights = 1, 1. Bias = -1.5
# Why -1.5? Because 1+1-1.5 = 0.5 (>0) -> 1
# But 1+0-1.5 = -0.5 (<=0) -> 0
and_perceptron = Perceptron(weights=[1, 1], bias=-1.5)

dataset = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

print("Input 1 | Input 2 | Expected | Prediction")
print("-" * 40)
for inputs, expected in dataset:
    prediction = and_perceptron.predict(inputs)
    match = "✅" if prediction == expected else "❌"
    print(f"   {inputs[0]}    |    {inputs[1]}    |    {expected}     |     {prediction} {match}")

# ─── 3. Failing the XOR Gate ──────────────────────────────────────────────────
print("\n" + "=" * 40)
print("  TEST 2: The XOR Gate (Non-Linear)")
print("=" * 40)
print("Attempting to solve XOR with a single Perceptron...")

# There is NO combination of weights and bias that solves this perfectly with one perceptron.
# Let's try some arbitrary values to show it fails.
xor_perceptron = Perceptron(weights=[1, 1], bias=-0.5)

xor_dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)  # This will fail!
]

print("Input 1 | Input 2 | Expected | Prediction")
print("-" * 40)
for inputs, expected in xor_dataset:
    prediction = xor_perceptron.predict(inputs)
    match = "✅" if prediction == expected else "❌"
    print(f"   {inputs[0]}    |    {inputs[1]}    |    {expected}     |     {prediction} {match}")

print("\n💡 Conclusion: A single Perceptron CANNOT solve the XOR problem.")
print("   We need a multi-layer network (FNN) with non-linear activation functions!")
print("=" * 40)
