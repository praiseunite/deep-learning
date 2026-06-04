"""
============================================================
  FILE: 01_single_neuron_demo.py
  SESSION: 02 — Artificial Neural Networks (ANN)
  PURPOSE: Demonstrate a single artificial neuron in full detail.
           Shows the forward pass step by step with printed output
           so students can follow exactly what is happening.

  CONCEPTS:
    - Weighted sum (z = W·X + b)
    - Sigmoid activation function
    - Neuron output interpretation
    - Effect of changing weights and biases

  HOW TO RUN:
    python 01_single_neuron_demo.py
============================================================
"""

import numpy as np   # NumPy for efficient numerical computation

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: The Sigmoid Activation Function
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    """
    Sigmoid activation function.
    
    Formula:  f(z) = 1 / (1 + e^(-z))
    
    Properties:
      - Output is ALWAYS between 0 and 1
      - When z is large positive → output approaches 1
      - When z is large negative → output approaches 0
      - When z = 0              → output = 0.5 (the midpoint)
    
    Use case:
      - Commonly used in output layers for binary classification
      - Interprets output as a PROBABILITY
    """
    return 1 / (1 + np.exp(-z))


def relu(z):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula:  f(z) = max(0, z)
    
    Properties:
      - Output is 0 for all negative inputs
      - Output equals input for all positive inputs
      - Simple and computationally efficient
    
    Use case:
      - Most common choice for hidden layers in deep networks
    """
    return np.maximum(0, z)   # np.maximum works element-wise


def tanh(z):
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    Formula:  f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    
    Properties:
      - Output is ALWAYS between -1 and 1
      - Zero-centered (unlike sigmoid which is centered at 0.5)
      - Stronger gradient than sigmoid
    
    Use case:
      - Hidden layers, especially in RNNs
    """
    return np.tanh(z)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Single Neuron — Detailed Step-by-Step Forward Pass
# ─────────────────────────────────────────────────────────────────────────────

def single_neuron_forward(inputs, weights, bias, activation_name='sigmoid'):
    """
    Compute the output of a single artificial neuron.
    Prints every step so you can follow along.
    
    Parameters:
        inputs          : list — input values [x1, x2, ...]
        weights         : list — weight values [w1, w2, ...]
        bias            : float — the bias term
        activation_name : str   — which activation to use
    
    Returns:
        z      : float — the weighted sum (pre-activation)
        output : float — the neuron's final output
    """
    # Convert to NumPy arrays for vector math
    x = np.array(inputs)
    w = np.array(weights)
    
    print(f"  {'─' * 45}")
    print(f"  NEURON FORWARD PASS")
    print(f"  {'─' * 45}")
    
    # ── Step 1: Show each weighted input individually ─────────────────────
    print(f"\n  STEP 1: Compute each weighted input (xᵢ × wᵢ)")
    total = 0
    for i, (xi, wi) in enumerate(zip(x, w)):
        contribution = xi * wi
        total += contribution
        print(f"    x{i+1}({xi}) × w{i+1}({wi}) = {contribution:.4f}")
    
    # ── Step 2: Add bias to get weighted sum z ────────────────────────────
    z = np.dot(x, w) + bias   # np.dot is the efficient dot product
    print(f"\n  STEP 2: Sum all contributions + bias")
    print(f"    Weighted contributions sum: {total:.4f}")
    print(f"    Bias: {bias}")
    print(f"    z = {total:.4f} + {bias} = {z:.4f}")
    
    # ── Step 3: Apply the activation function ─────────────────────────────
    print(f"\n  STEP 3: Apply '{activation_name}' activation function")
    
    if activation_name == 'sigmoid':
        output = sigmoid(z)
        print(f"    f(z) = 1 / (1 + e^(-{z:.4f}))")
        print(f"         = 1 / (1 + {np.exp(-z):.4f})")
        print(f"         = {output:.4f}")
        interpretation = f"Probability: {output:.2%}"
    elif activation_name == 'relu':
        output = relu(z)
        print(f"    f(z) = max(0, {z:.4f}) = {output:.4f}")
        interpretation = f"Positive signal: {output:.4f}"
    elif activation_name == 'tanh':
        output = tanh(z)
        print(f"    f(z) = tanh({z:.4f}) = {output:.4f}")
        interpretation = f"Signal in (-1,1): {output:.4f}"
    
    # ── Step 4: Interpret the result ──────────────────────────────────────
    print(f"\n  STEP 4: Interpret the output")
    print(f"    Output: {output:.4f}")
    print(f"    Interpretation: {interpretation}")
    
    if activation_name == 'sigmoid':
        if output > 0.5:
            print(f"    → Neuron is ACTIVATED ✅ (output > 0.5)")
        else:
            print(f"    → Neuron is SUPPRESSED ❌ (output < 0.5)")
    
    return z, output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Demonstrate the Effect of Weights and Bias
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_weight_effects():
    """
    Shows how changing weights affects a neuron's output.
    Runs the same inputs through neurons with different weights.
    """
    print("\n" + "=" * 60)
    print("  DEMO: How Weights Affect Output")
    print("=" * 60)
    
    # Fixed inputs and bias — only weights change
    inputs = [0.5, 0.8]
    bias   = 0.0
    
    weight_experiments = [
        ([0.5,  0.5 ], "Moderate positive weights"),
        ([2.0,  2.0 ], "Large positive weights"),
        ([0.01, 0.01], "Very small weights"),
        ([-0.5,-0.5 ], "Negative weights (suppression)"),
        ([1.0, -1.0 ], "Mixed: one positive, one negative"),
    ]
    
    print(f"\n  Fixed inputs: {inputs}, Fixed bias: {bias}")
    print(f"\n  {'Weights':<25} {'z':<10} {'Output':<12} {'Activated?'}")
    print(f"  {'─' * 60}")
    
    for weights, description in weight_experiments:
        z = np.dot(np.array(inputs), np.array(weights)) + bias
        output = sigmoid(z)
        activated = "✅ YES" if output > 0.5 else "❌ NO"
        print(f"  {str(weights):<25} {z:<10.4f} {output:<12.4f} {activated}")
        print(f"  {description}")
        print()


def demonstrate_bias_effects():
    """
    Shows how changing the bias affects a neuron's output.
    The bias acts as a threshold — making it easier or harder to activate.
    """
    print("\n" + "=" * 60)
    print("  DEMO: How Bias Affects Output")
    print("=" * 60)
    
    # Fixed everything except bias
    inputs  = [0.0, 0.0]   # ALL INPUTS ARE ZERO
    weights = [0.5, 0.5]
    
    bias_experiments = [-3.0, -1.0, 0.0, 1.0, 3.0]
    
    print(f"\n  Fixed inputs: {inputs} (all zeros!), Fixed weights: {weights}")
    print(f"  Note: With all-zero inputs, ONLY the bias affects the output")
    print(f"\n  {'Bias':<10} {'z':<10} {'Output':<12} {'Interpretation'}")
    print(f"  {'─' * 60}")
    
    for bias in bias_experiments:
        z = np.dot(np.array(inputs), np.array(weights)) + bias
        output = sigmoid(z)
        
        if bias < -1:
            interp = "Strong suppression (hard to activate)"
        elif bias < 0:
            interp = "Slight suppression"
        elif bias == 0:
            interp = "Neutral (output = 0.5)"
        elif bias < 2:
            interp = "Slight boost"
        else:
            interp = "Strong boost (easy to activate)"
        
        print(f"  {bias:<10} {z:<10.4f} {output:<12.4f} {interp}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Activation Function Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_activations():
    """
    Runs the same inputs through the same weights but with
    different activation functions to show their differences.
    """
    print("\n" + "=" * 60)
    print("  DEMO: Same Input — Different Activations")
    print("=" * 60)
    
    inputs  = [0.6, 0.4, 0.9]
    weights = [0.8, -0.3, 0.5]
    bias    = 0.2
    
    z = np.dot(np.array(inputs), np.array(weights)) + bias
    
    print(f"\n  Inputs:  {inputs}")
    print(f"  Weights: {weights}")
    print(f"  Bias:    {bias}")
    print(f"  Weighted Sum (z): {z:.4f}")
    print()
    print(f"  {'Activation':<15} {'Output':<15} {'Range':<15} {'Best Used For'}")
    print(f"  {'─' * 65}")
    print(f"  {'Sigmoid':<15} {sigmoid(z):<15.4f} {'(0, 1)':<15} Binary classification output")
    print(f"  {'ReLU':<15} {relu(z):<15.4f} {'(0, ∞)':<15} Hidden layers — fast, efficient")
    print(f"  {'Tanh':<15} {tanh(z):<15.4f} {'(-1, 1)':<15} Hidden layers — zero-centered")
    print()
    print(f"  📌 Key Insight:")
    print(f"     - Sigmoid & Tanh 'squash' output to a bounded range")
    print(f"     - ReLU passes positive values unchanged, kills negatives")
    print(f"     - Each has different strengths — we'll explore all in Session 4!")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run All Demonstrations
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("  Session 02: Artificial Neural Networks")
    print("  Code Demo: Single Neuron Forward Pass")
    print("=" * 60)
    
    # ── Demo 1: Detailed neuron forward pass (from In-Class Exercise A) ───
    print("\n📌 DEMO 1: From In-Class Exercise A")
    z_val, out = single_neuron_forward(
        inputs          = [0.6, 0.4, 0.9],
        weights         = [0.8, -0.3, 0.5],
        bias            = 0.2,
        activation_name = 'sigmoid'
    )
    
    # ── Demo 2: Same with ReLU ────────────────────────────────────────────
    print("\n📌 DEMO 2: Same values but with ReLU activation")
    single_neuron_forward(
        inputs          = [0.6, 0.4, 0.9],
        weights         = [0.8, -0.3, 0.5],
        bias            = 0.2,
        activation_name = 'relu'
    )
    
    # ── Demo 3: Effect of weights ─────────────────────────────────────────
    demonstrate_weight_effects()
    
    # ── Demo 4: Effect of bias ────────────────────────────────────────────
    demonstrate_bias_effects()
    
    # ── Demo 5: Activation comparison ────────────────────────────────────
    compare_activations()
    
    print("\n" + "=" * 60)
    print("  ✅ All demos complete!")
    print("  💡 Next: Session 03 — Feedforward Neural Networks")
    print("     We connect many neurons together and start training!")
    print("=" * 60)
