"""
============================================================
  FILE: 02_ann_from_scratch.py
  SESSION: 02 — Artificial Neural Networks (ANN)
  PURPOSE: Build a complete 2-layer ANN from scratch using
           only NumPy — no TensorFlow, no Keras.
           This is the FOUNDATION of everything in deep learning.

  CONCEPTS:
    - Multi-layer forward propagation
    - Complete neural network class
    - Making predictions
    - Visualizing the network structure

  WHY THIS MATTERS:
    When you later use TensorFlow/Keras, you'll understand
    EXACTLY what happens inside those "magic" functions.
    This code IS what Keras does — just without the optimizations.

  HOW TO RUN:
    python 02_ann_from_scratch.py
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducible results
# (Same seed = same random numbers every time = consistent output)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    """f(z) = 1 / (1 + e^-z) → Output: (0, 1)"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """f(z) = max(0, z) → Output: (0, ∞)"""
    return np.maximum(0, z)

def softmax(z):
    """
    Softmax function — converts a vector of numbers into probabilities.
    All outputs sum to 1.0 (like probabilities must).
    Used for MULTI-CLASS classification output layers.
    
    Formula: softmax(zᵢ) = e^zᵢ / Σe^z
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: The ANN Class — A Complete Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class SimpleANN:
    """
    A simple 2-layer Artificial Neural Network built from scratch.
    
    Architecture:
        Input Layer  → Hidden Layer → Output Layer
    
    This network demonstrates the core mechanics of ALL neural networks.
    
    Attributes:
        n_inputs  : Number of input features
        n_hidden  : Number of neurons in the hidden layer
        n_outputs : Number of output neurons
        W1, b1    : Weights and biases for Layer 1 (Input → Hidden)
        W2, b2    : Weights and biases for Layer 2 (Hidden → Output)
    """
    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initialize the ANN with random weights.
        
        Weight Initialization Strategy:
            We use small random values (multiplied by 0.01).
            Large initial weights cause problems during training.
            Small weights keep the network in a "neutral" starting state.
        
        Parameters:
            n_inputs  : int — size of input vector (number of features)
            n_hidden  : int — number of neurons in hidden layer
            n_outputs : int — number of output neurons
        """
        self.n_inputs  = n_inputs
        self.n_hidden  = n_hidden
        self.n_outputs = n_outputs
        
        # ── Layer 1 Parameters (Input → Hidden) ──────────────────────────
        # W1 shape: (n_hidden × n_inputs)
        # Each row = weights for one hidden neuron
        # Each column = how much that input affects all hidden neurons
        self.W1 = np.random.randn(n_hidden, n_inputs) * 0.01
        
        # b1 shape: (n_hidden × 1) — one bias per hidden neuron
        self.b1 = np.zeros((n_hidden, 1))
        
        # ── Layer 2 Parameters (Hidden → Output) ─────────────────────────
        # W2 shape: (n_outputs × n_hidden)
        self.W2 = np.random.randn(n_outputs, n_hidden) * 0.01
        
        # b2 shape: (n_outputs × 1) — one bias per output neuron
        self.b2 = np.zeros((n_outputs, 1))
        
        print(f"  ✅ ANN initialized:")
        print(f"     Architecture: {n_inputs} → {n_hidden} → {n_outputs}")
        print(f"     W1 shape: {self.W1.shape}  (hidden × inputs)")
        print(f"     b1 shape: {self.b1.shape}  (hidden × 1)")
        print(f"     W2 shape: {self.W2.shape}  (outputs × hidden)")
        print(f"     b2 shape: {self.b2.shape}  (outputs × 1)")
        total_params = (n_hidden * n_inputs + n_hidden + 
                       n_outputs * n_hidden + n_outputs)
        print(f"     Total parameters: {total_params}")
    
    def forward(self, X):
        """
        Forward propagation: Pass input X through the network.
        
        Data flow:
            X → [Layer1: W1·X + b1] → [ReLU] → 
                [Layer2: W2·A1 + b2] → [Sigmoid] → 
                Output
        
        Parameters:
            X : numpy array of shape (n_inputs, 1) or (n_inputs,)
                The input data
        
        Returns:
            A2 : numpy array — the network's output (prediction)
        
        Also stores intermediate values for educational display.
        """
        # Ensure X is a column vector (n_inputs × 1)
        X = np.array(X).reshape(-1, 1)
        
        # ── Layer 1: Hidden Layer ─────────────────────────────────────────
        # Weighted sum for layer 1
        # Z1 shape: (n_hidden × 1)
        self.Z1 = np.dot(self.W1, X) + self.b1
        
        # Apply ReLU activation to hidden layer
        # A1 shape: (n_hidden × 1) — "A" stands for Activation
        self.A1 = relu(self.Z1)
        
        # ── Layer 2: Output Layer ─────────────────────────────────────────
        # Weighted sum for layer 2
        # Z2 shape: (n_outputs × 1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        
        # Apply Sigmoid activation to output layer
        # A2 shape: (n_outputs × 1)
        self.A2 = sigmoid(self.Z2)
        
        # Save inputs for display
        self.last_input = X
        
        return self.A2
    
    def predict(self, X):
        """
        Make a binary prediction (0 or 1) from input X.
        
        Threshold: if output > 0.5 → class 1, else → class 0
        """
        output = self.forward(X)
        return 1 if output[0, 0] > 0.5 else 0
    
    def predict_proba(self, X):
        """Return the raw probability output (between 0 and 1)."""
        output = self.forward(X)
        return output[0, 0]
    
    def show_forward_pass(self, X, input_names=None):
        """
        Run a forward pass and print every intermediate value.
        Educational function to trace data through the network.
        
        Parameters:
            X           : input values
            input_names : optional list of names for inputs (e.g., ['Age', 'Income'])
        """
        X_arr = np.array(X)
        output = self.forward(X_arr)
        
        print(f"\n  {'─' * 50}")
        print(f"  FORWARD PASS TRACE")
        print(f"  {'─' * 50}")
        
        # Print inputs
        print(f"\n  📥 INPUTS:")
        for i, val in enumerate(X_arr.flatten()):
            name = input_names[i] if input_names else f"x{i+1}"
            print(f"     {name} = {val}")
        
        # Print Layer 1
        print(f"\n  🔵 HIDDEN LAYER (Layer 1 — {self.n_hidden} neurons, ReLU):")
        for j in range(self.n_hidden):
            z_val = self.Z1[j, 0]
            a_val = self.A1[j, 0]
            print(f"     Neuron {j+1}: z={z_val:.4f} → ReLU → a={a_val:.4f}")
        
        # Print Layer 2
        print(f"\n  🟠 OUTPUT LAYER (Layer 2 — {self.n_outputs} neuron(s), Sigmoid):")
        for j in range(self.n_outputs):
            z_val = self.Z2[j, 0]
            a_val = self.A2[j, 0]
            print(f"     Neuron {j+1}: z={z_val:.4f} → Sigmoid → output={a_val:.4f}")
        
        # Final prediction
        pred = self.predict(X_arr)
        prob = self.predict_proba(X_arr)
        print(f"\n  🎯 FINAL PREDICTION:")
        print(f"     Probability: {prob:.4f} ({prob:.2%})")
        print(f"     Class: {'1 (Positive / Yes)' if pred == 1 else '0 (Negative / No)'}")
        print(f"  {'─' * 50}")
        
        return output
    
    def count_parameters(self):
        """Return total number of learnable parameters in the network."""
        total = (self.W1.size + self.b1.size + 
                 self.W2.size + self.b2.size)
        return total
    
    def describe(self):
        """Print a complete description of the network architecture."""
        print(f"\n  {'═' * 50}")
        print(f"  NETWORK ARCHITECTURE SUMMARY")
        print(f"  {'═' * 50}")
        print(f"  Input Layer:   {self.n_inputs} neurons  (receives data)")
        print(f"  Hidden Layer:  {self.n_hidden} neurons  (ReLU activation)")
        print(f"  Output Layer:  {self.n_outputs} neuron(s) (Sigmoid activation)")
        print(f"  {'─' * 50}")
        print(f"  Layer 1 Weights (W1): shape {self.W1.shape}")
        print(f"  Layer 1 Biases  (b1): shape {self.b1.shape}")
        print(f"  Layer 2 Weights (W2): shape {self.W2.shape}")
        print(f"  Layer 2 Biases  (b2): shape {self.b2.shape}")
        print(f"  {'─' * 50}")
        print(f"  Total Learnable Parameters: {self.count_parameters()}")
        print(f"  {'═' * 50}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Visualize Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_functions():
    """
    Plot Sigmoid, ReLU, and Tanh activation functions side by side.
    This visualization helps understand how each function transforms input.
    """
    # Create x values from -6 to 6
    z = np.linspace(-6, 6, 300)
    
    # Set dark background style
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Activation Functions — The "Switch" Inside Each Neuron',
                 fontsize=14, color='white', fontweight='bold', y=1.02)
    
    # ── Sigmoid ───────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(z, sigmoid(z), color='#4FC3F7', linewidth=3, label='Sigmoid')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax1.axvline(x=0,   color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(z, sigmoid(z), alpha=0.15, color='#4FC3F7')
    ax1.set_title('Sigmoid\nf(z) = 1/(1+e⁻ᶻ)', color='white', fontsize=11)
    ax1.set_xlabel('z (weighted sum)', color='gray')
    ax1.set_ylabel('Activation Output', color='gray')
    ax1.set_ylim(-0.1, 1.1)
    ax1.text(2, 0.1, 'Output: (0, 1)', color='#4FC3F7', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # ── ReLU ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(z, relu(z), color='#FF8A65', linewidth=3, label='ReLU')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(z, relu(z), alpha=0.15, color='#FF8A65')
    ax2.set_title('ReLU\nf(z) = max(0, z)', color='white', fontsize=11)
    ax2.set_xlabel('z (weighted sum)', color='gray')
    ax2.set_ylabel('Activation Output', color='gray')
    ax2.set_ylim(-0.5, 6.5)
    ax2.text(1, 0.5, 'Output: (0, ∞)', color='#FF8A65', fontsize=10)
    ax2.text(-5, 1, 'Kills negatives', color='gray', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    # ── Tanh ──────────────────────────────────────────────────────────────
    ax3 = axes[2]
    tanh_vals = np.tanh(z)
    ax3.plot(z, tanh_vals, color='#CE93D8', linewidth=3, label='Tanh')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between(z, tanh_vals, alpha=0.15, color='#CE93D8')
    ax3.set_title('Tanh\nf(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)', color='white', fontsize=11)
    ax3.set_xlabel('z (weighted sum)', color='gray')
    ax3.set_ylabel('Activation Output', color='gray')
    ax3.set_ylim(-1.2, 1.2)
    ax3.text(1.5, -0.8, 'Output: (-1, 1)', color='#CE93D8', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, 
                bbox_inches='tight', facecolor='black')
    print("  📊 Activation functions plot saved as 'activation_functions.png'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run All Demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("  Session 02: Artificial Neural Networks")
    print("  Code Demo: Complete ANN From Scratch")
    print("=" * 60)
    
    # ── Demo 1: Build and describe the network ────────────────────────────
    print("\n📌 DEMO 1: Build a 2-Layer ANN")
    print("  Architecture: 2 inputs → 3 hidden → 1 output")
    
    ann = SimpleANN(n_inputs=2, n_hidden=3, n_outputs=1)
    ann.describe()
    
    # ── Demo 2: Predict student pass/fail ─────────────────────────────────
    print("\n📌 DEMO 2: Predict Student Exam Result")
    print("  Task: Will student pass? (1=Yes, 0=No)")
    print("  Inputs: [Hours_Studied, Attendance_Rate] (normalized 0-1)")
    
    students = [
        ([0.9, 0.95], "Student A (studied a lot, attended almost all classes)"),
        ([0.2, 0.3 ], "Student B (barely studied, low attendance)"),
        ([0.7, 0.5 ], "Student C (moderate study, moderate attendance)"),
    ]
    
    for student_input, description in students:
        print(f"\n  {description}")
        ann.show_forward_pass(
            X           = student_input,
            input_names = ['Hours_Studied', 'Attendance']
        )
    
    # ── Demo 3: Show how network size affects parameter count ─────────────
    print("\n📌 DEMO 3: How Architecture Affects Parameter Count")
    print("  (More parameters = more learning capacity, but needs more data)")
    
    architectures = [
        (2, 2,   1),
        (2, 10,  1),
        (2, 50,  1),
        (10, 100, 5),
    ]
    
    print(f"\n  {'Architecture':<25} {'Parameters':<15}")
    print(f"  {'─' * 40}")
    for inp, hid, out in architectures:
        temp_ann = SimpleANN.__new__(SimpleANN)
        temp_ann.n_inputs  = inp
        temp_ann.n_hidden  = hid
        temp_ann.n_outputs = out
        temp_ann.W1 = np.zeros((hid, inp))
        temp_ann.b1 = np.zeros((hid, 1))
        temp_ann.W2 = np.zeros((out, hid))
        temp_ann.b2 = np.zeros((out, 1))
        params = temp_ann.count_parameters()
        arch_str = f"{inp}→{hid}→{out}"
        print(f"  {arch_str:<25} {params:<15,}")
    
    print(f"\n  💡 GPT-3 has 175,000,000,000 (175 billion) parameters!")
    print(f"     Your laptop can handle networks with millions.")
    print(f"     That's why large models need specialized hardware (GPUs, TPUs).")
    
    # ── Demo 4: Plot activation functions ─────────────────────────────────
    print("\n📌 DEMO 4: Visualizing Activation Functions")
    print("  Opening a plot showing Sigmoid, ReLU, and Tanh...")
    plot_activation_functions()
    
    print("\n" + "=" * 60)
    print("  ✅ Session 02 Code Demo Complete!")
    print()
    print("  What you just ran IS a neural network.")
    print("  In Session 3, we'll make it LEARN — adjusting weights")
    print("  automatically to improve its predictions from data.")
    print("=" * 60)
