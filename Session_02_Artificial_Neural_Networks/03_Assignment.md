# 📋 Session 02 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Engineering Your Own Neural Network"
### Due: Before Session 3 begins
### Estimated Time: 2 – 2.5 hours

---

> **Professor's Note:** *"In Session 3, we will build our first real neural network using Python. This assignment ensures you arrive with the mathematical intuition and the clean Python coding habits to hit the ground running. The code you write tonight — a neuron class from scratch — will be the foundation upon which everything else is built."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Master the manual calculation of a neuron's output
- Build a reusable `Neuron` class in Python with clean, documented code
- Extend it to a simple `NeuralLayer` class
- Explore how different activation functions change a neuron's behavior
- Understand the effect of weights and biases intuitively

---

## 📝 TASK 1 — Written Questions (30 minutes)

Answer the following questions in your own words. Each answer should be **3–5 sentences minimum.** Do NOT copy from the notes — use your own words and analogies.

---

**Question 1:** In your own words, explain the difference between a **weight** and a **bias** in a neural network. Use an analogy that was NOT used in the lecture notes.

**Your Answer:** _______________________________________________

---

**Question 2:** Why is an **activation function** necessary? What would happen to a deep neural network if we removed all activation functions and used only weighted sums?

**Your Answer:** _______________________________________________

---

**Question 3:** You are a software engineer at a Nigerian fintech startup. Your team wants to build an AI system. For each scenario below, identify the **most appropriate ANN type** and explain why in one sentence:

| Scenario | Best ANN Type | Why? |
|----------|--------------|------|
| Detecting fraudulent bank transactions from account history over time | | |
| Verifying customer identity from a photo of their face | | |
| Recommending a loan amount based on customer profile data | | |
| Creating synthetic training data for a rare disease that has few real cases | | |

---

**Question 4:** A fellow student says: *"Since more layers means more powerful networks, I will always use 100 hidden layers for every problem."* Write a response explaining **why this is wrong** and what factors should guide the choice of network depth.

**Your Answer:** _______________________________________________

---

## 💻 TASK 2 — Python: Build a `Neuron` Class (45 minutes)

### Instructions:
Create a file called `neuron_class.py` inside your `C:\DL_Course\Session_02\` folder.

Build a **complete, well-documented `Neuron` class** that:
1. Stores weights and bias as attributes
2. Supports three activation functions: Sigmoid, ReLU, Tanh
3. Has a `forward()` method that computes the output
4. Has a `describe()` method that prints a summary of the neuron's setup

**Starter Template (complete the missing parts marked with `# TODO`):**

```python
"""
neuron_class.py
Assignment: Session 02 — Artificial Neural Networks
Student Name: [YOUR NAME HERE]
Date: [DATE HERE]

Description:
    A complete implementation of a single artificial neuron
    supporting multiple activation functions.
"""

import numpy as np
import math


class Neuron:
    """
    Represents a single artificial neuron.
    
    Attributes:
        weights (list or array): Connection weights for each input
        bias    (float)        : Bias term added before activation
        activation (str)       : Name of activation function to use
                                 Options: 'sigmoid', 'relu', 'tanh'
    """
    
    def __init__(self, weights, bias, activation='sigmoid'):
        """
        Initialize the neuron with weights, bias, and activation function.
        
        Parameters:
            weights    : list of floats — one weight per input
            bias       : float          — the bias term
            activation : str            — 'sigmoid', 'relu', or 'tanh'
        """
        self.weights    = np.array(weights)   # Store as NumPy array
        self.bias       = bias
        self.activation = activation.lower()  # Make lowercase for consistency
    
    # ── Activation Functions ────────────────────────────────────────────────
    
    def _sigmoid(self, z):
        """
        Sigmoid function: f(z) = 1 / (1 + e^-z)
        Output range: (0, 1)
        Used for: Binary classification output
        """
        # TODO: Implement and return the sigmoid formula
        pass
    
    def _relu(self, z):
        """
        ReLU (Rectified Linear Unit): f(z) = max(0, z)
        Output range: (0, infinity)
        Used for: Hidden layers in deep networks
        """
        # TODO: Return max(0, z) using numpy
        # Hint: np.maximum(0, z) works for both scalars and arrays
        pass
    
    def _tanh(self, z):
        """
        Hyperbolic Tangent: f(z) = (e^z - e^-z) / (e^z + e^-z)
        Output range: (-1, 1)
        Used for: Hidden layers (better than sigmoid for some cases)
        """
        # TODO: Use np.tanh(z) and return the result
        pass
    
    # ── Core Methods ────────────────────────────────────────────────────────
    
    def forward(self, inputs):
        """
        Compute the neuron's output for given inputs.
        
        Process:
            1. Convert inputs to NumPy array
            2. Calculate weighted sum: z = dot(weights, inputs) + bias
            3. Apply activation function
            4. Return both z (pre-activation) and output (post-activation)
        
        Parameters:
            inputs : list or array of input values
        
        Returns:
            z      : float — the weighted sum (pre-activation value)
            output : float — the activated output
        """
        inputs = np.array(inputs)
        
        # TODO: Step 1 — Calculate weighted sum z
        # Hint: Use np.dot(self.weights, inputs) + self.bias
        z = None   # Replace None with the formula
        
        # TODO: Step 2 — Apply the correct activation function
        # Use an if/elif/else to choose based on self.activation
        if self.activation == 'sigmoid':
            output = None   # Replace with self._sigmoid(z)
        elif self.activation == 'relu':
            output = None   # Replace with self._relu(z)
        elif self.activation == 'tanh':
            output = None   # Replace with self._tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}. "
                           f"Choose 'sigmoid', 'relu', or 'tanh'")
        
        return z, output
    
    def describe(self):
        """
        Print a human-readable summary of this neuron's configuration.
        
        Example output:
            ──────────────────────────────
            Neuron Configuration
            ──────────────────────────────
            Number of inputs : 3
            Weights          : [0.5, -0.3, 0.8]
            Bias             : 0.2
            Activation       : sigmoid
            ──────────────────────────────
        """
        # TODO: Print a formatted summary of this neuron
        # Hint: Use f-strings and print() statements
        pass


# ── Test Your Neuron Class ───────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 55)
    print("  TEST 1: Sigmoid Neuron (from In-Class Exercise A)")
    print("=" * 55)
    
    # Create a sigmoid neuron with the Exercise A values
    n1 = Neuron(
        weights    = [0.8, -0.3, 0.5],
        bias       = 0.2,
        activation = 'sigmoid'
    )
    
    n1.describe()
    
    z, output = n1.forward([0.6, 0.4, 0.9])
    print(f"\n  Input:  [0.6, 0.4, 0.9]")
    print(f"  z (weighted sum): {z:.4f}")
    print(f"  Output (sigmoid): {output:.4f}")
    print(f"  Activated? {'YES ✅' if output > 0.5 else 'NO ❌'}")
    
    print()
    print("=" * 55)
    print("  TEST 2: ReLU Neuron")
    print("=" * 55)
    
    # TODO: Create a ReLU neuron with weights=[0.5, 0.5], bias=-0.8
    # Test it with inputs [1.0, 0.5]
    # Print the z value and output
    # Is the output different from sigmoid? Why?
    
    print()
    print("=" * 55)
    print("  TEST 3: Tanh Neuron")
    print("=" * 55)
    
    # TODO: Create a Tanh neuron with weights=[0.3, -0.7, 0.4], bias=0.1
    # Test it with inputs [0.2, 0.8, 0.5]
    # Note: tanh output is between -1 and 1 (unlike sigmoid's 0 to 1)
    
    print()
    print("=" * 55)
    print("  TEST 4: Comparison — Same Inputs, Different Activations")
    print("=" * 55)
    
    # Create 3 neurons with SAME weights but DIFFERENT activations
    test_inputs  = [0.5, 0.8, -0.2]
    test_weights = [0.6, 0.4, 0.9]
    test_bias    = 0.1
    
    # TODO: Create sigmoid_n, relu_n, and tanh_n neurons
    # Run all three on test_inputs
    # Print a comparison table:
    #   Activation | z    | Output
    #   Sigmoid    | ___  | ___
    #   ReLU       | ___  | ___
    #   Tanh       | ___  | ___
```

---

### What You Need to Complete:
1. ✅ Implement `_sigmoid(z)` method
2. ✅ Implement `_relu(z)` method
3. ✅ Implement `_tanh(z)` method
4. ✅ Complete the `forward()` method (fill in z calculation and activation calls)
5. ✅ Implement the `describe()` method
6. ✅ Complete TEST 2 (ReLU neuron)
7. ✅ Complete TEST 3 (Tanh neuron)
8. ✅ Complete TEST 4 (Comparison table)

---

## 💻 TASK 3 — Python: Build a `NeuralLayer` Class (30 minutes)

### Instructions:
In the SAME `neuron_class.py` file (below your `Neuron` class), add a `NeuralLayer` class that represents a **full layer of neurons**.

```python
class NeuralLayer:
    """
    Represents one layer in a neural network.
    A layer contains multiple neurons, all sharing the same activation function
    and all receiving the same input vector.
    
    Attributes:
        neurons (list): List of Neuron objects in this layer
    """
    
    def __init__(self, n_neurons, n_inputs, activation='sigmoid'):
        """
        Initialize the layer with random weights.
        
        Parameters:
            n_neurons  : int — how many neurons in this layer
            n_inputs   : int — how many inputs each neuron receives
            activation : str — activation function for all neurons
        """
        self.neurons = []
        
        # TODO: Create n_neurons neurons, each with:
        #   - n_inputs random weights (use np.random.randn(n_inputs) * 0.1)
        #   - a random bias (use np.random.randn() * 0.1)
        #   - the given activation function
        # Append each neuron to self.neurons
        
        # Set random seed for reproducibility (always use this in testing!)
        np.random.seed(42)
        
        for i in range(n_neurons):
            # TODO: Create and append each neuron
            pass
    
    def forward(self, inputs):
        """
        Run a forward pass through all neurons in this layer.
        
        Each neuron receives the SAME inputs and produces its own output.
        The layer's output is the collection of all neuron outputs.
        
        Parameters:
            inputs : list — the input vector to the layer
        
        Returns:
            outputs : list — one output value per neuron
        """
        outputs = []
        
        # TODO: For each neuron, call neuron.forward(inputs)
        # Collect all outputs into the outputs list
        # Return the list of outputs
        
        return outputs
    
    def describe(self):
        """Print a summary of the layer."""
        print(f"\n  Layer: {len(self.neurons)} neurons, "
              f"activation={self.neurons[0].activation if self.neurons else 'N/A'}")
        for i, neuron in enumerate(self.neurons):
            print(f"    Neuron {i+1}: weights={neuron.weights.round(3).tolist()}, "
                  f"bias={neuron.bias:.3f}")


# ── Test the NeuralLayer ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # (Add this to the existing test block)
    
    print()
    print("=" * 55)
    print("  TEST 5: NeuralLayer — 3 neurons, 2 inputs")
    print("=" * 55)
    
    layer = NeuralLayer(n_neurons=3, n_inputs=2, activation='sigmoid')
    layer.describe()
    
    layer_inputs = [0.7, 0.5]
    layer_outputs = layer.forward(layer_inputs)
    
    print(f"\n  Layer Inputs:  {layer_inputs}")
    print(f"  Layer Outputs: {[round(o, 4) for o in layer_outputs]}")
    print(f"\n  ✅ This is how a full hidden layer works!")
    print(f"     Each of the 3 neurons processed the same 2 inputs")
    print(f"     and produced its own output.")
```

---

## 🏆 BONUS TASK — Two-Layer Network (For High Achievers)

### Challenge:
Using your `NeuralLayer` class, build a **complete 2-layer neural network** and run a full forward pass.

```python
"""
BONUS: mini_network.py

Build a 2-layer network:
  Layer 1 (Hidden):  3 neurons, 2 inputs, ReLU activation
  Layer 2 (Output):  1 neuron, 3 inputs (from Layer 1), Sigmoid activation

Test with inputs: [0.8, 0.3]

Print:
  - Layer 1 outputs
  - Final output (Layer 2)
  - Your interpretation: "This input would be classified as: YES/NO"
    (based on whether final output > 0.5)
"""
```

**Why this is significant:** What you're building IS a complete neural network. When you use TensorFlow or Keras in Session 6, you'll be asking it to do exactly what you've built here — but for millions of neurons and inputs simultaneously.

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1:** Written answers as a `.docx` or `.pdf`
2. ✅ **Task 2 & 3:** Complete `neuron_class.py` file with all TODOs filled
3. ✅ **Task 2 & 3:** Screenshot of the terminal output when you run the file
4. ⭐ **Bonus:** `mini_network.py` file (if attempted)

**File Naming:**
```
Firstname_Lastname_Session02_Written.docx
Firstname_Lastname_Session02_neuron_class.py
Firstname_Lastname_Session02_output_screenshot.png
```

---

## ⏰ Deadline
Submit before Session 3. We will build on this code in class!

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 02 — Assignment*
