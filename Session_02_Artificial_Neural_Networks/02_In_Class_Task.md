# 🧪 Session 02 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Build a Neural Network — With Your Hands!"
### Duration: 45–50 minutes

---

> **Instructor Note:** This task has two parts. Part 1 is a physical/paper exercise where students manually compute forward propagation — no computer needed. Part 2 introduces the first Python interaction with neurons. The physical calculation is CRITICAL — students who do the math by hand remember it forever.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Label all components of an ANN diagram from memory
- [ ] Manually compute the output of a single neuron step by step
- [ ] Trace a forward pass through a 2-layer network by hand
- [ ] Write Python code that simulates a single artificial neuron
- [ ] Identify the correct ANN type for a given problem

---

## 🛠️ What You Need
- ✅ Paper, pen, and a calculator (phone calculator is fine)
- ✅ Python installed with NumPy (`pip install numpy`)
- ✅ The Lecture Notes (01_Lecture_Notes.md) for reference

---

## 📋 PART 1 — Label the Network (5 minutes)

### Instructions:
Draw the following network structure on paper and label EVERY component.

```
     x₁ = 0.5 ──── ? ────┐
                          │
     x₂ = 0.3 ──── ? ────┤──► [   ?   ] ──► f(?) ──► ?
                          │         ↑
     x₃ = 0.8 ──── ? ────┘        bias = ?
```

**Fill in what each `?` represents:**

| Symbol | What is it called? | What does it do? |
|--------|--------------------|-----------------|
| x₁, x₂, x₃ | | |
| The lines connecting x to the circle | | |
| The circle `[ ]` | | |
| `bias` | | |
| `f()` | | |
| The final arrow output | | |

**✅ Completion Check:** Can you redraw this diagram completely from memory without looking? Try it!

---

## 📋 PART 2 — Manual Forward Pass (20 minutes)

### 🧮 Exercise A: Compute a Single Neuron Output

You have one neuron with the following setup:

```
Inputs:   x₁ = 0.6,   x₂ = 0.4,   x₃ = 0.9
Weights:  w₁ = 0.8,   w₂ = -0.3,  w₃ = 0.5
Bias:     b  = 0.2
Activation Function: Sigmoid → f(z) = 1 / (1 + e^(-z))
```

**Work through each step:**

**Step 1: Calculate each weighted input**
```
x₁ × w₁ = _____ × _____ = _____
x₂ × w₂ = _____ × _____ = _____   ← Note the NEGATIVE weight!
x₃ × w₃ = _____ × _____ = _____
```

**Step 2: Calculate the weighted sum z**
```
z = (x₁×w₁) + (x₂×w₂) + (x₃×w₃) + b
z = _______ + _______ + _______ + 0.2
z = _______
```

**Step 3: Apply the Sigmoid Activation**
```
f(z) = 1 / (1 + e^(-z))
     = 1 / (1 + e^(-_____))
     = 1 / (1 + _______)    ← Use your calculator: e^(-z)
     = 1 / _______
     = _______
```

**Step 4: Interpret the result**
```
Your output is between 0 and 1.
If output > 0.5 → Neuron is "activated" (fires strongly)
If output < 0.5 → Neuron is "weakly activated" (low signal)

My neuron output: _______
Is it activated (>0.5)?  YES / NO
```

> 💡 **Expected Answer:** z ≈ 0.89, f(z) ≈ 0.709

---

### 🧮 Exercise B: Full 2-Layer Network Forward Pass

Now trace data through a **complete 2-layer network**:

```
SCENARIO: Predicting if a customer will buy a product
Inputs:
  x₁ = 0.7  (Age, normalized)
  x₂ = 0.5  (Income, normalized)

NETWORK STRUCTURE:
  Input Layer:    2 neurons
  Hidden Layer:   2 neurons (use Sigmoid)
  Output Layer:   1 neuron  (use Sigmoid)

WEIGHTS AND BIASES:
  Hidden Neuron 1:  w₁ = 0.4, w₂ = 0.6,  b = 0.1
  Hidden Neuron 2:  w₁ = -0.2, w₂ = 0.9, b = 0.0
  Output Neuron:    w₁ = 0.7, w₂ = 0.3,  b = -0.1
```

**STEP 1: Calculate Hidden Neuron 1 output (h₁)**
```
z_h1 = (0.7 × ___) + (0.5 × ___) + ___
     = _____ + _____ + _____
     = _____
h₁ = Sigmoid(_____) = _____
```

**STEP 2: Calculate Hidden Neuron 2 output (h₂)**
```
z_h2 = (0.7 × ___) + (0.5 × ___) + ___
     = _____ + _____ + _____
     = _____
h₂ = Sigmoid(_____) = _____
```

**STEP 3: Calculate Output Neuron (using h₁ and h₂ as inputs)**
```
z_out = (h₁ × 0.7) + (h₂ × 0.3) + (-0.1)
      = (_____ × 0.7) + (_____ × 0.3) + (-0.1)
      = _____ + _____ + (-0.1)
      = _____
output = Sigmoid(_____) = _____
```

**STEP 4: Make a Decision**
```
If output > 0.5: Customer is LIKELY TO BUY  🛒
If output < 0.5: Customer is NOT LIKELY TO BUY  ❌

My prediction: output = _____
Decision: _______________
```

> 💡 **Expected Answers:** h₁ ≈ 0.627, h₂ ≈ 0.596, output ≈ 0.618 → Likely to Buy

---

### 🧮 Exercise C: The Effect of Negative Weights

Redo Exercise A but change w₂ from -0.3 to +0.3:
```
New Weights:  w₁ = 0.8, w₂ = +0.3, w₃ = 0.5, b = 0.2

New z = _______
New f(z) = _______
```

**Discussion Question:**
```
Original output (w₂ = -0.3): _______
New output      (w₂ = +0.3): _______

The output is (higher / lower) when w₂ is positive.
This shows that a NEGATIVE weight _____________ the influence
of that input on the neuron's output.
```

---

## 💻 PART 3 — Python: Build a Neuron from Scratch (20 minutes)

Open a new Python file called `my_first_neuron.py` and type (don't copy-paste!) the following code. Read every comment carefully.

```python
# ============================================================
# my_first_neuron.py
# Goal: Build a single artificial neuron from scratch in Python
# ============================================================

import numpy as np   # NumPy for math operations

# ── Step 1: Define the Sigmoid Activation Function ──────────
def sigmoid(z):
    """
    The Sigmoid function squashes any number to between 0 and 1.
    Formula: f(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))

# ── Step 2: Define a Single Neuron ──────────────────────────
def neuron(inputs, weights, bias):
    """
    A single artificial neuron.
    
    Parameters:
        inputs  : list of input values  [x1, x2, x3, ...]
        weights : list of weight values [w1, w2, w3, ...]
        bias    : single bias value b
    
    Returns:
        The neuron's output after applying sigmoid activation
    """
    # Convert to NumPy arrays for easy math
    inputs  = np.array(inputs)
    weights = np.array(weights)
    
    # Weighted sum: z = w1*x1 + w2*x2 + ... + b
    z = np.dot(inputs, weights) + bias
    
    # Apply activation function
    output = sigmoid(z)
    
    return z, output

# ── Step 3: Test with our Exercise A values ─────────────────
print("=" * 50)
print("  Testing: Exercise A values")
print("=" * 50)

inputs  = [0.6, 0.4, 0.9]
weights = [0.8, -0.3, 0.5]
bias    = 0.2

z, output = neuron(inputs, weights, bias)

print(f"  Inputs:  {inputs}")
print(f"  Weights: {weights}")
print(f"  Bias:    {bias}")
print(f"  Weighted Sum (z): {z:.4f}")
print(f"  Output (sigmoid): {output:.4f}")
print(f"  Neuron fires?    {'YES ✅' if output > 0.5 else 'NO ❌'}")
print()

# ── Step 4: Try YOUR OWN values ──────────────────────────────
print("=" * 50)
print("  Now try YOUR OWN values!")
print("=" * 50)

# ✏️ CHANGE THESE VALUES and see how the output changes!
my_inputs  = [1.0, 0.0, 0.5]   # ← change these
my_weights = [0.5, 0.5, 0.5]   # ← change these
my_bias    = 0.0                 # ← change this

z2, output2 = neuron(my_inputs, my_weights, my_bias)

print(f"  My Inputs:  {my_inputs}")
print(f"  My Weights: {my_weights}")
print(f"  My Bias:    {my_bias}")
print(f"  Weighted Sum (z): {z2:.4f}")
print(f"  Output: {output2:.4f}")
print()
print("  💡 Experiment: What happens when you make all weights very")
print("     large (e.g., 10.0)? What about very small (0.001)?")
```

**Run the file:**
```
python my_first_neuron.py
```

**Experiment Tasks:**
1. Set all weights to `0.0` — what is the output? Why?
2. Set one weight to `100.0` — what happens to the output?
3. Set the bias to `-5.0` — how does this affect the output?
4. Make ALL inputs `0.0` — what is the only thing affecting output now?

**Write your observations:**
```
Observation 1 (all weights = 0):   _______________________________
Observation 2 (one weight = 100):  _______________________________
Observation 3 (bias = -5.0):       _______________________________
Observation 4 (all inputs = 0):    _______________________________
```

---

## 📋 PART 4 — ANN Type Matching (5 minutes)

Match each problem to the correct ANN type:

| # | Problem Description | ANN Type |
|---|--------------------|---------:|
| 1 | Predict tomorrow's temperature from the last 30 days | ? |
| 2 | Classify a photo as "cat" or "dog" | ? |
| 3 | Predict house price from: size, bedrooms, age, location | ? |
| 4 | Generate a realistic human face that doesn't exist | ? |
| 5 | Compress a medical scan image for efficient storage | ? |
| 6 | Translate English text to French | ? |

**Options:** FNN, CNN, RNN, GAN, Autoencoder, RNN

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | All components correctly labeled | 15 |
| Part 2A | All steps computed correctly | 20 |
| Part 2B | Full 2-layer forward pass completed | 25 |
| Part 2C | Effect of negative weights explained | 10 |
| Part 3 | Python neuron runs + 4 experiments done | 20 |
| Part 4 | All 6 problems matched correctly | 10 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 02 — In-Class Task*
