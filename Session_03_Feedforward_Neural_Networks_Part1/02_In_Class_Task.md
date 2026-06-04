# 🧪 Session 03 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Bouncer and the Activation Lab"
### Duration: 40–50 minutes

---

> **Instructor Note:** In this task, students will manually simulate a Perceptron to understand linear classification, and then use Python to visualize how different activation functions change the shape of data.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Calculate the output of a Perceptron for various inputs.
- [ ] Understand how a Perceptron acts as an AND/OR logic gate.
- [ ] Use Python to visualize Sigmoid, ReLU, and Tanh.
- [ ] Visually explain why ReLU prevents the "Vanishing Gradient".

---

## 🛠️ What You Need
- ✅ Paper and pen.
- ✅ Python environment from Session 1.
- ✅ The Lecture Notes (01_Lecture_Notes.md).

---

## 📋 PART 1 — The Perceptron Bouncer (15 minutes)

Let's build a Perceptron that acts as a logical **AND gate**.
An AND gate only outputs 1 if **BOTH** inputs are 1. Otherwise, it outputs 0.

| Input 1 ($x_1$) | Input 2 ($x_2$) | Desired Output |
|-----------------|-----------------|----------------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

We will use a **Step Function**: If (Sum + Bias) > 0, Output = 1. Else, Output = 0.

### The Challenge:
We have set the weights: $w_1 = 1$, $w_2 = 1$.
**What must the Bias ($b$) be to make this network act exactly like an AND gate?**

### Step-by-Step Test:
Let's try a Bias of $b = -0.5$. Calculate the output for all 4 rows.
*Example for Row 2 ($x_1=0, x_2=1$):*
$Sum = (0 * 1) + (1 * 1) - 0.5 = 0.5$.
Since $0.5 > 0$, the Step Function outputs 1.
Wait, Row 2 should output 0! So $b = -0.5$ is **WRONG**.

**Your Task:**
Find a Bias ($b$) that makes the Perceptron output exactly [0, 0, 0, 1] for the four rows.
*(Hint: Think of a negative number).*

**My chosen Bias ($b$):** _________

**Prove it works (Show your math for all 4 rows):**
Row 1 (0,0): ___________________________________________ -> Output: ___
Row 2 (0,1): ___________________________________________ -> Output: ___
Row 3 (1,0): ___________________________________________ -> Output: ___
Row 4 (1,1): ___________________________________________ -> Output: ___

---

## 📋 PART 2 — The XOR Dilemma (10 minutes)

Now let's try the **XOR (Exclusive OR) gate**.
Output 1 ONLY if the inputs are different.

| $x_1$ | $x_2$ | Desired Output |
|-------|-------|----------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Discussion with your partner:**
Can you find ANY combination of $w_1, w_2$, and $b$ that solves this using a single Perceptron?
Try for 3 minutes.
*Hint: Write down the inequalities. To output 1 for Row 2, $w_2 + b > 0$. To output 0 for Row 4, $w_1 + w_2 + b \leq 0$. Is this mathematically possible?*

**Conclusion:** ______________________________________________________________
**What is required to solve this problem?** __________________________________

---

## 💻 PART 3 — Visualizing Activation Functions in Python (25 minutes)

Let's see these activation functions with our own eyes.

1. Open your code editor and create a new file `activation_lab.py` in your `Session_03` folder.
2. We are going to write a script that plots Sigmoid, Tanh, and ReLU.
3. Type the following code (do not copy-paste, type it to build muscle memory!):

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the input range (z values from -10 to 10)
# This creates an array of 100 numbers evenly spaced between -10 and 10
z = np.linspace(-10, 10, 100)

# 2. Define the Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# 3. Apply the functions to our input z
y_sigmoid = sigmoid(z)
y_relu = relu(z)
y_tanh = tanh(z)

# 4. Plotting the results
plt.figure(figsize=(12, 4))

# Plot Sigmoid
plt.subplot(1, 3, 1)
plt.plot(z, y_sigmoid, color='blue', linewidth=2)
plt.title("Sigmoid")
plt.grid(True)

# Plot Tanh
plt.subplot(1, 3, 2)
plt.plot(z, y_tanh, color='purple', linewidth=2)
plt.title("Tanh")
plt.grid(True)

# Plot ReLU
plt.subplot(1, 3, 3)
plt.plot(z, y_relu, color='orange', linewidth=2)
plt.title("ReLU")
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Run the script:** `python activation_lab.py`

### 🔍 Analysis Questions (Answer based on your graphs):

1. Look at the **Sigmoid** graph. What happens to the slope (the steepness of the line) when $z$ is 8? What happens when $z$ is -8?
   __________________________________________________________________________

2. This "flatness" at the edges is called the **Vanishing Gradient**. If a network learns by calculating slopes, why is a flat slope bad?
   __________________________________________________________________________

3. Look at the **ReLU** graph. Does the right side (positive $z$) ever go flat? How does this solve the Vanishing Gradient problem?
   __________________________________________________________________________

4. Look at the left side of the **ReLU** graph (negative $z$). The output is 0. This is called a "Dead Neuron". Why might a neuron that always outputs 0 be a problem?
   __________________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Found correct Bias for AND gate and showed math. | 30 |
| Part 2 | Explained why XOR cannot be solved linearly. | 20 |
| Part 3 | Successfully ran Python plotting script. | 20 |
| Part 3 | Answered the 4 analysis questions correctly. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 03 — In-Class Task*
