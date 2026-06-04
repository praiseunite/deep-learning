# 📋 Session 03 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Connecting the Layers"
### Due: Before Session 4 begins
### Estimated Time: 1.5 – 2 hours

---

> **Professor's Note:** *"You now understand perceptrons and activation functions. In Session 4, we will train a full FNN. This assignment bridges the gap by having you manually trace data through a Feedforward Network, and then implement a more advanced activation function in Python."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Map inputs to outputs in a complete Feedforward Neural Network.
- Understand how layer dimensions determine the number of weights (parameters).
- Research and implement the **Leaky ReLU** activation function in Python.

---

## 📝 TASK 1 — Network Architecture & Parameter Counting (30 minutes)

You are designing an FNN to predict whether a customer will default on a loan based on 5 features: Age, Income, Credit Score, Debt Amount, and Loan Duration.

Your architecture is:
- **Input Layer:** 5 neurons (for the 5 features).
- **Hidden Layer 1:** 10 neurons.
- **Hidden Layer 2:** 8 neurons.
- **Output Layer:** 1 neuron (Binary output: Default = 1, Safe = 0).

### Answer the following questions:
*Hint: A dense connection from a layer of size M to a layer of size N requires (M * N) weights, plus N biases.*

1. **How many weights (w)** connect the Input Layer to Hidden Layer 1?
2. **How many biases (b)** are in Hidden Layer 1?
3. **How many weights (w)** connect Hidden Layer 1 to Hidden Layer 2?
4. **How many biases (b)** are in Hidden Layer 2?
5. **How many total learnable parameters (weights + biases)** are in this entire network? Show your calculation.

**Write your answers clearly in your submission document.**

---

## 💻 TASK 2 — Python: Implementing "Leaky ReLU" (45 minutes)

In class, we discussed the "Dying ReLU" problem: if a neuron using ReLU gets a negative input, it outputs 0. If it stays negative, the gradient is 0, and the neuron never learns again—it is "dead".

To fix this, researchers invented **Leaky ReLU**. Instead of being completely flat (0) for negative numbers, Leaky ReLU has a very small slope (like 0.01) for negative numbers.

**Formula for Leaky ReLU:**
- If $z \geq 0$, output $= z$
- If $z < 0$, output $= 0.01 * z$

### Your Coding Mission:
1. Create a Python file named `advanced_activations.py`.
2. Write a Python function `leaky_relu(z)` that implements this formula. (You can use `np.where` or a simple if/else for arrays).
3. Test your function with a list of inputs: `[-5, -1, 0, 2, 10]`.
4. **BONUS:** Use Matplotlib to plot Leaky ReLU from -10 to 10, similar to what we did in the in-class task.

**Provide the Python code and a screenshot of the output/plot in your submission.**

---

## 📝 TASK 3 — Case Study Reflection (30 minutes)

Write a short paragraph (150-200 words) answering the following prompt:

*"You are building an AI to scan medical X-Rays and detect a rare bone disease. Your final output needs to be a probability percentage (e.g., 'There is an 87% chance this patient has the disease'). Which activation function must you use on the **final output neuron**, and why? Why would using ReLU on the final output neuron be a terrible idea for this specific task?"*

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1:** Parameter counting answers.
2. ✅ **Task 2:** Python code (`advanced_activations.py`) and a screenshot of its output/plot.
3. ✅ **Task 3:** Short written reflection.

**File Naming Convention:**
```
Firstname_Lastname_Session03_Assignment.docx
Firstname_Lastname_Session03_advanced_activations.py
```

---

## ⏰ Deadline
Submit **before the start of Session 4.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 03 — Assignment*
