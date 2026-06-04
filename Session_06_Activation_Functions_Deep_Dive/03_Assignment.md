# 📋 Session 06 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Dying ReLU Detective"
### Due: Before Session 7 begins
### Estimated Time: 1 hour

---

> **Professor's Note:** *"In class, you saw how different activation functions perform under normal conditions. In this homework, you will intentionally sabotage a network to force the 'Dying ReLU' problem to occur, and then you will prove how Leaky ReLU fixes it."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Understand the impact of negative numbers and bias on standard ReLU.
- Demonstrate practically how "Leaky ReLU" prevents dead neurons.
- Write a professional summary of activation function behavior.

---

## 💻 TASK 1 — Forcing a Dead Neuron (40 minutes)

You do not need Keras for this task. You will write pure, basic Python math.

Imagine a single neuron in a hidden layer. It receives three inputs from the previous layer, and calculates its weighted sum: $z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + b$.

### Your Coding Mission:
1. Create a new file named `dead_neuron_test.py`.
2. Write a Python function for standard ReLU.
3. Write a Python function for Leaky ReLU (use an alpha of `0.05`).
4. **The Scenario:** The neuron has learned terrible, highly negative weights.
   - Input $x = 100$
   - Weight $w = -5$
   - Bias $b = -50$
5. Calculate the raw sum $z$.
6. Pass $z$ through both your ReLU function and your Leaky ReLU function.
7. Print the outputs.

**Example Code Structure:**
```python
def standard_relu(z):
    # Your logic here

def leaky_relu(z, alpha=0.05):
    # Your logic here

# Calculate z
# Print results
```

---

## 📝 TASK 2 — The Architect's Report (20 minutes)

Write a short report (2-3 paragraphs) answering the following questions. Write as if you are explaining this to a junior developer on your team.

1. **The Sabotage:** In Task 1, what was the output of the Standard ReLU function? Why is this a major problem if this happens during epoch 1 of a 100-epoch training run? (Use the term "gradient" in your answer).
2. **The Rescue:** What was the output of the Leaky ReLU function? How does this specific output mathematically fix the problem mentioned in question 1?
3. **The Trade-off:** If Leaky ReLU and ELU are "better" at preventing dead neurons than Standard ReLU, why is Standard ReLU still the most popular default activation function in the world?

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1:** Your `dead_neuron_test.py` script.
2. ✅ **Task 1:** A screenshot of your terminal showing the printed outputs.
3. ✅ **Task 2:** A document (`.docx` or `.pdf`) containing your Architect's Report.

**File Naming Convention:**
```
Firstname_Lastname_Session06_dead_neuron.py
Firstname_Lastname_Session06_Terminal.png
Firstname_Lastname_Session06_Report.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 7.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 06 — Assignment*
