# 🧪 Session 05 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The 'Try It Yourself' Mega-Lab"
### Duration: 90 minutes

---

> **Instructor Note:** This lab compiles the "Try It Yourself" exercises from Chapters 1, 2, and 3 of the textbook. It is designed as an open-book challenge. Students may work in pairs but must write their own answers and code.

---

## 🎯 Lab Objectives

1. Defend DL concepts against common misconceptions.
2. Trace the mathematical logic of complex neural structures.
3. Diagnose and fix a broken Keras Neural Network script.

---

## 📝 PART 1 — Theory Defenses (From Textbook Session 1)

**Instructions:** You are the Lead AI Architect at a tech company. Your CEO has made three statements. You must politely correct the CEO using your knowledge from Session 1. Write 2-3 sentences for each.

**CEO Statement 1:** *"Let's just use Traditional Machine Learning to build our self-driving car. We can manually program it to recognize stop signs, pedestrians, and lane lines."*
**Your Correction:** ___________________________________________________________
______________________________________________________________________________

**CEO Statement 2:** *"I read that Deep Learning is basically just mapping inputs to outputs. It's essentially just a massive Excel spreadsheet doing linear regression."*
**Your Correction:** ___________________________________________________________
______________________________________________________________________________

**CEO Statement 3:** *"We have a very small dataset of 50 customer reviews. Deep Learning is the best, so let's build a 100-layer neural network to analyze them."*
**Your Correction:** ___________________________________________________________
______________________________________________________________________________

---

## 🧮 PART 2 — The Neuron Logic Puzzle (From Textbook Session 2)

**Instructions:** Look at the following network logic and answer the questions.

**Scenario:** A bank uses a single neuron to approve loans (Output 1 = Approve, Output 0 = Deny).
- **Inputs:** $x_1$ (Income > 50k), $x_2$ (No Criminal Record), $x_3$ (Has existing debt). All inputs are binary (1 or 0).
- **Weights:** $w_1 = 4$, $w_2 = 5$, $w_3 = -6$.
- **Bias:** $b = -2$.
- **Activation:** Step Function (If $Sum > 0 \rightarrow 1$, else $0$).

**Question 1:** A customer walks in. They make $60k ($x_1=1$), have no criminal record ($x_2=1$), but they have existing debt ($x_3=1$). Do they get the loan? (Show your math).
**Math & Answer:** _____________________________________________________________

**Question 2:** The bank changes its policy. Now, having a criminal record ($x_2=0$) means an AUTOMATIC denial, no matter how high the income is. To enforce this using the exact same neuron structure, what should the new Bias ($b$) and Weights ($w_1, w_2, w_3$) be?
*(Hint: Make the penalty for $x_2=0$ impossible to overcome).*
**New Configuration:** _________________________________________________________

---

## 💻 PART 3 — The Broken Code Challenge (From Textbook Session 3/4)

**Instructions:** A junior developer tried to write a script to classify the MNIST digits, but the code is full of errors and won't run. Your job is to debug it.

**The Broken Code:**
```python
import tensorflow as tf
from tensorflow import keras

# 1. Load data
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='linear'),
    keras.layers.Dense(1, activation='relu')
])

# 3. Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

# 4. Train model
model.fit(X_train, y_train, epochs=1)
```

**Find and fix the 5 major errors in this code.**

| Error # | Line of Code / Concept | Why it is wrong | How to fix it |
|---------|------------------------|-----------------|---------------|
| 1 | Preprocessing | | |
| 2 | Input Layer | | |
| 3 | Hidden Layer Activation | | |
| 4 | Output Layer | | |
| 5 | Loss Function | | |

---

## 🏆 PART 4 — Build It Clean (Practical Implementation)

Now that you have debugged the junior developer's code, open your IDE and write the **perfect, clean version** of the MNIST classifier from memory (or using your notes).

**Requirements for your script:**
1. Properly normalizes the data.
2. Uses 2 Hidden Layers (128 neurons and 64 neurons).
3. Uses ReLU for hidden layers and Softmax for the output layer.
4. Trains for 5 epochs.
5. Prints the final Test Accuracy.

**Run the script. If you get over 97% Test Accuracy, you have successfully completed the lab!**

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Corrected all 3 CEO statements with sound reasoning. | 25 |
| Part 2 | Solved both logic puzzles correctly. | 25 |
| Part 3 | Identified and explained the 5 major code errors. | 25 |
| Part 4 | Successfully wrote and ran the corrected Keras script. | 25 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 05 — In-Class Task*
