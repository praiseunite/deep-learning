# 🧪 Session 07 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Batch Size Experiment"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will manipulate the `batch_size` parameter in Keras to observe the real-world effects of Batch, Stochastic, and Mini-Batch Gradient Descent on training time and accuracy.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Understand how to configure batch sizes in Keras.
- [ ] Observe the training speed differences between SGD, Batch, and Mini-batch.
- [ ] Analyze the stability of the loss curve based on batch size.

---

## 🛠️ What You Need
- ✅ Python installed with TensorFlow/Keras.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 💻 PART 1 — The Theory Check (10 minutes)

Answer the following questions based on the lecture notes:

1. You have a dataset of 10,000 images. If you use **Mini-Batch Gradient Descent** with a batch size of `100`, how many times will the network update its weights in **one single Epoch**?
   *Math/Answer:* ________________________________

2. You have a dataset of 5,000 images. If you use **Batch Gradient Descent** (the whole dataset at once), how many times will the network update its weights in **10 Epochs**?
   *Math/Answer:* ________________________________

3. Why do we generally use batch sizes that are powers of 2 (e.g., 32, 64, 128, 256)? *(Hint: Think about the hardware running the math).*
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Code Experiment (40 minutes)

You will use Keras to run all three variants of Gradient Descent on the Fashion MNIST dataset.

### Step 1: Set up the script
Create a file named `batch_size_experiment.py`.
Copy the starter code provided by your instructor (or from the `Code_Snippets` folder).

### Step 2: Understand the Code
Look at the `model.fit()` line in the code provided:
```python
history = model.fit(X_train, y_train, epochs=3, batch_size=CURRENT_BATCH_SIZE)
```
Notice the `batch_size=` parameter. This is the magic command in Keras that switches between the variants!
- `batch_size = 1` $\rightarrow$ Stochastic Gradient Descent (SGD)
- `batch_size = 32` $\rightarrow$ Mini-Batch Gradient Descent
- `batch_size = len(X_train)` $\rightarrow$ Batch Gradient Descent

### Step 3: Run the Experiments

**Experiment A: Stochastic Gradient Descent (SGD)**
Set the batch size to `1`. Run the script.
*Note: This will be SLOW. Watch how the progress bar updates. It is doing 60,000 individual weight updates per epoch!*
- **Training Time (approx):** ________ seconds
- **Final Test Accuracy:** ________%

**Experiment B: Mini-Batch Gradient Descent (The Standard)**
Change the batch size to `32`. Run the script.
- **Training Time (approx):** ________ seconds
- **Final Test Accuracy:** ________%

**Experiment C: Large Mini-Batch**
Change the batch size to `1024`. Run the script.
- **Training Time (approx):** ________ seconds
- **Final Test Accuracy:** ________%

**Experiment D: Full Batch Gradient Descent**
Change the batch size to `60000` (the entire dataset size). Run the script.
- **Training Time (approx):** ________ seconds
- **Final Test Accuracy:** ________%

---

## 🔍 PART 3 — Analysis

Compare the results of your 4 experiments and answer these questions:

1. Which batch size trained the fastest? Why?
2. Which batch size resulted in the highest test accuracy? 
3. Look at Experiment D (Full Batch). Even though it was relatively fast per epoch, look at the final accuracy. Was it good or terrible? Why do you think a massive batch size hurts the network's ability to learn effectively in only 3 epochs?

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Solved the 3 theory questions correctly. | 20 |
| Part 2 | Completed all 4 coding experiments and recorded the data. | 50 |
| Part 3 | Answered the 3 analysis questions based on the data. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 07 — In-Class Task*
