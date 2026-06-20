# 🧪 Session 14 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Width vs. Depth Showdown"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will test the Universal Approximation Theorem. They will build two models that have roughly the exact same number of parameters (weights). One will be incredibly wide and shallow. The other will be narrow and deep. They will race them to see which one learns faster and better.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Define Hierarchical Feature Learning.
- [ ] Construct a shallow, wide neural network.
- [ ] Construct a deep, narrow neural network.
- [ ] Compare the parameter efficiency of depth vs. width.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (15 minutes)

Answer the following questions based on the lecture notes:

1. According to the Universal Approximation Theorem, how many hidden layers are mathematically required to solve any problem (assuming the layer has infinite neurons)?
   *Answer:* __________________________________________________________________

2. Why is a deep network better at recognizing a picture of a dog than a shallow network? Use the phrase "Hierarchical Feature Learning" in your answer.
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Code Showdown (30 minutes)

Create a file named `depth_vs_width.py` and copy the code from the `Code_Snippets` folder.

### Step 1: Analyze the Two Architectures
Look closely at the code for `Shallow_Model` and `Deep_Model`.

**The Shallow Model:**
- Has exactly **1 hidden layer**.
- That layer is incredibly wide: **512 neurons**.

**The Deep Model:**
- Has **4 hidden layers**.
- The layers are much narrower: **128 -> 64 -> 32 -> 16 neurons**.

Both models have roughly similar total parameter counts, meaning they take up the same amount of memory.

### Step 2: Run the Race
Run the script. It will train the Shallow model first, and then the Deep model, both for exactly 10 epochs on the Fashion MNIST dataset.

Watch the validation accuracy (`val_accuracy`) for both.

### Step 3: Record the Results
Write down the final `val_accuracy` (at Epoch 10) for both models.
- **Shallow Model Final Val_Accuracy:** ________
- **Deep Model Final Val_Accuracy:** ________

---

## 🔍 PART 3 — Analysis

1. Based on your results in Step 3, which architectural strategy (making the network wider, or making it deeper) was more efficient at learning the complex shapes of the clothing items?
   *Answer:* __________________________________________________________________

2. If a Deep network is so much smarter, why do you think early AI researchers in the 1990s only built Shallow networks? *(Hint: Think about what happens to the gradients when you have too many layers, and think about the hardware available in 1995).*
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 2 theory questions accurately. | 40 |
| Part 2 | Successfully executed the script and recorded the final accuracies. | 30 |
| Part 3 | Answered the 2 analysis questions. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 14 — In-Class Task*
