# 🧪 Session 17 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "From Images to Sentences"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will shift paradigms. They will execute a Python script that applies a `Conv1D` network to a natural language dataset (IMDB Movie Reviews) to predict whether the review is positive or negative.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Differentiate between the architectural components of LeNet and AlexNet.
- [ ] Understand the concept of an N-gram in text.
- [ ] Implement a `Conv1D` layer in Keras for sequential text data.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. AlexNet introduced two major architectural components/techniques that were not present in LeNet-5. What were they, and what did they solve?
   *Answer:* __________________________________________________________________

2. If you are building an AI to detect stop signs for a self-driving car, would you use `Conv1D` or `Conv2D`? Why?
   *Answer:* __________________________________________________________________

3. If you have a 1D Kernel of size `2` sliding across the sentence "I love deep learning", what are the three distinct word-pairs (bi-grams) the kernel will look at?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The NLP CNN Experiment (30 minutes)

Create a file named `cnn_nlp_demo.py` and copy the code from the `Code_Snippets` folder.

### Step 1: Analyze the Code
Look at the architecture. It is built for text, not images!
```python
model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
```
- Notice we use `Conv1D` instead of `Conv2D`.
- The `kernel_size=3` means the network will look at chunks of 3 words at a time (tri-grams) to determine if a movie review is positive or negative.
- Notice `GlobalMaxPooling1D`. Instead of shrinking the text gradually, this immediately grabs the most important feature from the entire sentence and throws the rest away.

### Step 2: Run the Script
Execute the script. It will train on a tiny subset of IMDB movie reviews.

### Step 3: Record the Results
Write down the final `val_accuracy` after the 3 epochs finish.
- **1D CNN Validation Accuracy:** ________

---

## 🔍 PART 3 — Analysis

1. Why do you think looking at 3 words at a time (`kernel_size=3`) is better for analyzing text than just looking at 1 word at a time? *(Hint: Think about the phrase "not very good").*
   *Answer:* __________________________________________________________________

2. Before AlexNet, neural networks took weeks to train. How did the creators of AlexNet solve this training speed bottleneck?
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions accurately. | 40 |
| Part 2 | Successfully executed the script and recorded the accuracy. | 30 |
| Part 3 | Answered the 2 analysis questions. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 17 — In-Class Task*
