# 🧪 Session 20 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Memory Test"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will execute a simple RNN on Kaggle. They will train the network to predict the next number in a sequence.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Articulate why standard Dense networks fail on sequential data.
- [ ] Explain the concept of Backpropagation Through Time (BPTT).
- [ ] Implement a `SimpleRNN` layer using Keras.

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. If you feed the sentence "The weather is very nice today" into an RNN, what information does the RNN use when processing the word "nice"?
   *Answer:* __________________________________________________________________

2. What does BPTT stand for, and how is it different from standard Backpropagation?
   *Answer:* __________________________________________________________________

3. Briefly explain the "Vanishing Gradient Problem" in your own words. Why does an RNN have a "short-term memory"?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Sequence Experiment (30 minutes)

Open your Kaggle Notebook. We are going to train an RNN to predict the next number in a simple time-series pattern:
`[10, 20, 30] -> Predicts 40`
`[20, 30, 40] -> Predicts 50`

### Step 1: The Code
Create a new cell in your Kaggle notebook, copy the code from `Code_Snippets/rnn_toy_sequence.py`, and hit Play.

### Step 2: Analyze the Output
Look at the final prediction the model makes at the very bottom of the output.

1. We asked the model to predict what comes after `[70, 80, 90]`. What number did the RNN predict?
   *Answer:* __________________________________________________________________

2. Look at the code. We used `keras.layers.SimpleRNN`. In the lecture, we learned this has a major flaw for long sequences. What is that flaw?
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions accurately. | 50 |
| Part 2 | Successfully ran the RNN script and analyzed the output. | 50 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 20 — In-Class Task*
