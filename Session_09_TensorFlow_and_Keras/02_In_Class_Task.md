# 🧪 Session 09 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Raw Tensors vs The Steering Wheel"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will drop down into "Raw TensorFlow" to manually create Tensors and perform math. Then, they will learn to rebuild their previous models using the advanced Keras Functional API instead of the Sequential API.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Initialize constant and variable Tensors using raw `tf`.
- [ ] Perform basic mathematical operations on Tensors.
- [ ] Construct a neural network using the Keras Functional API.

---

## 🛠️ What You Need
- ✅ Python installed with TensorFlow/Keras.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 💻 PART 1 — Playing with Raw Tensors (15 minutes)

Before Keras existed, Deep Learning engineers had to build networks using raw Tensor math. Let's see what that looks like.

Create a file named `01_tensor_basics.py`. Copy the code from the `Code_Snippets` folder and run it.

Look closely at the terminal output and answer the following:

1. What is the difference in output between a `tf.constant` and a `tf.Variable`? Why might a Neural Network use `tf.Variable` for its weights instead of `tf.constant`?
   *Answer:* __________________________________________________________________

2. In the script, we multiplied a 1D Tensor `[1, 2, 3]` by the scalar `5`. The output was `[5, 10, 15]`. This is called "Broadcasting" (TensorFlow automatically applies the scalar to every element). Why is this hardware-efficient on a GPU?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Functional API Challenge (30 minutes)

Up until now, you have built models like this (The Sequential API):
```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

**Your Challenge:**
Create a new Python script called `functional_api_lab.py`.
Re-write the exact model shown above, but use the **Keras Functional API** (as shown in Section 5 of your Lecture Notes).

*Checklist for success:*
- [ ] You must define an explicit `keras.Input` layer.
- [ ] You must pass the output of one layer directly into the next layer using parentheses, e.g., `layer(previous_output)`.
- [ ] You must use `keras.Model(inputs=..., outputs=...)` to finalize the model.
- [ ] Add `model.summary()` at the end of your script to prove the architecture is identical.

---

## 🔍 PART 3 — Concept Check

1. Look at your `model.summary()` output. It should show a column called `Connected to`. You never saw this column in the Sequential API. Why does the Functional API need to explicitly track what layer is connected to what?
   *Answer:* __________________________________________________________________

2. If Keras does all the heavy lifting, why is it still important for an AI engineer to understand that TensorFlow Computational Graphs are running underneath? *(Think about debugging or optimizing for speed).*
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 2 Tensor theory questions. | 30 |
| Part 2 | Successfully rebuilt the model using the Functional API and printed the summary. | 50 |
| Part 3 | Answered the 2 analysis questions. | 20 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 09 — In-Class Task*
