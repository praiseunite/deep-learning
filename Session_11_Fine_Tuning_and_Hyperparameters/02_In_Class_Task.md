# 🧪 Session 11 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Hyperparameter Hunt"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will stop guessing how many neurons they need. They will use the `keras_tuner` library to automate the discovery of the perfect neural network architecture for the Fashion MNIST dataset.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Differentiate between Parameters and Hyperparameters.
- [ ] Define a model-building function that accepts a `hp` (Hyperparameter) object.
- [ ] Execute a `RandomSearch` using Keras Tuner.
- [ ] Extract the optimal architecture from the tuner.

---

## 🛠️ What You Need
- ✅ Python installed.
- ✅ You must install Keras Tuner: Open your terminal and run `pip install keras-tuner`.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (10 minutes)

Answer the following questions:

1. Look at this line of code: `model.compile(optimizer=Adam(learning_rate=0.001))`
   Is the `learning_rate` a Parameter or a Hyperparameter? Why?
   *Answer:* __________________________________________________________________

2. When performing Fine-Tuning on a pre-trained model, why is it critical to use a very, very small Learning Rate (e.g., 1e-5)? *(Hint: Think about what large weight updates would do to the carefully learned "brain" of the pre-trained model).*
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Keras Tuner Experiment (35 minutes)

Create a file named `hyperparameter_tuning.py`. Copy the starter code provided in the `Code_Snippets` folder.

### Step 1: Understand the HP Object
Read the `build_model(hp)` function in the script.
Notice how we no longer hardcode the number of neurons. Instead, we give Keras Tuner a range:
```python
# Instead of Dense(128)...
hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
keras.layers.Dense(units=hp_units, activation='relu')
```
This tells the Tuner: *"Try 32 neurons, try 64, try 96... all the way up to 256, and tell me which is best."*

### Step 2: Run the Tuner
Run the script. 
The Tuner will start building and training 5 completely different models (`max_trials=5`). It only trains them for 3 epochs each to quickly gauge their potential.

Watch the terminal. You will see it print out statements like:
`Search: Running Trial 1`
`Value for 'units': 128`
`Value for 'learning_rate': 0.001`

### Step 3: Record the Winner
When the script finishes, it will print the "Best Hyperparameters Found". 
Record them here:
- **Best Number of Neurons (Units):** ________
- **Best Learning Rate:** ________

---

## 🔍 PART 3 — Analysis

1. We used `RandomSearch` for this experiment. Keras Tuner also offers `Hyperband` and `BayesianOptimization`. Based on the name alone, why might `RandomSearch` be inefficient if we were testing thousands of different combinations instead of just 5?
   *Answer:* __________________________________________________________________

2. Imagine you are working at a company and your boss asks you to improve an AI model. You decide to use Transfer Learning and download a massive model pre-trained by Google. Your boss asks, *"Why did you download Google's model instead of training our own from scratch?"* What are the two main business justifications you give?
   *Answer 1:* ________________________________________________________________
   *Answer 2:* ________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 2 theory questions. | 30 |
| Part 2 | Successfully installed keras-tuner, ran the script, and recorded the winning configuration. | 40 |
| Part 3 | Answered the 2 analysis questions. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 11 — In-Class Task*
