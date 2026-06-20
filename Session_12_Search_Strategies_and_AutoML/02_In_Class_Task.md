# 🧪 Session 12 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Bayesian Upgrade"
### Duration: 45–55 minutes

---

> **Instructor Note:** In the previous session, students used a basic `RandomSearch` tuner. Today, they will upgrade their tuning script to use intelligent `BayesianOptimization`, allowing the tuner to learn from its past mistakes.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Differentiate between Grid, Random, and Bayesian search strategies.
- [ ] Implement `kt.BayesianOptimization` using Keras Tuner.
- [ ] Define what AutoML does in the Machine Learning lifecycle.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ `keras-tuner` installed (`pip install keras-tuner`).
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. You are training a massive deep learning model that takes 4 hours to train per epoch. You need to tune 5 different hyperparameters. Why is **Grid Search** an absolutely terrible choice for this scenario?
   *Answer:* __________________________________________________________________

2. Explain the fundamental difference between **Random Search** and **Bayesian Optimization**. *(What does Bayesian do that Random does not?)*
   *Answer:* __________________________________________________________________

3. Name three tasks that an **AutoML** pipeline handles completely automatically, without human intervention.
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Bayesian Tuner Experiment (30 minutes)

Create a file named `search_strategies_demo.py` and copy the code from the `Code_Snippets` folder.

### Step 1: Analyze the Code
Look at the script. We are using the exact same Tunable Model (`build_model`) as Session 11. However, look at the Tuner initialization:

```python
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='bayesian_dir',
    project_name='fashion_bayes'
)
```
Instead of picking randomly, this tuner will use complex probability math to pick the next hyperparameter set based on the previous trial's success.

### Step 2: Run the Tuner
Run the script. Watch the terminal output carefully. 
Because it is using Bayesian Optimization, it might take a fraction of a second longer *between* trials to calculate the math for the next best guess.

### Step 3: Record the Results
- **Winning number of Neurons:** ________
- **Winning Learning Rate:** ________

---

## 🔍 PART 3 — Analysis

1. We only set `max_trials=5`. Why is `max_trials=5` actually a very unfair test for Bayesian Optimization compared to Random Search? *(Hint: Bayesian optimization needs to learn from its past. How much "past" history does it have after only 5 trials?)*
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions accurately. | 40 |
| Part 2 | Successfully executed the Bayesian Tuner and recorded results. | 40 |
| Part 3 | Answered the analysis question regarding Bayesian limitations. | 20 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 12 — In-Class Task*
