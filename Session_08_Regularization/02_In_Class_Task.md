# 🧪 Session 08 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Dropout Experiment"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will intentionally overfit a massive neural network on a small subset of data, and then use Keras `Dropout` layers to rescue it.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Identify the symptoms of Overfitting by comparing Training and Validation accuracy.
- [ ] Implement `keras.layers.Dropout()` in a Sequential model.
- [ ] Analyze the impact of Dropout rates (e.g., 0.2 vs 0.5) on network generalization.

---

## 🛠️ What You Need
- ✅ Python installed with TensorFlow/Keras.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (10 minutes)

Answer the following questions based on the lecture notes:

1. Look at the following training log for a neural network:
   - Epoch 1: Train Acc 80%, Val Acc 79%
   - Epoch 5: Train Acc 89%, Val Acc 88%
   - Epoch 15: Train Acc 96%, Val Acc 89%
   - Epoch 25: Train Acc 99%, Val Acc 86%
   **Question:** At approximately which epoch did the model begin to overfit? How do you know?
   *Answer:* __________________________________________________________________

2. Why is a Dropout layer automatically deactivated by Keras when you call `model.evaluate()` or `model.predict()`?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Keras Experiment (35 minutes)

We are going to use the Fashion MNIST dataset, but to force the model to overfit quickly, we are only going to let it see the first 5,000 images (instead of all 60,000). We will also give it a massive capacity (512 neurons).

### Step 1: Set up the script
Create a file named `dropout_experiment.py`. Copy the starter code provided in the `Code_Snippets` folder.

### Step 2: Run the Baseline (Un-regularized) Model
Read through the code. Notice we have a huge hidden layer (512 neurons) and we are training for 20 epochs on a small dataset.
Run the script as-is.

Look closely at the terminal output for the final epochs:
- **Baseline Training Accuracy:** ________%
- **Baseline Validation Accuracy:** ________%
- **Is it overfitting?** (Yes/No) ________

### Step 3: Implement Dropout
We are now going to add Dropout to prevent this memorization.

Un-comment the `Dropout` layer in the `build_dropout_model` function inside your script:
```python
def build_dropout_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'),
        
        # Add Dropout here! It will randomly disable 40% of the 512 neurons.
        keras.layers.Dropout(0.4), 
        
        keras.layers.Dense(10, activation='softmax')
    ])
    # ... compile ...
```
Change the variable at the bottom of the script to `run_dropout_experiment = True`. Run the script again.

- **Dropout Training Accuracy:** ________%
- **Dropout Validation Accuracy:** ________%

### Step 4: The Extreme Dropout Test
Change the Dropout rate from `0.4` (40%) to `0.9` (90%). Run the script.
- **Extreme Dropout Training Acc:** ________%
- **Extreme Dropout Validation Acc:** ________%
- *Why did the accuracy completely collapse?* ________________________________

---

## 🔍 PART 3 — Analysis

Compare the Baseline model to the 40% Dropout model. 

1. You should notice that the Training Accuracy was actually *lower* in the Dropout model than in the Baseline model. Why is a lower training accuracy considered a "success" in this context?
2. Did the Validation Accuracy improve when using 40% Dropout? 
3. If you were deploying this model to a real-world smartphone app, which model would you choose to deploy (Baseline or Dropout) and why?

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 2 theory questions correctly. | 20 |
| Part 2 | Successfully ran the Baseline, 40% Dropout, and 90% Dropout tests. | 50 |
| Part 3 | Answered the 3 analysis questions correctly. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 08 — In-Class Task*
