# 🧪 Session 06 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Activation Playground"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will replace the default ReLU activation function with advanced variants (Leaky ReLU, ELU, and Swish) using the Keras API, and observe how it impacts training speed and accuracy.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Understand the mathematical outputs of different activation variants.
- [ ] Implement advanced activation functions within a Keras Sequential model.
- [ ] Analyze the trade-offs between computation speed and model accuracy.

---

## 🛠️ What You Need
- ✅ Python installed with TensorFlow/Keras.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Math Check (15 minutes)

Before we code, let's ensure you understand how these functions process raw numbers.
Assume a neuron calculates a raw linear sum of **$z = -5.0$**. 

Calculate the output of this neuron if it uses the following activation functions:

1. **Standard ReLU** `max(0, z)`
   - Output: ________
2. **Leaky ReLU** (Assuming the "leak" alpha is 0.1, formula: `if z < 0: return z * 0.1`)
   - Output: ________
3. **Linear** (No activation function)
   - Output: ________

*Teacher check-in: Discuss why the linear output is dangerous for deep networks.*

---

## 💻 PART 2 — Keras Implementation: The Variant Race (40 minutes)

You will write a script to build **four identical neural networks**, differing ONLY in their hidden layer activation functions. We will train them all on the Fashion MNIST dataset to see which one performs best.

### Step 1: Set up the script
Create a file named `activation_playground.py`.
Import libraries and load the Fashion MNIST data (normalize it to 0-1, just like in Session 5).

### Step 2: Write a Model Builder Function
Instead of copy-pasting the model code four times, write a Python function that takes an activation name and returns a compiled model.

```python
import tensorflow as tf
from tensorflow import keras
import time # To track how long training takes

# ... (Load and normalize Fashion MNIST data here) ...

def build_and_compile_model(activation_function):
    """Builds a 2-hidden-layer network with the specified activation."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=activation_function),
        keras.layers.Dense(64, activation=activation_function),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
```

### Step 3: Run the Experiment Loop
We will loop through a list of activation functions, train a model for each, and record the time and accuracy.

```python
# The variants we want to test
activations_to_test = ['relu', 'leaky_relu', 'elu', 'swish']

print("--- STARTING THE ACTIVATION RACE ---")

for act in activations_to_test:
    print(f"\nTraining model with [ {act.upper()} ]...")
    
    model = build_and_compile_model(act)
    
    # Start a stopwatch
    start_time = time.time()
    
    # Train for 5 epochs (silent mode with verbose=0 so it doesn't flood the terminal)
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Stop the stopwatch
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Test accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Result for {act.upper()}:")
    print(f"  -> Accuracy: {accuracy*100:.2f}%")
    print(f"  -> Time: {time_taken:.2f} seconds")
```

### Step 4: Run and Analyze
Run the script. It will take a few minutes as it trains 4 separate neural networks in the background.

**Answer the following based on your terminal output:**
1. Which activation function achieved the **highest accuracy**?
2. Which activation function was the **fastest** to train?
3. Which activation function was the **slowest** to train?
4. **Critical Thinking:** Look at the slowest function. Why do you think it took longer than standard ReLU? (Refer to your Lecture Notes).

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Solved the math calculations correctly. | 20 |
| Part 2 | Successfully coded and ran the `activation_playground.py` loop. | 50 |
| Part 2 | Answered the 4 analysis questions comparing speed and accuracy. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 06 — In-Class Task*
