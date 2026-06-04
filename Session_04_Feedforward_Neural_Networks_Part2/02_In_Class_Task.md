# 🧪 Session 04 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Hello World of AI — Classifying MNIST"
### Duration: 45–55 minutes

---

> **Instructor Note:** This is a milestone session. Students will write their first complete, functional neural network using TensorFlow/Keras to classify real image data. Ensure all students have the required libraries installed before beginning.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Understand the concept of Gradient Descent visually via a Python simulation.
- [ ] Load and preprocess image data (MNIST) for a neural network.
- [ ] Build a Feedforward Neural Network using the Keras Sequential API.
- [ ] Compile and train the model, observing the Loss decrease across Epochs.

---

## 🛠️ What You Need
- ✅ Python installed with pip.
- ✅ Required Libraries: `pip install tensorflow numpy matplotlib`
- ✅ The Lecture Notes (01_Lecture_Notes.md) for reference.

---

## 📋 PART 1 — The Hiker in the Fog (Gradient Descent Demo) (15 minutes)

Before we build a massive network, let's look at how the "Hiker in the Fog" finds the bottom of the valley.

1. Open your IDE and create a file named `01_gradient_descent_demo.py` in your Session_04 folder.
2. Copy the code provided by your instructor (or from the Code Snippets folder).
3. Run the script: `python 01_gradient_descent_demo.py`

### 🔍 Experiment & Observe
The script simulates Gradient Descent trying to find the bottom of a curve (where $x=0$). 
Look at the terminal output and answer these questions:

1. Look at the `Loss` value at Step 1, Step 5, and Step 10. What is happening to the Loss over time?
   __________________________________________________________________________

2. **Change the Learning Rate:** In the code, find the variable `LEARNING_RATE = 0.1` and change it to `0.9` (A massive stride!). Run the script again. What happens to the "Current X" and "Loss"? Why did this happen?
   __________________________________________________________________________

3. **Change the Learning Rate:** Now change it to `0.001` (A tiny baby step). Run it again. Did it reach the bottom ($X=0$) by step 15? Why?
   __________________________________________________________________________

*Teacher check-in: Ensure everyone understands that the Learning Rate determines the step size!*

---

## 💻 PART 2 — Building the MNIST Digit Classifier (35 minutes)

This is it. Your first real AI model. We are going to build a Feedforward Neural Network (FNN) to recognize human handwriting.

### Step 1: Set up the file
Create a new file called `02_mnist_fnn.py`. We will build this step-by-step. Type the code carefully.

### Step 2: Import Libraries & Load Data
```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the MNIST dataset directly from Keras
print("Loading MNIST data...")
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training data shape: {X_train.shape}") # Should be (60000, 28, 28)
```

### Step 3: Preprocess the Data
Neural networks like small numbers (between 0 and 1). Pixel values are between 0 and 255. We must divide by 255.0 to normalize them.

```python
# 2. Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### Step 4: Build the Architecture (The FNN)
We need an Input Layer (flattened), a Hidden Layer (ReLU), and an Output Layer (10 digits).

```python
# 3. Build the Feedforward Neural Network
print("Building the model...")
model = keras.Sequential([
    # Input Layer: Flattens the 28x28 2D image into a 1D array of 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # Hidden Layer: 128 neurons, ReLU activation function
    keras.layers.Dense(128, activation='relu'),
    
    # Output Layer: 10 neurons (digits 0-9), Softmax outputs probabilities
    keras.layers.Dense(10, activation='softmax')
])

# Show a summary of the parameters
model.summary()
```

### Step 5: Compile the Model (Attach the Engine)
We must tell the model what Loss Function to use, and which Optimizer (Gradient Descent algorithm) to use.

```python
# 4. Compile the model
model.compile(
    optimizer='adam', # Adam is a smart, advanced version of Gradient Descent
    loss='sparse_categorical_crossentropy', # Standard loss for classification
    metrics=['accuracy'] # We want to track how accurate it is
)
```

### Step 6: Train the Model (The Workout)
```python
# 5. Train the model!
print("\n--- STARTING TRAINING ---")
# epochs=5 means the model will see the entire dataset 5 times
history = model.fit(X_train, y_train, epochs=5)
```

### Step 7: Evaluate & Test
```python
# 6. Evaluate on test data (data it has never seen before)
print("\n--- EVALUATING ON TEST DATA ---")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 7. Let's make a single prediction to see it work!
image_index = 0  # Look at the very first test image
prediction_array = model.predict(X_test[image_index].reshape(1, 28, 28))
predicted_digit = np.argmax(prediction_array)
actual_digit = y_test[image_index]

print(f"\nPrediction for image 0: Model guessed {predicted_digit}, Actual is {actual_digit}")
```

### 🚀 RUN THE SCRIPT
Save the file and run it: `python 02_mnist_fnn.py`

Watch the terminal as it trains across the 5 epochs.

### 🔍 Analysis Questions:
1. What was the network's accuracy on the training data at the end of Epoch 1? What was it at the end of Epoch 5?
   __________________________________________________________________________

2. What was the final Test Accuracy? Is it higher or lower than the Training Accuracy? (We will discuss why in the next session!)
   __________________________________________________________________________

3. Look at the `model.summary()` output in your terminal. How many total parameters (weights and biases) does this simple network have?
   __________________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Completed Gradient Descent experiment and answered 3 questions. | 30 |
| Part 2 | Successfully coded, compiled, and trained the MNIST model. | 50 |
| Part 2 | Answered the 3 analysis questions regarding model accuracy. | 20 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 04 — In-Class Task*
