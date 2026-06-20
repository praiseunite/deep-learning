# 🧪 Session 19 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "My First Training Run"
### Duration: 30 minutes

---

> **Instructor Note:** The goal here is sheer execution. Students will build a neural network line-by-line, running each small piece of code in its own cell to understand exactly how the pieces fit together.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Understand the difference between importing data, building a model, and training a model.
- [ ] Successfully execute a multi-layer Neural Network script step-by-step in Kaggle.
- [ ] Read the terminal output to identify "Loss" and "Accuracy" during training.

---

## 💻 The Step-by-Step Execution (20 minutes)

Open your Kaggle Notebook (with the GPU turned on). We are going to build our AI one block at a time.

### Cell 1: Getting the Data
Create a code cell, paste this code, and hit the **Play** button. This downloads 60,000 images of handwritten numbers.
```python
import tensorflow as tf
from tensorflow import keras

print("Downloading Data...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0  # Normalize pixel values
print("Data Ready!")
```

### Cell 2: Building the Brain
Create a *new* cell below the first one. Paste this code and hit **Play**. This constructs the empty neural network.
```python
print("Building Model Architecture...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Flatten the 2D image into a 1D line
    keras.layers.Dense(128, activation='relu'), # Hidden layer with 128 neurons
    keras.layers.Dense(10, activation='softmax') # Output layer (10 digits)
])
print("Model Built!")
```

### Cell 3: Compiling the Brain
Create a *new* cell. Paste this and hit **Play**. This tells the AI how to learn (its optimizer and loss function).
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Model Compiled and ready to learn.")
```

### Cell 4: Training (The Exciting Part)
Create your final cell. Paste this and hit **Play**. Watch the progress bar!
```python
print("Starting Training...")
# 'epochs=5' means the AI will look at the 60,000 images 5 times over.
model.fit(X_train, y_train, epochs=5)
```

---

## 🔍 Reading the Output

While Cell 4 is running, you will see output that looks like this:
`Epoch 1/5`
`1875/1875 [==============================] - 5s 2ms/step - loss: 0.2941 - accuracy: 0.9150`

What does this mean?
- **Epoch:** One full pass through the entire dataset.
- **1875/1875:** The data is broken up into 1,875 "batches". The bar fills up as it finishes each batch.
- **loss:** How wrong the model is. You want this number to go *down* over time.
- **accuracy:** How many images it guessed correctly (0.9150 means 91.5%). You want this number to go *up* over time.

---

## 📋 The Recording (15 minutes)

Wait for all 5 Epochs to finish. Look at the very last line of the output for `Epoch 5/5`.

1. What was the final `loss` number?
   *Answer:* ________________________________

2. What was the final `accuracy` number?
   *Answer:* ________________________________

3. Looking at the output for `Epoch 1` compared to `Epoch 5`, did the loss go down? Did the accuracy go up?
   *Answer:* ________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Successfully pasted and executed the code in Kaggle. | 50 |
| Part 2 | Recorded the final loss and accuracy correctly. | 50 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 19 — In-Class Task*
