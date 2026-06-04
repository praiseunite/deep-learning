# 📋 Session 04 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Tuning the Engine"
### Due: Before Session 5 begins
### Estimated Time: 1.5 – 2 hours

---

> **Professor's Note:** *"You have just built an AI that can read human handwriting with over 95% accuracy. A few decades ago, this was science fiction. But true engineering isn't just running code; it's breaking it to understand how it works. Your homework is to manipulate the Hyperparameters of your MNIST model and observe the consequences."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Understand the impact of **Epochs** on model training and overfitting.
- Understand the impact of **Hidden Layer size** on network capacity.
- Visualize predictions using Matplotlib to verify AI behavior.

---

## 💻 TASK 1 — Python: Hyperparameter Tuning (60 minutes)

You will use your `02_mnist_fnn.py` code from the in-class task as a starting point. Create a copy of it named `mnist_tuning.py`.

You are going to run **THREE different experiments** by changing specific lines of code. For each experiment, record the **Final Training Accuracy** and the **Final Test Accuracy**.

### 🧪 Experiment A: The Power of Deep Learning (Adding Capacity)
Change the architecture. Instead of one hidden layer with 128 neurons, create a much deeper network:
- Hidden Layer 1: 512 neurons, ReLU
- Hidden Layer 2: 256 neurons, ReLU
- Output Layer: 10 neurons, Softmax
*(Train for 5 epochs)*

**Results for Exp A:**
- Training Accuracy: ________
- Test Accuracy: ________

### 🧪 Experiment B: The "Underpowered" Network (Removing Capacity)
Change the architecture back to a single hidden layer, but give it only **10 neurons**.
- Hidden Layer 1: 10 neurons, ReLU
- Output Layer: 10 neurons, Softmax
*(Train for 5 epochs)*

**Results for Exp B:**
- Training Accuracy: ________
- Test Accuracy: ________
- **Question:** Why did the accuracy drop so significantly compared to 128 neurons? (Answer in 1-2 sentences in your document).

### 🧪 Experiment C: Over-training (The Overfitting Danger)
Set the architecture back to the original (1 Hidden Layer, 128 neurons, ReLU).
This time, change the training loop to train for **25 epochs** instead of 5.
```python
history = model.fit(X_train, y_train, epochs=25)
```

**Results for Exp C:**
- Training Accuracy at Epoch 25: ________
- Test Accuracy at the end: ________
- **Question:** You will likely notice that by Epoch 25, the Training Accuracy is extremely high (maybe 99.5%+), but the Test Accuracy stops improving and might even be lower. Explain why training a model for too many epochs can be a bad thing. (Use the term "memorization" or "overfitting").

---

## 🎨 TASK 2 — Python: Visualizing the Prediction (30 minutes)

At the very end of your script, we made a prediction for image `0`. But looking at numbers in a terminal is boring. Let's physically *see* the image and the prediction.

**Add this code to the very bottom of your script:**

```python
# --- VISUALIZING THE PREDICTION ---
# Let's look at image index 42 (You can change this number!)
idx = 42
image = X_test[idx]
true_label = y_test[idx]

# Ask the model to predict
pred_array = model.predict(image.reshape(1, 28, 28))
predicted_label = np.argmax(pred_array)

# Plot the image
plt.figure(figsize=(4,4))
plt.imshow(image, cmap='gray') # Show image in grayscale
plt.title(f"AI Guessed: {predicted_label} | Actual: {true_label}", 
          color=('green' if predicted_label == true_label else 'red'))
plt.axis('off')
plt.show()
```

### Your Mission:
1. Run the script with this new plotting code.
2. A window should pop up showing a handwritten number, with the AI's guess in the title (Green if correct, Red if wrong).
3. **Change the `idx` variable to 3 different numbers** (e.g., 8, 115, 9999).
4. Take a screenshot of ONE of the pop-up windows showing a successful prediction.

---

## 📝 TASK 3 — Concept Review (30 minutes)

Answer the following questions in your own words (3-5 sentences each).

1. Explain the difference between a **Loss Function** (like Cross-Entropy) and an **Optimizer** (like Gradient Descent/Adam). What is the specific job of each?
2. Explain the **Forward Pass** vs. the **Backward Pass** (Backpropagation). During which pass are the actual predictions made, and during which pass is the learning actually happening?

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1 & 3:** A document (`.docx` or `.pdf`) containing your experiment results, the answers to the experiment questions, and the answers to the Task 3 concept questions.
2. ✅ **Task 2:** A screenshot of the Matplotlib window showing the handwritten digit and the AI's prediction.
3. ✅ **Code:** Your final `mnist_tuning.py` script.

**File Naming Convention:**
```
Firstname_Lastname_Session04_Written.pdf
Firstname_Lastname_Session04_Image.png
Firstname_Lastname_Session04_mnist_tuning.py
```

---

## ⏰ Deadline
Submit **before the start of Session 5.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 04 — Assignment*
