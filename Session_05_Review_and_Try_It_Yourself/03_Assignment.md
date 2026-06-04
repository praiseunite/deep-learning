# 📋 Session 05 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Tackling a New Dataset: Fashion MNIST"
### Due: Before Session 6 begins
### Estimated Time: 1.5 – 2 hours

---

> **Professor's Note:** *"You have mastered the MNIST handwritten digits. But true mastery means applying your skills to unfamiliar data. For this assignment, you will step up to a harder challenge: classifying clothing items using the **Fashion MNIST** dataset."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Load and explore a brand new dataset from the Keras API.
- Adapt your existing neural network architecture to a new problem.
- Compare the difficulty of different datasets.

---

## 💻 TASK 1 — The Fashion MNIST Challenge (90 minutes)

The **Fashion MNIST** dataset is designed to be a direct drop-in replacement for the original MNIST digit dataset. It has the exact same image size (28x28) and the exact same splits (60,000 training, 10,000 testing).

However, instead of digits 0-9, it contains 10 categories of clothing:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

### Your Coding Mission:
1. Create a new Python file named `fashion_mnist_challenge.py`.
2. Write a complete Keras script to classify these images.
3. You must use a minimum of **2 Hidden Layers** with **ReLU** activation.
4. Train the model for **10 Epochs**.

*Hint: Loading the data is exactly the same, just change the dataset name:*
```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```

### Visualizing the Result (Optional but Highly Recommended):
Use the same Matplotlib code from Session 4 to visualize a prediction. Instead of numbers, the network is now identifying shirts and shoes!

---

## 📝 TASK 2 — Analysis Report (30 minutes)

After successfully training your Fashion MNIST model, answer the following questions in a short document.

1. **Accuracy Comparison:** What was your final Test Accuracy for Fashion MNIST? Was it higher or lower than your Test Accuracy for the original MNIST digits (from Session 4)?
2. **Difficulty Analysis:** Why do you think one dataset is harder for the Neural Network to classify than the other? Think about the shape of a "T-shirt" vs a "Shirt" compared to a "1" vs a "0".
3. **Hyperparameter Theory:** If you wanted to squeeze 2% more accuracy out of your Fashion MNIST model, name two things you could change in your code (e.g., architecture, epochs, optimizers) to try and achieve that.

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1:** Your complete, working `fashion_mnist_challenge.py` script.
2. ✅ **Task 1:** A screenshot of your terminal showing the final Test Accuracy.
3. ✅ **Task 2:** A document (`.docx` or `.pdf`) containing your Analysis Report answers.

**File Naming Convention:**
```
Firstname_Lastname_Session05_fashion.py
Firstname_Lastname_Session05_Terminal.png
Firstname_Lastname_Session05_Report.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 6.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 05 — Assignment*
