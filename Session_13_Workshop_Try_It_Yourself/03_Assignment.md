# 📋 Session 13 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Hyperparameter Tuning Challenge"
### Due: Before Session 14 begins
### Estimated Time: 60 minutes

---

> **Professor's Note:** *"In class today, you trained a model on CIFAR-10, but the accuracy was likely quite low (around 45-55%). Your job as an AI engineer is not just to build a model, but to optimize it. For your homework, you must drastically improve the performance of our CIFAR-10 model."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Experiment with hyperparameter tuning manually.
- Understand the relationship between model depth, width, and accuracy.
- Practice the iterative process of Deep Learning development.

---

## 💻 THE CODING CHALLENGE

Take the `end_to_end_training.py` script you fixed during the in-class lab. 
Save a copy of it as `cifar_optimized.py`.

Your goal is to modify the architecture and hyperparameters of the model so that the **Validation Accuracy (`val_accuracy`) reaches at least 55%** (0.5500) after 15 epochs.

**Rules & Guidelines:**
1. You may change the number of hidden layers (add more, or remove some).
2. You may change the number of neurons in those layers.
3. You may adjust the `Dropout` percentage.
4. You may change the number of `epochs` to 15.
5. You **must not** change the dataset.
6. You **must not** change the final output layer (it must remain 10 neurons with softmax).

*Hint: CIFAR-10 images are very complex. A single hidden layer of 128 neurons is not "smart" enough to understand them. Try making the network "deeper" (more layers) and "wider" (more neurons like 512 or 256).*

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ Your final `cifar_optimized.py` script.
2. ✅ A screenshot of your terminal showing the end of the final epoch (Epoch 15/15) where the `val_accuracy` is 0.5500 or higher.

**File Naming Convention:**
```
Firstname_Lastname_Session13_Optimized.py
Firstname_Lastname_Session13_Accuracy.png
```

---

## ⏰ Deadline
Submit **before the start of Session 14.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 13 — Assignment*
