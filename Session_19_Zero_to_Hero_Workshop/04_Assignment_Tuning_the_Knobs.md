# 📋 Session 19 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Tuning the Knobs"
### Due: Before Session 20 begins
### Estimated Time: 20 minutes

---

> **Professor's Note:** *"Now that you know how to run a model, it is time to act like an AI Engineer. You are going to change the code, re-run the training, and see if you can make the AI smarter."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Modify hyperparameters (specifically `epochs`) in a working neural network.
- Prove that longer training times can lead to better accuracy.

---

## 💻 THE EXPERIMENT

In the In-Class Task, you ran the model for **5 Epochs** (it studied the dataset 5 times). 
Your model probably achieved an accuracy around **0.97** (97%).

Can we get closer to 99% just by letting it study longer?

### Step 1: Modify the Code
Go back to your Kaggle notebook. Scroll down to the very bottom of the code cell where it says:
```python
# Train the model
model.fit(X_train, y_train, epochs=5)
```

Change `epochs=5` to `epochs=15`.

### Step 2: Run the Training
Hit the **Play** button on the cell again.
You will notice the training takes three times as long because it is looping through the 60,000 images 15 times instead of 5.

### Step 3: Record the Results
Look at the output for `Epoch 15/15`.

1. What is your new final `accuracy`?
   *Answer:* _______________

2. Compare your new accuracy (15 epochs) to your old accuracy (5 epochs). Did training the model for a longer amount of time make it smarter?
   *Answer:* _______________

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ A document containing your two answers from Step 3.
2. ✅ **A Screenshot:** Take a screenshot of your Kaggle notebook showing the final `Epoch 15/15` output so your instructor knows you actually ran the code!

**File Naming Convention:**
```
Firstname_Lastname_Session19_Tuning.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 20.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 19 — Assignment*
