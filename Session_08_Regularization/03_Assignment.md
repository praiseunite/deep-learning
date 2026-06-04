# 📋 Session 08 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Early Stopping Rescuer"
### Due: Before Session 9 begins
### Estimated Time: 1 hour

---

> **Professor's Note:** *"Dropout is fantastic, but sometimes the best way to prevent a model from studying too much is to simply take the textbook away. Keras has a built-in feature called an 'EarlyStopping Callback'. Your homework is to learn how to use it by reading the Keras documentation and implementing it in code."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Learn how to implement Keras Callbacks.
- Use `EarlyStopping` to automatically halt a training loop.
- Recover the "best" weights from a training run.

---

## 💻 TASK 1 — Researching Keras Callbacks (20 minutes)

A "Callback" in Keras is a piece of code that can pause the training loop at the end of every epoch, look at the metrics, and execute an action (like saving the model, or stopping the training).

Search the internet or the Keras Documentation for `tf.keras.callbacks.EarlyStopping`.

Answer the following questions in a text document:
1. What does the `monitor` argument do? (e.g., `monitor='val_loss'`)
2. What does the `patience` argument do? If `patience=3`, what exactly does the network wait for before stopping?
3. What does `restore_best_weights=True` do? Why is it critically important if you want to use the model after training stops?

---

## 💻 TASK 2 — Implementing Early Stopping (40 minutes)

You will modify your code from the In-Class Task to include an Early Stopping callback.

1. Make a copy of `dropout_experiment.py` and rename it `early_stopping_homework.py`.
2. Find the baseline model (the one *without* Dropout).
3. We are going to intentionally over-train it by setting `epochs=50`.
   ```python
   history = model.fit(X_train_small, y_train_small, epochs=50, validation_data=(X_val, y_val))
   ```
4. **Your Mission:** Define an EarlyStopping callback and pass it into the `model.fit()` function. 

**Code Hints:**
```python
# 1. Define the callback BEFORE you call model.fit()
my_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# 2. Add the callbacks array to the fit function
model.fit(..., epochs=50, callbacks=[my_callback])
```

### The Test
Run your script. 
Even though you told the model to train for 50 epochs, if you implemented the callback correctly, it should stop automatically somewhere between Epoch 10 and Epoch 25, printing a message that it halted early.

Take a screenshot of your terminal showing the training loop halting before 50.

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1:** A document (`.docx` or `.pdf`) containing your answers about the EarlyStopping arguments.
2. ✅ **Task 2:** Your final `early_stopping_homework.py` script.
3. ✅ **Task 2:** A screenshot of your terminal showing the training halting early.

**File Naming Convention:**
```
Firstname_Lastname_Session08_EarlyStop_Answers.pdf
Firstname_Lastname_Session08_early_stopping_homework.py
Firstname_Lastname_Session08_Terminal.png
```

---

## ⏰ Deadline
Submit **before the start of Session 9.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 08 — Assignment*
