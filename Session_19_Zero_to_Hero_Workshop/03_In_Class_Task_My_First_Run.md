# 🧪 Session 19 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "My First Training Run"
### Duration: 30 minutes

---

> **Instructor Note:** The goal here is sheer execution. Students will take a complete, pre-written script, paste it into Kaggle, and run it. The success criteria is watching the epoch progress bar and seeing the final accuracy.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Successfully execute a multi-layer Neural Network script in Kaggle.
- [ ] Read the terminal output to identify "Loss" and "Accuracy" during training.

---

## 💻 The Execution (15 minutes)

1. Open your Kaggle Notebook (with the GPU turned on).
2. Create a brand new code cell.
3. Open the file `Code_Snippets/my_first_model.py` provided by your instructor.
4. Highlight all the code, copy it, and paste it into your Kaggle cell.
5. Hit the **Play** button.

## 🔍 Reading the Output

While the code is running, you will see output that looks like this:
`Epoch 1/5`
`1875/1875 [==============================] - 5s 2ms/step - loss: 0.2941 - accuracy: 0.9150`

What does this mean?
- **Epoch:** One full pass through the entire dataset of 60,000 images.
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
