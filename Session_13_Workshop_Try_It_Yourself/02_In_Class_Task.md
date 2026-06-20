# 🧪 Session 13 — Try It Yourself Lab
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The CIFAR-10 Training Run"
### Duration: 90 minutes

---

> **Instructor Note:** This is the capstone coding lab for this section of the course. Students will open the `end_to_end_training.py` script. The script is broken. It is missing the Activation Functions, the Dropout Regularization, the Optimizer compilation, and the Training command. Students must fix it and watch the model train.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Apply Activation Functions to Keras layers.
- [ ] Insert Regularization (Dropout) to prevent overfitting.
- [ ] Compile a model with the Adam optimizer and Sparse Categorical Crossentropy loss.
- [ ] Execute `model.fit()` and interpret the real-time terminal output.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ The `end_to_end_training.py` script located in the `Code_Snippets` folder.

---

## 💻 PART 1 — Fixing the Code

Open `end_to_end_training.py`. Scroll down and find the `FIXME` comments.

### 1. Fix the Hidden Layers (Activation Functions)
In `STEP 2`, the hidden layers currently have no activation functions. 
Based on Session 6, apply the industry-standard activation function that prevents the vanishing gradient problem.
*Modify the code to add `activation='____'` to the two hidden Dense layers.*

### 2. Fix the Output Layer
This is a classification problem with exactly 10 categories (Airplanes, Cars, Cats, etc.).
The output layer must convert the network's raw numbers into 10 probabilities that sum to 100%.
*Modify the code to add the correct `activation='____'` to the final Dense layer.*

### 3. Add Regularization (Dropout)
In `STEP 3`, the network is currently prone to severe overfitting. 
Based on Session 8, add a `keras.layers.Dropout()` layer immediately after the first hidden layer. Set it to drop 30% of the neurons.

### 4. Compile the Model
In `STEP 4`, you must compile the model. 
- Set the `optimizer` to `'adam'` (the standard from Session 7).
- Set the `loss` to `'sparse_categorical_crossentropy'`.
- Set `metrics` to `['accuracy']`.

### 5. Start the Training Engine
In `STEP 5`, the `history` variable is empty.
Call the `model.fit()` function. 
- Pass it the `X_train` and `y_train` data.
- Set `epochs=10`.
- Set `validation_data=(X_test, y_test)`.

---

## 🚀 PART 2 — Execute and Observe

1. Save your script.
2. Open your terminal and run the script: `python end_to_end_training.py`.
   *(Note: The first time you run it, it will take a few seconds to download the 160MB CIFAR-10 dataset).*
3. **Watch the Training:** You will see a progress bar for each Epoch. Watch the `loss`, `accuracy`, `val_loss`, and `val_accuracy` values change.

---

## 🔍 PART 3 — Performance Analysis

Once the training has finished (after 10 epochs), answer these questions based on your terminal output:

1. What was your final Training Accuracy (`accuracy`)? ________%
2. What was your final Validation Accuracy (`val_accuracy`)? ________%
3. Did your model suffer from Overfitting? *(Is the training accuracy significantly higher than the validation accuracy?)*
   *Answer:* __________________________________________________________________
4. If you wanted to increase the validation accuracy, name two Hyperparameters you could try tuning in your next experiment:
   *Answer 1:* ________________________________________________________________
   *Answer 2:* ________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Correctly filled in all 5 `FIXME` sections in the code. | 50 |
| Part 2 | Successfully executed the script without syntax errors and completed 10 epochs. | 30 |
| Part 3 | Answered the 4 performance analysis questions based on their specific output. | 20 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 13 — Try It Yourself Lab*
