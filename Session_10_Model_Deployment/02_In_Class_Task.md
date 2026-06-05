# 🧪 Session 10 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Model Cryogenics Lab"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will experience the full lifecycle. They will train a simple model, save it to the hard drive in both H5 and SavedModel formats, intentionally delete the model from memory, load it back from the hard drive, and prove it still retains its knowledge.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Save a Keras model using `model.save()`.
- [ ] Differentiate between the `.h5` file and the `SavedModel` folder structure.
- [ ] Load a model from disk using `keras.models.load_model()`.
- [ ] Verify that a loaded model retains its trained weights.

---

## 🛠️ What You Need
- ✅ Python installed with TensorFlow/Keras.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (10 minutes)

Answer the following questions based on the lecture notes:

1. What is the process called that shrinks a model by converting 32-bit floats into 8-bit integers so it can fit on a smartphone?
   *Answer:* __________________________________________________________________

2. Which deployment target (TF Serving, TF Lite, or TF.js) would you use if you needed to run your AI completely offline on an iOS device?
   *Answer:* __________________________________________________________________

3. You deployed an AI to a website using TF.js. Who is paying for the server costs to do the actual mathematical predictions? You, or the user visiting the website? *(Hint: Where does the Javascript execute?)*
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Save & Load Experiment (35 minutes)

Create a file named `save_and_load.py`. Copy the starter code provided in the `Code_Snippets` folder.

### Step 1: Train and Save
Read through the code. We are training a very basic model on dummy data. 
Look at the bottom of the script. We are using two different save commands:
```python
model.save('my_legacy_model.h5')
model.save('my_modern_model')
```
Run the script.

### Step 2: Inspect your Hard Drive
Minimize your code editor and look at your actual computer's file explorer in the folder where your python script is saved.
1. Find `my_legacy_model.h5`. What is the file size? ________ KB
2. Open the `my_modern_model` folder. List the files/folders you see inside it:
   - __________________
   - __________________

### Step 3: Delete and Resurrect
Go back to your code editor. We are going to prove the model saved successfully by deleting it and bringing it back to life.

Add the following code to the very bottom of your script:

```python
import numpy as np

# 1. Delete the model from Python's memory completely
del model 

# 2. Try to use it (This will cause a fatal Error!)
try:
    print(model.predict(np.array([[0.5, 0.5]])))
except Exception as e:
    print(f"\nEXPECTED ERROR: {e}")

# 3. Resurrect it from the Hard Drive!
print("\nLoading model from hard drive...")
resurrected_model = keras.models.load_model('my_modern_model')

# 4. Prove it works
print("\nPrediction from resurrected model:")
print(resurrected_model.predict(np.array([[0.5, 0.5]])))
```

Run the script again. 
If successful, you will see an intentional error when you try to use the deleted model, followed by a successful prediction from the loaded model.

---

## 🔍 PART 3 — Analysis

1. In Step 3, why did we have to import `numpy` to test the prediction? What does `model.predict()` expect to receive?
   *Answer:* __________________________________________________________________

2. If you sent the `my_modern_model` folder to your friend, and they ran `keras.models.load_model()`, would they need to run `model.fit()` again to use it? Why or why not?
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions. | 30 |
| Part 2 | Successfully saved the model, inspected the files, and executed the resurrection code. | 50 |
| Part 3 | Answered the 2 analysis questions. | 20 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 10 — In-Class Task*
