# 📋 Session 11 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Transfer Learning Blueprint"
### Due: Before Session 12 begins
### Estimated Time: 45 minutes

---

> **Professor's Note:** *"You now know that you rarely have to build complex image models from scratch. In this homework, you will practice the exact syntax required to download a pre-trained model and lock its weights. You won't be training the model, you will just be architecting the blueprint."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Utilize `keras.applications` to download a pre-trained base.
- Freeze the weights of a base model.
- Attach a custom classification head using the Functional API.

---

## 💻 THE CODING CHALLENGE

You are building an AI to classify images of Dogs vs Cats. You decide to use Transfer Learning with **MobileNetV2** (a highly efficient model created by Google, pre-trained on ImageNet).

Create a python script named `transfer_learning_blueprint.py`.

Write the Python code to achieve the following:

**Step 1: Download the Base Model**
Use `keras.applications.MobileNetV2`.
- Set `input_shape=(128, 128, 3)`.
- Set `include_top=False` *(This is the most important part! It cuts off the 1000-class ImageNet head, leaving only the feature extractor).*
- Set `weights='imagenet'`.
Save this to a variable called `base_model`.

**Step 2: Freeze the Base**
We do not want to destroy Google's carefully trained weights.
Write the exact line of code that prevents `base_model` from being updated during training. *(Hint: It involves setting a boolean property on the base_model).*

**Step 3: Attach the Custom Head**
Use the Functional API to build the rest of the network:
1. `inputs = keras.Input(shape=(128, 128, 3))`
2. Pass the `inputs` into the `base_model`.
3. Pass the output of the base model into a `keras.layers.GlobalAveragePooling2D()` layer. *(This is just a smarter version of a Flatten layer).*
4. Pass that output into a final `keras.layers.Dense` layer with `1` neuron and a `sigmoid` activation (because we are predicting Dog vs Cat — a binary choice).

**Step 4: Finalize and Verify**
Create the final model using `keras.Model(inputs=inputs, outputs=outputs)`.
Call `model.summary()`.

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ Your `transfer_learning_blueprint.py` python script.
2. ✅ A screenshot of the terminal showing the `model.summary()` output. 

*Grading Check:* In your screenshot, look at the very bottom of the summary. 
It should show roughly **2,257,984 Total params**.
Crucially, it should show **2,257,984 Non-trainable params** (because you successfully froze the base model) and only a tiny number of **Trainable params** (your new Dense layer).

**File Naming Convention:**
```
Firstname_Lastname_Session11_TL_Blueprint.py
Firstname_Lastname_Session11_Summary.png
```

---

## ⏰ Deadline
Submit **before the start of Session 12.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 11 — Assignment*
