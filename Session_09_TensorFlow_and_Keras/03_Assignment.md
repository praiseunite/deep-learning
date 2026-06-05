# 📋 Session 09 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Multi-Input Architect"
### Due: Before Session 10 begins
### Estimated Time: 1 hour

---

> **Professor's Note:** *"The Sequential API is like building a straight road. The Functional API is like building a highway interchange. In this assignment, you will prove why the Functional API is necessary by designing an architecture that is impossible to build sequentially."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Design a complex, non-linear neural network architecture.
- Use the Keras Functional API to handle multiple inputs.
- Use the `keras.layers.concatenate` function.

---

## 📝 THE SCENARIO

You are building an AI for a real estate company to predict the **Price of a House**.
You have two completely different types of data for every house:
1. **Numerical Data:** (Square footage, number of bedrooms, age of the house). This is an array of 3 numbers.
2. **Image Data:** A 64x64 pixel grayscale photo of the front of the house.

You cannot pass a photo and a list of numbers into the exact same starting layer. You need a **Two-Headed Network**.

---

## 💻 THE CODING CHALLENGE

Create a python script named `multi_input_model.py`.

Use the Keras Functional API to build the following architecture:

**Branch A (The Numbers):**
1. Create an `Input` layer for the numerical data `shape=(3,)`.
2. Connect it to a `Dense` layer with 16 neurons and `relu` activation.

**Branch B (The Image):**
1. Create an `Input` layer for the image data `shape=(64, 64, 1)`.
2. Connect it to a `Flatten` layer.
3. Connect the Flatten layer to a `Dense` layer with 64 neurons and `relu` activation.

**The Merger:**
1. Use `keras.layers.concatenate([branch_a_output, branch_b_output])` to merge the two branches together.
2. Connect the merged data into a final `Dense` layer with `1` neuron (this is the final predicted price).

**The Finalization:**
Create the model using `keras.Model(inputs=[input_A, input_B], outputs=final_output)`.
Run `model.summary()`.

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ Your `multi_input_model.py` python script.
2. ✅ A screenshot of the terminal showing the `model.summary()` output. It must clearly show the `Concatenate` layer connecting to the two different branches.

**File Naming Convention:**
```
Firstname_Lastname_Session09_multi_input.py
Firstname_Lastname_Session09_Summary.png
```

---

## ⏰ Deadline
Submit **before the start of Session 10.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 09 — Assignment*
