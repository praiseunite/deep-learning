# 🧪 Session 18 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Importing the Titans"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will learn that they do not have to build VGGNet or ResNet from scratch. They will use `keras.applications` to import the architectures directly from TensorFlow.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Define the primary innovation of VGG, Inception, and ResNet.
- [ ] Utilize `tf.keras.applications` to instantiate complex architectures.
- [ ] Analyze the parameter counts of different famous architectures.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. VGGNet proved that you don't need large kernels (like 7x7 or 11x11). What is the only size of Convolutional Kernel used in VGGNet?
   *Answer:* __________________________________________________________________

2. What is the fundamental concept behind an **Inception Module**?
   *Answer:* __________________________________________________________________

3. What specific problem does the **Skip Connection** in ResNet solve, and how does it allow networks to be 150+ layers deep?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Architecture Import (30 minutes)

Create a file named `importing_architectures.py` and copy the code from the `Code_Snippets` folder.

### Step 1: Analyze the Code
Look at how easy it is to summon a world-class AI model:
```python
vgg_model = keras.applications.VGG16(weights=None, input_shape=(224, 224, 3))
```
- `weights=None` means we are just importing the *blueprint* (the empty architecture), not a pre-trained brain. We will cover pre-trained brains in a later session.

### Step 2: Run the Script
Execute the script. It will download the blueprints for VGG16 and ResNet50, and print their summaries.
*(Note: Be prepared to scroll up in your terminal, the summaries are very long!)*

### Step 3: Record the Results
Look at the very bottom of each `model.summary()` to find the total parameter count.
- **Total Parameters in VGG16:** ___________
- **Total Parameters in ResNet50:** ___________

---

## 🔍 PART 3 — Analysis

1. Based on your results in Step 3, which model is significantly "heavier" (takes up more memory and disk space)? 
   *Answer:* __________________________________________________________________

2. ResNet50 has **50 layers**. VGG16 only has **16 layers**. Even though ResNet is much deeper, it actually has *fewer* parameters! Based on your lecture notes, what mechanism did ResNet use to allow data to skip past dense parameter blocks?
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions accurately. | 40 |
| Part 2 | Successfully executed the script and recorded the parameter counts. | 30 |
| Part 3 | Answered the 2 analytical questions. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 18 — In-Class Task*
