# 🧪 Session 16 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Building the Feature Extractor"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will replace their standard `Dense` layers with `Conv2D` and `MaxPooling2D` layers. They will analyze the `model.summary()` to see how Max Pooling aggressively shrinks the spatial dimensions of an image.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Define the purpose of a Kernel/Filter.
- [ ] Implement `keras.layers.Conv2D`.
- [ ] Implement `keras.layers.MaxPooling2D`.
- [ ] Explain why CNNs maintain spatial relationships better than Dense networks.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. You feed an image into a standard neural network (using only `Dense` layers). What is the very first layer you must use, and why is this highly destructive to the image?
   *Answer:* __________________________________________________________________

2. What is the primary purpose of a **Kernel** (Filter) in a Convolutional layer?
   *Answer:* __________________________________________________________________

3. If you apply a 2x2 Max Pooling layer to an image that is currently 64x64 pixels in size, what will the new size of the image be? What happens to the other pixels?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The CNN Architecture Experiment (30 minutes)

Create a file named `cnn_basics.py` and copy the code from the `Code_Snippets` folder.

### Step 1: Analyze the Architecture
Look at the model defined in the code.
Notice the pattern:
`Conv2D` -> `MaxPooling2D` -> `Conv2D` -> `MaxPooling2D` -> `Flatten` -> `Dense`.
This is the classic CNN architecture!

Notice the arguments in the first layer:
```python
keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3))
```
- `filters=32` means we are asking the network to learn 32 *different* 3x3 magnifying glasses (e.g., one for horizontal edges, one for red dots, etc.).

### Step 2: Run the Script and Read the Summary
Run the script. It will not train the model; it will simply print the architectural summary.

Look closely at the **Output Shape** column in your terminal.
1. After the first `Conv2D`, the shape is `(None, 30, 30, 32)`.
2. After the first `MaxPooling2D`, the shape changes. What is it? _________

### Step 3: Record the Parameters
Look at the total parameter counts at the bottom of the summary.
- How many parameters are in the `Conv2D` layers compared to the final `Dense` layer? 
- **Total Params in the entire model:** _________

---

## 🔍 PART 3 — Analysis

1. In Step 2, you noticed the image size shrank from 30x30 to 15x15 after the Max Pooling layer. In your own words, why do we want the image to shrink as it goes deeper into the network? *(Hint: Think about computational memory and focusing on important features).*
   *Answer:* __________________________________________________________________

2. Look at the `Flatten` layer in the `model.summary()`. It converts the 3D data into a 1D array of `1024` numbers. Why is it acceptable to finally "destroy" the 2D spatial relationship at this specific point in the network, but it was a bad idea to do it at the very beginning?
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions accurately. | 40 |
| Part 2 | Successfully executed the script and recorded the summary shapes. | 30 |
| Part 3 | Answered the 2 analytical questions regarding spatial relationships. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 16 — In-Class Task*
