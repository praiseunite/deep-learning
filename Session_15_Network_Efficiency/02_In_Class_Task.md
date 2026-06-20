# 🧪 Session 15 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Shrink Ray"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this final lab, students will apply Post-Training Quantization. They will take a standard 32-bit TensorFlow model, shrink it using TF Lite, and measure exactly how many Megabytes of storage space they saved.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Define Pruning, Quantization, and Knowledge Distillation.
- [ ] Execute the TensorFlow Lite Converter.
- [ ] Apply `tf.lite.Optimize.DEFAULT` to quantize a model.
- [ ] Compare the file size of a standard model versus a quantized model.

---

## 🛠️ What You Need
- ✅ Python and TensorFlow installed.
- ✅ The Lecture Notes (`01_Lecture_Notes.md`).

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. You train a model, but you realize 30% of the weights in the dense layers are incredibly close to `0.0001`. What efficiency strategy should you use to optimize this model without losing accuracy?
   *Answer:* __________________________________________________________________

2. Why does converting a model's weights from 32-bit floats to 8-bit integers immediately reduce the model's file size by roughly 75% (a 4x reduction)?
   *Answer:* __________________________________________________________________

3. In Knowledge Distillation, why is it better for the Student model to learn from the Teacher's "Soft Targets" (e.g., 90% Dog, 10% Cat) rather than the raw "Hard Targets" (e.g., 100% Dog)? *(Hint: Does the 10% Cat tell the student something important about the shape of the animal?)*
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Quantization Experiment (30 minutes)

Create a file named `efficiency_demo.py` and copy the code from the `Code_Snippets` folder.

### Step 1: Analyze the Code
Look at the script. 
First, we build a heavy model with nearly 2 million parameters.
Then, we save it as a standard `.h5` file.
Finally, we use the `tf.lite.TFLiteConverter`. Notice the critical line of code:
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```
This single line tells TensorFlow to execute Post-Training Quantization, squishing all 32-bit floats down into 8-bit integers.

### Step 2: Run the Script
Execute the python script.

### Step 3: Record the Results
Look at the terminal output. It will calculate the file sizes of the two models on your hard drive.
- **Original Model Size:** ________ MB
- **Quantized (TF Lite) Size:** ________ MB

---

## 🔍 PART 3 — Analysis

1. Look at your results in Step 3. Roughly how many times smaller is the quantized model compared to the original? (Divide the original size by the quantized size).
   *Answer:* __________________________________________________________________

2. If you were deploying an AI model to a drone that had to process video at 60 Frames Per Second using a tiny battery, list the sequence of optimization strategies you would apply to ensure it runs as fast as possible.
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 theory questions accurately. | 40 |
| Part 2 | Successfully executed the script and recorded the file sizes. | 30 |
| Part 3 | Answered the 2 analysis questions. | 30 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 15 — In-Class Task*
