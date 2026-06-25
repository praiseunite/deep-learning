# 🧪 Session 18 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "Engineering the Blocks"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will stop relying on pre-built models. They will use the Keras Functional API to manually construct an Inception Block and a Residual Block, cementing their understanding of the underlying math and tensor shapes.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Understand the mathematical parameter savings of stacked 3x3 kernels.
- [ ] Implement a 1x1 Bottleneck and Parallel concatenation (Inception).
- [ ] Implement a Skip Connection using tensor addition (ResNet).

---

## 📋 PART 1 — The VGG Math (15 minutes)

VGGNet proved that stacking small 3x3 kernels is better than using large kernels. Let's do the math to prove it.

1. **Calculate Parameters for a 7x7 Kernel:**
   If you have a single convolutional layer with a 7x7 kernel, how many weights (parameters) does that single filter have?
   *Calculation:* ________________________________
   *Answer:* ________________________________

2. **Calculate Parameters for Stacked 3x3 Kernels:**
   To get the same receptive field as a 7x7 kernel, you need to stack **three** 3x3 kernels on top of each other. How many total parameters do those three 3x3 kernels have?
   *Calculation:* ________________________________
   *Answer:* ________________________________

3. **The Conclusion:**
   How many parameters did you save by stacking the 3x3 kernels instead of using the 7x7 kernel?
   *Answer:* ________________________________

---

## 💻 PART 2 — Building the Blocks (30 minutes)

Open your Python environment. You have been provided a script named `Code_Snippets/building_blocks.py`.

### Step 1: Analyze the Inception Block
Run the script. Look at the terminal output for the **Inception Block Summary**.

1. Look closely at the code for `path2`. It uses a 1x1 filter *before* the 3x3 filter. Why? What mathematical operation does this 1x1 filter perform on the incoming tensor?
   *Answer:* __________________________________________________________________

2. In the `inception_block` function, the very last line uses `layers.concatenate()`. If `path1` has 64 filters, `path2` has 64 filters, `path3` has 32 filters, and `path4` has 32 filters, what is the total depth (number of channels) of the final concatenated output?
   *Answer:* __________________________________________________________________

### Step 2: Analyze the Residual Block
Look at the terminal output for the **Residual Block Summary**.

3. Look at the code for the Residual Block. We use `layers.add([x, shortcut])`. What mathematical equation does this represent, and how does it prevent the Vanishing Gradient problem during backpropagation?
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Correctly calculated the parameter savings (49 vs 27). | 40 |
| Part 2 | Successfully ran the block builder and answered the tensor/math questions. | 60 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 18 — In-Class Task*
