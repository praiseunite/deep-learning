# 📋 Session 16 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Feature Map Investigator"
### Due: Before Session 17 begins
### Estimated Time: 30 minutes

---

> **Professor's Note:** *"To master CNNs, you must be able to perform the mathematical convolution operation in your head. Do not worry—it is just simple multiplication and addition. In this assignment, you will manually act as a Kernel sliding over an image."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Mathematically calculate a dot product for a Convolution operation.
- Demonstrate how a 3x3 kernel generates a single number on a feature map.
- Apply a Max Pooling operation manually.

---

## 📝 THE MANUAL CONVOLUTION

### Scenario 1: The Dot Product
Imagine you have a tiny 3x3 pixel area of an image. The pixel values (brightness) are:
```text
[ 1, 0, 1 ]
[ 0, 1, 0 ]
[ 1, 0, 1 ]
```

The CNN has learned a 3x3 **Kernel** that looks for "X" shapes. Its weights are:
```text
[ 1,-1, 1 ]
[-1, 1,-1 ]
[ 1,-1, 1 ]
```

**Your Task:**
Multiply each pixel by its corresponding kernel weight, and then add all 9 resulting numbers together. Show your work.

*Calculation:*
___________________________________________________________________________
___________________________________________________________________________
*Final Single Number Result:* ________

*(Note: Because the input image perfectly matches the kernel's "X" pattern, you should get a highly positive number! This is how a CNN knows it found a feature).*

### Scenario 2: Max Pooling
Imagine after several convolutions, you have a 4x4 Feature Map that looks like this:
```text
[ 2, 4,  1, 3 ]
[ 6, 8,  2, 0 ]
[ 1, 1,  9, 5 ]
[ 3, 2,  4, 7 ]
```

You apply a **2x2 Max Pooling Layer** with a stride of 2 (meaning it looks at four distinct 2x2 quadrants).

**Your Task:**
Write out the resulting 2x2 grid. (Find the maximum number in the top-left quadrant, top-right quadrant, bottom-left, and bottom-right).

*Resulting 2x2 Grid:*
```text
[ _, _ ]
[ _, _ ]
```

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ A document (`.docx` or `.pdf`) containing your mathematical calculation for Scenario 1 and your completed 2x2 grid for Scenario 2.

**File Naming Convention:**
```
Firstname_Lastname_Session16_CNN_Math.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 17.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 16 — Assignment*
