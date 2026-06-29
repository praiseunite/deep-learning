# 📋 Session 21 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Preparing Shakespeare"
### Due: Before Session 22 begins
### Estimated Time: 30 minutes

---

> **Professor's Note:** *"The hardest part of Natural Language Processing (NLP) is not building the neural network—it is formatting the data. If your data representation is wrong, your AI will speak garbage."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Demonstrate mastery of sequence slicing for text generation training.

---

## 📝 THE SCENARIO

You have downloaded the complete works of William Shakespeare. You want to train an RNN to write a new play.

Your training phrase is: `"TO BE OR NOT TO BE"`

### Task 1: Sequence Slicing
To train an RNN, you must slice your text into "Input Sequences" and "Target Characters". 
Assume your **Input Sequence Length is 4 characters**.

Write out the first three training examples from the phrase `"TO BE OR NOT TO BE"`. (Remember that spaces count as characters!)

- **Example 1:**
  - Input (4 chars): `[_, _, _, _]`
  - Target (1 char): `[_]`

- **Example 2:**
  - Input (4 chars): `[_, _, _, _]`
  - Target (1 char): `[_]`

- **Example 3:**
  - Input (4 chars): `[_, _, _, _]`
  - Target (1 char): `[_]`

### Task 2: The Vocabulary Size Issue
Shakespeare used uppercase letters, lowercase letters, commas, periods, exclamation marks, and question marks.
In total, there are about **65 unique characters** in his works.

If you decide to One-Hot Encode the dataset, what is the exact shape (length) of the array used to represent a *single* comma? 
Explain why.

**Your Answer:**
___________________________________________________________________________
___________________________________________________________________________

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ A document (`.docx` or `.pdf`) containing your sequence slices and your answer to the vocabulary issue.

**File Naming Convention:**
```
Firstname_Lastname_Session21_Shakespeare.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 22.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 21 — Assignment*
