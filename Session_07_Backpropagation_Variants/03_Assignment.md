# 📋 Session 07 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Gradient Descent Consultant"
### Due: Before Session 8 begins
### Estimated Time: 1 hour

---

> **Professor's Note:** *"You now know that Backpropagation is the math that calculates the error, and Gradient Descent is the engine that updates the weights. You also know that the Batch Size dictates the rhythm of that engine. In this homework, you will step away from the code and act as a consultant for three different tech startups, advising them on their training strategies."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Apply theoretical knowledge of Batch, Mini-Batch, and Stochastic Gradient Descent to real-world hardware scenarios.
- Identify the correct Backpropagation variant based on dataset size and computational constraints.

---

## 📝 THE SCENARIOS (60 minutes)

Write a short, professional response (3-5 sentences) to each of the following client scenarios. You must clearly state which Gradient Descent variant they should use (**Batch, Stochastic, or Mini-Batch**) and justify your answer using the concepts of memory (RAM), speed, and loss stability.

### 🏢 Client A: "The Medical Imaging Startup"
**The Scenario:** "We are training a neural network on 500,000 high-resolution MRI scans. Our investors bought us the best Nvidia GPUs on the market. However, each MRI image is massive (200MB each). Our Lead Engineer wants to use **Batch Gradient Descent** because he read it provides the most stable loss curve and guarantees finding the minimum error."
**Your Advice:**
1. State whether their Lead Engineer is correct or incorrect to suggest Batch Gradient Descent.
2. Explain *why* (Think about what happens when you try to load 500,000 200MB images at once).
3. Recommend the variant they *should* use.

### 🏢 Client B: "The IoT Smart Watch Company"
**The Scenario:** "We are deploying a tiny neural network directly onto a smartwatch to detect if an elderly person has fallen down. The watch learns from the user's daily movements continuously. The watch has almost no memory (RAM), so it can only look at the data stream one single second at a time. It cannot store data batches."
**Your Advice:**
1. Recommend the exact variant they must use based on their memory constraints.
2. Warn them about the major drawback of this variant (How will the loss curve look? Will it train smoothly?).

### 🏢 Client C: "The Social Media Giant"
**The Scenario:** "We are training a network on 2 Billion user text posts. We have massive clusters of GPUs. We tried Stochastic Gradient Descent (SGD) because we heard it starts learning instantly. However, our GPUs are running at only 5% utilization, and training is taking months. Why is our multi-million dollar hardware running so slowly with SGD?"
**Your Advice:**
1. Explain why SGD (batch size = 1) is wasting their expensive GPU hardware.
2. Recommend the industry-standard variant and explain why it will maximize their GPU utilization.

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ A document (`.docx` or `.pdf`) containing your consultant advice for Clients A, B, and C. 

**File Naming Convention:**
```
Firstname_Lastname_Session07_Consultant.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 8.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 07 — Assignment*
