# 📋 Session 10 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Deployment Architect"
### Due: Before Session 11 begins
### Estimated Time: 45 minutes

---

> **Professor's Note:** *"As a Deep Learning Architect, you must choose the right tool for the job. Not every model belongs on a massive cloud server, and not every model can fit on a smartphone. In this homework, you will analyze business requirements and recommend the correct TensorFlow deployment pipeline."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Understand the business logic behind choosing TF Serving vs TF Lite vs TFJS.
- Evaluate constraints such as privacy, latency, and hardware capabilities.

---

## 📝 THE SCENARIOS

Write a professional response (2-4 sentences) for each of the following business scenarios. 

**For each scenario, you MUST provide:**
1. Your specific deployment recommendation (**TensorFlow Serving**, **TensorFlow Lite**, or **TensorFlow.js**).
2. The primary reason *why* this is the only correct choice based on the business constraints.

### 🏢 Scenario A: "The Medical Startup"
**The Product:** An AI that looks at photos of skin moles to detect skin cancer.
**The Constraint:** Due to strict medical privacy laws (HIPAA), the photos of the patients' skin are legally NOT allowed to leave the patient's smartphone. The photos cannot be transmitted over the internet to a server.
**Your Recommendation & Justification:**
___________________________________________________________________________
___________________________________________________________________________

### 🏢 Scenario B: "The Global E-Commerce Giant"
**The Product:** A massive recommendation engine (like Amazon) that predicts what product a user will buy next based on their shopping history. 
**The Constraint:** The neural network is over 5 Gigabytes in size. It receives 10,000 prediction requests per second from users all over the world. It must be highly scalable.
**Your Recommendation & Justification:**
___________________________________________________________________________
___________________________________________________________________________

### 🏢 Scenario C: "The Browser Game"
**The Product:** A fun browser-based game where the user turns on their webcam and tries to match their facial expressions to emojis on the screen.
**The Constraint:** The developers have zero budget to pay for cloud servers. They need a solution where the AI runs entirely for free, utilizing the CPU/GPU of the user who is visiting the website.
**Your Recommendation & Justification:**
___________________________________________________________________________
___________________________________________________________________________

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ A document (`.docx` or `.pdf`) containing your deployment recommendations for Scenarios A, B, and C.

**File Naming Convention:**
```
Firstname_Lastname_Session10_Deployment_Architect.pdf
```

---

## ⏰ Deadline
Submit **before the start of Session 11.**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 10 — Assignment*
