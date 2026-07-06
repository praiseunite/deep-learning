# Session 31 — Final Capstone Assignment
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Build Your Own AI Pipeline"
### Submission Deadline: End of Course

---

## Overview

This is your **Final Capstone Assignment**. You have now learned all 5 Deep Learning Machines. This assignment asks you to combine **at least two of them** into one end-to-end AI pipeline that solves a real problem of your choosing.

This is your chance to show everything you have learned. Make it something you are proud of.

---

## The Challenge

Design and build a **mini AI pipeline** that chains at least 2 of the 5 machines together.

Below are 5 suggested pipeline ideas — pick ONE, or propose your own (get instructor approval first).

---

## 🛠️ Pipeline Option A — The AI Art Judge

**Machines Used:** Vision Machine + Art Machine

**What it does:**
1. The **Vision Machine** (VGG19) looks at a photo and identifies what is in it (e.g., "This is a mountain").
2. The **Art Machine** (Arbitrary Style Transfer) applies a style chosen based on the predicted category.
   - If it sees a mountain → apply a "Romantic landscape painting" style.
   - If it sees a dog → apply a "Cartoon illustration" style.
   - If it sees a city → apply a "Cubist painting" style.

**Deliverable:**
- A Kaggle notebook that takes any input image, classifies it, chooses a style, and produces the styled output.
- A `plt.subplots(1, 3)` plot: Original | Predicted Category (text) | Styled Output

---

## 🛠️ Pipeline Option B — The AI Novelist

**Machines Used:** Dream Machine + Text Machine

**What it does:**
1. The **Dream Machine** (CVAE) generates a handwritten digit or fashion item image.
2. The **Text Machine** (RNN) generates a short "product description" or "story" about the generated item.

For example: Generate a dress → RNN writes "A flowing summer dress with blue embroidery..."

**Deliverable:**
- A Kaggle notebook where for 5 different generated items, the RNN produces a short description.
- A display showing the generated image + its AI-written description side by side.

---

## 🛠️ Pipeline Option C — The AI Augmentor

**Machines Used:** Dream Machine + Classifier Machine

**What it does:**
1. The **Dream Machine** (CVAE) generates 500 extra synthetic images of a specific digit.
2. The **Classifier Machine** (ANN) is trained on the real dataset PLUS the 500 synthetic images.
3. Compare the accuracy of Classifier trained WITHOUT synthetic data vs WITH synthetic data.

**Deliverable:**
- Two trained classifiers: one baseline, one augmented.
- A comparison bar chart showing the test accuracy of both.
- A Markdown cell explaining why data augmentation from generative models helps.

---

## 🛠️ Pipeline Option D — The Art Style Classifier

**Machines Used:** Art Machine + Classifier Machine + Vision Machine

**What it does:**
1. The **Art Machine** generates 3 versions of the same photo: Van Gogh style, Hokusai style, and Kandinsky style.
2. The **Vision Machine** (VGG19 features) is used to extract feature maps from each styled version.
3. A small **Classifier Machine** (ANN on top of the features) is trained to predict which art style was applied.

**Deliverable:**
- A demonstration that the classifier can distinguish the 3 art styles from the VGG19 feature maps.
- A confusion matrix showing how well the classifier identifies each style.

---

## 🛠️ Pipeline Option E — Your Own Pipeline (Freestyle)

**Requirement:** Must chain at least 2 of the 5 machines. Must include a clear "problem statement" — what is your pipeline solving, and for who?

**Get instructor approval before starting.**

---

## Assignment Marking Scheme

| Criteria | Marks | Details |
|---------|-------|---------|
| **Pipeline Working End-to-End** | 40 | All cells run, input feeds correctly into each machine, final output is visible |
| **Code Quality & Comments** | 20 | Every major step has a 1-line comment explaining what it does |
| **Written Explanation** | 20 | A Markdown section explaining: What problem does this solve? Who would use it? What are its limitations? |
| **Creativity & Effort** | 20 | Original choice of content images/styles/themes, evidence of experimentation |
| **Total** | **100** | |

---

## Submission Requirements

Submit your **Kaggle Notebook URL** via the LMS.

- [ ] Notebook is **Public** (so instructor can access it)
- [ ] Your full name and student ID in the very first cell
- [ ] All cells are run with visible outputs
- [ ] The pipeline includes at least **2 of the 5 machines**
- [ ] A Markdown section titled "**My Pipeline Explanation**" answers: What does it do? Who is it for? What are the limits?

---

## 💡 Final Tips

- **Start simple.** Get Machine 1 working first, then add Machine 2.
- **Reuse your code.** All 5 machines are already coded in the In-Class Task. Copy and adapt.
- **Make it personal.** Use your own photos, your own text data, your own style choices.
- **Show your thinking.** The 20 marks for Written Explanation are easy marks — don't skip them!
- **Be bold.** The 20 creativity marks reward people who tried something unexpected.

> *"The best way to learn deep learning is to build something that you personally find exciting. Whatever you build today, you built it with skills that very few people in the world have. Be proud of that."*
>
> — Your Instructor

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 31 — Final Capstone*
