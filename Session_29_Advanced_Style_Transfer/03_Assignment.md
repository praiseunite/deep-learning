# Session 29 — Assignment
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Build Your Own Style Transfer App"
### Submission Deadline: Next Session

---

## Overview

In this assignment, you will take everything you have learned across Sessions 28 and 29 and build your own mini style transfer pipeline. You will apply it to a personal photo and experiment with the alpha and beta weights to understand how they control the final output.

---

## Task 1 — Style Your Own Photo (Kaggle) [40 marks]

Use the Arbitrary Style Transfer model from the in-class task, but this time use **your own images**.

### Steps:
1. Find a **content image** — a photo of your city, campus, food, or anything you like.
2. Find a **style image** — a painting, artwork, or texture you find interesting (use [WikiArt](https://www.wikiart.org/) or [Google Arts & Culture](https://artsandculture.google.com/) for inspiration).
3. Upload both images to Kaggle.
4. Run the style transfer using the TF Hub model.
5. Submit a screenshot of your final styled image alongside the original content and style images.

**Deliverable:** Three images in a `plt.subplots(1, 3)` grid: Content | Style | Result.

---

## Task 2 — The Alpha-Beta Experiment [30 marks]

Using the **Classic NST code from Session 28** (the 500-iteration version), run three experiments where you change ONLY the alpha and beta weights:

| Experiment | ALPHA (Content) | BETA (Style) | Expected Result |
|-----------|----------------|-------------|-----------------|
| A | 1e4 | 1e-2 | Strong content, weak style |
| B | 1e3 | 1e-2 | Balanced |
| C | 1e2 | 1e-2 | Weak content, strong style |

**Deliverable:** A figure showing all 3 experiment results side by side with titles showing the alpha/beta values.

### Thinking Questions (answer in a Markdown cell):
1. At which alpha/beta ratio did you prefer the output? Why?
2. Is there a setting where the output becomes completely unrecognizable as the content image?

---

## Task 3 — Short Written Reflection [30 marks]

In a Kaggle Markdown cell, write a short paragraph (minimum 100 words) answering:

> *"Imagine you work at a company that makes a photo-editing app using Arbitrary Style Transfer. A user uploads a photo of a famous person and applies a style to create an AI-edited image of them doing something they never did. What ethical responsibility does your company have? What technical safeguards could you add to prevent misuse?"*

**Hint:** Use the Spiderman Rule from Session 27, and think about watermarking, consent, and content filters.

---

## Submission Format

Submit your Kaggle Notebook link via your LMS (Learning Management System). Ensure the notebook has:
- [ ] All cells run with visible output
- [ ] Task 1: styled image comparison plot
- [ ] Task 2: three alpha-beta experiment plots
- [ ] Task 3: written reflection in a Markdown cell
- [ ] Your full name and student ID in the first cell

---

## Marking Scheme

| Task | Marks | Key Criteria |
|------|-------|-------------|
| Task 1: Style Your Own Photo | 40 | Images uploaded, model runs, output looks correct |
| Task 2: Alpha-Beta Experiment | 30 | All 3 experiments run, differences are visible and discussed |
| Task 3: Written Reflection | 30 | Thoughtful, coherent, >100 words, references at least 2 ethical issues |
| **Total** | **100** | |

---
*Session 29 | Deep Learning Using Neural Networks | Aptech*
