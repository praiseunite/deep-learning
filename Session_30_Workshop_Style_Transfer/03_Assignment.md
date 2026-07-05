# Session 30 — Assignment
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Create Your AI Art Portfolio"
### Submission Deadline: Next Session

---

## Overview

In this assignment, you will create a personal **AI Art Portfolio** — a single Kaggle Notebook that documents three pieces of AI-generated art, each using a different style transfer method. You will also reflect on the process and explain your technical choices in plain English.

Think of this as showing off your work! Your notebook should be clean, well-commented, and something you would be proud to share.

---

## Task 1 — Classic NST: Your Perfect Balance [30 marks]

Using the Classic NST code from the in-class task (Challenge 1), find the alpha/beta combination that you personally think produces the **best-looking output**.

**You must run at least 5 different alpha/beta combinations and pick your favourite.**

### Deliverable:
A `plt.subplots(2, 3)` figure showing:
- Row 1: Your 5 experiments (each labelled with its alpha/beta values)
- Row 2: Your final chosen "best" output, enlarged, with a Markdown cell below explaining WHY you chose it

---

## Task 2 — Arbitrary Style Transfer: Personal Style Gallery [40 marks]

Using the TF Hub model from the in-class task (Challenge 2), create a style gallery using:
- **1 content image:** A photo of yourself, your campus, your food, or anything personal
- **5 different style images:** Find 5 different paintings from [WikiArt.org](https://www.wikiart.org) (you must choose different styles — e.g., Impressionism, Cubism, Abstract, Baroque, Street Art)

### Deliverable:
A `plt.subplots(2, 5)` gallery showing:
- Row 1: The 5 style paintings with their name and art movement labelled
- Row 2: Your content photo styled as each of the 5 paintings

### Markdown cell required:
Answer these questions:
1. Which art movement produced the most recognisable transformation? Why?
2. Which art movement produced the least recognisable transformation? Why?
3. Which style would you choose if you were building a real photo-filter app for social media?

---

## Task 3 — Texture Synthesis: Design Your Own Fabric [30 marks]

Using the Texture Synthesis code from the in-class task (Challenge 3), run **two different texture experiments**:

**Experiment A:** Use a painting as the style source (e.g., Monet's Water Lilies)
**Experiment B:** Use a photograph of a real fabric/material as the style source (e.g., a photograph of wood grain, marble, or fabric)

For Experiment B, you may download any royalty-free texture photo from [Unsplash.com](https://unsplash.com) or [Pexels.com](https://www.pexels.com).

### Deliverable:
For each experiment, a 3-panel figure:
- Panel 1: Random Noise (start)
- Panel 2: Target Style Image
- Panel 3: Synthesized Texture (end)

### Markdown cell required:
Answer: "Which experiment produced a more 'usable' texture (e.g., for a game or app)? Why?"

---

## Marking Scheme

| Task | Marks | Key Criteria |
|------|-------|-------------|
| Task 1: Classic NST Tuning | 30 | ≥5 experiments run; best output chosen with written explanation |
| Task 2: Personal Gallery | 40 | Own photo used; 5 different styles from 5 different movements; gallery plot clean; all 3 questions answered |
| Task 3: Texture Synthesis | 30 | Both experiments A & B run; both 3-panel figures visible; written reflection present |
| **Total** | **100** | |

---

## Submission Requirements

Submit your **Kaggle Notebook URL** via the LMS. Ensure:
- [ ] All cells are run with visible outputs
- [ ] Notebook is set to **Public** so the instructor can view it
- [ ] Your full name and student ID are in the first Markdown cell
- [ ] Plots have proper titles and axis labels

---

## 💡 Tips for a Great Portfolio

- **Make it personal.** Use your own photos and styles that you genuinely find beautiful.
- **Comment your code.** A one-line comment above each major step shows you understand what it does.
- **Quality over quantity.** Five beautifully chosen styles beat ten random ones.
- **Be creative.** There is no wrong answer — this is art AND engineering!

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 30*
