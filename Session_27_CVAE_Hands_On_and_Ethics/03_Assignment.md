# Session 27 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "The Responsible AI Report"
### Due: Before Session 28 begins
### Estimated Time: 60 minutes

---

> **Professor's Note:** *"This assignment is intentionally different from previous ones. It combines coding AND writing. As AI engineers, you must be able to build AND to reason about the impact of what you build."*

---

## Assignment Objectives

By completing this assignment, you will:
- Build an improved CVAE with a larger latent space.
- Write a structured ethical analysis of your generative model.
- Demonstrate understanding of both technical capability and ethical responsibility.

---

## Part A: The Code (30 minutes)

### Task: Improve the CVAE

Open a new Kaggle Notebook with GPU. Build a CVAE on MNIST with these specifications:

1. **LATENT_DIM = 10** (instead of 2 -- more capacity for variation)
2. **Deeper architecture:** Encoder with 3 hidden layers (512, 256, 128), Decoder mirrors it
3. **Train for 30 epochs**
4. **Generate a 10x10 grid** of conditionally generated digits (same as Session 26)

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_train_flat = X_train.reshape(-1, 784)

NUM_CLASSES = 10
LATENT_DIM = 10  # Increased from 2!

y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)

# YOUR TASK: Build encoder and decoder with 3 hidden layers each
# Encoder: img(784) + label(10) -> Dense(512) -> Dense(256) -> Dense(128) -> z_mean, z_log_var
# Decoder: z(10) + label(10) -> Dense(128) -> Dense(256) -> Dense(512) -> output(784)

# Build your CVAE below (adapt from Session 26 code)
# ...
```

### What to Submit (Code):
1. Screenshot of your complete model architecture (or copy-paste the code)
2. Screenshot of the 10x10 conditional generation grid
3. Answer: How does the output quality compare to Session 26 (where LATENT_DIM was 2)? Are the digits sharper?

---

## Part B: The Ethics Report (30 minutes)

Write a **1-page** (300-500 words) ethical analysis of your CVAE model. Structure it as follows:

### Section 1: What Does My Model Do? (50 words)
Describe in plain English what your CVAE can generate and how it works.

### Section 2: Potential Misuse (150 words)
Describe at least **3 specific ways** your model (or a more powerful version of it) could be misused. For each, explain:
- What the misuse looks like
- Who could be harmed
- How severe the harm could be

### Section 3: Safeguards (150 words)
For each misuse you identified, propose at least **one technical or policy safeguard** that could prevent or mitigate it. Use the **DARE framework** from the lecture:
- **D**isclose: How would you ensure transparency?
- **A**ssess: How would you evaluate risks before deployment?
- **R**estrict: What technical limits would you build in?
- **E**ducate: How would you inform users?

### Section 4: My Commitment (50 words)
Write a personal statement about how you will approach responsible AI development in your career.

---

## Grading Criteria

| Criteria | Points |
|----------|--------|
| **Code:** CVAE runs with LATENT_DIM=10 and deeper architecture | 25 |
| **Code:** 10x10 generation grid shows correct, recognizable digits | 15 |
| **Code:** Quality comparison with Session 26 answered | 10 |
| **Ethics:** 3 specific misuse scenarios identified | 15 |
| **Ethics:** Safeguards are practical and specific (not generic) | 20 |
| **Ethics:** Personal commitment statement is thoughtful | 15 |
| **Total** | **100** |

---

## Submission Requirements

1. Kaggle notebook (or screenshots)
2. 10x10 generation grid screenshot
3. Written ethics report (1 page, submitted as .docx or .pdf)

---
*Session 27 | Deep Learning Using Neural Networks | Aptech*
