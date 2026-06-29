# Session 25 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Pick Your Project"
### Due: Before Session 26 begins
### Estimated Time: 60 minutes

---

> **Professor's Note:** *"This is your first open-ended assignment. You will choose ONE project from three options and complete it independently. This mirrors real-world AI development where you must choose the right tool for the job."*

---

## Assignment Objectives

By completing this assignment, you will:
- Independently build a complete generative AI project from start to finish.
- Make design decisions about architecture, hyperparameters, and evaluation.
- Document your process and results professionally.

---

## Choose ONE of the Following Projects

---

### Option A: "The AI Poet" (RNN)

**Goal:** Train a character-level RNN to generate poetry-style text.

**Dataset:** Use any short text you like. Suggested options:
- A collection of song lyrics (copy-paste 10-15 song lyrics into a single string)
- Famous quotes (combine 30-40 quotes)
- A paragraph from a book or article, repeated 20 times

**Steps:**
1. Open a new Kaggle Notebook.
2. Paste your chosen text as the training data.
3. Build a character-level RNN (use the Session 25 Challenge 1 code as your starting point).
4. Train for at least 100 epochs.
5. Generate 3 different text samples using different seed strings.
6. Experiment: try changing `SEQ_LENGTH` to 5, 10, and 20. Document which produces the best output.

**Submit:**
- Your Kaggle notebook (or screenshots of all cells and outputs)
- Your 3 generated text samples
- A paragraph explaining which `SEQ_LENGTH` worked best and why

---

### Option B: "The Number Factory" (VAE)

**Goal:** Build a VAE that generates specific types of handwritten digits by navigating the latent space.

**Steps:**
1. Open a new Kaggle Notebook with GPU.
2. Build and train a VAE on MNIST (use Session 22 code as starting point).
3. Use `LATENT_DIM = 2` so you can visualize.
4. Create a **latent space map**: a 15x15 grid of images by decoding evenly spaced points across the latent space.

```python
# Latent Space Map
n = 15
figure = np.zeros((28 * n, 28 * n))

# Create a grid of latent points
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        decoded = decoder.predict(z_sample, verbose=0)
        digit = decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(12, 12))
plt.imshow(figure, cmap='gray')
plt.title("VAE Latent Space Map: Every Point Generates a Different Digit", fontsize=14)
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.tight_layout()
plt.show()
```

5. Pick 3 pairs of digits and create interpolation strips (like Challenge 2 in class).

**Submit:**
- Your Kaggle notebook
- Screenshot of the 15x15 latent space map
- Screenshots of your 3 interpolation strips
- A paragraph describing what you observe in the latent space map (which regions produce which digits)

---

### Option C: "The Fashion Designer" (GAN)

**Goal:** Train a GAN on Fashion-MNIST and create a "timeline" showing how generated images improve over training.

**Steps:**
1. Open a new Kaggle Notebook with GPU.
2. Build and train a GAN on Fashion-MNIST (use Session 23 homework code as starting point).
3. Train for 300 epochs, saving snapshots every 30 epochs.
4. Create a training progress plot showing generated images at each snapshot.
5. Plot the Discriminator and Generator loss curves (like Challenge 3 in class).
6. Experiment: try changing the Generator's hidden layer sizes. Document the effect.

**Submit:**
- Your Kaggle notebook
- Training progress plot (generated images over time)
- Loss curves plot
- A paragraph describing: at which epoch did the images start looking realistic? What happened to the loss values at that point?

---

## Grading Criteria

| Criteria | Points |
|----------|--------|
| Code runs without errors | 30 |
| Generated outputs are reasonable quality | 30 |
| Written analysis is thoughtful and accurate | 25 |
| Experimentation (tried variations) | 15 |
| **Total** | **100** |

---

## Submission Requirements

1. Complete Kaggle notebook (or screenshots of all cells and outputs).
2. All required plots/screenshots as specified in your chosen project.
3. Written analysis paragraphs.

---
*Session 25 | Deep Learning Using Neural Networks | Aptech*
