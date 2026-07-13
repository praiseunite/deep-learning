# Lab Guide 03: "The Text Machine" -- Build an AI Text Generator
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Sessions 20-21 (RNNs, BPTT, LSTM, Character-level Text Generation)
### Estimated Time: 2-3 hours (including deployment)
---

> *"By the end of this guide, you will have a working AI that writes text on its own. Give it a few starting words, and it will continue writing -- sometimes nonsense, sometimes surprisingly poetic. You will deploy it so anyone can type a seed phrase and watch the AI generate text."*

---

## What You Will Build

A **Character-Level Text Generator**. You feed it a collection of famous quotes, and it learns the patterns of English text character by character. Then you give it a starting phrase like "the meaning of" and it generates a continuation. The final app lives on Hugging Face where anyone can try it.

---

## What You Need Before Starting

- [ ] A **Kaggle** account with phone verification (for GPU)
- [ ] A **Hugging Face** account (free)
- [ ] No software to install

---

# PHASE 1: Get the Data

## Step 1.1: Create a New Notebook

1. Go to [kaggle.com](https://www.kaggle.com) -> Click **Create** -> **New Notebook**.
2. Turn on GPU: Right panel -> **Settings** -> **Accelerator** -> **GPU T4 x2**.

## Step 1.2: Prepare the Text Corpus

We will create a text corpus directly in the notebook using famous quotes and proverbs. This way there are no dataset dependencies.

### Cell 1: Create the Training Text

```python
# ============================================================
# CELL 1: CREATE THE TEXT CORPUS
# What this cell does: Builds a collection of text for the AI to learn from
# ============================================================

# We write a large block of text for the AI to study.
# The more text, the better the AI learns. But even this amount
# will teach it basic English patterns.
#
# Why quotes and proverbs? They are short, meaningful sentences
# with varied vocabulary. Perfect for learning language patterns.

text = """
the only way to do great work is to love what you do.
in the middle of difficulty lies opportunity.
life is what happens when you are busy making other plans.
the future belongs to those who believe in the beauty of their dreams.
it does not matter how slowly you go as long as you do not stop.
the best time to plant a tree was twenty years ago the second best time is now.
an investment in knowledge pays the best interest.
the only limit to our realization of tomorrow will be our doubts of today.
tell me and i forget teach me and i remember involve me and i learn.
do not go where the path may lead go instead where there is no path and leave a trail.
success is not final failure is not fatal it is the courage to continue that counts.
the mind is everything what you think you become.
strive not to be a success but rather to be of value.
the best revenge is massive success.
believe you can and you are halfway there.
everything you have ever wanted is on the other side of fear.
happiness is not something ready made it comes from your own actions.
if you want to lift yourself up lift up someone else.
whatever the mind of man can conceive and believe it can achieve.
the only impossible journey is the one you never begin.
education is the most powerful weapon which you can use to change the world.
the best way to predict the future is to create it.
it is during our darkest moments that we must focus to see the light.
life is ten percent what happens to us and ninety percent how we react to it.
when you reach the end of your rope tie a knot in it and hang on.
always remember that you are absolutely unique just like everyone else.
the purpose of our lives is to be happy.
you miss one hundred percent of the shots you do not take.
i have learned that people will forget what you said people will forget what you did but people will never forget how you made them feel.
whether you think you can or you think you cannot you are right.
the secret of getting ahead is getting started.
quality is not an act it is a habit.
the only person you are destined to become is the person you decide to be.
we become what we think about most of the time.
your time is limited so do not waste it living someone else life.
if you look at what you have in life you will always have more.
a person who never made a mistake never tried anything new.
it is not what you look at that matters it is what you see.
the greatest glory in living lies not in never falling but in rising every time we fall.
the way to get started is to quit talking and begin doing.
""" * 3  # Repeat 3 times to give the AI more data to learn from.

# Convert to lowercase for simplicity (fewer characters to learn).
text = text.lower().strip()

print(f"Total characters in text: {len(text):,}")
print(f"First 200 characters:")
print(text[:200])
```

---

# PHASE 2: Explore the Data

### Cell 2: Understand the Character Set

```python
# ============================================================
# CELL 2: ANALYZE THE CHARACTERS
# What this cell does: Finds all unique characters in the text
# ============================================================

# A character-level model learns one character at a time.
# It needs to know every possible character it might encounter.
# sorted() puts them in alphabetical order for consistency.
chars = sorted(set(text))
VOCAB_SIZE = len(chars)

print(f"Unique characters: {VOCAB_SIZE}")
print(f"Characters: {chars}")
print()
print("This includes letters, spaces, and punctuation.")
print("The AI will learn to predict the next character given a sequence of previous characters.")
```

### Cell 3: Create Character Mappings

```python
# ============================================================
# CELL 3: CREATE CHARACTER MAPPINGS
# What this cell does: Converts characters to numbers and back
# ============================================================

# Neural networks work with numbers, not letters.
# We need two dictionaries:
# 1. char_to_idx: 'a' -> 0, 'b' -> 1, etc.
# 2. idx_to_char: 0 -> 'a', 1 -> 'b', etc.

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Example:
print("Character to number mapping:")
for ch in chars[:10]:
    print(f"  '{ch}' -> {char_to_idx[ch]}")

print(f"\nExample: 'hello' -> {[char_to_idx[c] for c in 'hello']}")
print(f"Back:    {[char_to_idx[c] for c in 'hello']} -> {''.join(idx_to_char[i] for i in [char_to_idx[c] for c in 'hello'])}")
```

---

# PHASE 3: Prepare the Data

### Cell 4: Create Training Sequences

```python
# ============================================================
# CELL 4: CREATE INPUT/OUTPUT SEQUENCES
# What this cell does: Chops the text into overlapping windows
# ============================================================

import numpy as np

# SEQ_LENGTH is how many characters the AI sees before predicting the next one.
# Like reading a sentence: if you see "the cat sat on th", you predict "e".
SEQ_LENGTH = 60

# We create training pairs:
# Input:  "the cat sat on th" (60 characters)
# Output: "e"                 (the next character)

# Convert entire text to numbers.
text_as_numbers = np.array([char_to_idx[ch] for ch in text])

# Create overlapping sequences.
# Example with SEQ_LENGTH=5 and text "hello world":
#   Input: "hello" -> Output: " "
#   Input: "ello " -> Output: "w"
#   Input: "llo w" -> Output: "o"
#   etc.

X = []  # Will hold input sequences
y = []  # Will hold the target (next character)

for i in range(len(text_as_numbers) - SEQ_LENGTH):
    # Take SEQ_LENGTH characters as input.
    input_seq = text_as_numbers[i : i + SEQ_LENGTH]
    
    # The character right after is the target.
    target_char = text_as_numbers[i + SEQ_LENGTH]
    
    X.append(input_seq)
    y.append(target_char)

X = np.array(X)
y = np.array(y)

print(f"Total training sequences: {len(X):,}")
print(f"Input shape: {X.shape}")   # (num_sequences, 60)
print(f"Output shape: {y.shape}")  # (num_sequences,)
print(f"\nExample:")
print(f"  Input:  '{''.join(idx_to_char[i] for i in X[0])}'")
print(f"  Target: '{idx_to_char[y[0]]}'")
```

### Cell 5: Create TensorFlow Dataset

```python
# ============================================================
# CELL 5: CREATE TENSORFLOW DATASET
# What this cell does: Packages the data for efficient training
# ============================================================

import tensorflow as tf

BATCH_SIZE = 128

# tf.data.Dataset is TensorFlow's way of efficiently feeding data to the model.
# .from_tensor_slices() takes our NumPy arrays and makes them iterable.
# .shuffle() randomizes the order each epoch.
# .batch() groups sequences into batches of 128.
# .prefetch() loads the next batch while the current one trains.

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Check one batch.
for batch_x, batch_y in dataset.take(1):
    print(f"Batch input shape: {batch_x.shape}")   # (128, 60)
    print(f"Batch output shape: {batch_y.shape}")   # (128,)

print("Dataset ready!")
```

---

# PHASE 4: Build the Model

### Cell 6: Build the LSTM Model

```python
# ============================================================
# CELL 6: BUILD THE LSTM TEXT GENERATOR
# What this cell does: Creates a recurrent neural network
# ============================================================

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(name="Text_Generator")

# Layer 1: Embedding
# Converts each character number into a dense vector of 64 numbers.
# Why? Raw numbers (0, 1, 2...) don't capture relationships between characters.
# An embedding learns that 'a' and 'e' (both vowels) are "similar" in some
# mathematical sense, while 'a' and '.' are very different.
# input_dim = VOCAB_SIZE (how many unique characters exist)
# output_dim = 64 (each character becomes a vector of 64 numbers)
model.add(layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64))

# Layer 2: LSTM (Long Short-Term Memory)
# This is the "memory" layer. Unlike Dense layers that process each input
# independently, LSTM reads the sequence one character at a time and
# REMEMBERS what it read before.
#
# 128 = the size of the LSTM's memory (128 numbers it uses to remember).
# return_sequences=True = output the memory state at EVERY step,
#   not just the last one. This feeds into the next LSTM layer.
model.add(layers.LSTM(128, return_sequences=True))

# Layer 3: Another LSTM
# Two LSTM layers stacked = the model can learn more complex patterns.
# return_sequences=False = only output the FINAL memory state.
# We only need the final state because we predict one character.
model.add(layers.LSTM(128, return_sequences=False))

# Layer 4: Dropout
model.add(layers.Dropout(0.3))

# Layer 5: Dense hidden layer
model.add(layers.Dense(64, activation='relu'))

# Layer 6: Output layer
# VOCAB_SIZE neurons = one for each possible next character.
# Softmax = converts to probabilities (which character is most likely next).
model.add(layers.Dense(VOCAB_SIZE, activation='softmax'))

model.summary()
```

### Cell 7: Compile

```python
# ============================================================
# CELL 7: COMPILE THE MODEL
# What this cell does: Sets up the learning algorithm
# ============================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled! Ready to learn how to write.")
```

---

# PHASE 5: Train the Model

### Cell 8: Train

```python
# ============================================================
# CELL 8: TRAIN THE MODEL
# What this cell does: The model reads the text over and over to learn patterns
# ============================================================

# 30 epochs gives the model enough time to learn character patterns.
# Text generation models need more epochs than classifiers because
# predicting the next character is harder than classifying an image.
history = model.fit(
    dataset,
    epochs=30,
    verbose=1
)

print("\nTraining complete!")
```

### Cell 9: Plot Training

```python
# ============================================================
# CELL 9: PLOT TRAINING PROGRESS
# What this cell does: Shows how loss decreased over training
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], linewidth=2, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final loss: {history.history['loss'][-1]:.4f}")
print("Lower loss = the model is better at predicting the next character.")
```

---

# PHASE 6: Test the Model

### Cell 10: The Text Generation Function

```python
# ============================================================
# CELL 10: TEXT GENERATION FUNCTION
# What this cell does: Uses the trained model to write text
# ============================================================

def generate_text(seed_text, length=200, temperature=0.7):
    """
    Generate text starting from a seed phrase.
    
    Parameters:
    - seed_text: The starting phrase (e.g., "the best way to")
    - length: How many characters to generate
    - temperature: Controls creativity.
        - Low (0.2) = very predictable, repetitive, "safe" text
        - Medium (0.7) = balanced, readable text
        - High (1.5) = wild, creative, sometimes nonsensical
    """
    
    # Start with the seed text.
    result = seed_text.lower()
    
    # Generate one character at a time.
    for _ in range(length):
        # Take the last SEQ_LENGTH characters as input.
        # If the result is shorter than SEQ_LENGTH, pad with spaces.
        current_seq = result[-SEQ_LENGTH:]
        if len(current_seq) < SEQ_LENGTH:
            current_seq = ' ' * (SEQ_LENGTH - len(current_seq)) + current_seq
        
        # Convert characters to numbers.
        input_array = np.array([[char_to_idx.get(ch, 0) for ch in current_seq]])
        
        # Get the model's prediction (probabilities for each character).
        predictions = model.predict(input_array, verbose=0)[0]
        
        # Apply temperature scaling.
        # Temperature adjusts how "peaked" or "flat" the probability distribution is.
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / predictions.sum()
        
        # Randomly sample from the distribution.
        # np.random.choice picks a character index based on the probabilities.
        # This adds variety -- the model doesn't always pick the highest probability.
        next_char_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_char_idx]
        
        # Append the new character to our result.
        result += next_char
    
    return result

print("Text generation function ready!")
```

### Cell 11: Generate Sample Texts

```python
# ============================================================
# CELL 11: GENERATE SAMPLE TEXTS
# What this cell does: Shows the AI writing with different settings
# ============================================================

seeds = ["the best way to", "life is", "education is", "if you want to"]

print("=" * 60)
print("AI TEXT GENERATION RESULTS")
print("=" * 60)

for seed in seeds:
    print(f"\nSeed: '{seed}'")
    print("-" * 40)
    
    generated = generate_text(seed, length=150, temperature=0.7)
    print(generated)
    print()

print("=" * 60)
print("\nNote: The text might not always make perfect sense.")
print("This is a character-level model learning from a small corpus.")
print("With more data and training, the results improve dramatically!")
```

### Cell 12: Temperature Comparison

```python
# ============================================================
# CELL 12: COMPARE TEMPERATURES
# What this cell does: Shows how temperature affects creativity
# ============================================================

seed = "the secret of"

print("Same seed, different temperatures:\n")

for temp in [0.2, 0.5, 0.7, 1.0, 1.5]:
    print(f"Temperature {temp}:")
    print("-" * 40)
    print(generate_text(seed, length=100, temperature=temp))
    print()

print("Notice:")
print("  Low temperature (0.2) = repetitive but safe")
print("  High temperature (1.5) = creative but sometimes nonsensical")
```

---

# PHASE 7: Save the Model

### Cell 13: Save Everything

```python
# ============================================================
# CELL 13: SAVE THE MODEL AND METADATA
# What this cell does: Saves the model and character mappings
# ============================================================

import json

# Save the model.
model.save('text_generator.keras')

# Save the character mappings and config.
# The Hugging Face app needs these to convert between characters and numbers.
config = {
    'chars': chars,
    'char_to_idx': char_to_idx,
    'idx_to_char': {str(k): v for k, v in idx_to_char.items()},
    'seq_length': SEQ_LENGTH,
    'vocab_size': VOCAB_SIZE
}

with open('model_config.json', 'w') as f:
    json.dump(config, f)

print("Saved:")
print("  1. text_generator.keras (the trained model)")
print("  2. model_config.json (character mappings)")
print("\nDownload BOTH files from the Output panel.")
```

### Cell 14: Verify

```python
# ============================================================
# CELL 14: VERIFY SAVED MODEL
# What this cell does: Loads and tests the saved model
# ============================================================

loaded_model = keras.models.load_model('text_generator.keras')

with open('model_config.json', 'r') as f:
    loaded_config = json.load(f)

print(f"Loaded config: {len(loaded_config['chars'])} characters, seq_length={loaded_config['seq_length']}")

# Quick test.
test_input = np.array([[char_to_idx.get(ch, 0) for ch in "the best way to learn is to practice an"[-SEQ_LENGTH:]]])
test_pred = loaded_model.predict(test_input, verbose=0)
predicted_char = idx_to_char[np.argmax(test_pred)]
print(f"Test prediction: next character after '...practice an' -> '{predicted_char}'")
print("Saved model works!")
```

---

# PHASE 8: Deploy to Hugging Face

## Step 8.1: Download from Kaggle

Download both files from the Output panel:
- `text_generator.keras`
- `model_config.json`

## Step 8.2: Create Hugging Face Space

1. [huggingface.co](https://huggingface.co) -> Profile -> **New Space**.
2. Name: `ai-text-generator`
3. SDK: **Gradio**
4. Hardware: **CPU Basic** (free)
5. **Create Space**.

## Step 8.3: Upload Files

1. **Files** tab -> **Add file** -> **Upload files**.
2. Upload `text_generator.keras` AND `model_config.json`.
3. **Commit changes**.

## Step 8.4: Create `requirements.txt`

```
tensorflow
gradio
numpy
spaces
```

## Step 8.5: Create `app.py`

```python
# ============================================================
# app.py -- AI Text Generator Web App
# Type a seed phrase and the AI continues writing.
# ============================================================

import os
# CRITICAL: Force TensorFlow to CPU-only BEFORE importing it.
# Hugging Face's ZeroGPU injects CUDA libraries that conflict with TF.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gradio as gr
import numpy as np
import json
import spaces

# Import TensorFlow AFTER disabling CUDA.
import tensorflow as tf

# -----------------------------------------------
# STEP 1: Load model and config
# -----------------------------------------------
model = tf.keras.models.load_model('text_generator.keras')

# Dummy function to satisfy Hugging Face's ZeroGPU requirement.
# DO NOT put @spaces.GPU on the actual predict function.
@spaces.GPU
def dummy_gpu():
    pass

with open('model_config.json', 'r') as f:
    config = json.load(f)

chars = config['chars']
char_to_idx = config['char_to_idx']
idx_to_char = {int(k): v for k, v in config['idx_to_char'].items()}
SEQ_LENGTH = config['seq_length']

# -----------------------------------------------
# STEP 2: Text generation function
# -----------------------------------------------
def generate(seed_text, length=200, temperature=0.7):
    """Generate text from a seed phrase."""
    
    seed_text = seed_text.lower().strip()
    if not seed_text:
        seed_text = "the best way to"
    
    result = seed_text
    
    for _ in range(int(length)):
        current_seq = result[-SEQ_LENGTH:]
        if len(current_seq) < SEQ_LENGTH:
            current_seq = ' ' * (SEQ_LENGTH - len(current_seq)) + current_seq
        
        input_array = np.array([[char_to_idx.get(ch, 0) for ch in current_seq]])
        predictions = model.predict(input_array, verbose=0)[0]
        
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / predictions.sum()
        
        next_idx = np.random.choice(len(predictions), p=predictions)
        result += idx_to_char[next_idx]
    
    return result

# -----------------------------------------------
# STEP 3: Gradio interface
# -----------------------------------------------
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Seed Text", placeholder="Type a starting phrase...",
                   value="the best way to"),
        gr.Slider(minimum=50, maximum=500, value=200, step=50,
                  label="Length (characters to generate)"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                  label="Temperature (creativity)")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=8),
    title="AI Text Generator (LSTM)",
    description="Enter a seed phrase and the AI will continue writing. "
                "Adjust temperature: low (0.2) = safe/repetitive, high (1.5) = creative/wild. "
                "Trained on famous quotes using a character-level LSTM.",
)

demo.launch()
```

Commit the file. Wait 2-5 minutes for the build.

---

# PHASE 9: Test It Live

## Test 1: Try Different Seeds

Enter these phrases one at a time and see what the AI writes:
- "the meaning of life"
- "education is"
- "if you believe"
- "success comes from"

## Test 2: Play with Temperature

- Set temperature to **0.2** -- notice how the text is safe but repetitive.
- Set temperature to **1.5** -- notice how it's creative but sometimes makes no sense.
- **0.7** is usually the sweet spot.

## Test 3: Share the URL

Send the link to classmates. Let them type their own seed phrases!

---

# Troubleshooting

### Problem: Generated text is just repeated characters
**Solution:** Train for more epochs (try 50). The model hasn't learned enough patterns yet.

### Problem: "KeyError" when generating
**Solution:** The seed text contains a character not in the training data. Use only lowercase letters, spaces, and basic punctuation.

### Problem: Model file too large for Hugging Face
**Solution:** The LSTM model should be under 5MB. If it's larger, reduce the LSTM units from 128 to 64.

### Problem: App loads slowly
**Solution:** The first prediction takes a few seconds because TensorFlow initializes. After that, it's fast.

---

## What You Learned

| Concept | Where You Used It |
|---------|------------------|
| RNNs (Session 20) | The LSTM layers that process sequences |
| BPTT (Session 20) | Happens inside `model.fit()` -- backpropagation through time |
| LSTM (Session 20) | The memory cells that remember long-range patterns |
| Text Generation (Session 21) | The `generate_text()` function with temperature sampling |
| Embeddings | Converting characters to dense vectors |
| Temperature Sampling | Controlling creativity vs. predictability |

---
*Lab Guide 03 | Deep Learning Using Neural Networks | Aptech*
