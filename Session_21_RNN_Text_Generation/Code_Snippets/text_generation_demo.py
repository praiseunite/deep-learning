"""
============================================================
  FILE: text_generation_demo.py
  SESSION: 21 — RNN Text Generation
  PURPOSE: Demonstrate mapping characters to integers and
           running an autoregressive text generation loop.
============================================================
"""

import numpy as np

print("==================================================")
print("  Aptech AI Lab: Autoregressive Text Generation   ")
print("==================================================")

# ─── 1. THE DATA PREPARATION ─────────────────────────────────────────────────
text_data = "APTECH"
print(f"\nTraining Word: {text_data}")

# Build the Vocabulary (unique characters)
# For APTECH, it is: A, P, T, E, C, H
vocab = list(set(text_data))
vocab.sort() # Sorting gives us ['A', 'C', 'E', 'H', 'P', 'T']

print(f"Vocabulary: {vocab}")

# Create dictionaries to translate Chars to Ints, and Ints back to Chars
char_to_int = {char: i for i, char in enumerate(vocab)}
int_to_char = {i: char for i, char in enumerate(vocab)}

print(f"Integer Mapping: {char_to_int}")


# ─── 2. CREATING SEQUENCES (Sliding Window) ──────────────────────────────────
# If sequence length is 3, we look at 3 letters and predict the 4th.
seq_length = 3
print("\n--- Slicing Sequences ---")

for i in range(len(text_data) - seq_length):
    input_seq = text_data[i : i + seq_length]
    target_char = text_data[i + seq_length]
    
    print(f"Input Sequence: {list(input_seq)}  ---> Target Next Char: '{target_char}'")


# ─── 3. A FAKE TRAINED MODEL ─────────────────────────────────────────────────
# To keep this demo simple without requiring 15 minutes of GPU training, 
# we create a 'dummy' prediction function that perfectly memorized the word APTECH.
def dummy_predict_next_char(current_sequence):
    # This fake model looks at the last 3 characters and knows what comes next
    memory_bank = {
        "APT": "E",
        "PTE": "C",
        "TEC": "H"
    }
    
    # Get the last 3 characters of whatever sequence we are given
    last_three = current_sequence[-3:]
    
    # Predict the next character based on memory
    return memory_bank.get(last_three, "?")


# ─── 4. AUTOREGRESSIVE TEXT GENERATION LOOP ──────────────────────────────────
print("\n--- Autoregressive Text Generation ---")

# We start with a seed prompt
current_sequence = "APT"
print(f"Seed Sequence: {current_sequence}")

# We want the AI to write 3 new letters for us
letters_to_generate = 3

for i in range(letters_to_generate):
    # 1. Ask the model to predict the next character based on the current sequence
    next_char = dummy_predict_next_char(current_sequence)
    
    print(f"Loop {i+1}: AI read '{current_sequence[-3:]}' and predicted '{next_char}'")
    
    # 2. Glue the predicted character to the end of our sequence! (Autoregression)
    current_sequence = current_sequence + next_char

print(f"\nFinal Generated Text: {current_sequence}")
print("==================================================")
