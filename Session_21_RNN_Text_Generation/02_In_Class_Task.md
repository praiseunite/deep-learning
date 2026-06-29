# 🧪 Session 21 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Number Cruncher"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will manually perform the "Words to Numbers" data preparation phase, and then run a basic text generation script to see Autoregression in action.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Construct a character vocabulary and map it to integers.
- [ ] Understand the dimensions of One-Hot Encoded data.
- [ ] Observe an Autoregressive loop generating text.

---

## 📋 PART 1 — Manual Data Representation (15 minutes)

You are an AI Engineer. Your client wants to train an RNN on exactly one sentence: `"DEEP LEARNING IS DEEP"`

1. **Build the Vocabulary:** List all the *unique* characters in that sentence (including the space character). Note: There are 10 unique characters total.
   *Answer:* __________________________________________________________________

2. **Integer Mapping:** Assign the integer `0` to the letter `D`, `1` to `E`, `2` to `P`, and `3` to ` ` (space). 
   Write out the integer sequence for the word `"DEEP "`.
   *Answer:* __________________________________________________________________

3. **One-Hot Encoding:** If there are 10 unique characters total in the vocabulary, how many numbers (zeros and ones) will it take to One-Hot Encode the single letter `D`?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Autoregressive Loop (30 minutes)

Open your Python environment. You have been provided a script named `Code_Snippets/text_generation_demo.py`.

### Step 1: Execute the Code
Run the script. This code trains a very tiny RNN to learn the spelling of the word `"APTECH"`.

### Step 2: Analyze the Output
Look closely at the terminal output.

1. When we give the network the input sequence `['P', 'T', 'E']`, what does the script say the "Target Next Char" is?
   *Answer:* __________________________________________________________________

2. Look at the section titled **Autoregressive Text Generation**. We fed the model the "Seed Sequence" of `"APT"`. The model then enters a `for` loop to predict the next 3 letters one by one.
   Write down the exact text the model finally generated.
   *Answer:* __________________________________________________________________

3. Open the `text_generation_demo.py` script. Look at the `for` loop under step 4. Find the line of code that takes the newly predicted character and "glues" it to the end of our current sequence so it can be fed back into the network. Write that line of code below:
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Correctly mapped the vocabulary and identified One-Hot dimensions. | 40 |
| Part 2 | Successfully ran the generation script and identified the autoregression loop. | 60 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 21 — In-Class Task*
