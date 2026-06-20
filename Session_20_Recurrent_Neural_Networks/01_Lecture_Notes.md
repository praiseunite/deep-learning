# 📘 Session 20 — Recurrent Neural Networks (RNNs)
### Course: Deep Learning Using Neural Networks | Aptech
### Duration: 2 Hours (TL20)
---

> **Professor's Opening Note:**
> *"Feedforward networks and CNNs are great at looking at a single image and making a decision. But what if the data has a specific order, like the words in a sentence or the daily price of a stock? Today, we introduce 'Memory' into our networks using Recurrent Neural Networks (RNNs)."*

---

## 📚 Table of Contents
1. [The Problem with Standard Networks](#1-the-problem-with-standard-networks)
2. [The Architecture of an RNN](#2-the-architecture-of-an-rnn)
3. [Training: Backpropagation Through Time (BPTT)](#3-training-backpropagation-through-time-bptt)
4. [Limitations: The Vanishing Gradient](#4-limitations-the-vanishing-gradient)
5. [Recommended Videos](#5-recommended-videos)

---

## 1. The Problem with Standard Networks

If you feed the word "Apple" into a standard Dense network, it will process it and forget it instantly. If you then feed it the word "Tree", it processes "Tree" completely independently. It has no idea that "Apple" came first.

For tasks like language translation, speech recognition, or stock market prediction, **the sequence matters**. You need a network that remembers the past.

---

## 2. The Architecture of an RNN

A **Recurrent Neural Network (RNN)** solves this by introducing a "loop" into the network.

When an RNN processes the word "Apple", it creates an output, but it also creates a **Hidden State** (a memory). 
When it moves to the next word, "Tree", it doesn't just look at "Tree". It looks at "Tree" *plus* the memory of "Apple".

**Unrolling Through Time:**
To understand an RNN, we "unroll" it. Imagine looking at a timeline:
- **Time 1 (t=1):** Input is "I". The network creates Memory 1.
- **Time 2 (t=2):** Input is "Love". The network combines "Love" with Memory 1 to create Memory 2.
- **Time 3 (t=3):** Input is "Code". The network combines "Code" with Memory 2 to make its final prediction.

---

## 3. Training: Backpropagation Through Time (BPTT)

How does an RNN learn? In a standard network, we use Backpropagation to send the error signal backwards through the layers.

In an RNN, we use **Backpropagation Through Time (BPTT)**. 
Because the network is "unrolled" across a timeline, the error signal doesn't just travel backwards through the layers—it literally travels backwards in time, from step `t=3` back to `t=2` and `t=1`, adjusting the weights at every step.

---

## 4. Limitations: The Vanishing Gradient

While RNNs sound perfect for sequences, they have a massive flaw: **Short-Term Memory**.

Because BPTT forces the error signal to travel backwards through time, it must undergo continuous mathematical multiplication.
If you have a sentence that is 50 words long, the error signal has to multiply itself 50 times to get back to the first word.

If those gradient numbers are smaller than 1 (e.g., 0.5), multiplying them 50 times causes the number to shrink to essentially zero (0.5 * 0.5 * 0.5... = 0.0000001).

This is called the **Vanishing Gradient Problem**. The gradient vanishes before it reaches the start of the sentence, meaning the RNN completely forgets the beginning of a long paragraph!

---

## 5. 🎬 Recommended Videos

Because "unrolling through time" is highly visual, you *must* watch these animations to fully grasp the concept:

### 🥇 Video 1 — The Visual Breakdown
**"Recurrent Neural Networks (RNNs) | LSTMs, GRUs, BPTT by ByteQuest"**
- 📺 Channel: Search YouTube for "ByteQuest RNN".
- 🎯 Why Watch: It provides incredible visual animations of the "unrolling" process and shows exactly how BPTT works math-wise.

### 🥈 Video 2 — The Math of the Vanishing Gradient
**"Vanishing Gradients: Why Training RNNs is Hard"**
- 📺 Channel: Search YouTube for "Machine Learning with Phil Vanishing Gradient".
- 🎯 Why Watch: Explains exactly why multiplying numbers smaller than 1 across time causes the network to suffer from "amnesia".

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 20*
