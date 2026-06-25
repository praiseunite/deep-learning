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
Because the network is "unrolled" across a timeline, the error signal doesn't just travel backwards through the layers—it literally travels backwards in time, from step `t=3` back to `t=2` and `t=1`.

### The Mathematics of the Hidden State
At any given time step $t$, the RNN calculates its new memory (Hidden State, $h_t$) using this exact equation:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

Where:
- $h_{t-1}$ is the memory from the previous time step.
- $x_t$ is the current input (e.g., the current word).
- $W_h$ and $W_x$ are the weights the network is trying to learn.
- $\tanh$ is the activation function that keeps the numbers squashed between -1 and 1 so they don't explode.

### The Chain Rule in Time
To update the weights $W_h$, we must calculate the derivative (gradient) of the Loss with respect to $W_h$. According to the Calculus Chain Rule, the gradient at time step $t=3$ depends on the gradient at $t=2$, which depends on $t=1$.
$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_h}$$

---

## 4. Limitations: The Vanishing Gradient

While RNNs sound perfect for sequences, they have a massive flaw: **Short-Term Memory**.

Because BPTT forces the error signal to travel backwards through time via the Chain Rule, it must undergo continuous mathematical multiplication. 
Look at the hidden state equation again. Every time we step backward in time, we multiply by the weight matrix $W_h$.

If you have a sentence that is 50 words long, the error signal has to multiply by $W_h$ 50 times to get back to the first word ($W_h^{50}$).

- **Vanishing:** If the values in $W_h$ are smaller than 1 (e.g., 0.5), multiplying them 50 times causes the gradient to shrink to essentially zero (0.5 * 0.5 * 0.5... = 0.0000001).
- **Exploding:** If the values are larger than 1 (e.g., 1.5), the gradient explodes to infinity.

When the gradient vanishes, the weights for the early time steps never update. This means the RNN completely forgets the beginning of a long paragraph!

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
