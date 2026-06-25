# 🧪 Session 20 — In-Class Task
### Deep Learning Using Neural Networks | Aptech
### Task Title: "The Math of Memory"
### Duration: 45–55 minutes

---

> **Instructor Note:** In this lab, students will step away from Keras and manually execute the math behind an RNN's Hidden State. This removes the "black box" nature of Keras and proves exactly how memory is carried forward.

---

## 🎯 Learning Objectives

By the end of this task, students will be able to:
- [ ] Calculate a Hidden State $h_t$ using previous memory $h_{t-1}$ and current input $x_t$.
- [ ] Understand why $W_h$ controls the rate at which gradients vanish or explode.
- [ ] Explain the Chain Rule in the context of Backpropagation Through Time (BPTT).

---

## 📋 PART 1 — The Theory Check (15 minutes)

1. Write out the full equation for calculating the Hidden State $h_t$. 
   *Answer:* __________________________________________________________________

2. What is the purpose of the $\tanh$ activation function in the hidden state equation? What would happen to the memory numbers if we didn't use it?
   *Answer:* __________________________________________________________________

3. In BPTT, if the hidden state weight $W_h$ is `1.8`, what happens to the error signal as it travels backward through 50 time steps? Does it vanish or explode?
   *Answer:* __________________________________________________________________

---

## 💻 PART 2 — The Manual Forward Pass (30 minutes)

Open your Python environment. You have been provided a script named `Code_Snippets/manual_rnn_forward_pass.py`.

### Step 1: The Forward Pass
Run the script. Look at the terminal output for the **Manual Forward Pass Through Time**.
1. At Time Step 1, the previous memory ($h_0$) is exactly `0.0`. Why is it zero?
   *Answer:* __________________________________________________________________

2. Look at Time Step 2. The script calculates: `tanh( (0.9 * 0.3799) + (0.6 * 0.2) + 0.1 )`. 
   Where did the number `0.3799` come from?
   *Answer:* __________________________________________________________________

### Step 2: The Vanishing Gradient
Look at the bottom of the terminal output under **Why Gradients Vanish**.

3. We simulated an error signal of `1.0` traveling backward through 50 time steps where $W_h$ = 0.9. What was the exact final value of the error signal after 50 multiplications?
   *Answer:* __________________________________________________________________

4. **Experiment:** Open the python script. Change `W_h = 0.9` to `W_h = 1.1`. Run the script again. What happens to the final error signal now? (This is called the Exploding Gradient!)
   *Answer:* __________________________________________________________________

---

## ✅ Task Completion Criteria

| Part | Requirement | Points |
|------|-------------|--------|
| Part 1 | Answered the 3 math/theory questions accurately. | 40 |
| Part 2 | Successfully ran the manual pass and simulated the exploding gradient. | 60 |
| **Total** | | **100** |

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 20 — In-Class Task*
