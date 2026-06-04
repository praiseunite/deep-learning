# 📋 Session 01 — Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Deep Learning in My World"
### Due: Before Session 2 begins
### Estimated Time: 1.5 – 2 hours

---

> **Professor's Note:** *"This assignment is designed to connect what you learned in class to the world around you. A great engineer is not just someone who can code — they are someone who understands the CONTEXT of the technology they build. This assignment develops that critical thinking."*

---

## 🎯 Assignment Objectives

By completing this assignment, you will:
- Demonstrate your understanding of the AI → ML → DL hierarchy
- Research a real-world DL application in depth
- Analyze an ethical AI issue critically
- Set up your Python environment (ready for coding in Session 3)

---

## 📝 TASK 1 — Written Reflection (30 minutes)

### Instructions:
Write a **minimum 400-word** reflection (in your own words — do NOT copy from the internet) responding to the following prompt:

---

> **Prompt:**
> *"Deep Learning is often described as a 'black box' — it works, but we don't always know why. Choose ONE application of Deep Learning from the lecture (e.g., medical diagnosis, criminal risk scoring, loan approval, hiring, or self-driving cars) and discuss: (a) the benefit it brings to society, (b) the specific ethical risks it creates, and (c) what safeguards you would implement if you were the engineer building it."*

---

### Format Requirements:
- **Minimum:** 400 words
- **Structure:** Use clear paragraphs with the following headings:
  - `## Introduction` — Introduce the application you chose
  - `## The Benefits` — What good does this technology do?
  - `## The Ethical Risks` — What could go wrong?
  - `## My Safeguards` — What would YOU do to prevent harm?
  - `## Conclusion` — Final thoughts

### Grading Criteria:
| Criterion | Marks |
|-----------|-------|
| Demonstrates understanding of DL concepts | 20 |
| Identifies specific and realistic ethical risks | 20 |
| Proposes thoughtful, practical safeguards | 20 |
| Quality of writing and structure | 20 |
| Minimum word count met | 20 |
| **Total** | **100** |

---

## 💻 TASK 2 — Environment Setup (45 minutes)

### ⚠️ IMPORTANT: You MUST complete this before Session 3!
We will start writing Python code in Session 3. If your environment is not set up, you will fall behind.

Follow these steps EXACTLY:

---

### Step 1: Install Python 3.10+

1. Go to: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download the **latest Python 3.10+** version for Windows
3. During installation: ✅ **CHECK** "Add Python to PATH" (very important!)
4. Click "Install Now"

**Verify Installation:**
Open Command Prompt (search for "cmd" in Windows) and type:
```
python --version
```
You should see: `Python 3.10.x` or higher ✅

---

### Step 2: Install Pip (Python Package Manager)

Pip should already be installed with Python. Verify:
```
pip --version
```
You should see: `pip 23.x.x` or higher ✅

---

### Step 3: Install Jupyter Notebook

```
pip install notebook
```

**Test it works:**
```
jupyter notebook
```
A browser window should open showing a file explorer ✅
Press `Ctrl+C` in the terminal to stop it.

---

### Step 4: Install Required Deep Learning Libraries

Run these commands ONE BY ONE in your Command Prompt:

```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install keras
```

> ⚠️ **Note:** TensorFlow installation may take 5-10 minutes. This is normal.

**After all installations, verify TensorFlow works:**
```
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

You should see: `TensorFlow version: 2.x.x` ✅

---

### Step 5: Create Your Working Folder

Create this folder structure on your computer:
```
C:\DL_Course\
├── Session_01\
├── Session_02\
├── Session_03\
└── datasets\
```

**Windows Command:**
Open Command Prompt and run:
```
mkdir C:\DL_Course\Session_01
mkdir C:\DL_Course\Session_02
mkdir C:\DL_Course\Session_03
mkdir C:\DL_Course\datasets
```

---

### Step 6: Run the Verification Script

Download and run the `environment_check.py` file from the `Code_Snippets` folder.

```
python C:\DL_Course\Session_01\environment_check.py
```

If everything is installed correctly, you will see:
```
✅ Python: OK
✅ NumPy: OK
✅ Pandas: OK
✅ Matplotlib: OK
✅ Scikit-Learn: OK
✅ TensorFlow: OK
✅ Keras: OK
🎉 Your environment is ready for Deep Learning!
```

**Screenshot this output and submit it with your assignment.**

---

### Troubleshooting Common Issues:

| Problem | Solution |
|---------|----------|
| `python` not found | Reinstall Python and check "Add to PATH" |
| `pip` not found | Run: `python -m ensurepip --upgrade` |
| TensorFlow install fails | Run: `pip install tensorflow --upgrade` |
| Import error | Make sure you spelled the library name correctly |

---

## 🏆 BONUS TASK (Optional — For High Achievers)

### Bonus A: Research Report
Choose ONE of these revolutionary DL breakthroughs and write a **1-page summary** explaining:
- What problem it solved
- How Deep Learning was used
- Its impact on the world

**Topics to choose from:**
1. **AlexNet (2012)** — The model that started the DL revolution
2. **AlphaGo (2016)** — How DeepMind's AI defeated the world Go champion
3. **AlphaFold (2020)** — How DeepMind solved protein folding (a 50-year biology mystery)
4. **GPT-3 (2020)** — The model that changed natural language understanding
5. **DALL-E (2021)** — Text-to-image generation

**Search Strategy:**
- Google: "[topic name] explained"
- YouTube: "[topic name] documentary" or "[topic name] explained simply"
- Wikipedia for a quick overview, then dig deeper

---

### Bonus B: Find a Local AI Company
Research ONE Nigerian or African company that is using Deep Learning or AI in their products. Write:
- Company name and website
- What DL application they use
- What problem it solves for African users

**Starting Points:**
- Flutterwave (payments & fraud detection)
- Andela (tech talent matching)
- Zindi Africa (AI/ML competition platform)
- InstaDeep (AI for healthcare and logistics)
- Wadhwani AI (agriculture AI)

---

## 📦 Submission Requirements

Submit the following to your instructor:
1. ✅ **Task 1:** Written reflection (minimum 400 words) — as a `.docx` or `.pdf` file
2. ✅ **Task 2:** Screenshot of the environment verification output
3. ⭐ **Bonus A (if attempted):** 1-page research report
4. ⭐ **Bonus B (if attempted):** African AI company summary

**File Naming Convention:**
```
Firstname_Lastname_Session01_Assignment.docx
Firstname_Lastname_Session01_EnvScreenshot.png
```

---

## ⏰ Deadline
Submit **before the start of Session 2.**
Late submissions will receive a 10-mark deduction per day.

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 01 — Assignment*
