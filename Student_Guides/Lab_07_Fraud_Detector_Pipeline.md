# Lab Guide 07: "The Fraud Detector" -- Build an AI Fraud Detection System
### Course: Deep Learning Using Neural Networks | Aptech
### Covers: Session 31 (Combining Machines, Real-World Deployment, Business Value)
### Estimated Time: 2-3 hours (including deployment)
---

> *"By the end of this guide, you will have built an AI that detects fraudulent credit card transactions. Enter transaction details and the AI instantly tells you whether it's legitimate or suspicious. You will deploy it as a web app -- the same kind of system that real banks use to protect millions of customers."*

---

## What You Will Build

A **Credit Card Fraud Detector**. The user enters transaction details (amount, time features) and the AI flags it as "Legitimate" or "Fraudulent". This is a real-world use case -- Nigerian banks, fintech apps, and payment processors use exactly this kind of AI to protect their customers.

---

## What You Need Before Starting

- [ ] A **Kaggle** account with phone verification (for GPU)
- [ ] A **Hugging Face** account (free)
- [ ] No software to install

---

# PHASE 1: Get the Data

## Step 1.1: Find the Dataset

1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Search for **"Credit Card Fraud Detection"** in the search bar.
3. Click on the dataset by **mlg-ulb** (the most popular one with 284,807 transactions).
4. Click **"New Notebook"** to create a notebook with data attached.

## Step 1.2: Turn on GPU

Right panel -> **Settings** -> **Accelerator** -> **GPU T4 x2**.

## Step 1.3: Understand the Dataset

This dataset contains real anonymized credit card transactions from European cardholders in September 2013. It has:
- 284,807 total transactions
- 492 fraudulent (0.17%) -- extremely imbalanced!
- 28 anonymized features (V1-V28) created by a mathematical technique called PCA
- 2 readable features: `Time` (seconds since first transaction) and `Amount` (in euros)
- 1 label: `Class` (0 = Legitimate, 1 = Fraud)

---

# PHASE 2: Explore the Data

### Cell 1: Load and Inspect

```python
# ============================================================
# CELL 1: LOAD THE DATA
# What this cell does: Reads the CSV file and shows its structure
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file. Each row is one credit card transaction.
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

print(f"Total transactions: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nFirst 3 rows:")
df.head(3)
```

### Cell 2: Check the Class Balance

```python
# ============================================================
# CELL 2: CHECK CLASS DISTRIBUTION
# What this cell does: Shows how many legitimate vs fraudulent transactions
# ============================================================

fraud_count = df['Class'].value_counts()

print("Class distribution:")
print(f"  Legitimate (0): {fraud_count[0]:,} ({fraud_count[0]/len(df)*100:.2f}%)")
print(f"  Fraudulent (1): {fraud_count[1]:,} ({fraud_count[1]/len(df)*100:.3f}%)")

# Plot.
fig, ax = plt.subplots(figsize=(6, 4))
fraud_count.plot(kind='bar', color=['steelblue', 'crimson'], ax=ax)
ax.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
ax.set_ylabel('Count')
ax.set_title('Transaction Class Distribution')
for i, v in enumerate(fraud_count):
    ax.text(i, v + 2000, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

print("\nThe dataset is EXTREMELY imbalanced!")
print("Only 0.17% of transactions are fraud.")
print("This is realistic -- in real life, most transactions are legitimate.")
```

### Cell 3: Explore Features

```python
# ============================================================
# CELL 3: EXPLORE THE FEATURES
# What this cell does: Shows summary statistics and distributions
# ============================================================

# Compare average transaction amount for legit vs fraud.
print("Average Transaction Amount:")
print(f"  Legitimate: ${df[df['Class']==0]['Amount'].mean():.2f}")
print(f"  Fraudulent: ${df[df['Class']==1]['Amount'].mean():.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Amount distribution.
ax1.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Legitimate', color='steelblue')
ax1.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='crimson')
ax1.set_xlabel('Transaction Amount ($)')
ax1.set_ylabel('Count')
ax1.set_title('Transaction Amount Distribution')
ax1.set_xlim(0, 500)
ax1.legend()

# Time distribution.
ax2.hist(df[df['Class']==0]['Time'], bins=50, alpha=0.7, label='Legitimate', color='steelblue')
ax2.hist(df[df['Class']==1]['Time'], bins=50, alpha=0.7, label='Fraud', color='crimson')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Count')
ax2.set_title('Transaction Time Distribution')
ax2.legend()

plt.tight_layout()
plt.show()
```

---

# PHASE 3: Prepare the Data

### Cell 4: Scale the Features

```python
# ============================================================
# CELL 4: SCALE THE FEATURES
# What this cell does: Standardizes Amount and Time columns
# ============================================================

from sklearn.preprocessing import StandardScaler

# The V1-V28 columns are already scaled (they came from PCA).
# But 'Amount' and 'Time' are NOT scaled.
# StandardScaler transforms them so mean=0 and std=1.
# This is important because neural networks train better when all
# features are on the same scale.

scaler_amount = StandardScaler()
scaler_time = StandardScaler()

df['Amount_scaled'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_scaled'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original unscaled columns.
df = df.drop(['Amount', 'Time'], axis=1)

print("Scaled Amount and Time.")
print(f"Amount_scaled range: [{df['Amount_scaled'].min():.2f}, {df['Amount_scaled'].max():.2f}]")
print(f"Time_scaled range: [{df['Time_scaled'].min():.2f}, {df['Time_scaled'].max():.2f}]")
```

### Cell 5: Split into Train and Test

```python
# ============================================================
# CELL 5: SPLIT THE DATA
# What this cell does: Separates data into training and test sets
# ============================================================

from sklearn.model_selection import train_test_split

# Separate features (X) from labels (y).
X = df.drop('Class', axis=1).values  # All columns except 'Class'
y = df['Class'].values                # Just the 'Class' column

# Split: 80% training, 20% testing.
# stratify=y ensures both sets have the same fraud ratio.
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training: {len(X_train):,} transactions ({y_train.sum():,} fraud)")
print(f"Testing:  {len(X_test):,} transactions ({y_test.sum():,} fraud)")
print(f"Features per transaction: {X_train.shape[1]}")

# Save feature names for the deployment app.
feature_names = df.drop('Class', axis=1).columns.tolist()
print(f"\nFeature names: {feature_names}")
```

### Cell 6: Handle Class Imbalance

```python
# ============================================================
# CELL 6: HANDLE CLASS IMBALANCE WITH CLASS WEIGHTS
# What this cell does: Tells the model to pay MORE attention to fraud cases
# ============================================================

from sklearn.utils.class_weight import compute_class_weight

# Because fraud is only 0.17% of data, the model could achieve 99.83% accuracy
# by ALWAYS predicting "Legitimate". That's useless!
#
# Class weights tell the model: "When you get a fraud case wrong, it counts
# as 577x worse than getting a legitimate case wrong."
# This forces the model to take fraud seriously.

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"Class weights:")
print(f"  Legitimate (0): {class_weights[0]:.4f}")
print(f"  Fraud (1): {class_weights[1]:.4f}")
print(f"\nThe model will treat each fraud case as {class_weights[1]:.0f}x more important!")
```

---

# PHASE 4: Build the Model

### Cell 7: Build the Neural Network

```python
# ============================================================
# CELL 7: BUILD THE FRAUD DETECTION MODEL
# What this cell does: Creates a Dense neural network for binary classification
# ============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_DIM = X_train.shape[1]  # 30 features

model = keras.Sequential(name="Fraud_Detector")

# Layer 1: First hidden layer.
# 64 neurons. Each receives all 30 input features.
model.add(layers.Dense(64, activation='relu', input_shape=(INPUT_DIM,)))

# Dropout: turn off 40% of neurons randomly during training.
# Extra aggressive dropout because the data is imbalanced --
# we don't want the model to memorize the majority class.
model.add(layers.Dropout(0.4))

# Layer 2: Second hidden layer.
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.4))

# Layer 3: Third hidden layer.
model.add(layers.Dense(16, activation='relu'))

# Layer 4: Output layer.
# 1 neuron with sigmoid activation.
# sigmoid outputs a value between 0 and 1:
#   Close to 0 = Legitimate
#   Close to 1 = Fraud
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
```

### Cell 8: Compile

```python
# ============================================================
# CELL 8: COMPILE THE MODEL
# What this cell does: Sets up the optimizer and loss function
# ============================================================

model.compile(
    optimizer='adam',
    # 'binary_crossentropy' because there are only 2 classes (fraud / not fraud).
    loss='binary_crossentropy',
    # We track both accuracy and AUC (Area Under ROC Curve).
    # AUC is better than accuracy for imbalanced data.
    # AUC = 1.0 means perfect, AUC = 0.5 means random guessing.
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("Model compiled!")
```

---

# PHASE 5: Train the Model

### Cell 9: Train with Class Weights

```python
# ============================================================
# CELL 9: TRAIN THE MODEL
# What this cell does: Trains with extra emphasis on fraud detection
# ============================================================

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=2048,        # Large batch because we have 227K samples.
    validation_split=0.15,
    class_weight=class_weight_dict,  # This is the key -- makes fraud count more!
    verbose=1
)

print("\nTraining complete!")
```

### Cell 10: Plot Training

```python
# ============================================================
# CELL 10: PLOT TRAINING CURVES
# What this cell does: Shows accuracy, loss, and AUC over time
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history.history['auc'], label='Train')
axes[2].plot(history.history['val_auc'], label='Val')
axes[2].set_title('AUC (Area Under ROC Curve)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

# PHASE 6: Test the Model

### Cell 11: Evaluate

```python
# ============================================================
# CELL 11: EVALUATE ON TEST DATA
# What this cell does: Checks performance on unseen transactions
# ============================================================

test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

### Cell 12: Confusion Matrix

```python
# ============================================================
# CELL 12: CONFUSION MATRIX
# What this cell does: Shows exactly how many frauds the model caught
# ============================================================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get predictions.
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probability to 0 or 1.

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\nHow to read this:")
print(f"  True Negatives (correct 'Legit'):  {cm[0,0]:,}")
print(f"  False Positives (wrongly flagged):  {cm[0,1]:,}")
print(f"  False Negatives (missed fraud!):    {cm[1,0]:,}")
print(f"  True Positives (caught fraud!):     {cm[1,1]:,}")
print(f"\nFraud Detection Rate: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
print(f"False Alarm Rate: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.2f}%")

print("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
```

### Cell 13: Test on Specific Examples

```python
# ============================================================
# CELL 13: TEST ON INDIVIDUAL TRANSACTIONS
# What this cell does: Shows predictions for specific transactions
# ============================================================

print("Sample Predictions:")
print("-" * 60)

# Show 5 legitimate and 5 fraud predictions.
legit_idx = np.where(y_test == 0)[0][:5]
fraud_idx = np.where(y_test == 1)[0][:5]

for idx in list(legit_idx) + list(fraud_idx):
    prob = y_pred_proba[idx]
    pred = "FRAUD" if prob > 0.5 else "LEGIT"
    true = "FRAUD" if y_test[idx] == 1 else "LEGIT"
    match = "CORRECT" if pred == true else "WRONG"
    print(f"  Prob: {prob:.4f} | Predicted: {pred:>5s} | True: {true:>5s} | {match}")
```

---

# PHASE 7: Save the Model

### Cell 14: Save Model and Scalers

```python
# ============================================================
# CELL 14: SAVE EVERYTHING FOR DEPLOYMENT
# What this cell does: Saves the model and preprocessing info
# ============================================================

import json
import joblib

# Save the model.
model.save('fraud_detector.keras')

# Save the scalers (we need these to process new transactions).
joblib.dump(scaler_amount, 'scaler_amount.pkl')
joblib.dump(scaler_time, 'scaler_time.pkl')

# Save feature info.
config = {
    'feature_names': feature_names,
    'input_dim': INPUT_DIM,
    'threshold': 0.5
}

with open('model_config.json', 'w') as f:
    json.dump(config, f)

print("Saved:")
print("  1. fraud_detector.keras")
print("  2. scaler_amount.pkl")
print("  3. scaler_time.pkl")
print("  4. model_config.json")
print("\nDownload ALL 4 files from the Output panel.")
```

### Cell 15: Export for Static Deployment (Optional)

> If Gradio deployment fails due to TensorFlow/Python version conflicts on Hugging Face,
> you can deploy as a **Static** site using TensorFlow.js instead.

```python
# ============================================================
# CELL 15: EXPORT MODEL TO TENSORFLOW.JS FORMAT (OPTIONAL)
# What this cell does: Converts model for browser deployment
# ============================================================

!pip install tensorflowjs -q

import tensorflowjs as tfjs

# Convert the Keras model to TensorFlow.js format.
tfjs.converters.save_keras_model(model, 'tfjs_model')

# Save scaler parameters into the config so JavaScript can use them.
# (The .pkl files only work in Python -- JavaScript needs the raw numbers.)
config_static = {
    'feature_names': feature_names,
    'input_dim': INPUT_DIM,
    'threshold': 0.5,
    'scaler_amount_mean': float(scaler_amount.mean_[0]),
    'scaler_amount_scale': float(scaler_amount.scale_[0]),
    'scaler_time_mean': float(scaler_time.mean_[0]),
    'scaler_time_scale': float(scaler_time.scale_[0])
}

with open('model_config_static.json', 'w') as f:
    json.dump(config_static, f)

print("TensorFlow.js model saved to 'tfjs_model/'")
print("Static config saved to 'model_config_static.json'")
print("\nDownload 'tfjs_model/' folder AND 'model_config_static.json' for Static deployment.")
```

---

# PHASE 8: Deploy to Hugging Face

## Step 8.1: Download from Kaggle

Download all 4 files from the Output panel.

## Step 8.2: Create Hugging Face Space

Name: `fraud-detector` | SDK: **Gradio** | Hardware: **CPU Basic**

## Step 8.3: Upload Files

Upload all 4 files: `fraud_detector.keras`, `scaler_amount.pkl`, `scaler_time.pkl`, `model_config.json`.

## Step 8.4: Create `requirements.txt`

```
tensorflow
gradio
numpy
scikit-learn
joblib
spaces
```

## Step 8.5: Create `app.py`

```python
# ============================================================
# app.py -- Fraud Detection Web App
# Enter transaction details and the AI flags fraud.
# ============================================================

import os
# CRITICAL: Force TensorFlow to CPU-only BEFORE importing it.
# Hugging Face's ZeroGPU injects CUDA libraries that conflict with TF.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gradio as gr
import numpy as np
import json
import joblib
import spaces

# Import TensorFlow AFTER disabling CUDA.
import tensorflow as tf

# -----------------------------------------------
# STEP 1: Load model, scalers, and config
# -----------------------------------------------
model = tf.keras.models.load_model('fraud_detector.keras')

# Dummy function to satisfy Hugging Face's ZeroGPU requirement.
# DO NOT put @spaces.GPU on the actual predict function.
@spaces.GPU
def dummy_gpu():
    pass
scaler_amount = joblib.load('scaler_amount.pkl')
scaler_time = joblib.load('scaler_time.pkl')

with open('model_config.json', 'r') as f:
    config = json.load(f)

# -----------------------------------------------
# STEP 2: Prediction function
# -----------------------------------------------
def detect_fraud(amount, time_seconds, v1, v2, v3, v4, v5, v6, v7, v8,
                 v9, v10, v11, v12, v13, v14):
    """
    Predict whether a transaction is fraudulent.
    For simplicity, we use Amount, Time, and 14 of the 28 V-features.
    The remaining V-features default to 0.
    """
    
    # Scale amount and time.
    amount_scaled = scaler_amount.transform([[amount]])[0][0]
    time_scaled = scaler_time.transform([[time_seconds]])[0][0]
    
    # Build the full feature vector (30 features).
    # V1-V14 from user input, V15-V28 default to 0.
    v_features = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14]
    v_remaining = [0.0] * 14  # V15-V28 default to zero.
    
    features = np.array(v_features + v_remaining + [amount_scaled, time_scaled])
    features = features.reshape(1, -1).astype('float32')
    
    # Predict.
    probability = model.predict(features, verbose=0)[0][0]
    
    if probability > 0.5:
        verdict = f"FRAUDULENT (Confidence: {probability*100:.1f}%)"
    else:
        verdict = f"LEGITIMATE (Confidence: {(1-probability)*100:.1f}%)"
    
    return verdict, float(probability)

# -----------------------------------------------
# STEP 3: Gradio interface
# -----------------------------------------------
demo = gr.Interface(
    fn=detect_fraud,
    inputs=[
        gr.Number(label="Transaction Amount ($)", value=50.0),
        gr.Number(label="Time (seconds since first transaction)", value=40000),
        gr.Number(label="V1", value=0.0),
        gr.Number(label="V2", value=0.0),
        gr.Number(label="V3", value=0.0),
        gr.Number(label="V4", value=0.0),
        gr.Number(label="V5", value=0.0),
        gr.Number(label="V6", value=0.0),
        gr.Number(label="V7", value=0.0),
        gr.Number(label="V8", value=0.0),
        gr.Number(label="V9", value=0.0),
        gr.Number(label="V10", value=0.0),
        gr.Number(label="V11", value=0.0),
        gr.Number(label="V12", value=0.0),
        gr.Number(label="V13", value=0.0),
        gr.Number(label="V14", value=0.0),
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Fraud Probability (0 = Legit, 1 = Fraud)")
    ],
    title="Credit Card Fraud Detector",
    description="Enter transaction details and the AI will predict whether the transaction "
                "is fraudulent. Trained on 284,807 real anonymized credit card transactions. "
                "The V-features are anonymized components -- in a real banking system, these "
                "would be derived from the raw transaction data.",
)

demo.launch()
```

Commit and wait for the build.

---

# ALTERNATIVE: Static Deployment (If Gradio Fails)

> **When to use this:** If the Gradio deployment fails because of TensorFlow/Python 3.12
> incompatibilities, CUDA errors, or dependency conflicts, use this method instead.
> No Python needed -- runs entirely in the browser.

**Requirement:** You must have run Cell 15 in Phase 7 to export the TFJS model and `model_config_static.json`.

## Static Step 1: Create a Static Space

1. [huggingface.co](https://huggingface.co) -> Profile -> **New Space**.
2. Name: `fraud-detector-static`
3. SDK: **Static** (NOT Gradio)
4. **Create Space**.

## Static Step 2: Upload Files

1. Upload ALL files from `tfjs_model/` (`model.json`, `.bin` files).
2. Upload `model_config_static.json`.
3. Commit.

## Static Step 3: Create `index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detector</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0;
            min-height: 100vh; display: flex; justify-content: center; padding: 20px;
        }
        .container { max-width: 650px; width: 100%; margin-top: 30px; }
        h1 { font-size: 1.6rem; margin-bottom: 8px; color: #f8fafc; }
        .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 0.9rem; line-height: 1.5; }
        .status { text-align: center; padding: 10px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem; }
        .status.loading { background: #1e3a5f; color: #7dd3fc; }
        .status.ready { background: #14532d; color: #86efac; }
        label { display: block; font-weight: 600; margin: 10px 0 4px; color: #cbd5e1; font-size: 0.85rem; }
        input[type="number"] {
            width: 100%; padding: 10px; border: 1px solid #334155; border-radius: 6px;
            background: #1e293b; color: #f1f5f9; font-size: 0.95rem; outline: none;
        }
        input:focus { border-color: #6366f1; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }
        button {
            width: 100%; padding: 14px; border: none; border-radius: 8px;
            background: #6366f1; color: white; font-size: 1rem; font-weight: 600;
            cursor: pointer; margin: 20px 0 16px; transition: background 0.2s;
        }
        button:hover { background: #4f46e5; }
        button:disabled { background: #334155; cursor: not-allowed; }
        .verdict {
            text-align: center; padding: 20px; border-radius: 10px;
            font-size: 1.2rem; font-weight: bold;
        }
        .verdict.legit { background: #14532d; color: #86efac; }
        .verdict.fraud { background: #450a0a; color: #fca5a5; }
        .footer { text-align: center; color: #64748b; margin-top: 24px; font-size: 0.75rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Credit Card Fraud Detector</h1>
        <p class="subtitle">
            Enter transaction details and the AI flags fraud.
            Runs entirely in your browser using TensorFlow.js.
        </p>
        <div id="status" class="status loading">Loading AI model...</div>

        <label for="amount">Transaction Amount ($)</label>
        <input type="number" id="amount" value="50.0" step="0.01">

        <label for="time_sec">Time (seconds since first transaction)</label>
        <input type="number" id="time_sec" value="40000" step="1">

        <p style="margin-top:16px;color:#94a3b8;font-size:0.85rem">
            V-Features (anonymized transaction components). Default to 0 for testing:
        </p>
        <div class="grid" id="vInputs"></div>

        <button id="predictBtn" onclick="predict()" disabled>Analyze Transaction</button>
        <div id="result"></div>
        <p class="footer">Fraud Detector | Deep Learning | Aptech<br>Powered by TensorFlow.js</p>
    </div>

    <script>
    let model = null;
    let config = null;

    // Create V1-V14 input fields.
    const vGrid = document.getElementById('vInputs');
    for (let i = 1; i <= 14; i++) {
        vGrid.innerHTML += `<div><label for="v${i}">V${i}</label>
            <input type="number" id="v${i}" value="0" step="0.01"></div>`;
    }

    async function init() {
        const s = document.getElementById('status');
        try {
            const resp = await fetch('model_config_static.json');
            config = await resp.json();
            model = await tf.loadLayersModel('model.json');
            s.className = 'status ready';
            s.textContent = 'Model loaded! Enter transaction details.';
            document.getElementById('predictBtn').disabled = false;
        } catch(e) {
            s.textContent = 'Error: ' + e.message;
            s.style.background = '#450a0a'; s.style.color = '#fca5a5';
        }
    }

    async function predict() {
        if (!model || !config) return;

        const amount = parseFloat(document.getElementById('amount').value);
        const timeSec = parseFloat(document.getElementById('time_sec').value);

        // Scale amount and time using saved scaler parameters.
        const amountScaled = (amount - config.scaler_amount_mean) / config.scaler_amount_scale;
        const timeScaled = (timeSec - config.scaler_time_mean) / config.scaler_time_scale;

        // Gather V1-V14.
        const vFeatures = [];
        for (let i = 1; i <= 14; i++) {
            vFeatures.push(parseFloat(document.getElementById('v' + i).value) || 0);
        }

        // Build 30-feature vector: V1-V14, V15-V28 (zeros), amount_scaled, time_scaled.
        const features = [...vFeatures, ...new Array(14).fill(0), amountScaled, timeScaled];
        const inputTensor = tf.tensor2d([features], [1, 30]);

        const prediction = await model.predict(inputTensor).data();
        inputTensor.dispose();

        const prob = prediction[0];
        const resultDiv = document.getElementById('result');

        if (prob > 0.5) {
            resultDiv.className = 'verdict fraud';
            resultDiv.innerHTML = `FRAUDULENT<br><span style="font-size:0.9rem">Confidence: ${(prob * 100).toFixed(1)}%</span>`;
        } else {
            resultDiv.className = 'verdict legit';
            resultDiv.innerHTML = `LEGITIMATE<br><span style="font-size:0.9rem">Confidence: ${((1 - prob) * 100).toFixed(1)}%</span>`;
        }
    }

    init();
    </script>
</body>
</html>
```

Commit the file. The page loads instantly -- no build step needed.

---

# PHASE 9: Test It Live

## Test 1: Normal Transaction

Enter typical values: Amount = $25.00, Time = 50000, all V-features = 0. The model should predict "LEGITIMATE".

## Test 2: Suspicious Transaction

Try unusual patterns from the dataset. Go back to your Kaggle notebook and add:

```python
# ============================================================
# GET SAMPLE FRAUD TRANSACTIONS FOR TESTING
# ============================================================
fraud_samples = df[df['Class'] == 1].head(5)
print("Sample fraud transactions (copy these V-values to Hugging Face):")
for idx, row in fraud_samples.iterrows():
    print(f"\nFraud #{idx}:")
    for feat in feature_names[:14]:
        print(f"  {feat}: {row[feat]:.4f}")
    print(f"  Amount_scaled: {row['Amount_scaled']:.4f}")
    break
```

Copy those values into your Hugging Face app and verify it flags them as fraud.

## Test 3: Share

Send the URL to classmates and challenge them: "Can you find a combination of numbers that tricks the AI?"

---

# Troubleshooting

### Problem: Model always predicts "Legitimate"
**Solution:** The class weights might not be working. Make sure you passed `class_weight=class_weight_dict` to `model.fit()`. Also try lowering the threshold from 0.5 to 0.3.

### Problem: Too many false alarms
**Solution:** Increase the threshold from 0.5 to 0.7. This makes the model more conservative about flagging fraud.

### Problem: "ModuleNotFoundError: joblib"
**Solution:** Add `joblib` to your `requirements.txt` on Hugging Face.

### Problem: App shows wrong number of features
**Solution:** Make sure the total features in `app.py` equal 30 (14 V-features from input + 14 zeros + amount_scaled + time_scaled).

---

## What You Learned

| Concept | Where You Used It |
|---------|------------------|
| ANN Classification (Sessions 1-5) | The Dense network architecture |
| Class Imbalance | Class weights to handle 0.17% fraud ratio |
| Feature Scaling (Session 9) | StandardScaler for Amount and Time |
| Binary Classification | Sigmoid output, binary crossentropy loss |
| AUC Metric | Better than accuracy for imbalanced data |
| Confusion Matrix | Evaluating true/false positives and negatives |
| Real-World Pipeline (Session 31) | End-to-end from data to deployed app |
| Business Context | FinTech fraud detection scenario |

---

## The Business Value Argument

When presenting this to a non-technical stakeholder (like a bank manager), frame it this way:

**Before AI:** Human auditors review flagged transactions. They catch maybe 60% of fraud, and it takes hours.

**After AI:** The AI reviews every transaction in milliseconds. It catches 90%+ of fraud instantly, and the false alarm rate is under 1%.

**The math:** If the bank loses $10 million/year to fraud and the AI catches 30% more, that's $3 million saved. The AI costs $50,000/year to run. That's a 60x return on investment.

This is how you sell deep learning to a business.

---
*Lab Guide 07 | Deep Learning Using Neural Networks | Aptech*
