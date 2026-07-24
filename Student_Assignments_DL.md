# 🚀 Advanced Deep Learning Capstone Assignments
## Monetizable, Production-Grade AI Solutions & Defense Framework

> **Course:** Deep Learning & Applied AI Engineering  
> **Target Students:** JP, Mfon, Elisha  
> **Objective:** Design, train, evaluate, deploy, and monetize production-ready Deep Learning applications using core curriculum concepts (ANNs, CNNs, Transfer Learning, RNNs/LSTMs, Generative Models, Neural Style Transfer, Hyperparameter Tuning, and Deployment).

---

## 🏗️ General Technical & Architectural Requirements

Every student project must satisfy the following **5-Layer Engineering Standard**:

```
[Layer 1: Data Pipeline] ➔ [Layer 2: Model Architecture] ➔ [Layer 3: Optimization & Evaluation] ➔ [Layer 4: Web Deployment] ➔ [Layer 5: Business Engine]
```

1. **Data Engineering & Augmentation:** Ingestion, cleaning, label encoding/normalization, dynamic image/sequence augmentation pipelines.
2. **Deep Learning Model Architecture:** Proper layer selection, regularizers (Dropout, BatchNormalization, L2 weight decay), and activation function selection.
3. **Hyperparameter Tuning & Evaluation:** Systematic search (KerasTuner / GridSearch), loss curve analysis, overfitting mitigation, and comprehensive metrics (Precision, Recall, F1-Score, Confusion Matrix, BLEU/Perplexity, or Loss terms).
4. **Interactive Web App & Cloud Deployment:** Web-based GUI (Streamlit / Flask / FastAPI + HTML5/CSS) deployed live online (Hugging Face Spaces, Render, Railway, or Docker).
5. **Business Plan & Monetization (`BUSINESS_PLAN.md`):** Problem-solution fit, Unique Value Proposition (UVP), pricing tier, revenue projections, and go-to-market strategy.

---

# 📋 Student Project Assignments

---

## 🌾 Assignment 1: Assigned to **JP**
### **AgriVision AI — Plant Disease & Crop Health Diagnostic SaaS**

* **Domain:** Agritech / Smart Farming
* **Primary Objective:** Build a real-time computer vision system that diagnoses crop diseases from leaf imagery, visualizes spatial model focus using Grad-CAM, provides automated treatment recommendations, and operates as a monetizable farm advisory SaaS.

---

### 🧠 Model Architecture & Deep Learning Specifications
* **Core Model:** **Transfer Learning** with **EfficientNetB3** or **ResNet50V2** (pretrained on ImageNet, fine-tuned on multi-class crop datasets like *PlantVillage* enriched with local Nigerian crops: cassava, maize, tomato, yam).
* **Techniques & Concepts Implemented:**
  * **Data Augmentation:** Random Flip, Rotation, Zoom, Contrast/Brightness Jitter to prevent background shortcut learning.
  * **Fine-Tuning Strategy:** Phase 1 (Freeze backbone, train top dense classifier head). Phase 2 (Unfreeze top 30-50 convolutional layers with a reduced learning rate `1e-5`).
  * **Explainable AI (XAI):** **Grad-CAM (Gradient-weighted Class Activation Mapping)** output layer to generate visual heatmaps showing *where* the CNN looks on the leaf to make its decision.
  * **Optimization:** Adam optimizer, Cosine Annealing learning rate schedule, EarlyStopping, ReduceLROnPlateau.

---

### 💻 Web App & Live Deployment
* **Frontend/Backend:** Flask or Streamlit web application.
* **Features:**
  * Image upload / Mobile camera upload.
  * Instant disease prediction with confidence score (e.g., *Cassava Mosaic Disease — 96.4% confidence*).
  * **Interactive Grad-CAM Heatmap overlay** showing infected leaf regions.
  * Downloadable **Treatment Plan PDF Report** generated dynamically.
* **Live Deployment:** Deployed on **Hugging Face Spaces (Streamlit)** or **Render (Flask/FastAPI)**.

---

### 💰 Monetization & Business Model (`BUSINESS_PLAN.md`)
* **Target Audience:** Smallholder farmers, commercial farm managers, agricultural extension agents, agro-chemical distributors.
* **Revenue Streams:**
  1. **Freemium Tier:** 3 free leaf diagnostics per week.
  2. **Pro Farmer Plan (₦2,500 / month):** Unlimited scans, downloadable PDF reports, offline-capable PWA, automated weekly crop health alerts.
  3. **B2B Agro-Dealer Marketplace (₦15,000 / month):** Local agro-dealers pay to be listed as verified treatment suppliers when a specific disease is diagnosed in their geographic area.

---

### 🎯 AI-Proof Defense Questions (Required in Student Submission)
> *Note: These questions require empirical proof from the student's training logs, visual evidence, and deep architectural understanding. They CANNOT be answered by generic AI queries.*

1. **Gradient & Heatmap Analysis:**  
   *"Upload a photo of a diseased leaf resting on noisy background soil. Show your model's Grad-CAM heatmap. Did the model focus on the leaf lesions or the background soil? If it focused on the soil, what exact data augmentation change did you make to break this shortcut learning?"*
2. **Fine-Tuning & Loss Curve Evidence:**  
   *"Show your exact Validation Loss vs. Epoch graph during Phase 1 (frozen backbone) vs Phase 2 (unfrozen backbone). At what exact epoch did overfitting begin in Phase 2, and what learning rate threshold resolved it?"*
3. **Out-of-Distribution Handling:**  
   *"If a user uploads an image of a non-crop object (e.g., a shoe or a car), raw Softmax will still output high confidence for one of your plant classes. How did you implement an Out-Of-Distribution (OOD) threshold or background class to reject non-plant inputs?"*

---

---

## 🩺 Assignment 2: Assigned to **Mfon**
### **SkinCheck AI — Dermatological Lesion Triage & Risk Assessment API**

* **Domain:** MedTech / Healthcare AI
* **Primary Objective:** Build a medical-grade dermatological lesion classifier that detects high-risk skin anomalies (e.g., Melanoma, Basal Cell Carcinoma, Nevus) across diverse skin tones, addresses severe class imbalance, and provides an API service for clinics and patients.

---

### 🧠 Model Architecture & Deep Learning Specifications
* **Core Model:** **MobileNetV3-Large** or **ResNet34** vision backbone fine-tuned on the **ISIC (International Skin Imaging Collaboration)** dataset + synthetic skin-tone balanced samples.
* **Advanced Generative Component:** **Conditional VAE (CVAE)** or **DCGAN** trained to generate synthetic minority-class lesion samples across varying skin pigmentation levels to reduce algorithmic bias.
* **Techniques & Concepts Implemented:**
  * **Class Imbalance Loss:** **Focal Loss** (\(\gamma = 2.0\)) or Weighted Categorical Cross-Entropy to penalize misclassification of rare, highly fatal lesions (Melanoma).
  * **Regularization & Optimization:** Label Smoothing, Dropout (\(0.4\)), CutMix / MixUp data augmentation.
  * **Quantization & Edge Prep:** Convert trained model to **TensorFlow Lite (TFLite)** or **ONNX Runtime** format for sub-100ms inference time.

---

### 💻 Web App & Live Deployment
* **Frontend/Backend:** FastAPI REST backend + Streamlit / React web UI.
* **Features:**
  * Image upload with automatic image quality verification (sharpness / lighting check).
  * Risk Score Triage Matrix (Low Risk, Medium Risk, High Triage Urgency).
  * Feature Breakdown: Asymmetry, Border Irregularity, Color Variation scores.
  * Tele-Dermatology Referral Finder & PDF Triage summary for physicians.
* **Live Deployment:** API and Web app deployed on **Render** / **Railway** with swagger docs (`/docs`).

---

### 💰 Monetization & Business Model (`BUSINESS_PLAN.md`)
* **Target Audience:** Telehealth companies, private medical clinics, community pharmacies, health-conscious individuals.
* **Revenue Streams:**
  1. **Pay-Per-Scan (B2C):** ₦1,000 per detailed dermatological risk report.
  2. **B2B API License (Telemedicine Platforms):** ₦150 per API call for telehealth apps integrating automated pre-consultation triage.
  3. **Pharmacy Clinic Terminal (₦25,000 / month):** Retail pharmacies pay a monthly subscription to offer walk-in skin screening kiosks to customers.

---

### 🎯 AI-Proof Defense Questions (Required in Student Submission)
> *Note: These questions require empirical proof from the student's training logs, visual evidence, and deep architectural understanding. They CANNOT be answered by generic AI queries.*

1. **Medical Metrics & Thresholding Trade-Off:**  
   *"In medical triage, is Precision or Recall more critical? Present your model's ROC Curve. What exact classification threshold (e.g., \(t = 0.25\) vs \(t = 0.50\)) did you select for Melanoma to achieve a Recall \(\ge 95\%\), and how many False Positives did this shift introduce?"*
2. **Focal Loss vs. Standard Cross-Entropy Empirical Comparison:**  
   *"Provide the confusion matrix from your initial run using Standard Categorical Cross-Entropy versus your run using Focal Loss. Exactly how many malignant melanoma cases were missed (False Negatives) in each run?"*
3. **Edge Optimization & Quantization Validation:**  
   *"What was the file size (in MB) and average inference latency (in milliseconds) of your full Keras model versus your quantized TFLite/ONNX model? Did quantization cause any drop in F1-score?"*

---

---

## 🎨 Assignment 3: Assigned to **Elisha**
### **NeuralBrand AI — Automated Product Design, Style Transfer & Ad Copy Engine**

* **Domain:** E-Commerce / Creative AdTech & Generative AI
* **Primary Objective:** Build a dual-engine creative suite combining Neural Style Transfer (VGG19) to transform plain product photos into high-end branded artistic visuals, paired with an LSTM/GRU sequence model that automatically generates high-converting marketing captions and ad copy.

---

### 🧠 Model Architecture & Deep Learning Specifications
* **Visual Engine (Neural Style Transfer - NST):**
  * **Backbone:** Pretrained **VGG19** feature extractor.
  * **Loss Function Architecture:** Custom optimization loop calculating **Content Loss** (from `block4_conv2` layer) and **Style Loss** (Gram Matrices computed across `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`).
  * **Optimization:** Optimization via **L-BFGS** or **Adam** over target image pixels.
* **Text Engine (Ad Copy Sequence Model):**
  * **Architecture:** Stacked **LSTM / GRU** Recurrent Neural Network (or fine-tuned lightweight Transformer) trained on e-commerce product titles, descriptions, and high-performing social media ad captions.
  * **Techniques:** Word/Character Embeddings, Temperature Scaling (\(T \in [0.2, 1.2]\)) for controlling generation creativity, Gradient Clipping (`clipnorm=1.0`).

---

### 💻 Web App & Live Deployment
* **Frontend/Backend:** Flask or FastAPI web portal with a modern visual canvas.
* **Features:**
  * Dual Input Studio: Upload Product Photo + Select/Upload Preset Brand Art Style.
  * Real-Time / Asynchronous Style Transfer Renderer.
  * Interactive Copy Generator: Input product category & key benefits \(\rightarrow\) output 3 AI marketing captions (Hook, Body, Call-to-Action).
  * Export Ad Campaign Kit (Stylized Image + Matching Ad Copy + Hashtags).
* **Live Deployment:** Deployed on **Render** or **Hugging Face Spaces** utilizing asynchronous background processing (e.g., Python threading/background queue) to handle computationally intensive style transfer iterations without timing out the web server.

---

### 💰 Monetization & Business Model (`BUSINESS_PLAN.md`)
* **Target Audience:** Instagram/TikTok e-commerce sellers, digital marketing agencies, brand managers, small business owners.
* **Revenue Streams:**
  1. **Freemium Tier:** 3 watermarked designs + basic captions per month.
  2. **Pro Creator Subscription (₦5,000 / month):** 50 high-res style transfer renders, unlimited AI ad captions, custom brand preset storage, watermark removal.
  3. **Agency Pro Plan (₦25,000 / month):** Batch product image processing, commercial license, custom style image uploads, API access.

---

### 🎯 AI-Proof Defense Questions (Required in Student Submission)
> *Note: These questions require empirical proof from the student's training logs, visual evidence, and deep architectural understanding. They CANNOT be answered by generic AI queries.*

1. **Content vs. Style Loss Weighting Ratio (\(\alpha / \beta\)):**  
   *"Show 3 generated images of the same product using your VGG19 NST engine at \(\alpha/\beta = 10^{-3}\), \(\alpha/\beta = 10^{-5}\), and \(\alpha/\beta = 10^{-7}\). Explain visually what happens to the underlying product geometry when style loss dominates content loss."*
2. **Gram Matrix Mathematical Intuition:**  
   *"Explain in your own words how the Gram Matrix captures texture and artistic style without capturing spatial object location. Why do we compute inner products across feature channels rather than spatial dimensions?"*
3. **RNN Temperature Scaling Dynamics:**  
   *"Generate ad copy for the same product at Temperature = 0.2, Temperature = 0.7, and Temperature = 1.4. Copy and paste the raw text outputs in your defense document. Why does higher temperature cause word repetition or hallucination in your LSTM?"*

---

---

## 📊 Evaluation & Grading Rubric (100 Marks Total)

| Component | Description | Marks |
|---|---|---|
| **Data Engineering & Preprocessing** | Proper dataset structure, cleaning, augmentation pipelines, and handling imbalance/normalization | **15 Marks** |
| **Model Architecture & Training** | Appropriate model choice (CNN, Transfer Learning, RNN/LSTM, GAN/VAE, NST), proper hyperparameter selection, and loss convergence | **25 Marks** |
| **Model Evaluation & Optimization** | Hyperparameter tuning, confusion matrix/loss curve analysis, metric selection (F1, Recall, Loss balance), and optimization | **15 Marks** |
| **Web App & Cloud Deployment** | Functional web frontend/API, real-time input processing, deployed live on Hugging Face / Render / Railway | **20 Marks** |
| **Monetization & Business Plan** | Complete `BUSINESS_PLAN.md` with problem, solution, UVP, pricing model, revenue forecast, and GTM strategy | **15 Marks** |
| **AI-Proof Defense Performance** | Satisfactory empirical answers & logs submitted for the 3 Defense Questions | **10 Marks** |
| **Total** | | **100 Marks** |

---

## 📁 Required Submission Repository Structure

```
StudentName_DeepLearning_Capstone/
│
├── BUSINESS_PLAN.md          ← Detailed business & monetization model
├── DEFENSE_ANSWERS.md        ← Empirical answers + plots for the 3 AI-Proof Defense Questions
├── README.md                 ← Project overview, architecture diagram, and deployment link
├── requirements.txt          ← Python dependencies
│
├── data/
│   └── dataset_loader.py     ← Ingestion, cleaning, and augmentation scripts
│
├── models/
│   ├── train.py              ← Model definition, training loop, KerasTuner, and callbacks
│   ├── evaluate.py           ← Metrics, confusion matrices, loss curves, and Grad-CAM/Gram-matrix scripts
│   └── saved_models/         ← Trained model artifacts (.h5, .keras, .tflite, or .onnx)
│
├── app/
│   ├── app.py                ← Flask / FastAPI / Streamlit main web server
│   ├── templates/            ← HTML templates (if Flask/FastAPI)
│   └── static/               ← CSS, JS, generated images
│
└── feature_log.md            ← Agile sprint build log (commits and feature progression)
```

---

> 💡 **Instructor Note:** Instruct students to push their code to GitHub, deploy live, and prepare a **5-minute live pitch & technical defense** addressing their 3 unique defense questions.
