# 🎒 The Full Flow: From Kaggle to Hugging Face
### Building the Nomentral Land Document Fraud Detector
---

> **Instructor's Note:**
> *"You asked: 'How do I actually build this fraud detector and put it on Hugging Face?' This is the complete, unbroken pipeline. Phase 1 happens in Kaggle to train the brain. Phase 2 happens in Hugging Face Spaces to build the website. By the end of this guide, you will have a live URL that anyone can use to test your AI."*

---

## 🏗️ Phase 1: Building the Brain (in Kaggle)

Open a new Kaggle Notebook. Turn on the GPU. We are going to build a Vision Machine (CNN) that can tell the difference between "Authentic" land documents and "Fraudulent/Forged" ones. 

Since we don't have Nomentral's private company data, we will generate dummy data to prove the concept works.

### Step 1: Generate Dummy Land Documents
Copy this into Kaggle and click Play. This creates 100 fake images of "Authentic" documents and 100 "Fraudulent" ones. (In real life, you would just upload a ZIP file of real photos).

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Create folders
os.makedirs("dataset/authentic", exist_ok=True)
os.makedirs("dataset/fraudulent", exist_ok=True)

print("Generating dummy land documents for training...")
# Authentic: Mostly white pages with a solid BLUE "seal" in the top-left
for i in range(100):
    img = np.ones((100, 100, 3)) * 255 # White page
    img[10:30, 10:30] = [0, 0, 255]    # Blue square seal
    tf.keras.preprocessing.image.save_img(f"dataset/authentic/auth_{i}.jpg", img)

# Fraudulent: Mostly white pages, but the "seal" is RED and in the bottom-right
for i in range(100):
    img = np.ones((100, 100, 3)) * 255
    img[70:90, 70:90] = [255, 0, 0]    # Red fake seal
    tf.keras.preprocessing.image.save_img(f"dataset/fraudulent/fraud_{i}.jpg", img)
print("Data ready!")
```

### Step 2: Load the Data into Keras
```python
# 2. Load the images into a Keras Dataset
train_dataset = keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(100, 100),
    batch_size=32
)

# See what classes it found (0 = authentic, 1 = fraudulent)
class_names = train_dataset.class_names
print("Classes:", class_names)
```

### Step 3: Build and Train the CNN Fraud Detector
```python
# 3. Build a simple Vision Machine (CNN)
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(100, 100, 3)), # Normalize pixels
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # 0 = Authentic, 1 = Fraudulent
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training the AI to spot fakes...")
model.fit(train_dataset, epochs=5)
print("Training Complete!")
```

### Step 4: SAVE THE MODEL (Crucial Step!)
If you skip this, your AI dies when you close Kaggle.
```python
# 4. Save the trained brain to a file
model.save("nomentral_fraud_ai.keras")
print("Model saved! Download this file to your computer.")
```
**👉 Action:** On the right side of Kaggle, look under **Output**. You will see `nomentral_fraud_ai.keras`. Click the three dots next to it and download it to your laptop.

---

## 🚀 Phase 2: Deploying the Web App (in Hugging Face)

Now we leave Kaggle. We have our trained brain (`.keras` file). We need to put it on the web.

### Step 1: Create a Hugging Face Space
1. Go to [HuggingFace.co](https://huggingface.co/) and create a free account (or log in).
2. Click your profile picture top right ➔ **New Space**.
3. Space Name: `Nomentral-Fraud-Detector`
4. License: `MIT`
5. Space Hardware: **Gradio**
6. Space Hardware: Free (CPU basic)
7. Click **Create Space**.

### Step 2: Upload Your Model
1. In your new Space, click the **Files** tab.
2. Click **Add file** ➔ **Upload files**.
3. Drag and drop the `nomentral_fraud_ai.keras` file you downloaded from Kaggle.
4. Click **Commit changes to main**.

### Step 3: Tell Hugging Face What Libraries You Need
1. Click **Add file** ➔ **Create new file**.
2. Name the file exactly: `requirements.txt`
3. Paste this exactly into the box:
```text
tensorflow
numpy
gradio
Pillow
spaces
```
4. Click **Commit new file to main**.

### Step 4: Write the Web App Code
This is the magic part. We will write 20 lines of Python that turns your AI into a website.
1. Click **Add file** ➔ **Create new file**.
2. Name the file exactly: `app.py`
3. Paste this code:

```python
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import spaces

# 1. Load the AI brain you uploaded
model = tf.keras.models.load_model('nomentral_fraud_ai.keras')

# Dummy function to satisfy Hugging Face's ZeroGPU requirement
@spaces.GPU
def dummy_function():
    pass

# 2. Define the prediction function (Runs safely on CPU to avoid CUDA conflicts)
def predict_document(image):
    # Resize the user's uploaded image to exactly what the AI expects (100x100)
    image = image.resize((100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, 0) # Create a batch

    # Make the prediction
    prediction = model.predict(img_array)[0][0]
    
    # Format the result for the user
    if prediction > 0.5:
        confidence = prediction * 100
        return f"🚨 FRAUDULENT DOCUMENT DETECTED (Confidence: {confidence:.2f}%)"
    else:
        confidence = (1 - prediction) * 100
        return f"✅ AUTHENTIC DOCUMENT (Confidence: {confidence:.2f}%)"

# 3. Build the User Interface
iface = gr.Interface(
    fn=predict_document,
    inputs=gr.Image(type="pil", label="Upload Land Document Scan"),
    outputs=gr.Text(label="AI Analysis Result"),
    title="Nomentral AI - Land Document Verification",
    description="Upload a scanned land document. The AI will verify the security seals to detect forgeries."
)

# 4. Launch the app!
iface.launch()
```
4. Click **Commit new file to main**.

### Step 5: The "Building" Phase
Click back to the **App** tab at the top. You will see a yellow circle that says **"Building"**. 
Hugging Face is currently downloading TensorFlow and setting up your web server. This takes about 2 to 3 minutes.

Once the circle turns **Green (Running)**, your app is live!

### Step 6: Test It!
You now have a live URL (e.g., `https://huggingface.co/spaces/yourusername/Nomentral-Fraud-Detector`).
You can text this link to your phone, send it to your CEO, or share it on LinkedIn. 

Anyone can open it, upload an image from their photo gallery, and watch your Kaggle AI process it in real-time.

**Congratulations. You have completed the Full Flow. You didn't just write code — you shipped a product!**

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks*
