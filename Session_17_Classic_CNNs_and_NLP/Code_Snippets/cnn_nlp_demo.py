"""
============================================================
  FILE: cnn_nlp_demo.py
  SESSION: 17 — Classic CNNs and NLP
  PURPOSE: Build a 1D Convolutional Neural Network to 
           classify IMDB movie reviews as Positive/Negative.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: 1D CNN for Text Classification")
print("=" * 50)

# ─── 1. LOAD AND PREP DATA ────────────────────────────────────────────────────
print("\nDownloading IMDB Dataset (this may take a moment)...")
max_features = 5000  # Only use the top 5000 most common words
maxlen = 400         # Cut reviews after 400 words

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# Pad sequences so every review is exactly 400 words long
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)


# ─── 2. BUILD THE 1D CNN ARCHITECTURE ─────────────────────────────────────────
print("\nBuilding the 1D CNN Model...")
model = keras.Sequential()

# Step 1: Embedding Layer (Turns integer words into dense vectors)
model.add(keras.layers.Embedding(max_features, 50, input_length=maxlen))

# Step 2: The 1D Convolution
# It will look at 3 words at a time (kernel_size=3) to find sentiment patterns
model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))

# Step 3: Global Max Pooling
# Instantly extracts the strongest signal from the entire sentence
model.add(keras.layers.GlobalMaxPooling1D())

# Step 4: The Dense Classifier
model.add(keras.layers.Dense(250, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid')) # Output: 0 = Negative, 1 = Positive

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ─── 3. TRAIN THE MODEL ───────────────────────────────────────────────────────
print("\nTraining the model on Movie Reviews...")
# We use a small batch size and just 3 epochs for classroom speed
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=3,
                    validation_data=(X_test, y_test))

print("\n" + "=" * 50)
print(f"  FINAL VALIDATION ACCURACY: {history.history['val_accuracy'][-1]:.4f}")
print("=" * 50)
