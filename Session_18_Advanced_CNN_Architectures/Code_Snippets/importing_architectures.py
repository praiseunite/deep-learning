"""
============================================================
  FILE: importing_architectures.py
  SESSION: 18 — Advanced CNN Architectures
  PURPOSE: Demonstrate how to instantiate world-class 
           architectures directly from Keras Applications.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: Importing the Titans")
print("=" * 50)

# ─── 1. IMPORTING VGG16 ───────────────────────────────────────────────────────
print("\n[1/2] Instantiating VGG16 Architecture...")
print("      (This may take a moment to download the blueprint)")

# weights=None means we get the empty architecture, not pre-trained weights
# We specify a standard 224x224 RGB image input
vgg_model = keras.applications.VGG16(weights=None, input_shape=(224, 224, 3), classes=1000)

print("\n--- VGG16 Summary ---")
vgg_model.summary()


# ─── 2. IMPORTING RESNET50 ────────────────────────────────────────────────────
print("\n" + "="*50)
print("[2/2] Instantiating ResNet50 Architecture...")

resnet_model = keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=1000)

print("\n--- ResNet50 Summary ---")
resnet_model.summary()

print("\n" + "="*50)
print("  Look at the bottom of the summaries above.")
print("  Compare the Total Parameters between VGG16 and ResNet50!")
print("="*50)
