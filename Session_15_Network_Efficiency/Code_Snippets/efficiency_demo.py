"""
============================================================
  FILE: efficiency_demo.py
  SESSION: 15 — Network Efficiency
  PURPOSE: Demonstrate Post-Training Quantization by 
           shrinking a heavy model and comparing file sizes.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: The Shrink Ray (Quantization)")
print("=" * 50)

# ─── 1. BUILD A HEAVY MODEL ───────────────────────────────────────────────────
print("\nBuilding a heavy model...")
# We use a massive dense layer to artificially inflate the model size
heavy_model = keras.Sequential([
    keras.layers.Dense(2000, activation='relu', input_shape=(1000,)),
    keras.layers.Dense(10, activation='softmax')
])

# This model has over 2 Million parameters (Weights).
heavy_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


# ─── 2. SAVE THE ORIGINAL 32-BIT MODEL ────────────────────────────────────────
original_file = "heavy_model_32bit.h5"
heavy_model.save(original_file)
print(f"Original model saved to: {original_file}")


# ─── 3. QUANTIZE TO 8-BIT USING TF LITE ───────────────────────────────────────
print("\nFiring the Shrink Ray (Applying 8-bit Quantization)...")

# Initialize the TF Lite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(heavy_model)

# THIS IS THE MAGIC LINE: Tell it to optimize for size (Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_quant_model = converter.convert()

# Save the tiny model
quantized_file = "tiny_model_8bit.tflite"
with open(quantized_file, 'wb') as f:
    f.write(tflite_quant_model)
print(f"Quantized model saved to: {quantized_file}")


# ─── 4. COMPARE THE SIZES ─────────────────────────────────────────────────────
def get_file_size_mb(file_path):
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

original_mb = get_file_size_mb(original_file)
quantized_mb = get_file_size_mb(quantized_file)

print("\n" + "=" * 50)
print("  SIZE COMPARISON RESULTS")
print("=" * 50)
print(f"Original Model (32-bit Float) : {original_mb:.2f} MB")
print(f"Quantized Model (8-bit Int)   : {quantized_mb:.2f} MB")
print(f"Reduction Factor              : {original_mb / quantized_mb:.1f}x Smaller!")
print("=" * 50)
