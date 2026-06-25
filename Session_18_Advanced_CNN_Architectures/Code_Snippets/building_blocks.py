"""
============================================================
  FILE: building_blocks.py
  SESSION: 18 — Advanced CNN Architectures
  PURPOSE: Manually implement an Inception Block and a 
           Residual Block using Keras Functional API.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("==================================================")
print("  Aptech AI Lab: Engineering CNN Building Blocks  ")
print("==================================================")

# =====================================================================
# 1. THE INCEPTION BLOCK (Parallelism & 1x1 Bottleneck)
# =====================================================================
def inception_block(x):
    print("\n--- Building Inception Block ---")
    
    # Path 1: 1x1 Convolution (Looks at depth only)
    path1 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    
    # Path 2: 1x1 Bottleneck -> 3x3 Convolution
    # Notice how we reduce the depth first to save computational cost!
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    
    # Path 3: 1x1 Bottleneck -> 5x5 Convolution
    path3 = layers.Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(path3)
    
    # Path 4: Max Pooling -> 1x1 Convolution
    path4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate all paths together along the depth axis (axis=-1)
    output = layers.concatenate([path1, path2, path3, path4], axis=-1)
    return output

# Create a dummy model to view the Inception Block summary
inputs = keras.Input(shape=(32, 32, 256)) # Suppose we have 256 channels coming in
outputs = inception_block(inputs)
inception_model = keras.Model(inputs=inputs, outputs=outputs)
print("\nInception Block Summary:")
inception_model.summary()


# =====================================================================
# 2. THE RESIDUAL BLOCK (Conquering the Vanishing Gradient)
# =====================================================================
def residual_block(x):
    print("\n--- Building Residual Block ---")
    
    # Save the original input for the skip connection [The 'x' in F(x) + x]
    shortcut = x
    
    # The main convolutional path [The 'F(x)']
    # We use padding='same' to ensure the spatial dimensions don't shrink, 
    # so we can mathematically add it to the shortcut later.
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(x)
    
    # The Skip Connection: H(x) = F(x) + x
    # This guarantees a gradient of at least 1 during backpropagation!
    x = layers.add([x, shortcut])
    
    # Apply final activation
    output = layers.Activation('relu')(x)
    return output

# Create a dummy model to view the Residual Block summary
# Input needs to match the filter count (64) for the element-wise addition to work
res_inputs = keras.Input(shape=(32, 32, 64)) 
res_outputs = residual_block(res_inputs)
res_model = keras.Model(inputs=res_inputs, outputs=res_outputs)
print("\nResidual Block Summary:")
res_model.summary()
