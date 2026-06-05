"""
============================================================
  FILE: hyperparameter_tuning.py
  SESSION: 11 — Fine-Tuning & Hyperparameters
  PURPOSE: Automate the discovery of the best neural 
           network architecture using Keras Tuner.
============================================================
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("  Aptech AI Lab: The Hyperparameter Hunt")
print("=" * 50)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0


# ─── 2. DEFINE THE TUNABLE MODEL ──────────────────────────────────────────────
def build_model(hp):
    """
    Instead of hardcoding values, we use the `hp` object to 
    define ranges of possible hyperparameters.
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # Let the Tuner choose the number of neurons! (Between 32 and 256, in steps of 32)
    hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Let the Tuner choose the best Learning Rate!
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# ─── 3. CONFIGURE THE TUNER ───────────────────────────────────────────────────
print("\nInitializing Random Search Tuner...")
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',  # We want the highest validation accuracy
    max_trials=5,              # Try 5 different combinations
    executions_per_trial=1,    # Run each combination once
    directory='tuning_dir',    # Where to save the logs
    project_name='fashion_tuning'
)

# ─── 4. START THE HUNT ────────────────────────────────────────────────────────
print("\nStarting the Hyperparameter Search. This will take a moment...\n")
tuner.search(X_train, y_train, epochs=3, validation_split=0.2)

# ─── 5. GET THE RESULTS ───────────────────────────────────────────────────────
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "=" * 50)
print("  SEARCH COMPLETE! WINNING CONFIGURATION:")
print("=" * 50)
print(f"Optimal number of neurons : {best_hps.get('units')}")
print(f"Optimal learning rate     : {best_hps.get('learning_rate')}")
print("=" * 50)
