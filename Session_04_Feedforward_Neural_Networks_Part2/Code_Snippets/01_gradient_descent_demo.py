"""
============================================================
  FILE: 01_gradient_descent_demo.py
  SESSION: 04 — Feedforward Neural Networks
  PURPOSE: Visually simulate how Gradient Descent steps down
           a loss curve to find the minimum error.
============================================================
"""

# Let's imagine a very simple Loss Function curve that looks like a bowl (parabola)
# Formula: Loss = x^2
# The gradient (derivative, or slope) of x^2 is 2x.

def loss_function(x):
    """Calculates the error (loss) given a weight x."""
    return x ** 2

def gradient(x):
    """Calculates the slope of the curve at a given weight x."""
    return 2 * x

print("=" * 50)
print("  GRADIENT DESCENT SIMULATOR")
print("=" * 50)

# 1. Start with a random, terrible guess for our weight (X)
current_x = 10.0  
print(f"Starting Position (Random Guess): X = {current_x}, Initial Loss = {loss_function(current_x):.2f}")
print("Goal: Find the bottom of the valley where X = 0 and Loss = 0.\n")

# 2. Define the Learning Rate (The size of the hiker's step)
# IN-CLASS TASK: You will change this number to 0.9 and then to 0.001!
LEARNING_RATE = 0.1  

# 3. Take 15 steps down the mountain
epochs = 15

for step in range(1, epochs + 1):
    # Calculate the slope at our current position
    current_gradient = gradient(current_x)
    
    # UPDATE RULE: New X = Old X - (Learning Rate * Gradient)
    # We subtract because if the slope is positive (uphill right), we want to move left (downhill).
    step_size = LEARNING_RATE * current_gradient
    current_x = current_x - step_size
    
    # Calculate our new error
    current_loss = loss_function(current_x)
    
    # Print the progress
    print(f"Step {step:2d} | Current X: {current_x:7.4f} | Loss: {current_loss:7.4f}")

print("\n" + "=" * 50)
if abs(current_x) < 0.1:
    print("✅ SUCCESS! We reached the bottom of the valley (minimum loss)!")
else:
    print("❌ FAILED! We did not reach the bottom. Check your Learning Rate.")
print("=" * 50)
