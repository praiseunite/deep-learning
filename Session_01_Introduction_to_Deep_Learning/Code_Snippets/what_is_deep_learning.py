"""
============================================================
  FILE: what_is_deep_learning.py
  SESSION: 01 — Introduction to Deep Learning
  PURPOSE: Demonstrate the DIFFERENCE between traditional 
           programming and machine learning/deep learning
           using a simple, concrete example.
  
  CONCEPT: This script compares TWO approaches to a problem:
    Approach 1: TRADITIONAL PROGRAMMING — you write the rules
    Approach 2: MACHINE LEARNING     — the machine learns the rules
  
  HOW TO RUN:
    python what_is_deep_learning.py
============================================================
"""

# ─── Import Libraries ──────────────────────────────────────────────────────────
# We only use numpy (numerical Python) — a basic math library
# No deep learning libraries needed yet! That comes in Session 3.
import numpy as np

# ─── SECTION 1: Traditional Programming (Rule-Based) ─────────────────────────

print("=" * 65)
print("  APPROACH 1: TRADITIONAL PROGRAMMING (Rule-Based)")
print("=" * 65)
print()
print("  Scenario: Classify a number as 'Small', 'Medium', or 'Large'")
print()

def classify_number_with_rules(number):
    """
    Traditional programming approach.
    A HUMAN writes the rules manually.
    
    Rules:
      - If number < 10  → 'Small'
      - If 10–50        → 'Medium'  
      - If > 50         → 'Large'
    
    Problem: What if the rules are wrong? What if the data changes?
             A human must manually update every single rule!
    """
    # Rule 1: Check if number is small
    if number < 10:
        return "Small"
    
    # Rule 2: Check if number is medium
    elif number <= 50:
        return "Medium"
    
    # Rule 3: Everything else is large
    else:
        return "Large"

# Test the rule-based classifier
test_numbers = [3, 25, 75, 8, 42, 100]

print("  Testing the RULE-BASED classifier:")
print(f"  {'Number':<10} {'Category':<15}")
print(f"  {'-'*25}")

for num in test_numbers:
    category = classify_number_with_rules(num)
    print(f"  {num:<10} {category:<15}")

print()
print("  ⚠️  Problem: The rules were WRITTEN by a human.")
print("       If the pattern changes, a human must update all the rules manually.")
print()

# ─── SECTION 2: Machine Learning Approach ─────────────────────────────────────

print("=" * 65)
print("  APPROACH 2: MACHINE LEARNING (Learning from Data)")
print("=" * 65)
print()
print("  Scenario: The machine LEARNS what 'Small/Medium/Large' means")
print("            from EXAMPLES — no human writes the rules!")
print()

# Step 1: We provide training DATA (examples)
# These are pairs of (input_number, correct_category)
# In real ML, you'd have thousands or millions of examples
training_data = [
    # (number, label)
    # Label: 0 = Small, 1 = Medium, 2 = Large
    (2,  0),   # 2 → Small
    (5,  0),   # 5 → Small
    (8,  0),   # 8 → Small
    (15, 1),   # 15 → Medium
    (25, 1),   # 25 → Medium
    (40, 1),   # 40 → Medium
    (60, 2),   # 60 → Large
    (80, 2),   # 80 → Large
    (95, 2),   # 95 → Large
]

print("  📚 Training Data Provided to the Machine:")
print(f"  {'Number':<10} {'Label (0=Small, 1=Medium, 2=Large)':<40}")
print(f"  {'-'*50}")

for num, label in training_data:
    label_names = {0: "Small (0)", 1: "Medium (1)", 2: "Large (2)"}
    print(f"  {num:<10} {label_names[label]:<40}")

print()

# Step 2: The machine "learns" boundaries from the data
# (In real DL, this is done by a neural network with gradient descent)
# Here we simulate the concept with simple statistics

# Extract numbers and labels separately
numbers = [d[0] for d in training_data]    # [2, 5, 8, 15, 25, 40, 60, 80, 95]
labels  = [d[1] for d in training_data]    # [0, 0, 0, 1,  1,  1,  2,  2,  2]

# Find the boundary between Small and Medium (max of Small class)
small_numbers  = [numbers[i] for i in range(len(numbers)) if labels[i] == 0]
medium_numbers = [numbers[i] for i in range(len(numbers)) if labels[i] == 1]
large_numbers  = [numbers[i] for i in range(len(numbers)) if labels[i] == 2]

# The machine discovers the boundaries from the data
boundary_1 = (max(small_numbers) + min(medium_numbers)) / 2   # ~11.5
boundary_2 = (max(medium_numbers) + min(large_numbers)) / 2   # ~50.0

print(f"  🤖 What the Machine LEARNED from the data:")
print(f"     The machine discovered these boundaries:")
print(f"     → Small / Medium boundary: ~{boundary_1}")
print(f"     → Medium / Large boundary: ~{boundary_2}")
print()
print(f"     Rules the machine INFERRED (we did NOT write these!):")
print(f"     → If number < {boundary_1:.0f} → Small")
print(f"     → If {boundary_1:.0f} ≤ number < {boundary_2:.0f} → Medium")
print(f"     → If number ≥ {boundary_2:.0f} → Large")
print()

# Step 3: The machine makes predictions on new, unseen data
def classify_with_learned_rules(number, b1, b2):
    """
    Classification using boundaries LEARNED from data.
    The human did NOT write these rules — the machine found them!
    """
    if number < b1:
        return "Small"
    elif number < b2:
        return "Medium"
    else:
        return "Large"

# Test on new data the machine has never seen
new_data = [4, 30, 70, 12, 55, 88]

print("  🔮 Predicting on NEW DATA (never seen during training):")
print(f"  {'Number':<10} {'Prediction':<15} {'Traditional':<15}")
print(f"  {'-'*40}")

for num in new_data:
    ml_prediction = classify_with_learned_rules(num, boundary_1, boundary_2)
    rule_prediction = classify_number_with_rules(num)
    
    # Check if they agree
    match = "✅" if ml_prediction == rule_prediction else "⚠️ Different!"
    print(f"  {num:<10} {ml_prediction:<15} {rule_prediction:<15} {match}")

# ─── SECTION 3: Key Takeaways ─────────────────────────────────────────────────

print()
print("=" * 65)
print("  🎯 KEY TAKEAWAYS — What This Demonstrates")
print("=" * 65)
print()
print("  TRADITIONAL PROGRAMMING:")
print("    ✍️  Humans write the rules")
print("    ✍️  Rules must be updated manually when things change")
print("    ✍️  Works well for simple, well-defined problems")
print()
print("  MACHINE LEARNING / DEEP LEARNING:")
print("    🤖 Machine DISCOVERS the rules from data")
print("    🤖 Adapts automatically when given new data")
print("    🤖 Works for complex problems (faces, speech, language)")
print()
print("  DEEP LEARNING specifically adds:")
print("    🧠 Multiple LAYERS of rule discovery (depth)")
print("    🧠 Can learn patterns humans cannot articulate")
print("    🧠 Needs large amounts of data to learn effectively")
print()
print("  💡 In the next sessions, you will build REAL neural networks")
print("     that learn patterns from image data — no rules required!")
print()
print("=" * 65)
