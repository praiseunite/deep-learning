"""
============================================================
  FILE: dl_timeline_visualization.py
  SESSION: 01 — Introduction to Deep Learning
  PURPOSE: Create a visual timeline of Deep Learning milestones
           using Matplotlib (Python's plotting library)
  
  CONCEPTS DEMONSTRATED:
    - Using Matplotlib to create visualizations
    - Plotting a timeline (a common data visualization task)
    - How Deep Learning evolved over decades
  
  HOW TO RUN:
    python dl_timeline_visualization.py
    
  OUTPUT:
    A window will open showing the timeline.
    Close the window to end the program.
============================================================
"""

# ─── Import Libraries ──────────────────────────────────────────────────────────
# matplotlib.pyplot is the main plotting module
# We call it 'plt' for short — this is standard convention
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For creating colored legend patches
import numpy as np                     # For numerical operations

# ─── Data: Deep Learning Milestones ──────────────────────────────────────────

# Each entry is: (year, milestone_title, category, description)
# Category determines the color of the marker on our timeline
milestones = [
    # --- Artificial Intelligence Era ---
    (1950, "Turing Test",          "AI",  "Alan Turing proposes a test for machine intelligence"),
    (1956, "AI is Born",           "AI",  "Dartmouth Conference — 'Artificial Intelligence' coined"),
    (1966, "ELIZA Chatbot",        "AI",  "First chatbot that simulates conversation"),
    
    # --- Machine Learning Era ---
    (1980, "Backpropagation",      "ML",  "Rumelhart & Hinton enable training of multi-layer networks"),
    (1989, "LeNet (First CNN)",    "ML",  "Yann LeCun creates first CNN for handwritten digits"),
    (1997, "Deep Blue",            "ML",  "IBM's Deep Blue defeats chess world champion Kasparov"),
    
    # --- Deep Learning Revolution ---
    (2006, "DL Revival",           "DL",  "Hinton shows deep networks can be pre-trained effectively"),
    (2012, "AlexNet",              "DL",  "AlexNet wins ImageNet — DL beats everything else by 10%"),
    (2014, "GAN Invented",         "DL",  "Goodfellow invents Generative Adversarial Networks"),
    (2016, "AlphaGo",              "DL",  "DeepMind's AlphaGo defeats world Go champion"),
    (2017, "Transformers",         "DL",  "Google publishes 'Attention is All You Need'"),
    (2020, "GPT-3",                "DL",  "OpenAI's GPT-3: 175 billion parameters language model"),
    (2020, "AlphaFold",            "DL",  "DeepMind solves 50-year-old protein folding problem"),
    (2022, "ChatGPT",              "DL",  "OpenAI releases ChatGPT — 100M users in 2 months"),
    (2024, "Gemini & GPT-4",       "DL",  "Multimodal AI models understand text, image, audio, video"),
]

# ─── Color Scheme ─────────────────────────────────────────────────────────────
# Each category gets its own color for easy visual differentiation
category_colors = {
    "AI": "#4A90D9",   # Blue  — Artificial Intelligence era
    "ML": "#9B59B6",   # Purple — Machine Learning era  
    "DL": "#E74C3C",   # Red   — Deep Learning era
}

category_labels = {
    "AI": "Artificial Intelligence (1950s–1970s)",
    "ML": "Machine Learning (1980s–2000s)",
    "DL": "Deep Learning (2006–Present)",
}

# ─── Create the Figure ────────────────────────────────────────────────────────

# Create a large figure to fit all the milestones
# figsize=(width, height) in inches
fig, ax = plt.subplots(figsize=(16, 9))

# Set a dark background for modern look
fig.patch.set_facecolor('#0D1117')   # Dark background for the entire figure
ax.set_facecolor('#0D1117')          # Dark background for the plot area

# ─── Draw the Main Timeline Line ─────────────────────────────────────────────
# A horizontal line from 1945 to 2026
ax.axhline(y=0, color='#4A4A4A', linewidth=2, zorder=1)

# ─── Plot Each Milestone ─────────────────────────────────────────────────────
# We alternate between above (y=1) and below (y=-1) the line
# to avoid text overlapping

for i, (year, title, category, description) in enumerate(milestones):
    
    # Alternate: even index → above line, odd index → below line
    y_pos = 1 if i % 2 == 0 else -1
    
    # Get the color for this category
    color = category_colors[category]
    
    # Draw a vertical connector line from the timeline to the label
    ax.plot([year, year], [0, y_pos * 0.85], 
            color=color, linewidth=1.5, alpha=0.6, zorder=2)
    
    # Draw the circle marker on the timeline
    ax.scatter(year, 0, 
               color=color, s=80, zorder=3, 
               edgecolors='white', linewidths=1)
    
    # Add the year label
    ax.text(year, y_pos * 0.88, str(year), 
            ha='center', va='bottom' if y_pos > 0 else 'top',
            color='white', fontsize=7.5, fontweight='bold',
            fontfamily='monospace')
    
    # Add the milestone title
    ax.text(year, y_pos * 1.0, title, 
            ha='center', va='bottom' if y_pos > 0 else 'top',
            color=color, fontsize=8.5, fontweight='bold',
            wrap=True)
    
    # Add the description (smaller text)
    ax.text(year, y_pos * 1.08, description, 
            ha='center', va='bottom' if y_pos > 0 else 'top',
            color='#AAAAAA', fontsize=6.5,
            wrap=True, style='italic')

# ─── Style the Plot ───────────────────────────────────────────────────────────

# Set the x-axis range with a small margin
ax.set_xlim(1945, 2027)

# Set the y-axis range to give space for labels
ax.set_ylim(-1.8, 1.8)

# Remove the y-axis ticks (we don't need them)
ax.set_yticks([])

# Style the x-axis
ax.set_xticks(range(1950, 2025, 5))
ax.tick_params(axis='x', colors='white', labelsize=9)

# Remove the frame/border
for spine in ax.spines.values():
    spine.set_visible(False)

# ─── Add Title ────────────────────────────────────────────────────────────────
ax.set_title("Timeline of Artificial Intelligence, Machine Learning & Deep Learning\n"
             "From Turing (1950) to Gemini & GPT-4 (2024)",
             color='white', fontsize=14, fontweight='bold', pad=20)

# ─── Add Legend ───────────────────────────────────────────────────────────────
# Create colored patches for the legend
legend_patches = [
    mpatches.Patch(color=color, label=category_labels[cat])
    for cat, color in category_colors.items()
]

ax.legend(handles=legend_patches, 
          loc='lower right',
          facecolor='#1C1C1C',
          edgecolor='#4A4A4A',
          labelcolor='white',
          fontsize=9)

# ─── Add Subtitle / Footer ────────────────────────────────────────────────────
fig.text(0.5, 0.02, 
         "Aptech — Deep Learning Using Neural Networks | Session 01",
         ha='center', color='#666666', fontsize=9)

# ─── Adjust Layout and Display ────────────────────────────────────────────────
plt.tight_layout(pad=2)

# Save the figure to the Assets folder as well
plt.savefig('dl_timeline.png', dpi=150, 
            bbox_inches='tight', facecolor='#0D1117')
print("📊 Timeline saved as 'dl_timeline.png'")

# Display the plot
# (A window will pop up — close it to end the program)
print("🖼️  Opening timeline visualization...")
print("   Close the window to exit.")
plt.show()
