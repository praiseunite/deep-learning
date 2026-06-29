# Session 28 -- Assignment (Homework)
### Deep Learning Using Neural Networks | Aptech
### Assignment Title: "Gallery Night"
### Due: Before Session 29 begins
### Estimated Time: 50 minutes

---

> **Professor's Note:** *"Tonight, you become an AI artist. You will apply neural style transfer to three different content-style combinations and build your own AI art gallery."*

---

## Assignment Objectives

By completing this assignment, you will:
- Apply style transfer to multiple content-style pairings.
- Experiment with alpha/beta weights to control the visual outcome.
- Analyze why different styles transfer differently.

---

## THE SCENARIO

You are curating an AI Art Exhibition. You need to create 3 styled artworks using Neural Style Transfer. Each piece must use a different style image, and you must write a short "artist statement" explaining your creative choices and observations.

---

## Step-by-Step Instructions (Kaggle)

Open a **new** Kaggle Notebook with the GPU enabled.

### Cell 1: Setup (Same as In-Class)
Copy the imports, image loading functions, feature extractor, and loss functions from the in-class task (Cells 1-4).

### Cell 2: Define a Reusable Style Transfer Function

```python
def run_style_transfer(content_path, style_path, alpha=1e3, beta=1e-2, 
                       iterations=500, img_height=192, img_width=192):
    """Run complete style transfer and return the result."""
    
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    
    # Extract targets
    all_feat = feature_extractor(content)
    target_content = all_feat[:len(CONTENT_LAYERS)]
    all_feat = feature_extractor(style)
    target_style = all_feat[len(CONTENT_LAYERS):]
    
    generated = tf.Variable(content, dtype=tf.float32)
    optimizer = keras.optimizers.Adam(learning_rate=5.0)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            gen_feat = feature_extractor(generated)
            gen_content = gen_feat[:len(CONTENT_LAYERS)]
            gen_style = gen_feat[len(CONTENT_LAYERS):]
            
            c_loss = sum(compute_content_loss(tc, gc)
                        for tc, gc in zip(target_content, gen_content))
            s_loss = sum(compute_style_loss(ts, gs)
                        for ts, gs in zip(target_style, gen_style))
            total = alpha * c_loss + beta * s_loss
        
        grads = tape.gradient(total, generated)
        optimizer.apply_gradients([(grads, generated)])
        
        if i % 100 == 0:
            print(f"  Iter {i} | Loss: {total:.2f}")
    
    return deprocess_image(generated.numpy())

print("Style transfer function ready!")
```

### Cell 3: Download Multiple Style Images
```python
# Content image
content_path = keras.utils.get_file(
    'content.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)

# Style 1: Kandinsky (Abstract)
style1_path = keras.utils.get_file(
    'kandinsky.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)

# Style 2: Japanese Wave (Hokusai)
style2_path = keras.utils.get_file(
    'wave.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/800px-Tsunami_by_hokusai_19th_century.jpg'
)

# Style 3: Mosaic/Abstract pattern
style3_path = keras.utils.get_file(
    'towers.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_graridge.jpg'
)

# Display all style options
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, path, name in zip(axes, 
    [style1_path, style2_path, style3_path],
    ['Kandinsky', 'Hokusai Wave', 'Nature Texture']):
    img = keras.preprocessing.image.load_img(path, target_size=(192, 192))
    ax.imshow(img)
    ax.set_title(name, fontsize=12)
    ax.axis('off')
plt.suptitle("Available Style Images", fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 4: Create Your Gallery
Run style transfer with each style image. This will take 3-5 minutes per style.

```python
print("=== Artwork 1: Kandinsky Style ===")
result1 = run_style_transfer(content_path, style1_path, alpha=1e3, beta=1e-2, iterations=400)

print("\n=== Artwork 2: Hokusai Wave Style ===")
result2 = run_style_transfer(content_path, style2_path, alpha=1e3, beta=1e-2, iterations=400)

print("\n=== Artwork 3: Nature Texture Style ===")
result3 = run_style_transfer(content_path, style3_path, alpha=1e3, beta=1e-2, iterations=400)

print("\nAll artworks generated!")
```

### Cell 5: Display Your Gallery
```python
content_img = keras.preprocessing.image.load_img(content_path, target_size=(192, 192))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top row: style images
styles = [
    (style1_path, "Kandinsky Style"),
    (style2_path, "Hokusai Style"),
    (style3_path, "Nature Style")
]
for i, (path, name) in enumerate(styles):
    img = keras.preprocessing.image.load_img(path, target_size=(192, 192))
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Style: {name}", fontsize=11)
    axes[0, i].axis('off')

# Bottom row: results
results = [result1, result2, result3]
labels = ["Artwork 1", "Artwork 2", "Artwork 3"]
for i, (res, label) in enumerate(zip(results, labels)):
    axes[1, i].imshow(res)
    axes[1, i].set_title(label, fontsize=11)
    axes[1, i].axis('off')

plt.suptitle("AI Art Gallery: Style Images (Top) and Generated Art (Bottom)", fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 6: Experiment with Weights
Choose one style and try different alpha/beta ratios.

```python
print("Experiment: Varying Style Weight")
result_low_style = run_style_transfer(content_path, style1_path, alpha=1e3, beta=1e-3, iterations=300)
result_high_style = run_style_transfer(content_path, style1_path, alpha=1e3, beta=1e-1, iterations=300)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
ax1.imshow(result_low_style)
ax1.set_title("Low Style (beta=0.001)", fontsize=11)
ax1.axis('off')
ax2.imshow(result1)
ax2.set_title("Medium Style (beta=0.01)", fontsize=11)
ax2.axis('off')
ax3.imshow(result_high_style)
ax3.set_title("High Style (beta=0.1)", fontsize=11)
ax3.axis('off')
plt.suptitle("Effect of Style Weight on Output", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Written Questions (Your "Artist Statement")

For each of your 3 artworks, write 2-3 sentences answering:

1. **Artwork 1 (Kandinsky):** What elements of the Kandinsky style transferred to the photo? (Colors? Shapes? Textures?)

2. **Artwork 2 (Hokusai):** How does the wave style compare to Kandinsky? Which style is more "aggressive" in changing the content?

3. **Artwork 3 (Nature):** Is a non-painting "style" (like a nature photo) effective for style transfer? Why or why not?

4. **Weight Experiment:** Describe the visual difference between low, medium, and high style weights. At what point does the content become unrecognizable?

---

## Submission Requirements

1. Screenshot of your full gallery (Cell 5).
2. Screenshot of the weight experiment (Cell 6).
3. Written artist statement (answers to all 4 questions).

---
*Session 28 | Deep Learning Using Neural Networks | Aptech*
