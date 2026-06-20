# 📘 Session 19 — Getting and Seeing Data
### Course: Deep Learning Using Neural Networks | Aptech
---

Now that you have a powerful Kaggle computer running, you need something to feed it. A neural network without data is like a car without fuel.

## The Easy Way: `keras.datasets`

Because downloading CSV files and uploading them to the cloud can be annoying for beginners, the creators of Keras built several famous datasets *directly into the code library*.

You don't need to download any files. You just write one line of code, and Keras pulls the data from the internet straight into your notebook's memory.

### The MNIST Dataset
We are going to use the **MNIST dataset**. It is a collection of 60,000 tiny, black-and-white images of handwritten numbers (0 through 9). 

To download it into your notebook, you use this command:
```python
import tensorflow as tf

# This one line downloads 60,000 images!
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```

### Seeing is Believing
It is hard to trust data you can't see. Is `X_train` actually full of images? Let's write some code to look at the very first image in the dataset.

**Try this in your Kaggle Notebook:**
1. Create a new cell.
2. Paste the following code:
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Download the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Show the very first image
plt.imshow(X_train[0], cmap='gray')
plt.title(f"The computer says this is a: {y_train[0]}")
plt.show()
```
3. Hit the **Play** button.

You should see a grainy, pixelated image of a handwritten number appear on your screen! You have successfully imported data into the cloud.

---
*© 2024 Aptech Limited | Deep Learning Using Neural Networks | Session 19*
