# 🧠 Deep Learning Using Neural Networks

Welcome to the **Deep Learning Using Neural Networks** repository! This course is designed to take you from an absolute beginner in Artificial Intelligence to a confident Deep Learning engineer. 

This curriculum follows a heavily structured, pedagogical approach. We prioritize understanding the "why" before the "how," using real-world analogies to break down complex mathematical concepts like Gradient Descent, Backpropagation, and Regularization.

---

## 🎯 What is this course about?
Traditional Machine Learning relies on humans manually telling the computer what features to look for. **Deep Learning** removes the human bottleneck. In this course, you will learn how to build Artificial Neural Networks (ANNs) capable of automatically extracting features from raw data. You will learn the architecture of these networks, the calculus that allows them to learn from their mistakes, and the modern techniques used to prevent them from "cheating" by memorizing data.

---

## 🛠️ Requirements & Tools Needed
To succeed in this course, you will need the following:

### Knowledge Prerequisites
- **Basic Python Programming:** You should be comfortable with variables, loops, functions, and lists.
- **Basic Algebra:** You don't need to be a calculus master, but you should understand the equation of a line ($y = mx + b$).

### Software Requirements
- **Python 3.8+** installed on your machine.
- An IDE or Text Editor (We highly recommend **Visual Studio Code** or **Jupyter Notebooks**).
- The following Python libraries installed via `pip`:
  - `tensorflow` (and `keras`)
  - `numpy`
  - `matplotlib`

### Hardware & Cloud Platforms (Free GPU Access)
You **do not** need an expensive computer or a dedicated graphics card (GPU) to start this course. The early sessions will run perfectly fine on a standard laptop CPU. However, as we progress to more complex models (like CNNs), a GPU will speed up training from *hours* to *minutes*.

If you do not have a dedicated GPU, we highly recommend using these **100% free** cloud platforms that run directly in your browser:
- 🌟 **[Google Colab](https://colab.research.google.com/):** The industry standard for beginners. It gives you a free Jupyter Notebook environment with TensorFlow pre-installed and allows you to connect to a free NVIDIA GPU.
- 📊 **[Kaggle Notebooks](https://www.kaggle.com/code):** Similar to Colab, but deeply integrated with thousands of open-source datasets. Also provides free GPU access.

*(Instructions for installing these libraries are covered in Session 1).*

---

## 📚 Course Table of Contents

Navigate to the folders below to access the Lecture Notes, In-Class Tasks, Homework Assignments, and Code Snippets for each session.

| Session | Topic | Focus |
| :---: | :--- | :--- |
| **[Session 01](./Session_01_Introduction_to_Deep_Learning/)** | **Introduction to Deep Learning** | The history of AI, ML vs DL, and AI Ethics. |
| **[Session 02](./Session_02_Artificial_Neural_Networks/)** | **Artificial Neural Networks** | The biological inspiration, the artificial Neuron, and Forward Propagation. |
| **[Session 03](./Session_03_Feedforward_Neural_Networks_Part1/)** | **Feedforward Neural Networks (Part 1)** | Building networks with hidden layers, and the limitations of linear models (XOR problem). |
| **[Session 04](./Session_04_Feedforward_Neural_Networks_Part2/)** | **Feedforward Neural Networks (Part 2)** | Loss Functions, Gradient Descent, Backpropagation, and the Keras MNIST digit classifier. |
| **[Session 05](./Session_05_Review_and_Try_It_Yourself/)** | **Review & "Try It Yourself" Lab** | Consolidating Sessions 1-4 with a massive debugging lab and the Fashion MNIST dataset. |
| **[Session 06](./Session_06_Activation_Functions_Deep_Dive/)** | **Activation Functions Deep Dive** | "Warping space", the Dying ReLU problem, and variants like Leaky ReLU, ELU, and Swish. |
| **[Session 07](./Session_07_Backpropagation_Variants/)** | **Backpropagation Variants** | The difference between Batch, Stochastic, and Mini-Batch Gradient Descent and hardware limits. |
| **[Session 08](./Session_08_Regularization/)** | **Regularization** | Combating Overfitting using L1/L2 penalties, Dropout layers, and Early Stopping Callbacks. |
| **[Session 09](./Session_09_TensorFlow_and_Keras/)** | **TensorFlow and Keras** | Computational Graphs, Tensors, and the Keras Functional API. |
| **[Session 10](./Session_10_Model_Deployment/)** | **Model Deployment** | Saving models (SavedModel/H5) and deploying via TF Serving, TF Lite, and TF.js. |
| **[Session 11](./Session_11_Fine_Tuning_and_Hyperparameters/)** | **Fine-Tuning & Hyperparameters** | Transfer Learning with pre-trained models (MobileNet) and automated Hyperparameter tuning using Keras Tuner. |
| **[Session 12](./Session_12_Search_Strategies_and_AutoML/)** | **Search Strategies & AutoML** | Grid vs Random vs Bayesian Optimization, and the philosophy of AutoML. |
| **[Session 13](./Session_13_Workshop_Try_It_Yourself/)** | **Workshop: End-to-End Training** | A hands-on coding lab covering Sessions 4-7. Building, compiling, and training a model on CIFAR-10. |
| **[Session 14](./Session_14_Deep_vs_Shallow_Networks/)** | **Deep vs Shallow Networks** | Universal Approximation Theorem and the power of Hierarchical Feature Learning. |
| **[Session 15](./Session_15_Network_Efficiency/)** | **Network Efficiency** | Shrinking models for mobile deployment using Pruning, Quantization, and Knowledge Distillation. |
| **[Session 16](./Session_16_Convolutional_Neural_Networks/)** | **Convolutional Neural Networks** | Introduction to CNNs, Kernels, Convolution operations, and Spatial Hierarchy. |
| **[Session 17](./Session_17_Classic_CNNs_and_NLP/)** | **Classic CNNs & NLP** | History (LeNet to AlexNet) and 1D Convolutions for sequential text data. |
| **[Session 18](./Session_18_Advanced_CNN_Architectures/)** | **Advanced CNN Architectures** | Deep dive into the ImageNet titans: VGGNet, InceptionNet, and ResNet. |
| **[Session 19](./Session_19_Zero_to_Hero_Workshop/)** | **Zero-to-Hero Workshop** | Hand-holding setup of Kaggle, Cloud GPUs, and executing a first model. |
| **[Session 20](./Session_20_Recurrent_Neural_Networks/)** | **Recurrent Neural Networks** | Sequential data, Backpropagation Through Time (BPTT), and the Vanishing Gradient problem. |
| **[Session 21](./Session_21_RNN_Text_Generation/)** | **RNN Text Generation** | Sequential data representation, One-Hot Encoding, and Autoregressive generation. |

> **Note to Students:** The `Solutions/` directories are intentionally excluded from this public repository. You must complete the in-class tasks and assignments yourself!

---
*© 2024 Aptech Limited Curriculum Format*
