# CIFAR-10 Image Classification with Convolutional Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-70.03%25-success?style=for-the-badge)

A professional implementation of a Convolutional Neural Network (CNN) built from scratch using TensorFlow and Keras to classify the benchmark CIFAR-10 dataset across 10 distinct object classes.

---

## 📌 Project Overview

This repository contains the complete implementation and lab submission for the Computer Vision CNN assignment. The primary objective is to design, train, and evaluate a deep learning architecture capable of correctly identifying low-resolution ($32 \times 32$ pixels) RGB images from the CIFAR-10 dataset.

The workflow demonstrates standard professional deep learning practices, including rigorous data preprocessing, architectural design with spatial dimension preservation (`padding='same'`), optimization curve analysis, and evaluation on unseen test data.

---

## 📊 Dataset & Preprocessing

The **CIFAR-10** dataset consists of 60,000 $32 \times 32$ color images split across 10 mutually exclusive classes:
*Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.*

### Data Splits
- **Training Set**: 45,000 images
- **Validation Set**: 5,000 images (derived via stratified split from the training data)
- **Testing Set**: 10,000 images

### Preprocessing Pipeline
1. **Float Conversion**: Image pixel intensities are cast to `float32` and scaled from $[0, 255]$ to $[0.0, 1.0]$.
2. **Standardization (Z-Score Normalization)**: Pixel values are zero-centered and normalized across the training distribution:
   $$\hat{X} = \frac{X - \mu}{\sigma}$$
   *Note: To prevent data leakage, validation and test sets are normalized strictly using the training set's empirical mean ($\mu$) and standard deviation ($\sigma$).*
3. **Label Encoding**: Target classes are converted to one-hot encoded categorical vectors matching the network's 10-dimensional output softmax layer.

---

## 🧠 Network Architecture

The model is constructed using the Keras Sequential API, featuring two sequential convolutional blocks followed by a dense classification head. Zero-padding (`padding='same'`) is applied to preserve spatial resolution across convolutional layers before downsampling.

| Layer (Type) | Output Shape | Parameters | Activation / Details |
| :--- | :--- | :--- | :--- |
| **Input Layer** | `(32, 32, 3)` | 0 | $32 \times 32$ RGB images |
| **Conv2D (Block 1)** | `(32, 32, 32)` | 896 | Kernel: $3 \times 3$, `padding='same'`, ReLU |
| **MaxPooling2D** | `(16, 16, 32)` | 0 | Pool: $2 \times 2$, Stride: 2 |
| **Conv2D (Block 2)** | `(16, 16, 64)` | 18,496 | Kernel: $3 \times 3$, `padding='same'`, ReLU |
| **MaxPooling2D** | `(8, 8, 64)` | 0 | Pool: $2 \times 2$, Stride: 2 |
| **Flatten** | `(4096)` | 0 | Flattens 3D feature maps to 1D vector |
| **Dense (Hidden)** | `(128)` | 524,416 | Fully Connected, ReLU |
| **Dense (Output)** | `(10)` | 1,290 | Fully Connected, Softmax (10 classes) |

**Total Trainable Parameters**: `545,098`

---

## 🚀 Training Configuration & Results

### Hyperparameters
- **Optimizer**: Adam ($\text{learning rate} = 0.001$)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 10

### Model Performance
After 10 epochs of training, the model successfully generalizes without severe overfitting, achieving the following final evaluation metrics on the test set:

- **Final Test Loss**: `0.8719`
- **Final Test Accuracy**: `70.03%`

```
Test Evaluation:
313/313 [==============================] - 1s 4ms/step - loss: 0.8719 - accuracy: 0.7003
Test loss: 0.8719114661216736
Test accuracy: 0.7002999782562256
```

### Learning Curves Analysis
Training curves (accessible within the notebook) indicate smooth convergence:
- **Accuracy Curve**: Steady increase across both training and validation sets, stabilizing around epoch 8-10.
- **Loss Curve**: Validation loss closely tracks training loss, confirming the robustness of the feature extraction layers and validating the chosen architecture.

---

## 📁 Repository Structure

```
.
├── CIFAR10_Assignment.ipynb    # Main completed assignment notebook with all outputs
├── CV_Lab_CNN.ipynb            # Reference laboratory manual notebook
├── CV_Lab_CNN.pdf              # Reference laboratory manual PDF instructions
├── requirements.txt            # Required Python packages and dependencies
├── .gitignore                  # Git ignore configurations
└── README.md                   # Project documentation
```

---

## 🛠️ Getting Started / Quickstart

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/CIFAR10-CNN-Classification.git
   cd CIFAR10-CNN-Classification
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook to explore the code and execute cells:
   ```bash
   jupyter notebook CIFAR10_Assignment.ipynb
   ```

---

## 📜 License & Acknowledgments
Developed as part of the Computer Vision course lab assignments. Utilizing the standard CIFAR-10 benchmark provided by the Canadian Institute for Advanced Research.
