# Task 3 – Digit Recognition Using MNIST and Keras

## 📌 Objective
Build and train a neural network using TensorFlow and Keras to classify handwritten digits (0–9) from the MNIST dataset.

## 📁 Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (28x28 pixels).
  - 60,000 training samples
  - 10,000 test samples

## 🧠 Model Architecture
- **Input**: Flatten layer to convert 28x28 image to a 784-dimensional vector
- **Hidden Layers**:
  - Dense(128) with ReLU activation
  - Dense(64) with ReLU activation
- **Output**: Dense(10) with softmax for digit classification

## ⚙️ Dependencies
- Python 3.x
- TensorFlow

## 🛠️ How to Run
1. Install TensorFlow (if not installed):
   ```bash
   pip install tensorflow
