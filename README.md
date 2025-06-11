# 🧠 Deep Learning Task - 2 (CodTech Internship)

## ✅ Task Objective
Build a deep learning model for **image classification** using **TensorFlow**.

## 📌 Dataset
- **MNIST Handwritten Digits**
- 60,000 training images + 10,000 test images
- Grayscale, size 28x28
- 10 classes (digits 0 to 9)

## 🧠 Model Architecture
Input: 28x28 → Flatten → Dense(128, ReLU) → Dense(10, Softmax)

## 📁 Files
- `model.py` – Model training, evaluation, accuracy/loss graph
- `visualize.py` – Prediction and result visualization
- `mnist_model.h5` – Saved trained model
- `training_results.png` – Accuracy/Loss plots

## ▶️ How to Run

### 1. Train the Model
```bash
python model.py
