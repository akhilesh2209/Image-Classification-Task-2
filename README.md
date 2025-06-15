🧠 Deep Learning Task - 2 (CodTech Internship)
✅ Objective
Build a deep learning model for image classification using TensorFlow on the MNIST handwritten digit dataset.

📂 Dataset: MNIST Handwritten Digits
60,000 training images
10,000 test images
Grayscale format (28x28 pixels)
10 classes (digits from 0 to 9)
🧠 Model Architecture
Input: 28x28 → Flatten → Dense(128, ReLU) → Dense(10, Softmax)

Flatten: Converts image into 1D vector
Dense (128, ReLU): Fully connected layer
Dense (10, Softmax): Output layer for classification
📁 Project Files
File	Description
model.py	Trains the model, evaluates performance, and plots training graphs
visualize.py	Loads the saved model and visualizes predictions
mnist_model.h5	Trained model saved in HDF5 format
training_results.png	Accuracy and loss graph
▶️ How to Run
1️⃣ Train the Model:
python model.py
2️⃣ Visualize Predictions:

python visualize.py
⚙️ Dependencies Make sure you have the following Python libraries installed: tensorflow matplotlib

Install via pip:

pip install tensorflow matplotlib