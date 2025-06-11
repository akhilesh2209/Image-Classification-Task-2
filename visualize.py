import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

# Load and preprocess test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Predict
predictions = model.predict(x_test)

# Display first 5 predictions
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]
    color = 'green' if predicted_label == actual_label else 'red'
    plt.title(f"Predicted: {predicted_label} | Actual: {actual_label}", color=color)
    plt.axis('off')
    plt.show()
