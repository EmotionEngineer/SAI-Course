# 🌐 Introduction to Neural Networks

Neural Networks, modeled loosely on the human brain, are powerful tools in **Deep Learning**. They’ve revolutionized fields like computer vision, natural language processing, and robotics. In this guide, we'll walk through the structure, concepts, and practical code examples to illuminate how neural networks function.

---

## 🧠 What is a Neural Network?

A **Neural Network (NN)** is a set of algorithms modeled after the human brain that aims to recognize underlying relationships in data. Neural Networks consist of interconnected layers and nodes, capable of learning patterns and making decisions from data.

### 🔑 Key Concepts:
1. **Neuron (Node)**: The fundamental processing unit, which receives inputs, transforms them, and passes an output.
2. **Layer**: Layers of nodes, which transform data step-by-step:
   - **Input Layer**: Receives the raw data.
   - **Hidden Layers**: Transform data to detect patterns.
   - **Output Layer**: Provides the final output (e.g., a prediction).
3. **Weights**: Values that adjust the influence of an input on the neuron’s output.
4. **Bias**: A constant added to the input, helping with fine-tuning.
5. **Activation Function**: Introduces non-linearity, enabling the network to model complex patterns.

### Code Example: Simple Neuron Calculation
```python
import numpy as np

# Inputs (example: two features)
inputs = np.array([1.5, 2.0])

# Weights and bias
weights = np.array([0.5, -1.2])
bias = 0.7

# Calculating the neuron's output
output = np.dot(inputs, weights) + bias
print("Neuron Output:", output)
```

---

## 🌉 Architecture of a Neural Network

### Types of Neural Networks:
- **Feedforward Neural Networks (FNN)**: Data flows in one direction, from input to output.
- **Convolutional Neural Networks (CNN)**: Optimized for images, CNNs can detect features like edges and shapes.
- **Recurrent Neural Networks (RNN)**: Designed for sequential data, RNNs are useful for time series and language processing.

### Layers and Neurons
A **Dense (Fully Connected)** Neural Network connects each neuron in one layer to every neuron in the next. A **Multi-layer Perceptron (MLP)** contains one or more hidden layers, capable of handling non-linear tasks.

### Visual Example of Layers:
```plaintext
Input Layer ---> Hidden Layer 1 ---> Hidden Layer 2 ---> Output Layer
```

---

## 🔢 Forward Propagation: Making Predictions

**Forward Propagation** is the mechanism through which data passes through the network. Each neuron computes a **weighted sum** of its inputs, applies an activation function, and sends the output to the next layer.

1. **Weighted Sum**: Compute a sum of inputs and weights, then add the bias.
2. **Activation**: Apply an activation function (like ReLU or Sigmoid) to introduce non-linearity.
3. **Output Layer**: Produces the final output for prediction or classification.

### Code Example: Forward Propagation
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example inputs, weights, and bias for a simple network
inputs = np.array([0.6, 1.2])
weights = np.array([0.8, -0.5])
bias = 0.1

# Calculating weighted sum and applying the sigmoid activation
weighted_sum = np.dot(inputs, weights) + bias
output = sigmoid(weighted_sum)
print("Forward Propagation Output:", output)
```

---

## 🔄 Backpropagation: How Neural Networks Learn

**Backpropagation** adjusts weights to reduce error, enabling the network to learn from data. Through **gradient descent** (or variants), the network iteratively reduces prediction errors, refining its ability to make accurate predictions.

1. **Error Calculation**: Find the difference between the predicted and actual output.
2. **Gradient Calculation**: Compute the gradient of the loss function, indicating the error's sensitivity to each weight.
3. **Weight Update**: Adjust weights in the direction that minimizes the error.

### Example Loss Functions
- **Mean Squared Error (MSE)**: Common in regression tasks.

  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

- **Cross-Entropy**: Used in classification tasks.

  $$\text{Cross-Entropy} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

### Code Example: Gradient Descent Step
```python
# Learning rate
learning_rate = 0.01

# Example gradient for a single weight
gradient = -0.5  # Example value

# Update the weight
updated_weight = weights[0] - learning_rate * gradient
print("Updated Weight:", updated_weight)
```

---

## 🧩 Key Activation Functions

Activation functions introduce **non-linearity** to the model, making it possible to learn complex patterns. Below are some of the most common activation functions:

1. **Sigmoid**: Maps values between 0 and 1, ideal for probabilities.

   $$\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$
3. **ReLU (Rectified Linear Unit)**: Sets negative values to 0, allowing only positive values to pass.

   $$\text{ReLU}(z) = \max(0, z)$$
5. **Tanh**: Centers values between -1 and 1.

   $$\text{Tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

### Code Example: Activation Functions
```python
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Sample input
input_value = -0.5

# Applying ReLU and Tanh
print("ReLU Output:", relu(input_value))
print("Tanh Output:", tanh(input_value))
```

---

## 📉 Training a Neural Network

Training a neural network involves multiple **epochs** where the network:
1. **Initializes Weights**: Starts with random weights.
2. **Performs Forward Propagation**: Outputs predictions.
3. **Calculates Loss**: Measures prediction errors.
4. **Backpropagates the Error**: Updates weights to reduce error.
5. **Repeats the Process**: Iterates over the data multiple times, refining accuracy.

### Code Example: Training Loop
```python
# Dummy data for training example
inputs = np.array([0.8, -0.1])
weights = np.array([0.5, 0.2])
bias = 0.0
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Forward pass
    weighted_sum = np.dot(inputs, weights) + bias
    prediction = sigmoid(weighted_sum)

    # Loss calculation (dummy target of 0.5 for illustration)
    target = 0.5
    loss = (prediction - target) ** 2

    # Backpropagation (gradient for this example)
    gradient = 2 * (prediction - target) * prediction * (1 - prediction)  # Sigmoid derivative

    # Weight update
    weights -= learning_rate * gradient * inputs
    bias -= learning_rate * gradient

print("Final Weights:", weights)
print("Final Bias:", bias)
```

---

## 🚀 Applications of Neural Networks

Neural Networks have transformed many fields and industries:

- **Image Recognition**: Powering facial recognition, medical imaging, and self-driving cars.
- **Natural Language Processing (NLP)**: BERT, GPT, and other models enable machines to understand and generate human language.
- **Healthcare**: Neural networks assist in diagnosing diseases and predicting patient outcomes.
- **Finance**: Fraud detection, automated trading, and credit assessment rely on neural networks.

---

## 🧩 Neural Networks in Summary

Neural Networks have become essential for building intelligent applications that learn from data, transforming AI and many related fields. Through interconnected layers, forward and backward propagation, and non-linear activations, neural networks capture complex patterns and make high-level decisions.

### Quick Recap:
- **Forward Propagation** allows data to flow from input to output.
- **Backpropagation** and **Gradient Descent** adjust weights to minimize error.
- **Activation Functions** introduce non-linearity, enabling learning of complex patterns.
