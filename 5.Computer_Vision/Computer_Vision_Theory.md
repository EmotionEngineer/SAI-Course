# 📸 Introduction to Computer Vision

Computer Vision (CV) is a field of artificial intelligence focused on enabling machines to interpret and understand the visual world. CV drives applications from image recognition and object detection to more complex tasks like video analysis and scene understanding.

---

## 🔍 What is Computer Vision?

Computer Vision aims to enable machines to **analyze and understand visual data** (images and videos) by mimicking aspects of human visual processing. Techniques from **image processing**, **machine learning**, and **deep learning** all contribute to this field.

---

## 📏 Key Concepts in Computer Vision

### 1. **Image Representation**

Images are stored as matrices of pixels, with each pixel representing a part of the image:
   - **Grayscale Images**: Each pixel has a single intensity value (0–255).
   - **Color Images**: Each pixel contains three values for RGB (Red, Green, Blue) channels.
   - **Resolution**: Defined by the width and height (in pixels), indicating detail.

### 2. **Convolution**

Convolution is a core operation in CV, where a **kernel** (or filter) slides across the image, applying transformations to detect features:

$$(I * K)(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x+i, y+j) \cdot K(i, j)$$

Where:
- $I(x, y)$: Input image pixel value at location $(x, y)$
- $K(i, j)$: Kernel weight at location $(i, j)$

This operation highlights features like edges, shapes, or patterns, essential for neural networks to learn meaningful details.

### 3. **Feature Extraction**

Feature extraction involves identifying unique aspects of an image, such as edges, shapes, and textures, which can distinguish objects. Features are vital for tasks like **object detection** and **classification**.

### 4. **Object Detection and Recognition**

   - **Object Detection** locates and categorizes objects in an image, drawing bounding boxes around them.
   - **Recognition** identifies the object itself, often through a classification model (e.g., recognizing faces, animals, or specific objects).

### 5. **Segmentation**

Segmentation divides an image into segments or regions:
   - **Semantic Segmentation**: Labels each pixel based on the object class (e.g., "cat," "road").
   - **Instance Segmentation**: Distinguishes between instances of the same object class.

---

## 🔠 Neural Networks in Computer Vision

### Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)** are designed to handle image data. A CNN learns spatial hierarchies from basic edges in early layers to more complex structures in deeper layers.

#### CNN Layers and Their Roles

1. **Convolutional Layer**: Detects features by convolving filters over the image.
   - **ReLU Activation**: Applied to introduce non-linearity.
  
2. **Pooling Layer**: Reduces the spatial dimensions of feature maps, retaining essential information while minimizing computational load.

3. **Fully Connected Layer**: Combines learned features to make predictions.

#### CNN Formula

In a convolutional layer, each neuron's output can be defined as:

$$y = f\left(\sum_{i=1}^n w_i \cdot x_i + b\right)$$

Where:
- $y$: Output of the neuron,
- $w_i$: Weight of each input,
- $x_i$: Input value (pixel intensity),
- $b$: Bias,
- $f$: Activation function, e.g., ReLU.

---

## 🤖 Implementing a Simple CNN in Python (using TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define a simple CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()
```

---

## 🔄 Workflow of a Computer Vision Model

Building a CV model involves multiple stages:
1. **Data Collection**: Gather labeled images for training (e.g., animals, vehicles).
2. **Preprocessing**: Resize, normalize, and augment images for consistency.
3. **Feature Extraction**: Apply convolutional layers to learn key patterns.
4. **Model Training**: Train the CNN model using backpropagation.
5. **Evaluation**: Test the model on new data, optimizing for accuracy.
6. **Deployment**: Integrate the trained model into applications (often in real-time).

---

## 📐 Image Processing Techniques

Image preprocessing enhances images for analysis. Common techniques include:

- **Rescaling**: Adjusting image size to a standard resolution.
- **Normalization**: Scaling pixel values, often to a 0–1 range.
- **Blurring**: Reduces noise by averaging nearby pixels.
- **Edge Detection**: Highlights object outlines using algorithms like Sobel or Canny.

```python
import cv2
import matplotlib.pyplot as plt

# Load and preprocess an image
image = cv2.imread("sample_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image, (128, 128))

# Edge detection
edges = cv2.Canny(resized_image, 100, 200)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(resized_image)
plt.title("Resized Image")
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")
plt.show()
```

---

## 🔥 Key Computer Vision Models

Several foundational CNN models have revolutionized CV:

1. **LeNet-5**: Early CNN for digit recognition.
2. **AlexNet**: Demonstrated CNN's power in the 2012 ImageNet competition.
3. **VGGNet**: Deep network with stacked convolution layers.
4. **ResNet**: Introduced residual connections, enabling deeper networks.
5. **YOLO (You Only Look Once)**: Real-time object detection model that processes images in one pass.

---

## 🧩 Computer Vision in Summary

Computer Vision enables machines to "see" and interpret visual data. Through **image processing techniques** and **neural networks**, CV models learn complex patterns, making high-level predictions possible in real time.

### Quick Recap:
- **Image Processing** refines raw images for analysis.
- **Convolutional Neural Networks** excel in identifying image features.
- **Computer Vision Applications** span healthcare, retail, autonomous driving, and more.
