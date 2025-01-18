# Handwritten Digit Classification with CNNs

## Project Overview
This project involves building a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) using the **MNIST dataset**. The model predicts the digit present in an image with high accuracy, and it has been tested on custom handwritten digit images.

---

## Dataset
- **Name**: MNIST Handwritten Digit Dataset
- **Source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mnist)
- **Description**: The dataset consists of 60,000 training images and 10,000 test images of handwritten digits (28x28 grayscale).

---

## Model Architecture
1. **Convolutional Layers**:
   - Extract features from input images using filters.
2. **MaxPooling Layers**:
   - Downsample feature maps to reduce dimensions.
3. **Dropout Layers**:
   - Prevent overfitting by randomly deactivating neurons during training.
4. **Dense Layers**:
   - Fully connected layers for digit classification.
5. **Output Layer**:
   - 10 neurons with Softmax activation for multi-class classification (digits 0-9).

---

## Implementation Steps

### 1. Data Preparation
- **Normalization**: Pixel values were normalized to the range [0, 1].
- **Reshaping**: Data reshaped to include a channel dimension (28x28x1) for grayscale images.

### 2. Model Training
- **Optimizer**: Adam optimizer for adaptive learning.
- **Loss Function**: Sparse Categorical Crossentropy for multi-class classification.
- **Metrics**: Accuracy to evaluate performance.
- **Epochs**: 10 epochs with a batch size of 64.

### 3. Evaluation
- Achieved **99.22% accuracy** on the test dataset.
- Visualized training and validation accuracy/loss over epochs.

### 4. Testing with Custom Images
- A custom handwritten digit image was preprocessed (grayscale, resized, normalized).
- Model predicted the digit correctly with high confidence.

---

## Results
- **Training Accuracy**: ~98.80%
- **Validation Accuracy**: ~99.22%
- **Custom Image Prediction**: Successfully predicted the digit in a custom image.

---

## Visualization
1. **Training Accuracy and Loss**:
   - Accuracy and loss curves are plotted to analyze model performance over epochs.

2. **Custom Image Prediction**:
   - The custom image is displayed with the predicted digit.

---

## Applications
- **Digit Recognition**: Automated processing of handwritten documents.
- **Education**: Training tools for understanding deep learning.
- **Image Classification**: A foundational project for building advanced image recognition systems.
