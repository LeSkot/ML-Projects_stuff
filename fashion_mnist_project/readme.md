# Fashion-MNIST Image Classification (CNN)

## Overview
This project uses a Convolutional Neural Network (CNN) to classify
grayscale clothing images from the Fashion-MNIST dataset.

The goal of this project is to build intuition around image-based
classification and CNN behavior using a simpler dataset.

## Dataset
Fashion-MNIST consists of 28×28 grayscale images across 10 clothing
categories such as T-shirts, trousers, shoes, and bags.

- 60,000 training images
- 10,000 test images

## Model
The CNN is implemented using Keras and includes:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers for classification
- Softmax output layer

Input images are normalized before training.

## Training
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Evaluation metric: Accuracy

The model is trained on the training set and evaluated on unseen
test data.

## What I Learned
- Differences between grayscale and RGB image pipelines
- CNN architecture design for simple datasets
- Label encoding and loss function choices
- Model evaluation and interpretation of accuracy

## Notes
This project focuses on understanding deep learning fundamentals
and building clean, readable ML code while studying A-levels.

