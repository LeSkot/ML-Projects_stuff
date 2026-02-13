> CIFAR-10 Image Classification (CNN)

> Overview
This project implements a Convolutional Neural Network (CNN)
to classify images from the CIFAR-10 dataset into 10 categories.

The project was built while studying A-levels, with the goal of
understanding core deep learning concepts rather than achieving
state-of-the-art performance.

> Dataset
CIFAR-10 consists of 60,000 color images (32×32 pixels) across
10 classes such as airplanes, cars, animals, and ships.

- 50,000 training images
- 10,000 test images

> Model
The model is a simple CNN built using TensorFlow / Keras and includes:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected (Dense) layers
- Softmax output for multi-class classification

Images are normalized by scaling pixel values to the range [0, 1].

> Training
- Optimizer: Adam
- Loss function: Categorical Cross entropy
- Evaluation metric: Accuracy

The model is trained and evaluated on the CIFAR-10 dataset to
observe learning behavior and generalization.

> What I Learned
- How CNNs extract spatial features from images
- The importance of normalization for training stability
- Overfitting and validation performance
- Structuring a basic deep learning training pipeline

## Notes
This project prioritizes clarity, correctness, and learning
fundamentals over maximum accuracy.

