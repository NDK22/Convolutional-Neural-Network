# Convolutional Neural Network (CNN) with TensorFlow and Keras

**Author:** [Your Name]
**Date:** [Date]
**Description:** This repository contains Python code for building and training a Convolutional Neural Network (CNN) using TensorFlow and Keras.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Usage](#usage)
4. [API Documentation](#api-documentation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Model Loading and Saving](#model-loading-and-saving)
8. [Customization](#customization)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

Convolutional Neural Networks (CNNs) are a class of deep neural networks often used for image classification and recognition tasks. This repository provides a flexible and customizable CNN implementation using TensorFlow and Keras.

## Features

- **Input Layer**: Add an input layer with customizable shape.
- **Dense Layer**: Append dense layers with various activation functions.
- **Conv2D Layer**: Add convolutional layers with options for kernel size, padding, and activation functions.
- **MaxPooling2D Layer**: Incorporate max-pooling layers.
- **Flatten Layer**: Flatten the output for fully connected layers.
- **Customization**: Set layer training flags, load pre-trained models, and customize loss functions and metrics.
- **Training**: Train the CNN with specified hyperparameters.
- **Evaluation**: Evaluate the model's performance.
- **Model Loading and Saving**: Save and load model architectures and weights.
- **TensorFlow and Keras**: Utilizes TensorFlow and Keras for deep learning capabilities.

## Usage

Here's a basic example of how to use the CNN class:

```python
# Initialize the CNN
cnn = CNN()

# Add an input layer
cnn.add_input_layer(shape=(64, 64, 3), name="input_layer")

# Append dense and convolutional layers
cnn.append_dense_layer(num_nodes=128, activation="relu", name="dense_1")
cnn.append_conv2d_layer(num_of_filters=32, kernel_size=3, activation="relu", name="conv2d_1")
cnn.append_maxpooling2d_layer(pool_size=2, name="maxpooling2d_1")
cnn.append_flatten_layer(name="flatten")

# Set loss, optimizer, and metrics
cnn.set_loss_function(loss="SparseCategoricalCrossentropy")
cnn.set_optimizer(optimizer="SGD", learning_rate=0.001)
cnn.set_metric(metric="accuracy")

# Train the model
loss_list = cnn.train(X_train, y_train, batch_size=32, num_epochs=10)

# Evaluate the model
loss, accuracy = cnn.evaluate(X_test, y_test)

# Save the trained model
cnn.save_model("trained_cnn_model.h5")

API Documentation
The CNN class offers a set of methods to customize, train, and evaluate your model. Refer to the API Documentation section in the code for detailed explanations of each method and its parameters.

Training
To train the model, use the train method with your training data and specified hyperparameters. The method returns a list of loss values, which can be helpful for monitoring training progress.

Evaluation
Evaluate the model's performance using the evaluate method with your test data. The method returns the loss and metric values (e.g., accuracy) to assess how well the model performs on unseen data.

Model Loading and Saving
You can load pre-trained models or save your trained model architecture and weights using the load_a_model and save_model methods.

Customization
Customize your CNN by adding layers, setting layer training flags, choosing loss functions, optimizers, and metrics. Refer to the code and API Documentation for customization options.

Contributing
Feel free to contribute to this project by forking the repository and submitting pull requests.
