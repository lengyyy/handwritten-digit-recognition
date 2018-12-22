# Handwritten-digit-recognition
The project 4 of CS303 Artificial Intelligence.

## Preliminaries
The idea is to take a large number of handwritten digits, known as training examples, and then develop a system which can learn from those training examples. In other words, the neural network uses the examples to automatically infer rules for recognizing handwritten digits. Furthermore, by increasing the number of training examples, the network can learn more about handwriting, and so improve its accuracy.In this project, I wrote a computer program implementing a neural network that learns to recognize handwritten digits. This short program can recognize digits with an accuracy over 95 percent.

## Methodology
#### Overview
The procedure:
1. Data processing
2. Weight initialization
3. Train of network
  * Forward propagation
  * Back propagation
  * Gradient update
  * Test for accuracy
4. Plot the loss, accuracy

#### Initialization — Xavier Initialization
#### Optimization — Back propagation
#### Activation Functions
- ReLU
- Softmax
#### Mini-Batch Gradient Descent

## Empirical Verification
The data used in this lab is from a famous dataset called MNIST, which is composed of many hand written digit images.

The following ﬁgure describes the loss and the test accuracy. The accuracy gets 94% when 15 epochs. After 100 epochs, the accuracy of the test set is more than 95%.

<img src="https://github.com/lengyyy/handwritten-digit-recognition/blob/master/report/pictures/experiment.png" width="750">


Details are shown in [report](https://github.com/lengyyy/handwritten-digit-recognition/blob/master/report/Report.pdf).
