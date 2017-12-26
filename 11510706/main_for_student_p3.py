import copy
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Definition of functions and parameters


def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    return out


def relu_forward(x):
    out = np.maximum(0, x)
    return out


def softmax_forward(a):
    logits_exp = np.exp(a)
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)


def affine_relu_forward(x, w, b):
    a = affine_forward(x, w, b)
    out = relu_forward(a)
    return out


def affine_softmax_forward(x, w, b):
    a = affine_forward(x, w, b)
    out = softmax_forward(a)
    return out


def softmax_loss(x, y):
    N = x.shape[0]
    loss = -np.sum(np.log(x[np.arange(N), y])) / N
    return loss


def regularization_L2_softmax_loss(reg_lambda, weight1, weight2, weight3):
    weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
    weight3_loss = 0.5 * reg_lambda * np.sum(weight3 * weight3)
    return weight1_loss + weight2_loss + weight3_loss


def accuracy(predictions, labels):
    preds_correct_boolean = np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy


# for example
EPOCH = 100
loss_list = []
train_list = []
accuracy_list = []

# Read all data from .pkl
(train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'), encoding='latin1')

### 1. Data preprocessing: normalize all pixels to [0,1) by dividing 256
preprocessed_train_images = copy.deepcopy(train_images).astype(float)
preprocessed_test_images = copy.deepcopy(test_images).astype(float)
for i in range(preprocessed_train_images.shape[0]):
    preprocessed_train_images[i] = preprocessed_train_images[i]/256

train_labels_onehot = np.zeros((train_labels.shape[0], 10)).astype(int)
train_labels_onehot[np.arange(len(train_labels)), train_labels.astype(int)] = 1
test_labels_onehot = np.zeros((test_labels.shape[0], 10)).astype(int)
test_labels_onehot[np.arange(len(test_labels)), test_labels.astype(int)] = 1

for i in range(preprocessed_test_images.shape[0]):
    preprocessed_test_images[i] = preprocessed_test_images[i]/256

### 2. Weight initialization: Xavier
w1 = np.random.rand(784, 300) * 2 * np.sqrt(6) / np.sqrt(784 + 300) - np.sqrt(6)/np.sqrt(784 + 300)
w2 = np.random.rand(300, 100) * 2 * np.sqrt(6) / np.sqrt(300 + 100) - np.sqrt(6)/np.sqrt(300 + 100)
w3 = np.random.rand(100, 10) * 2 * np.sqrt(6) / np.sqrt(100 + 10) - np.sqrt(6)/np.sqrt(100 + 10)
b1 = np.zeros(300)
b2 = np.zeros(100)
b3 = np.zeros(10)

### 3. training of neural network
for epoch in range(EPOCH):


    for iteration in range(100):
        x = preprocessed_train_images[iteration * 100:(iteration + 1) * 100]
        # Forward propagation
        hidden_layer1 = affine_relu_forward(x, w1, b1)
        hidden_layer2 = affine_relu_forward(hidden_layer1, w2, b2)
        output_probs = affine_softmax_forward(hidden_layer2, w3, b3)

        # Back propagation
        output_error_signal = output_probs - train_labels_onehot[iteration*100:(iteration+1)*100]
        error_signal_hidden2 = np.dot(output_error_signal, w3.T)
        error_signal_hidden2[hidden_layer2 == 0] = 0

        error_signal_hidden1 = np.dot(error_signal_hidden2, w2.T)
        error_signal_hidden1[hidden_layer1 == 0] = 0

        gradient_layer3_weights = np.dot(hidden_layer2.T, output_error_signal)/100
        gradient_layer3_bias = np.mean(output_error_signal, axis=0, keepdims=False)

        gradient_layer2_weights = np.dot(hidden_layer1.T, error_signal_hidden2)/100
        gradient_layer2_bias = np.mean(error_signal_hidden2, axis=0, keepdims=False)

        gradient_layer1_weights = np.dot(x.T, error_signal_hidden1)/100
        gradient_layer1_bias = np.mean(error_signal_hidden1, axis=0, keepdims=False)

        reg_lambda = 0.0005
        gradient_layer3_weights += reg_lambda * w3
        gradient_layer2_weights += reg_lambda * w2
        gradient_layer1_weights += reg_lambda * w1
        # Loss calculation
        loss = softmax_loss(output_probs, train_labels[iteration * 100:(iteration + 1) * 100])
        loss += regularization_L2_softmax_loss(reg_lambda, w1, w2, w3)
        # Gradient update
        learning_rate = 0.1
        if epoch >= 50:
            learning_rate = 0.01
        w1 -= learning_rate * gradient_layer1_weights
        b1 -= learning_rate * gradient_layer1_bias
        w2 -= learning_rate * gradient_layer2_weights
        b2 -= learning_rate * gradient_layer2_bias
        w3 -= learning_rate * gradient_layer3_weights
        b3 -= learning_rate * gradient_layer3_bias

    loss_list.append(loss)
    print('epoch ' + str(epoch) + ': ' + str(loss))

    hidden_layer1 = affine_relu_forward(preprocessed_train_images, w1, b1)
    hidden_layer2 = affine_relu_forward(hidden_layer1, w2, b2)
    output_probs = affine_softmax_forward(hidden_layer2, w3, b3)
    print('Train accuracy: {0}%'.format(accuracy(output_probs, train_labels_onehot)))
    train_list.append(accuracy(output_probs, train_labels_onehot))

    hidden_layer1 = affine_relu_forward(preprocessed_test_images, w1, b1)
    hidden_layer2 = affine_relu_forward(hidden_layer1, w2, b2)
    output_probs = affine_softmax_forward(hidden_layer2, w3, b3)
    print('Test accuracy: {0}%'.format(accuracy(output_probs, test_labels_onehot)))
    accuracy_list.append(accuracy(output_probs, test_labels_onehot))

### 4. Plot
# for example
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(211)
ax1.plot(np.array(range(EPOCH)), np.asarray(loss_list))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
ax2 = plt.subplot(212)
ax2.plot(np.array(range(EPOCH)), np.asarray(accuracy_list))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid()
plt.tight_layout()
plt.savefig('figure.pdf', dbi=300)
