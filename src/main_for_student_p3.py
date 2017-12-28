import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def relu(a):
    return np.maximum(0.0, a)


def softmax(a):
    a = np.exp(a)
    return a/np.sum(a, axis=0)


def d_relu(a0):
    a = np.zeros(a0.shape)
    a[a0 > 0] = 1
    return a


def data_preprocess():
    """Data preprocessing: normalize all pixels to [0,1) by dividing 256"""
    global train_images, train_labels, test_images, test_labels
    train_images = (train_images / 256).T
    test_images = (test_images / 256).T
    train_labels_onehot = np.zeros((10, 10000))
    train_labels_onehot[train_labels.astype(int), np.arange(10000)] = 1
    test_labels_onehot = np.zeros((10, 1000))
    test_labels_onehot[test_labels.astype(int), np.arange(1000)] = 1
    return train_labels_onehot, test_labels_onehot


def weight_biase_init():
    """Weight initialization: Xavier"""
    sizes = [784, 300, 100, 10]
    border = [np.sqrt(6)/np.sqrt(x+y) for x, y in zip(sizes[:-1], sizes[1:])]
    weights = [np.random.uniform(-i, i, (y, x)) for i, x, y in zip(border, sizes[:-1], sizes[1:])]
    biases = [np.zeros((x, 1)) for x in sizes[1:]]
    return weights, biases


def train_neuralnetwork():
    """Training of neural network"""
    loss_list = np.zeros(EPOCH)
    accuracy_list = np.zeros(EPOCH)

    for epoch in range(0, EPOCH):
        for i in range(0, interation):
            learning_rate = 0.1
            if epoch >= 50:
                learning_rate = 0.01

            # Forward propagation
            x = train_images[:, 100 * i: 100 * (i + 1)]
            z1 = relu(np.dot(weights[0], x) + biases[0])
            z2 = relu(np.dot(weights[1], z1) + biases[1])
            y = softmax(np.dot(weights[2], z2) + biases[2])
            loss = -np.sum(np.log(y)[train_labels[100 * i: 100 * (i + 1)], np.arange(100)]) / 100

            # Back propagation, Gradient update
            delta_3 = y - train_labels_onehot[:,100*i: 100*(i+1)]
            delta_2 = np.dot(weights[2].T, delta_3)*d_relu(z2)
            delta_1 = np.dot(weights[1].T, delta_2)*d_relu(z1)
            weights[2] -= learning_rate*(np.dot(delta_3, z2.T)/100 + 0.0005*weights[2])
            biases[2] -= learning_rate*(np.sum(delta_3, axis=1, keepdims=True)/100)
            weights[1] -= learning_rate*(np.dot(delta_2, z1.T)/100 + 0.0005*weights[1])
            biases[1] -= learning_rate*(np.sum(delta_2, axis=1, keepdims=True)/100)
            weights[0] -= learning_rate * (np.dot(delta_1, x.T) / 100 + 0.0005 * weights[0])
            biases[0] -= learning_rate * (np.sum(delta_1, axis=1, keepdims=True) / 100)

        # Testing for accuracy
        z1_test = relu(np.dot(weights[0], test_images) + biases[0])
        z2_test = relu(np.dot(weights[1], z1_test) + biases[1])
        y_test = softmax(np.dot(weights[2], z2_test) + biases[2])
        accuracy = sum(np.argmax(test_labels_onehot, axis=0) == np.argmax(y_test, axis=0))/test_images.shape[1]
        loss_list[epoch] = loss
        accuracy_list[epoch] = accuracy

    return loss_list, accuracy_list


def plot():
    """Plot the loss, accuracy """
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(211)
    ax1.plot(np.arange(100), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    ax2 = plt.subplot(212)
    ax2.plot(np.arange(100), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.tight_layout()
    plt.savefig('figure.pdf', dbi=300)


if __name__ == '__main__':
    EPOCH = 100
    batchsize = 100
    interation = 100

    # Read all data from .pkl
    (train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),encoding='latin1')
    # Data preprocessing
    train_labels_onehot, test_labels_onehot = data_preprocess()
    # Weight initialization
    weights, biases = weight_biase_init()
    # Training of neural network
    loss_list, accuracy_list = train_neuralnetwork()
    # Plot the loss, accuracy
    plot()