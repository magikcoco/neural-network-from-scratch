import numpy as np  # for math
import pandas as pd  # for importing dataset
from matplotlib import pyplot as plt  # for plotting data

# importing and organizing the data
raw_data = pd.read_csv("mnist-csv/train.csv")  # training data
raw_data = np.array(raw_data)  # want numpy array instead of pandas dataframe
m, n = raw_data.shape  # dimensions of the data, m is rows and n is features+1
np.random.shuffle(raw_data)  # shuffle the data randomly around
data_dev = raw_data[0:1000].T  # transpose the first 1000 data
x_dev = data_dev[1:n]
y_dev = data_dev[0]
x_dev = x_dev / 255.
training_data = raw_data[1000:m].T  # transpose the last piece of data
x_train = training_data[1:n]
y_train = training_data[0]
x_train = x_train / 255.
_, m_train = x_train.shape


def initialize_parameters():
    w1 = np.random.rand(10, 784) - 0.5  # random values between 0 and 1
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5  # random values between 0 and 1
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def forward_propagation(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def one_hot(y):
    # for each row, go to colum in y and set it to one
    one_hot_y = np.zeros((y.size, y.max() + 1))  # array of zeroes with tuple of its size
    one_hot_y[np.arange(y.size), y] = 1  # np.arange(y.size) creates array that is a range from 0 to m
    one_hot_y = one_hot_y.T
    return one_hot_y


def derivative_relu(z):
    return z > 0


def backward_propagation(z1, a1, z2, a2, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * derivative_relu(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = initialize_parameters()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i % 10) == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print("Accuracy: ", get_accuracy(predictions, y))
    return w1, b1, w2, b2


w1, b1, w2, b2 = gradient_descent(x_train, y_train, 500, 0.1)
