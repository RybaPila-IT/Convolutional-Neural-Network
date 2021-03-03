import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from layers import FullyConnectedLayer, ConvolutionalPoolLayer
from network import NeuralNetwork
import numpy as np

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    network = NeuralNetwork([FullyConnectedLayer(784, 100),
                             FullyConnectedLayer(100, 10)])

    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    training_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (arr == y_) * 1)
                     for x_, y_ in zip(x_train[:50000], y_train[:50000])]
    validation_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (arr == y_) * 1)
                       for x_, y_ in zip(x_train[50000:], y_train[50000:])]
    test_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (arr == y_) * 1)
                 for x_, y_ in zip(x_test, y_test)]

    network.sgd(training_data, test_data=validation_data)
