import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from layers import FullyConnectedLayer, ConvolutionalPoolLayer
from network import NeuralNetwork
import numpy as np


def accuracy(net, test_data):
    correct = 0

    for x_, y_ in test_data:
        if np.array_equal(net.predict(x_), y_):
            correct += 1

    return correct / len(test_data)

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    network1 = NeuralNetwork([FullyConnectedLayer(784, 100),
                             FullyConnectedLayer(100, 10)])

    pred_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    training_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (pred_arr == y_) * 1)
                     for x_, y_ in zip(x_train[:400], y_train[:400])]
    validation_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (pred_arr == y_) * 1)
                       for x_, y_ in zip(x_train[59800:], y_train[59800:])]
    test_data_1 = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (pred_arr == y_) * 1)
                 for x_, y_ in zip(x_test[:900], y_test[:900])]

    network1.sgd(training_data, eta=0.028, epochs=15, validation_data=validation_data)

    training_data = [(x_ / 255, (pred_arr == y_) * 1) for x_, y_ in zip(x_train[:500], y_train[:500])]
    validation_data = [(x_ / 255, (pred_arr == y_) * 1) for x_, y_ in zip(x_train[59800:], y_train[59800:])]
    test_data_2 = [(x_ / 255, (pred_arr == y_) * 1) for x_, y_ in zip(x_test[:300], y_test[:300])]

    network2 = NeuralNetwork([ConvolutionalPoolLayer((28, 28), [(3, 3), (3, 3), (3, 3), (5, 5), (5, 5)]),
                              FullyConnectedLayer(482 + 169 + 144, 10)])

    network2.sgd(training_data, eta=0.0018, epochs=8, validation_data=validation_data)

    print('Regular network accuracy: {:.3f}'.format(accuracy(network1, test_data_1)))
    print('Convolutional network accuracy: {:.3f}'.format(accuracy(network2, test_data_2)))
