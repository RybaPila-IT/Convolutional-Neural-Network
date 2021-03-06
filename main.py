import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from layers import FullyConnectedLayer, ConvolutionalPoolLayer, FlattenLayer
from network import NeuralNetwork
from PIL import Image
import numpy as np


def show_image(image_data):

    size = (512, 512)
    image = Image.fromarray(obj=image_data.astype(np.uint8), mode='L')
    image = image.resize(size=size)
    image.show()


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    network = NeuralNetwork([FullyConnectedLayer(784, 100),
                             FullyConnectedLayer(100, 10)])

    pred_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    training_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (pred_arr == y_) * 1)
                     for x_, y_ in zip(x_train[:50000], y_train[:50000])]
    validation_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (pred_arr == y_) * 1)
                       for x_, y_ in zip(x_train[50000:], y_train[50000:])]
    test_data = [(np.reshape(x_, np.size(x_)).astype(np.float32) / 255, (pred_arr == y_) * 1)
                 for x_, y_ in zip(x_test, y_test)]

    # network.sgd(training_data, epochs=15, test_data=validation_data)

    network2 = NeuralNetwork([ConvolutionalPoolLayer((28, 28), [(3, 3), (3, 3), (5, 5)]),
                              FullyConnectedLayer(482, 10)])

    print(network2.gradient_descent_step(training_data[0:1], eta=0.01))
    # print(network2.backpropagation(x_train[0], training_data[0][1]))

    layer = FlattenLayer([(2, 2), (3, 3)])
    list = [np.array([[1, 2],
                      [4, 5]]),
            np.array([[3, 4, 5],
                      [10, 9, 18],
                      [0, 8, 7]])]

    #output = layer.feedforward(list)
    #print(layer.backpropagation(output))
