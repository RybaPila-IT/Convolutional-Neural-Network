from network import FullyConnectedLayer, ConvolutionalPoolLayer
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # a = 5
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    layer1 = ConvolutionalPoolLayer((28, 28), [(10, 10), (4, 4), (5, 5)])
    layer2 = FullyConnectedLayer(413, 10)
    print(layer2.pass_through(layer1.pass_through(x_train[0])))
