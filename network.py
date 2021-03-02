import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.activations import sigmoid

_GLOBAL_SEED_ = 1


class FullyConnectedLayer:

    def __init__(self, in_, out_, act_fn=sigmoid):

        self.params = (in_, out_)
        self.weights = tf.random.normal([out_, in_], seed=_GLOBAL_SEED_)
        self.bias = tf.random.normal([out_], seed=_GLOBAL_SEED_)
        self.act_fn = act_fn

    def pass_through(self, x, z_arr=None, a_arr=None):

        if a_arr is None:
            a_arr = []
        if z_arr is None:
            z_arr = []

        z = tf.tensordot(tf.cast(x, dtype=tf.float32), tf.transpose(self.weights), axes=1) \
            + self.bias
        a = self.act_fn(z)

        z_arr.append(z), a_arr.append(a)

        return a


class ConvolutionalPoolLayer:

    def __init__(self, in_s, filter_s, pool_s=(2, 2), act_fn=sigmoid):

        self.in_size = in_s
        self.weight = [tf.random.normal(i, seed=_GLOBAL_SEED_) for i in filter_s]
        self.biases = [tf.random.normal([1], seed=_GLOBAL_SEED_)] * len(filter_s)
        self.pool_s = pool_s
        self.act_fn = act_fn

    def pass_through(self, x):

        filter_results, pool_results = [], []

        # Performing image feature activations for each filter.
        for f, b in zip(self.weight, self.biases):
            result_dim = (x.shape[0] - f.shape[0] + 1, x.shape[1] - f.shape[1] + 1)
            result = np.zeros(shape=result_dim)

            for i in range(result_dim[0]):
                for j in range(result_dim[1]):
                    z = tf.reduce_sum(f * x[i : i + f.shape[0], j : j + f.shape[1]]) + b
                    a = self.act_fn(z)
                    result[i][j] = a

            filter_results.append(result)

        # Performing pooling of the results for each filter.
        for result in filter_results:
            pool = np.zeros((int((result.shape[0] + result.shape[0] % self.pool_s[0]) / self.pool_s[0]),
                             int((result.shape[1] + result.shape[1] % self.pool_s[0]) / self.pool_s[1])),
                            dtype=np.float32)

            for i in range(0, result.shape[0], self.pool_s[0]):
                for j in range(0, result.shape[1], self.pool_s[1]):
                    pool[int(i / self.pool_s[0]), int(j / self.pool_s[1])] = \
                        tf.reduce_max(result[i : i + self.pool_s[0], j : j + self.pool_s[1]])

            pool_results.append(pool)

        # Performing filter outputs concatenation.
        output = tf.reshape(pool_results[0], [tf.size(pool_results[0])])

        for i in pool_results[1:]:
            output = tf.concat([output, tf.reshape(i, [tf.size(i)])], 0)

        return output
