import numpy as np
from utility import Sigmoid, ReLu, convolve_2d

_GLOBAL_SEED_ = 1

np.random.seed(_GLOBAL_SEED_)


class FullyConnectedLayer:

    def __init__(self, in_, out_, reg_term=0.1, act_fn=Sigmoid):

        self.params = (in_, out_)
        self.weights = np.random.normal(loc=0, scale=0.012, size=(out_, in_))
        self.biases = np.random.normal(loc=0, scale=0.012, size=out_)
        self.reg_term = reg_term
        self.act_fn = act_fn

    def feedforward(self, x, z_arr=None, a_arr=None):

        if a_arr is None:
            a_arr = [x]
        if z_arr is None:
            z_arr = []

        z = np.dot(x, self.weights.T) + self.biases
        a = self.act_fn.activate(z)

        z_arr.append(z), a_arr.append(a)

        return a

    def backpropagation(self, layer_z, next_l_weights, prev_l_act, delta):

        z_d = self.act_fn.derivative(layer_z)
        delta = np.dot(next_l_weights.T, np.sum(delta, axis=0)) * z_d

        return np.dot(delta.T, prev_l_act), np.sum(delta, axis=0), delta

    def weights_cost(self):

        return self.reg_term * sum(self.weights ** 2)


class ConvolutionalPoolLayer:

    def __init__(self, in_s, filter_s, pool_s=(2, 2), act_fn=ReLu):

        self.in_size = in_s
        self.weight = [np.random.normal(loc=0, scale=0.12, size=i) for i in filter_s]
        self.biases = [np.random.normal(loc=0, scale=0.12, size=1)] * len(filter_s)
        self.pool_layer = MaxPoolLayer(pool_s)
        self.act_fn = act_fn

    def feedforward(self, a, z_arr=None, a_arr=None):

        if z_arr is None:
            z_arr = []
        if a_arr is None:
            a_arr = [a]

        filter_results, pool_results = [], []

        # Performing image feature activations for each filter.
        for f, b in zip(self.weight, self.biases):
            filter_results.append(self.act_fn.activate(convolve_2d(a, f) + b))

        # Performing pooling of the results for each filter.
        for result in filter_results:
            pool_results.append(self.pool_layer.feedforward(result))

        # Performing filter outputs concatenation.
        output = np.reshape(pool_results[0], [np.size(pool_results[0])])

        for i in pool_results[1:]:
            output = np.concatenate((output, np.reshape(i, [np.size(i)])), 0)

        z_arr.append(filter_results), a_arr.append(pool_results)

        return output


class MaxPoolLayer:

    def __init__(self, pool_s=(2, 2)):
        self.pool_s = pool_s

    def feedforward(self, x):

        pool = np.zeros((int((x.shape[0] + x.shape[0] % self.pool_s[0]) / self.pool_s[0]),
                         int((x.shape[1] + x.shape[1] % self.pool_s[0]) / self.pool_s[1])),
                        dtype=np.float32)

        for i in range(0, x.shape[0], self.pool_s[0]):
            for j in range(0, x.shape[1], self.pool_s[1]):
                pool[int(i / self.pool_s[0]), int(j / self.pool_s[1])] = \
                    np.amax(x[i: i + self.pool_s[0], j: j + self.pool_s[1]])

        return pool
