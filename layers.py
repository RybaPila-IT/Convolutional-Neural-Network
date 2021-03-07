import math
import numpy as np
from utility import Sigmoid, ReLu, convolve_2d, full_convolve_2d

_GLOBAL_SEED_ = 1

np.random.seed(_GLOBAL_SEED_)


class FullyConnectedLayer:

    def __init__(self, in_, out_, reg_term=0.1, act_fn=Sigmoid):

        self.weights_sizes = [(in_, out_)]
        self.biases_sizes = [out_]
        self.weights_u = np.zeros((out_, in_))
        self.weights = np.random.normal(loc=0, scale=0.012, size=(out_, in_))
        self.biases = np.random.normal(loc=0, scale=0.012, size=out_)
        self.biases_u = np.zeros(out_)
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
        delta = np.dot(next_l_weights.T, delta) * z_d

        self.biases_u += delta
        self.weights_u += np.dot(np.array([delta]).T, np.array([prev_l_act]))

        return delta

    def update(self, eta):

        self.biases -= self.biases_u * eta
        self.weights -= self.weights_u * eta
        self.biases_u *= 0
        self.weights_u *= 0

    def weights_cost(self):

        return self.reg_term * sum(self.weights ** 2)


class ConvolutionalPoolLayer:

    def __init__(self, in_s, filter_s, pool_s=(2, 2), flattens=True):

        self.in_size = in_s
        self.weights_u = [np.zeros(i) for i in filter_s]
        self.biases_u = [np.zeros(1)] * len(filter_s)
        self.weights = [np.random.normal(loc=0, scale=0.12, size=i) for i in filter_s]
        self.biases = [np.random.normal(loc=0, scale=0.12, size=1)] * len(filter_s)
        self.pool_layer = MaxPoolLayer(pool_s)
        self.flatten_layer = FlattenLayer([(int(math.ceil((in_s[0] - f_s[0] + 1) / pool_s[0])),
                                            int(math.ceil((in_s[1] - f_s[1]) + 1) / pool_s[1]))
                                           for f_s in filter_s])
        self.act_fn = ReLu
        self.flattens = flattens

    def feedforward(self, a, z_arr=None, a_arr=None):

        if z_arr is None:
            z_arr = []
        if a_arr is None:
            a_arr = [a]

        filter_results, pool_results = [], []

        # Performing image feature activations for each filter.
        for f, b in zip(self.weights, self.biases):
            filter_results.append(self.act_fn.activate(convolve_2d(a, f) + b))

        # Performing pooling of the results for each filter.
        for result in filter_results:
            pool_results.append(self.pool_layer.feedforward(result))

        # Performing filter outputs flattening
        output = self.flatten_layer.feedforward(pool_results) if self.flattens \
            else pool_results

        z_arr.append(filter_results), a_arr.append(output)

        return output

    def backpropagation(self, layer_z, next_l_weights=None, prev_l_act=None, delta=None):

        filter_pool_deltas = self.flatten_layer.backpropagation(delta, next_l_weights) if self.flattens \
            else delta

        filter_deltas = []

        for idx, f in enumerate(self.weights):
            filter_deltas.append(self.pool_layer.backpropagation(layer_z[idx], filter_pool_deltas[idx]))

        for idx, del_t in enumerate(filter_deltas):
            self.weights_u[idx] += convolve_2d(prev_l_act, del_t)
            self.biases_u[idx] += np.sum(del_t)

        # Section commented because of huge slow down
        # due to the fact of computations involved with
        # full_convolve_2d function

        # delta = []

        # for filter, del_t in zip(self.weights, filter_deltas):
          #  delta.append(full_convolve_2d(del_t, filter))

        return delta

    def update(self, eta):

        for i in range(len(self.weights_u)):
            self.weights[i] -= eta * self.weights_u[i]
            self.biases[i] -= eta * self.biases_u[i]
            self.biases_u[i] *= 0
            self.weights_u[i] *= 0

class MaxPoolLayer:

    def __init__(self, pool_s=(2, 2)):
        self.pool_s = pool_s

    def feedforward(self, x):

        pool = np.zeros((math.ceil(x.shape[0] / self.pool_s[0]),
                         math.ceil(x.shape[1] / self.pool_s[1])),
                        dtype=np.float32)

        for i in range(0, x.shape[0], self.pool_s[0]):
            for j in range(0, x.shape[1], self.pool_s[1]):
                pool[int(i / self.pool_s[0]), int(j / self.pool_s[1])] = \
                    np.amax(x[i: i + self.pool_s[0], j: j + self.pool_s[1]])

        return pool

    def backpropagation(self, prev_a, e):

        result = np.zeros(prev_a.shape)

        _y_s = self.pool_s[0]
        _x_s = self.pool_s[1]

        for i in range(e.shape[0]):
            for j in range(e.shape[0]):
                result[i * _y_s: (i + 1) * _y_s, j * _x_s: (j + 1) * _x_s] = \
                    (prev_a[i * _y_s: (i + 1) * _y_s, j * _x_s: (j + 1) * _x_s] ==
                     np.amax(prev_a[i * _y_s: (i + 1) * _y_s, j * _x_s: (j + 1) * _x_s])) * \
                    (0 if e[i][j] == 0 else e[i][j])

        return result


class FlattenLayer:

    def __init__(self, in_s):
        self.in_s = in_s

    @staticmethod
    def feedforward(inp):

        flat = np.array([])

        for i in inp:
            flat = np.concatenate((flat, np.reshape(i, [np.size(i)])), 0)

        return flat

    def backpropagation(self, inp, next_l_weights):

        inp = np.dot(next_l_weights.T, inp)

        output = []
        offset = 0

        for shape in self.in_s:
            size = shape[0] * shape[1]
            output.append(np.reshape(inp[offset: offset + size], shape))
            offset += size

        return output
