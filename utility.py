import numpy as np


class AccuracyMetric:
    """Class computing accuracy of predicted samples.
    Accuracy is expressed as the ratio: correct_guesses / all_guesses.
    """

    @staticmethod
    def metric_value(x, y):
        """Metric computes ration of correct guesses to all guesses."""
        return (sum(np.all(x == y, axis=1) * 1) / len(x)) * 100


class Sigmoid:
    """Class representing sigmoid activation function.
    Sigmoid class implements popular logistic function called sigmoid function.
    Sigmoid function for argument z is specified as follows:
        1 + / (1 + e ^ (-z))
    """

    @staticmethod
    def activate(z):
        """Sigmoid activation function.
        Sigmoid activation function for arg z is given as:
            1 + / (1 + e ^ (-z))
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        """Sigmoid derivative.
        Sigmoid derivative for arg z is given as:
            sig(z) * (1 - sig(z))   where sig(z) is sigmoid activation function.
        """
        return Sigmoid.activate(z) * (1 - Sigmoid.activate(z))


class ReLu:
    """Class representing rectifier activation function.

    Rectifier is defined as passing non-negative function values.
    Generally speaking rectifier computes following function:
        f(x) = max(0, x)
    """

    @staticmethod
    def activate(z):

        return np.maximum(z, 0)

    @staticmethod
    def derivative(z):

        return np.array((z >= 0) * 1)


class SquaredSum:
    """Class representing squared sum cost function.
    Squared sum cost function is given as:
        sum(for all k) [a_k - y] ^ 2    where a_k is the prediction of y value.
    """

    @staticmethod
    def prediction_cost(a, y):
        """Squared sum prediction cost for single activation a.
        Squared sum cost function is given as:
            sum(for all k) [a_k - y_k] ^ 2  where a_k is the prediction of y_k value.
        """
        return np.sum((a - y) ** 2) / 2

    @staticmethod
    def derivative(a, y, z):
        """Squared sum derivative with respect to last neural network layer.
        Squared sum derivative is given as:
            (a - y) * sig_d(z)  where sig_d is sigmoid derivative of last layer weighted input.
        """
        return (a - y) * Sigmoid.derivative(z)


class CrossEntropy:
    """Cross entropy cost function.
    Cross entropy cost function is given as:
        sum(for all k) [ -{y_k * ln(a_k) + (1 - y_k) * ln(1 - a_k)}]    where a_k is prediction of y_k value
    """

    @staticmethod
    def prediction_cost(a, y):
        """Cross entropy prediction cost.
        Cross entropy cost function is given as:
            sum(for all k) [ -{y_k * ln(a_k) + (1 - y_k) * ln(1 - a_k)}]    where a_k is prediction of y_k value
        """
        return np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a)))

    @staticmethod
    def derivative(a, y, z):
        """Derivative of cross entropy cost function with respect to last layer weights.
        Cross function derivative is given as:
            a - y   where a is prediction of y
        """
        return a - y + (z * 0)


def convolve_2d(s, f):
    """Perform the convolution operation.
    Convolution is being applied between two matrix arguments of the function.
    Function returns matrix, which is the result of operation.

    :param s - source on which the convolution will be performed.
    :param f - filter which will be slided across the `s` parameter.

    :returns matrix representing the convolution.
    """
    f_x_, f_y_ = f.shape[0], f.shape[1]

    result_dim = (s.shape[0] - f_x_ + 1, s.shape[1] - f_y_ + 1)
    result = np.zeros(result_dim)

    for i in range(result_dim[0]):
        for j in range(result_dim[1]):
            result[i][j] = np.sum(f * s[i: i + f_x_, j: j + f_y_])

    return result


def full_convolve_2d(src, fil):
    """Performs special convolution type.
    Function performs operation known as the full convolution.
    This operation is being used while computing gradient in
    convolutional network for previous layer.
    For more information one should get familiar with backpropagation
    in convolutional layer.
    :param src - source on which the 'fil' will be slided.
    :param fil - filter which will be slided across the `src`.
    :returns full convolution result."""
    fil = np.rot90(fil, 2)

    result_dim = (src.shape[0] + fil.shape[0] - 1, src.shape[1] + fil.shape[1] - 1)
    result = np.zeros(result_dim)

    for j in range(result_dim[0]):
        for i in range(result_dim[1]):

            src_slice = src[max(0, j + 1 - fil.shape[0]): min(j + 1, src.shape[0]),
                            max(0, i + 1 - fil.shape[1]): min(i + 1, src.shape[1])]
            fil_slice = fil[max(fil.shape[0] - j - 1, 0): min(fil.shape[0], src.shape[0] + 1 - j),
                            max(fil.shape[1] - i - 1, 0): min(fil.shape[1], src.shape[1] + 1 - i)]

            result[j][i] = np.sum(src_slice * fil_slice)

    return result
