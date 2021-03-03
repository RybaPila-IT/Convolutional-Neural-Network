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
