import matplotlib.pyplot as plt
import numpy as np
from utility import CrossEntropy, AccuracyMetric

_GLOBAL_SEED_ = 1


class NeuralNetwork:

    def __init__(self, layers, cost_fn=CrossEntropy, metric=AccuracyMetric):
        self.layers = layers
        self.cost_fn = cost_fn
        self.metric = metric

    def feedforward(self, a):
        z_arr, a_arr = [], [a]

        for layer in self.layers:
            a = layer.feedforward(a, z_arr, a_arr)

        return a, z_arr, a_arr

    def predict(self, x):

        feed_result = self.feedforward(x)[0]

        return (feed_result == np.max(feed_result)) * 1

    def backpropagation(self, x, y):

        result = self.feedforward(x)
        z_arr, a_arr = result[1], result[2]

        # Last layer error calculation.
        # NOTE: There is the assumption that the last layer is the
        # fully connected layer.
        delta = self.cost_fn.derivative(a_arr[-1], y, z_arr[-1])
        last_lay_err_b = delta
        last_lay_err_w = np.dot(np.array([delta]).T, np.array([a_arr[-2]]))

        # Saving layer updates by hand.
        self.layers[-1].biases_u += last_lay_err_b
        self.layers[-1].weights_u += last_lay_err_w

        # Performing chain rule.
        for l_ in range(2, len(self.layers) + 1):
            delta = self.layers[-l_].backpropagation(z_arr[-l_], self.layers[-l_ + 1].weights, a_arr[-l_ - 1], delta)


    def gradient_descent_step(self, batch, eta):

        for x, y in batch:
            self.backpropagation(x, y)

        for layer in self.layers:
            layer.update(eta)

    def sgd(self, training_data, epochs=30, batch_size=10, eta=0.01, test_data=None):

        costs, accuracies = [], []

        for e in range(epochs):
            np.random.shuffle(training_data)
            batches = [np.array(training_data[k:k + batch_size], dtype=object)
                       for k in range(0, len(training_data), batch_size)]

            for batch in batches:
                self.gradient_descent_step(batch, eta)

            cost = self.cost_fn.prediction_cost(self.feedforward([x_ for x_, y_ in training_data])[0],
                                                np.array([y_ for x_, y_ in training_data]))
            costs.append(cost)
            print('Epoch {} ended with cost: {:.3f}'.format(e + 1, cost))

            if test_data:
                accuracies.append(self.metric.metric_value(self.predict([x_ for x_, y_ in test_data]),
                                                           [y_ for x_, y_ in test_data]))

        self.plot(costs, 'Costs graph for eta: {:.3f}'.format(eta), 'Epochs', 'Cost', 'ro-')

        if len(accuracies) > 0:
            self.plot(accuracies, 'Accuracies graph for eta: {:.3f}'.format(eta), 'Epochs', 'Accuracy', 'g*-')

    @staticmethod
    def plot(y, title, x_label, y_label, styling):

        plt.plot(y, styling)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
