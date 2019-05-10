from digit_recognizer.util import *


class NeuralNetwork:
    def __init__(self, learning_rate):
        self.x = None
        self.a_1 = None
        self.a_2 = None
        self.learning_rate = learning_rate
        self.weights1 = set_random_weights(np.zeros((784, 28)))
        self.weights2 = set_random_weights(np.zeros((28, 11)))
        pass

    def forward(self, x):
        self.x = x
        self.a_1 = sigmoid(np.dot(x, self.weights1))
        self.a_2 = sigmoid(np.dot(self.a_1, self.weights2))
        return self.a_2

    def backward(self, y):
        d_a_2 = 2 * (y - self.a_2) * d_sigmoid(self.a_2)
        d_a_1 = np.dot(d_a_2, self.weights2.T) * d_sigmoid(self.a_1)

        d_weights2 = np.dot(np.transpose(np.expand_dims(self.a_1, axis=0)), np.expand_dims(d_a_2, axis=0))
        d_weights1 = np.dot(np.transpose(np.expand_dims(self.x, axis=0)), np.expand_dims(d_a_1, axis=0))

        # update the weights with the derivative (slope) of the loss function
        self.weights2 += self.learning_rate * d_weights2
        self.weights1 += self.learning_rate * d_weights1
