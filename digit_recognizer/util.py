import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def set_random_weights(weights):
    rows, columns = weights.shape
    for i in range(rows):
        for j in range(columns):
            random_int = random.randint(1, 2)
            value = random.random()
            if random_int is 1:
                value = -1 * value
            weights[i][j] = value

    return weights

