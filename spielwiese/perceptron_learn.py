
import random
import numpy as np

def perceptron_learn(X, results, weights, iterations):

    errors_list = []
    weights_list = []

    for i in range(iterations):
        index = random.randint(0, 149)
        x = X[index, :]
        y = results[index]

        z = x @ weights
        o = np.heaviside(z, 1)

        error = y - o

        weights_list.append(weights.copy())
        errors_list.append(error)

        delta_w = error*x
        weights = weights + delta_w
    
    return errors_list, weights_list
