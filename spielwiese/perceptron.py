import numpy as np
import matplotlib.pyplot as plt

def perceptron_eval(X, y):

    w = np.array([-1.5, 1.0, 1.0])

    z = X @ w
    print(w, X, z)
    o = np.heaviside(z, 1)

    gesamtfehler = np.sum(np.abs(o - y))

    return gesamtfehler

X = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 0, 0, 1])

res = perceptron_eval(X, y)
print(res)
