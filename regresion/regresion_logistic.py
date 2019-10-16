import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values.astype(float)

def gradient(XX, Y, thetas):
    H = h(XX, thetas)
    grad = (1/len(Y)) * np.dot(XX.T, H-Y)

    return grad

def cost(X, Y, thetas):
    m = np.shape(X)[0]
    H = h(X, thetas)

    c *= (- 1 / m) * np.dot(Y, np.log(H)) +  np.dot((1 - Y), np.log(1 - H))

    return c

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def h(X, thetas):
    return sigmoid(np.dot(X, thetas))

def show_border(X, Y, theta):
    plt.figure()

    pos = np.where(Y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parametro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("images/regresion_logistic_border.png")
    plt.show()
    plt.close()

def main():
    # DATA PREPROCESSING
    datos = load_csv("data/ex2data1.csv")
    X = datos[:, :-1]
    Y = datos[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    #X = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X)[1]

    thetas = np.zeros((n, 1), dtype=float)

    show_border(X, Y, thetas)


if __name__ == "__main__":
    main()
