import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def load_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values

    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def gradient_descent(X, Y, alpha, iterations):
    m = float(np.shape(X)[0])
    thetas = np.zeros((2, 1), dtype=float)
    costs = np.empty(iterations)
    thetasHistory = np.empty((iterations, np.shape(thetas)[0]))

    for i in range(iterations):
        H = h(X, thetas)
        thetas = thetas - (alpha * (1/m) * (X.T.dot(H - Y)))

        thetasHistory[i] = thetas.T
        costs[i] = cost(X, Y, thetas)
        print("Cost [", i, "]: ", costs[i])

    return thetasHistory, costs

def cost(X, Y, thetas):
    m = np.shape(X)[0]
    return ((h(X, thetas)-Y)**2).sum() / (2*m)

def h(X, thetas):
    return np.dot(X, thetas)

def main():
    # DATA PREPROCESSING
    datos = load_csv("data/ex1data1.csv")
    X = datos[:, :-1]
    Y = datos[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01

    # TRAINING
    thetas, costs = gradient_descent(X, Y, alpha, 1000)

    # PREDICTION
    prediction = h(X, thetas[-1,:][np.newaxis].T)

    # RESULTS
    plt.figure()
    plt.scatter(X[:,1], Y[:,0], c='red')
    plt.plot(X[:,1], prediction)
    plt.savefig('images/regresion_simple_results.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(costs)
    plt.savefig('images/regresion_simple_gradient_descent.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
