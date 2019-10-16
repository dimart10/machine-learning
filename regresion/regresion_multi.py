import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values.astype(float)

def gradient_descent(X, Y, alpha, iterations):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    thetas = np.zeros((n, 1), dtype=float)
    costs = np.empty(iterations)
    thetasHistory = np.empty((iterations, np.shape(thetas)[0]))

    for i in range(iterations):
        H = h(X, thetas)
        thetas = thetas - (alpha * (1/float(m)) * (X.T.dot(H - Y)))

        thetasHistory[i] = thetas.T
        costs[i] = cost(X, Y, thetas)
        print("Cost [", i, "]: ", costs[i])

    return thetasHistory, costs

def cost(X, Y, thetas):
    m = np.shape(X)[0]
    H = h(X, thetas)
    aux = H-Y
    return np.dot(aux.T, aux) / (2*m)

def h(X, thetas):
    return np.dot(X, thetas)

def main():
    # DATA PREPROCESSING
    datos = load_csv("data/ex1data2.csv")
    X = datos[:, :-1]
    X = normalize(X)[0]
    Y = datos[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01

    # TRAINING
    thetas, costs = gradient_descent(X, Y, alpha, 500)

    # PREDICTION
    prediction = h(X, thetas[-1,:][np.newaxis].T)

    # i changes the case you output (for test purposes)
    i = 40
    print("Case %i | %f:%f = %f | Real value: %f" %(i, X[i, 1], X[i, 2], prediction[0], Y[i]))

    # RESULTS
    plt.figure()
    plt.plot(costs)
    plt.savefig('images/regresion_multi_gradient_descent.png')
    plt.show()
    plt.close()


def normalize(X):
    mean = np.mean(X, 0)
    deviation = np.std(X, 0)
    normalized = (X-mean)/deviation

    return (normalized, mean, deviation)

if __name__ == "__main__":
    main()
