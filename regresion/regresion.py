import numpy as np
from pandas.io.parsers import read_csv

def load_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values

    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def gradient_descent(X, Y, alpha, iterations):
    m = np.shape(X)[0]
    thetas = np.zeros((2, m), dtype=float)
    
    for i in range(iterations):
        H = h(X, thetas)
        thetas = thetas - (alpha*(1/m) * (X.T.dot(H - Y)))

        print("Cost [", i, "]: ", cost(X, Y, thetas))

    return [0, 0]

def cost(X, Y, thetas):
    m = np.shape(X)[0]
    return ((h(X, thetas)-Y)**2).sum() / (2*m)

def h(X, thetas):
    return np.dot(X, thetas)

def main():
    datos = load_csv("ex1data1.csv")
    X = datos[:, :-1]
    Y = datos[:, -1]

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01

    thetas, costs = gradient_descent(X, Y, alpha, 10000)

if __name__ == "__main__":
    main()
