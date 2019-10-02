import numpy as np
from pandas.io.parsers import read_csv

def load_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values

    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def gradient_descent(X, Y, alpha):
    return [0, 0]

def cost(X, Y, theta):
    return 0

def h(theta, x):
    return t1 + t2*x

def main():
    datos = load_csv("ex1data1.csv")
    X = datos[:, :-1]
    Y = datos[:, -1]

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01

    thetas, costs = gradient_descent(X, Y, alpha)

if __name__ == "__main__":
    main()
