import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def load_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values

    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def normal_equation (X, Y):
    thetas = np.dot(X.T, X)
    thetas = np.linalg.inv(thetas)
    thetas = np.dot(np.dot(thetas, X.T), Y)
    return thetas

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
    Y = datos[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01

    # TRAINING
    thetas = normal_equation(X, Y)

    # PREDICTION
    prediction = h(X, thetas)
    
    # i changes the case you output (for test purposes)
    i = 40
    print("Case %i | %f:%f = %f | Real value: %f" %(i, X[i, 1], X[i, 2], prediction[0], Y[i]))

if __name__ == "__main__":
    main()
