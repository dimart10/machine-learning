import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt

def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values.astype(float)

def gradient(thetas, XX, Y):
    H = h(thetas, XX)
    grad = (1/len(Y)) * np.dot(XX.T, H-Y)

    return grad

def cost(thetas, X, Y):
    m = np.shape(X)[0]
    H = h(thetas, X)

    c = (-1/(len(X))) * (np.dot(Y.T, np.log(H)) + np.dot((1-Y).T, np.log(1-H)))

    return c

def sigmoid(Z):
    return 1/(1 + np.e**(-Z))

def h(thetas, X):
    return np.c_[sigmoid(np.dot(X, thetas))]

def show_border(thetas, X, Y):
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    
    H = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(thetas))

    H = H.reshape(xx1.shape)

    plt.figure()

    positives = np.where(Y == 1)
    negatives = np.where(Y == 0)
    plt.scatter(X[positives, 1], X[positives, 2], marker='+', color='blue')
    plt.scatter(X[negatives, 1], X[negatives, 2], color='red')

    plt.contour(xx1, xx2, H, [0.5], linewidths=1, colors='b')
    plt.savefig("images/regresion_logistic_border.png")
    plt.show()
    plt.close()

def evaluate(thetas, X, Y):
    result = h(thetas, X)
    passed_missed = np.logical_and((result >= 0.5), (Y == 0)).sum()
    failed_missed = np.logical_and((result < 0.5), (Y == 1)).sum()

    errors = (passed_missed + failed_missed)

    return errors/(result.shape[0])

def main():
    # DATA PREPROCESSING
    datos = load_csv("data/ex2data1.csv")
    X = datos[:, :-1]
    Y = datos[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X)[1]

    thetas = np.zeros((n, 1), dtype=float)

    result = opt.fmin_tnc(func=cost, x0=thetas, fprime=gradient, args=(X, Y))
    thetas = result[0]

    print("Error percentage: ", evaluate(thetas, X, Y)*100, "%")
    show_border(thetas, X, Y)


if __name__ == "__main__":
    main()
