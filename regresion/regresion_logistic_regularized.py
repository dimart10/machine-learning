import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values.astype(float)

def gradient(thetas, XX, Y, lamb):
    m = np.shape(XX)[0]

    H = h(thetas, XX)
    grad = (1/len(Y)) * np.dot(XX.T, H-Y)

    print(grad)

    grad += (lamb/m) * np.c_[thetas]

    return grad

def cost(thetas, X, Y, lamb):
    m = np.shape(X)[0]
    H = h(thetas, X)

    c = (-1/(len(X))) * (np.dot(Y.T, np.log(H)) + np.dot((1-Y).T, np.log(1-H)))
    c += (lamb/(2*m)) * (thetas**2).sum()

    return c

def sigmoid(Z):
    return 1/(1 + np.e**(-Z))

def h(thetas, X):
    return np.c_[sigmoid(np.dot(X, thetas))]

def show_decision_boundary(thetas, X, Y, poly):
    plt.figure()

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(thetas))

    h = h.reshape(xx1.shape)
    
    positives = np.where(Y == 1)
    negatives = np.where(Y == 0)
    plt.scatter(X[positives, 0], X[positives, 1], marker='+', color='blue')
    plt.scatter(X[negatives, 0], X[negatives, 1], color='red')

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')    
    plt.savefig("images/regresion_logistic_regularized.png")    
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
    datos = load_csv("data/ex2data2.csv")
    X = datos[:, :-1]
    Y = datos[:, -1][np.newaxis].T

    poly = PolynomialFeatures(6)
    X_poly = poly.fit_transform(X)

    lamb = 1

    m = np.shape(X_poly)[0]
    n = np.shape(X_poly)[1]

    thetas = np.zeros((n, 1), dtype=float)

    result = opt.fmin_tnc(func=cost, x0=thetas, fprime=gradient, args=(X_poly, Y, lamb))
    thetas = result[0]

    print("Error percentage: ", evaluate(thetas, X_poly, Y)*100, "%")
    show_decision_boundary(thetas, X, Y, poly)


if __name__ == "__main__":
    main()
