import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from  scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

def gradient(thetas, X, Y, lamb):
    m = np.shape(X)[0]
    H = h(thetas, X)

    grad = (1/m) * np.dot(X.T, H-Y)
    grad[1:] += (lamb/m) * np.c_[thetas[1:]] # Regularization

    return grad

def cost(thetas, X, Y, lamb):
    m = np.shape(X)[0]

    cost = ((h(thetas, X)-Y)**2).sum() / (2*m)
    cost += (lamb/(2*m)) * (thetas**2).sum() # Regularization

    return cost

def regression(thetasVector, X_poly, Y, lamb):
    thetas = np.reshape(thetasVector, (X_poly.shape[1], 1))

    c = cost(thetas, X_poly, Y, lamb)
    c = np.ravel(c)

    grad = gradient(thetas, X_poly, Y, lamb)
    grad = np.ravel(grad)

    return (c, grad)

def h(thetas, X):
    return np.dot(X, thetas)

def generate_plot(X, Y, prediction):
    plt.figure()
    plt.scatter(X, Y[:,0], c='red')
    plt.plot(X, prediction)
    plt.savefig('images/regression_lineal_regularized_results.png')
    plt.show()
    plt.close()

def train(X, Y, degree=1, lamb=1, testModel=False):
    X_poly = getXPoly(X, degree)

    m = np.shape(X_poly)[0]
    n = np.shape(X_poly)[1]

    thetas = np.ones((n, 1), dtype=float)
    thetasVector = np.ravel(thetas)

    results = opt.minimize(fun = regression, x0 = thetasVector,
                    args = (X_poly, Y, lamb), method = 'TNC',
                    jac = True, options = {'maxiter': 50})

    thetas = np.reshape(results['x'], (X_poly.shape[1], 1))

    if (testModel): testModel(thetas, X_poly, Y, lamb)

    return thetas, X_poly

def getXPoly(X, degree):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

def testModel(thetas, X_test, Y_test, degree, lamb):
    X_poly = getXPoly(X_test, degree)

    print("Cost: ", cost(thetas, X_poly, Y_test, lamb))
    print("Gradient: ", gradient(thetas, X_poly, Y_test, lamb))
    generate_plot(X_test, Y_test, h(thetas, X_poly))

def main():
    # DATA PREPROCESSING
    data = loadmat("../data/ex5data1.mat")
    X_val, Y_val = data['Xval'], data['yval']
    X_test, Y_test = data['Xtest'], data['ytest']

    # Model training
    trainingResults = train(X_val, Y_val, lamb=0)

    # Model testing
    testModel(trainingResults[0], X_test, Y_test, 1, 0)

if __name__ == "__main__":
    main()
