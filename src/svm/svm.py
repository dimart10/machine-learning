import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import loadmat
from sklearn.svm import SVC

def showBoundary(svm, X, Y):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
    x1, x2 = np.meshgrid(x1, x2)

    Yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
    pos = (Y == 1).ravel()
    neg = (Y == 0).ravel()

    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='Yellow',
     edgecolors='black', marker='o')
    plt.contour(x1, x2, Yp)
    plt.savefig('images/svm.png')
    plt.show()

def test(svm, X, Y):
    prediction = svm.predict(X)
    accuracy = (prediction[:,np.newaxis] == Y).sum() / X.shape[0]

    return accuracy

def train(X, Y, C, sigma, degree = 3, coef0 = 0, kernel = 'rbf', verbose = True):
    svm = SVC(kernel=kernel, C=C, gamma=1 / (2*sigma**2), verbose=verbose, degree=degree, coef0=coef0)
    svm.fit(X, Y.ravel())

    return svm

def findBestSVM(X, Y, X_val, Y_val, Cs, Sigmas, showResults=True):
    bestSvm = None
    bestAccuracy = -1
    bestSigma = None
    bestC = None
    for C in Cs:
        for sigma in Sigmas:
            current_svm = train(X, Y, C, sigma)
            current_accuracy = test(current_svm, X_val, Y_val)

            if (current_accuracy > bestAccuracy):
                bestSvm = current_svm
                bestAccuracy = current_accuracy
                bestC = C
                bestSigma = sigma

    if (showResults):
        print("Best values found [", bestAccuracy*100, "% accuracy ]: ",
                "C =", bestC, "|", "Sigma =", bestSigma)

    return bestSvm, bestC, bestSigma, bestAccuracy

def main():
    # DATA PREPROCESSING
    data = loadmat("../data/ex6data3.mat")
    X, Y = data['X'], data['y']
    X_val, Y_val = data['Xval'], data['yval']

    possible_values = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)

    bestSvm = findBestSVM(X, Y, X_val, Y_val, possible_values, possible_values)[0]

    showBoundary(bestSvm, X, Y)

if __name__ == "__main__":
    main()
