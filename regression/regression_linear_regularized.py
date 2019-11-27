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

def generate_plot(X, Y, X_toPredict, Y_predicted):
    order = np.ravel(np.argsort(X_toPredict, 0))

    X_toPredict_ordered = X_toPredict[order, :]
    Y_predicted_ordered = Y_predicted[order, :]

    plt.figure()
    plt.scatter(X, Y[:,0], c='red')
    plt.plot(X_toPredict_ordered, Y_predicted_ordered, c='blue')
    plt.savefig('images/regression_lineal_regularized_results.png')
    plt.show()
    plt.close()

def getXPoly(X, degree, normalizeValues=True, mean=np.array([]), deviation=np.array([])):
    m = np.shape(X)[0]

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    if (normalizeValues):
        normalization_results = normalize(X_poly[:,1:], mean, deviation)
        X_poly = normalization_results[0]
        mean = normalization_results[1]
        deviation = normalization_results[2]

    X_poly = np.hstack([np.ones([m, 1]), X_poly])

    return X_poly, mean, deviation

def normalize(X, mean=np.array([]), deviation=np.array([])):
    if (mean.size == 0): mean = np.mean(X, 0)
    if (deviation.size == 0): deviation = np.std(X, 0)

    normalized = (X-mean)/deviation

    return (normalized, mean, deviation)

def testModel(thetas, X, Y, degree, lamb, genPlot=False, normalize_X=True, mean=np.array([]), deviation=np.array([])):
    X_poly = getXPoly(X, degree, normalize_X, mean, deviation)[0]

    # Test the model
    c = cost(thetas, X_poly, Y, lamb)
    grad = gradient(thetas, X_poly, Y, lamb)

    # Show the results
    print("Cost: ", c)
    print("Gradient: ", grad)

    if (genPlot):
        X_plot = np.arange(X.min(), X.max(), 0.05)[:, np.newaxis]
        print(X_plot)
        X_plot_poly = getXPoly(X_plot, degree, normalize_X, mean, deviation)[0]
        generate_plot(X, Y, X_plot, h(thetas, X_plot_poly))

    return (c, grad)

def crossValidation_plot(trainingGroups, group_costs, crossValidation_costs):
    plt.figure()
    plt.plot(trainingGroups, group_costs, c='red')
    plt.plot(trainingGroups, crossValidation_costs, c='blue')
    plt.savefig('images/regression_lineal_regularized_crossValidation.png')
    plt.show()
    plt.close()

def train(X, Y, degree=1, lamb=1, thetas=np.array([]), normalize_X=True, mean=np.array([]), deviation=np.array([])):
    X_poly_results = getXPoly(X, degree, normalize_X)
    X_poly = X_poly_results[0]
    if (mean.size == 0): mean = X_poly_results[1]
    if (deviation.size == 0): deviation = X_poly_results[2]

    m = np.shape(X_poly)[0]
    n = np.shape(X_poly)[1]

    if (thetas.size == 0): thetas = np.ones((n, 1), dtype=float)
    thetasVector = np.ravel(thetas)

    results = opt.minimize(fun = regression, x0 = thetasVector,
                    args = (X_poly, Y, lamb), method = 'TNC',
                    jac = True, options = {'maxiter': 20})

    thetas = np.reshape(results['x'], (X_poly.shape[1], 1))

    return thetas, X_poly, mean, deviation

def train_crossValidation(k, X, Y, X_val, Y_val, degree=1, lamb=1, normalize_X=True):
    m = np.shape(X)[0]

    X_poly_results = getXPoly(X, degree, normalize_X)
    mean = X_poly_results[1]
    deviation = X_poly_results[2]
    X_val_poly = getXPoly(X_val, degree, normalize_X, mean, deviation)[0]

    if (k >= m):
        print("Error: k should be lower than the total number of training examples.")
        return

    crossValidation_costs = []
    group_costs = []

    trainingGroups = list(range(k, m, k))
    if (trainingGroups[-1] != m): trainingGroups.append(m)

    thetas=np.array([])
    for lastIndex in trainingGroups:
        X_group = X[0:lastIndex]
        Y_group = Y[0:lastIndex]

        group_results = train(X_group, Y_group, degree, lamb, thetas,
                                normalize_X, mean, deviation)

        thetas = group_results[0]
        group_cost = cost(thetas, group_results[1], Y_group, lamb)
        validation_cost = cost(thetas, X_val_poly, Y_val, lamb)

        group_costs.append(group_cost)
        crossValidation_costs.append(validation_cost)

        startIndex = lastIndex

    crossValidation_plot(trainingGroups, group_costs, crossValidation_costs)

    return thetas, X_poly_results[0], mean, deviation

def train_lamba_selector(X, Y, X_val, Y_val, degree=1, lambs=(0, 1), normalize_X=True, showResults=True):
    X_poly_results = getXPoly(X, degree, normalize_X)
    mean = X_poly_results[1]
    deviation = X_poly_results[2]
    X_val_poly = getXPoly(X_val, degree, normalize_X, mean, deviation)[0]

    bestLambda = 0
    bestCost = -1
    for lamb in lambs:
        trainingResult = train(X, Y, degree=degree, lamb=lamb)
        currentCost = cost(trainingResult[0], X_val_poly, Y_val, lamb)

        if (bestCost == -1 or currentCost < bestCost):
            bestLambda = lamb
            bestCost = currentCost
            thetas = trainingResult[0]

    if (showResults):
        print("Best lambda|cost: ", bestLambda, "|", bestCost)

    return thetas, bestLambda, bestCost, X_poly_results[0], mean, deviation

def main():
    # DATA PREPROCESSING
    data = loadmat("../data/ex5data1.mat")
    X, Y = data['X'], data['y']
    X_val, Y_val = data['Xval'], data['yval']
    X_test, Y_test = data['Xtest'], data['ytest']

    degree = 8
    lamb = 3
    lambs = (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
    k = 1

    # Model training
    #trainingResult = train(X, Y, degree=degree, lamb=lamb)
    #trainingResult = train_crossValidation(k, X, Y, X_val, Y_val, degree, lamb)
    trainingResult = train_lamba_selector(X, Y, X_val, Y_val, degree, lambs)

    thetas = trainingResult[0]

    # Model testing
    testModel(thetas, X_test, Y_test, degree, lamb, True, True, trainingResult[-2], trainingResult[-1])

if __name__ == "__main__":
    main()
