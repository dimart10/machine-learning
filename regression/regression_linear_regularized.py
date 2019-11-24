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
    order = np.ravel(np.argsort(X, 0))

    X = X[order, :]
    Y = Y[order, :]
    prediction = prediction[order, :]

    plt.figure()
    plt.scatter(X, Y[:,0], c='red')
    plt.plot(X, prediction, c='blue')
    plt.savefig('images/regression_lineal_regularized_results.png')
    plt.show()
    plt.close()

def getXPoly(X, degree, normalizeValues=True):
    m = np.shape(X)[0]

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    if (normalizeValues): X_poly = normalize(X_poly[:,1:])[0]

    X_poly = np.hstack([np.ones([m, 1]), X_poly])

    return X_poly

def normalize(X):
    mean = np.mean(X, 0)
    deviation = np.std(X, 0)

    normalized = (X-mean)/deviation

    return (normalized, mean, deviation)

def testModel(thetas, X, Y, degree, lamb, genPlot=False, normalize_X=True):
    X_poly = getXPoly(X, degree, normalize_X)

    # Test the model
    c = cost(thetas, X_poly, Y, lamb)
    grad = gradient(thetas, X_poly, Y, lamb)

    # Show the results
    print("Cost: ", c)
    print("Gradient: ", grad)

    if (genPlot): generate_plot(X, Y, h(thetas, X_poly))

    return (c, grad)

def crossValidation_plot(trainingGroups, group_costs, crossValidation_costs):
    plt.figure()
    plt.plot(trainingGroups, group_costs, c='red')
    plt.plot(trainingGroups, crossValidation_costs, c='blue')
    plt.savefig('images/regression_lineal_regularized_crossValidation.png')
    plt.show()
    plt.close()

def train(X, Y, degree=1, lamb=1, testModel=False, thetas=np.array([]), normalize_X=True):
    X_poly = getXPoly(X, degree, normalize_X)

    m = np.shape(X_poly)[0]
    n = np.shape(X_poly)[1]

    if (thetas.size == 0): thetas = np.ones((n, 1), dtype=float)
    thetasVector = np.ravel(thetas)

    results = opt.minimize(fun = regression, x0 = thetasVector,
                    args = (X_poly, Y, lamb), method = 'TNC',
                    jac = True, options = {'maxiter': 20})

    thetas = np.reshape(results['x'], (X_poly.shape[1], 1))

    if (testModel): testModel(thetas, X_poly, Y, lamb)

    return thetas, X_poly

def train_crossValidation(k, X, Y, X_val, Y_val, degree=1, lamb=1, testModel=False, normalize_X=True):
    m = np.shape(X)[0]
    X_val_poly = getXPoly(X_val, degree, normalize_X)

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

        group_results = train(X_group, Y_group, lamb = lamb, thetas = thetas, normalize_X=normalize_X)

        thetas = group_results[0]
        group_cost = cost(thetas, group_results[1], Y_group, lamb)
        validation_cost = cost(thetas, X_val_poly, Y_val, lamb)

        group_costs.append(group_cost)
        crossValidation_costs.append(validation_cost)

        startIndex = lastIndex

    crossValidation_plot(trainingGroups, group_costs, crossValidation_costs)

    return thetas

def main():
    # DATA PREPROCESSING
    data = loadmat("../data/ex5data1.mat")
    X, Y = data['X'], data['y']
    X_val, Y_val = data['Xval'], data['yval']
    X_test, Y_test = data['Xtest'], data['ytest']

    degree = 8
    lamb = 0

    # Model training
    #thetas = train_crossValidation(1, X, Y, X_val, Y_val, lamb=0)

    trainingResult = train(X, Y, degree=degree, lamb=lamb)
    thetas = trainingResult[0]

    # Model testing
    testModel(thetas, X, Y, degree, lamb, True)

if __name__ == "__main__":
    main()
