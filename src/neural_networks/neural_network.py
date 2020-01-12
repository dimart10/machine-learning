from  scipy.io import loadmat
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def forward_propagation(X, thetas):
    m = X.shape[0]
    prediction = np.zeros((m, 1))

    # Input layer
    a = np.hstack([np.ones([m, 1]), X])

    # Neural net parameters storage
    A = [a]
    Z = []

    # Hidden layers
    for i in range(0, len(thetas) - 1):
            z = np.dot(a, thetas[i].T)
            a = np.hstack([np.ones([m, 1]), sigmoid(z)])
            A.append(a)
            Z.append(z)

    # Output layer
    z = np.dot(a, thetas[-1].T)
    Z.append(z)
    h = sigmoid(z)

    prediction = np.argmax(h, 1)

    return (A, Z, h, np.reshape(prediction, (5000, 1)))

def backwards_propagation(thetasVector, X, Y, layerStructure, num_labels, lamb):
    m = X.shape[0]

    thetas = unVectorize(thetasVector, layerStructure)

    A, Z, h = forward_propagation(X, thetas)[:-1]

    deltas = []
    for theta in thetas:
        deltas.append(np.zeros(theta.shape))
    deltas = np.array(deltas)

    for example in range(m):
        last_d = h[example, :] - Y[example]
        for layer in range(len(A[:-1]), -1, -1):
            theta = thetas[layer]

            a = A[layer][example, :]
            d = np.dot(theta.T, last_d) * (a * (1 - a))

            deltas[layer] += np.dot(last_d[:, np.newaxis], a[np.newaxis, :])
            last_d = d[1:]

    gradient = (1 / m) * deltas
    gradient += (lamb / m) * deltas # Regularization
    gradientVector = vectorize(gradient)

    cost = cost_function(thetas, X, Y, h, lamb, num_labels)

    return (cost, gradientVector)

def cost_function(thetas, X, Y_onehot, h, lamb, num_labels):
    m = X.shape[0]

    cost = (-Y_onehot * np.log(h)) - ((1 - Y_onehot) * np.log(1 - h))
    cost = 1/m * cost.sum()

    # Regularization
    for theta in thetas:
            cost += (lamb / (2 * m)) * (theta**2).sum()

    return cost

def sigmoid(Z):
    return 1/(1 + np.e**(-Z))

def derivative_sigmoid(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)

def random_weights(L_in, L_out, init_range):
    return np.random.rand(L_in, L_out) * (init_range*2) - init_range

def evaluate(thetas, X, Y):
    prediction = forward_propagation(X, thetas)[-1]
    m = prediction.shape[0]

    return ((prediction == Y).sum()/m)


def vectorize(npArray):
    vec = np.empty((0))
    for arrayElement in npArray:
        vec = np.concatenate((vec, np.ravel(arrayElement)))

    return vec

def unVectorize(vector, layerStructure):
    temp = []
    last_index = 0
    for i in range(len(layerStructure) - 1):
        layerSize = layerStructure[i] + 1
        nextLayerSize = layerStructure[i + 1]
        totalSize = layerSize * nextLayerSize
        temp.append(np.reshape(vector[last_index:last_index+totalSize],(nextLayerSize, layerSize)))
        last_index = totalSize

    return np.array(temp)

def train(X, Y, layerStructure, num_labels, lamb):
    # Number of examples
    m = X.shape[0]

    Y_onehot = np.zeros((m, num_labels))
    for i in range(m):
            Y_onehot[i][Y[i]] = 1

    thetas = []
    for i in range(len(layerStructure)-1):
        thetas.append(random_weights(layerStructure[i+1], layerStructure[i]+1, 0.12))
    thetasVector = vectorize(thetas)

    fmin = minimize(fun = backwards_propagation, x0 = thetasVector,
                    args = (X, Y_onehot, layerStructure, num_labels, lamb), method = 'TNC',
                    jac = True, options = {'maxiter': 70})

    thetas = unVectorize(fmin['x'], layerStructure)

    return thetas

def main():
    data = loadmat("../../data/ex4data1.mat")
    X, Y = data['X'], data['y']
    Y = Y-1

    thetas = train(X, Y, (400, 25, 10), 10, 1)
    print("Precision:", evaluate(thetas, X, Y)*100, "%")

if __name__ == "__main__":
        main()
