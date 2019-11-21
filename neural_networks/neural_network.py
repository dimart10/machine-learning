from  scipy.io import loadmat
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

def backwards_propagation(thetas, X, Y, reg):
    m = X.shape[0]
    A, Z, h = forward_propagation(X, thetas)[:-1]

    for example in range(m):
        last_d = h[example, :] - Y[example]
        for layer in range(len(A[:-1]), -1, -1):
            theta = thetas[layer]
            a = A[layer][example, :]
            d = np.dot(theta.T, last_d) * (a * (1 - a)) # (1, 26)
            theta += np.dot(last_d[:, np.newaxis], a[np.newaxis, :])
            last_d = d[1:]

    return thetas

def sigmoid(Z):
    return 1/(1 + np.e**(-Z))

def evaluate(prediction, realY):
        m = prediction.shape[0]

        print("Precision", ((prediction+1 == realY).sum()/m)*100, "%")

def main():
        weights = loadmat("../data/ex4weights.mat")
        data = loadmat ("../data/ex4data1.mat")

        X, Y = data['X'], data['y']
        theta1, theta2 = weights['Theta1'], weights['Theta2']
        thetas = np.array([theta1, theta2])

        thethas = backwards_propagation(thetas, X, Y, 0)

        prediction = forward_propagation(X, thetas)[-1]
        evaluate(prediction, Y)

if __name__ == "__main__":
        main()
