from  scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def forward_propagation(X, thetas):
        m = X.shape[0]
        prediction = np.zeros((m, 1))

        # Input layer
        a = np.hstack([np.ones([m, 1]), X])

        # Hidden layers
        for i in range(0, len(thetas) - 1):
                z = np.dot(a, thetas[i].T)
                a = np.hstack([np.ones([m, 1]), sigmoid(z)])

        # Output layer
        z = np.dot(a, thetas[-1].T)
        h = sigmoid(z)

        prediction = np.argmax(h, 1)

        return np.reshape(prediction, (5000, 1))

def backwards_propagation():
    m = X.shape[0]
    ...
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    ...
    for t in range(m):
     a1t = a1[t, :] # (1, 401)
     a2t = a2[t, :] # (1, 26)
     ht = h[t, :] # (1, 10)
     yt = y[t] # (1, 10)
     d3t = ht - yt # (1, 10)
     d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)
     delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
     delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

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
        prediction = forward_propagation(X, np.array([theta1, theta2]))
        evaluate(prediction, Y)

if __name__ == "__main__":
        main()
