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

def sigmoid(Z):
    return 1/(1 + np.e**(-Z))

def evaluate(prediction, realY):
        m = prediction.shape[0]

        print("Precision", ((prediction+1 == realY).sum()/m)*100, "%")

def main():
        weights = loadmat("../data/ex3weights.mat")
        data = loadmat ("../data/ex3data1.mat")

        X, Y = data['X'], data['y']
        theta1, theta2 = weights['Theta1'], weights['Theta2']
        prediction = forward_propagation(X, np.array([theta1, theta2]))
        evaluate(prediction, Y)


if __name__ == "__main__":
        main()
