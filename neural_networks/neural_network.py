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

    deltas = thetas
    for example in range(m):
        last_d = h[example, :] - Y[example]
        for layer in range(len(A[:-1]), -1, -1):
            delta = deltas[layer]
            
            a = A[layer][example, :]
            d = np.dot(delta.T, last_d) * (a * (1 - a))

            delta += np.dot(last_d[:, np.newaxis], a[np.newaxis, :])
            last_d = d[1:]

    gradient = (1 / m) * deltas
    
    cost = cost_function(thetas, X, Y, h, 1, 10)

    return (cost, gradient)

def cost_function(thetas, X, Y, h, lamb, num_labels):
        m = X.shape[0]

        y_onehot = np.zeros((m, num_labels))
        for i in range(m):
                y_onehot[i][Y[i]] = 1

        cost = -y_onehot * np.log(h) - (1 - y_onehot) * np.log(1 - h)
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

def evaluate(prediction, realY):
        m = prediction.shape[0]

        print("Precision", ((prediction+1 == realY).sum()/m)*100, "%")

def main():
        #weights = loadmat("../data/ex4weights.mat")
        data = loadmat ("../data/ex4data1.mat")

        X, Y = data['X'], data['y']
        Y = Y-1

        theta1 = random_weights(25, 401, 0.12)
        theta2 = random_weights(10, 26, 0.12)

        thetas = np.array([theta1, theta2])

        backs_results = backwards_propagation(thetas, X, Y, 0)

        forward_prop = forward_propagation(X, thetas)
        print("Cost: ", backs_results[0])
        evaluate(forward_prop[-1], Y)

if __name__ == "__main__":
        main()
