from  scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import regresion_logistic_regularized as model

def train_models(X, Y):
    models = np.zeros((1, 10))

    # TRAIN 10 MODELS, ONE FOR EACH NUMBER

    #model.train(X, Y, False, 2)

def main():
    data = loadmat ("../data/ex3data1.mat")
    Y = data['y']
    X = data['X']

    #sample = np.random.choice(X.shape[0], 10)
    #plt.imshow(X[sample, :].reshape(-1, 20).T)
    #plt.show()

    train_models(X, Y)

if __name__ == "__main__":
    main()