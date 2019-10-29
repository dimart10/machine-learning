from  scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import regresion_logistic_regularized as model

def train_models(X, Y):
    models = list()

    # TRAIN 10 MODELS, ONE FOR EACH NUMBER
    for i in range(1, 11):
        print("Training model for number %i\n" %(i))
        formattedY = (Y == i).astype(int)
        models.append(model.train(X, formattedY, 2, 1, evaluateResults=False))

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
