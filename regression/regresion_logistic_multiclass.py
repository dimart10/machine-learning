from  scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import regresion_logistic_regularized as model
import sys

def train_models(X, Y):
        modelThetas = list()
        X_poly = []

        # TRAIN 10 MODELS, ONE FOR EACH NUMBER
        for i in range(1, 11):
                print("Training model for number %i\n" %(i))
                formattedY = (Y == i).astype(int)

                trainingResult = model.train(X, formattedY, 1, 0.1, evaluateResults=False)
                X_poly = trainingResult[0]

                if (i != 10):
                        modelThetas.append(trainingResult[1])
                else:
                        modelThetas.insert(0, trainingResult[1])

        return X_poly, modelThetas

def predict(X_poly, modelThetas, realY=[], showResults=True):
        m = X_poly.shape[0]
        predictions = np.zeros((m, len(modelThetas)))
        finalPrediction = np.zeros((m, 1))

        for i in range(len(modelThetas)):
                predictions[:, i] = model.h(modelThetas[i], X_poly).flatten()

        finalPrediction = np.argmax(predictions, 1)
        finalPrediction = np.reshape(finalPrediction, (5000, 1))
        if (not showResults): return finalPrediction
        
        if (len(realY) > 0):
                print("Precision", ((finalPrediction == realY).sum()/m)*100, "%")
                
        return finalPrediction

def main():
        data = loadmat ("../data/ex3data1.mat")
        Y = data['y']
        X = data['X']

        #sample = np.random.choice(X.shape[0], 10)
        #plt.imshow(X[sample, :].reshape(-1, 20).T)
        #plt.show()

        trainedModels = train_models(X, Y)
        predict(trainedModels[0], trainedModels[1], Y)

if __name__ == "__main__":
        main()
