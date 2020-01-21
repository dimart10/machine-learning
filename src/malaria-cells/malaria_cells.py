import os
import argparse
import sys
sys.path.append('../')
sys.path.append('../../utils')

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas
import scipy.optimize as opt
import data_preprocessing as Preprocessing
from PIL import Image

# Custom algorithms
import regression.regresion_logistic as LogRegression
import neural_networks.neural_network as NeuralNetwork
import regression.regresion_logistic_regularized as LogRegressionRegularized
import svm.svm as SVM


DATA_PATH = "../../data/heavy/malaria-cells/"
PARASITIZED_FOLDER = "Parasitized/"
UNINFECTED_FOLDER = "Uninfected/"
CACHE_FOLDER = ".cache/"
PARASITIZED_CACHE_FILE = "parasitized_cache"
UNINFECTED_CACHE_FILE = "uninfected_cache"

PARASITIZED_PATH = DATA_PATH + PARASITIZED_FOLDER
UNINFECTED_PATH = DATA_PATH + UNINFECTED_FOLDER
CACHE_PATH = DATA_PATH + CACHE_FOLDER

FILE_EXTENSION = ".png"
CACHE_EXTENSION = ".npy"
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60

TRAINING_BATCH = 1000
VAL_BATCH = 300
TEST_BATCH = 500

# Load & preprocess data
def loadData(cache = True):
    print("\nLoading data...")

    # First, check if the data has already been cached, if so load it and return
    if (cache and os.path.isfile(CACHE_PATH + PARASITIZED_CACHE_FILE + CACHE_EXTENSION)
            and os.path.isfile(CACHE_PATH + UNINFECTED_CACHE_FILE + CACHE_EXTENSION)):
        print("Image data is being loaded from cache")
        parasitized = np.load(CACHE_PATH + PARASITIZED_CACHE_FILE + CACHE_EXTENSION)
        uninfected = np.load(CACHE_PATH + UNINFECTED_CACHE_FILE + CACHE_EXTENSION)

    else:
        # Check if the folders exists and contain the data
        if (not os.path.isdir(PARASITIZED_PATH) or not os.path.isdir(UNINFECTED_PATH)):
            print("\nData directories don't exist, please read the README.md of this folder"
                    + "to learn how to download and setup the data\n")
            return False

        parasitized_filenames = [f for f in os.listdir(PARASITIZED_PATH) if f.endswith(FILE_EXTENSION)]
        uninfected_filenames = [f for f in os.listdir(UNINFECTED_PATH) if f.endswith(FILE_EXTENSION)]

        if (len(parasitized_filenames) <= 0 or len(uninfected_filenames) <= 0):
            print("\nData is missing or missplaced, please read the README.md of this folder"
                    + "to learn how to download and setup the data")
            return False

        # Load every image, resizes it to 80x80 and saves it as a 1D array
        parasitized = []
        for filename in parasitized_filenames:
            temp = cv2.imread(PARASITIZED_PATH + filename)
            temp = cv2.resize(temp, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
            parasitized.append(temp.ravel())
        parasitized = np.array(parasitized)

        uninfected = []
        for filename in uninfected_filenames:
            temp = cv2.imread(UNINFECTED_PATH + filename)
            temp = cv2.resize(temp, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
            uninfected.append(temp.ravel())
        uninfected = np.array(uninfected)

        # Cache the parsed data to avoid reading the images on every execution
        if (cache):
            if (not os.path.isdir(CACHE_PATH)): os.mkdir(CACHE_PATH)
            np.save(CACHE_PATH + PARASITIZED_CACHE_FILE, parasitized)
            np.save(CACHE_PATH + UNINFECTED_CACHE_FILE, uninfected)

    # Randomize the order of the images before splitting into sets
    np.random.shuffle(parasitized)
    np.random.shuffle(uninfected)

    # Split the images into the different sets
    startIndex = 0
    trainingSet = formatData(parasitized[:TRAINING_BATCH,:],
                             uninfected[:TRAINING_BATCH,:])

    startIndex += TRAINING_BATCH
    validationSet = formatData(parasitized[startIndex:startIndex + VAL_BATCH,:],
                               uninfected[startIndex:startIndex + VAL_BATCH,:])

    startIndex += VAL_BATCH
    testSet = formatData(parasitized[startIndex:startIndex + TEST_BATCH,:],
                         uninfected[startIndex:startIndex + TEST_BATCH,:])

    return (True, trainingSet, validationSet, testSet)

# Returns the data loaded from disk in the standard format
def formatData(parasitized, uninfected):
    n = parasitized.shape[1]

    formatted = np.vstack((np.insert(parasitized, n, 1, 1),
                            np.insert(uninfected, n, 0, 1)))
    formatted = Preprocessing.separate_data(formatted)

    return formatted

def logRegression(data, verbose, hideOutput, dontSave):
    print("\n*** Running logistic regression algorithm ***\n")
    thetas = LogRegression.train(data[0], data[1])

    if (not hideOutput):
        print("Error percentage: ", LogRegression.evaluate(thetas, X, Y)*100, "%")

def logRegressionRegularized(data, verbose, hideOutput, dontSave, lamb):
    print("\n*** Running regularized logistic regression algorithm ***\n")

    thetas = LogRegressionRegularized.train(data[0], data[1], 1, lamb)

    if (not hideOutput):
        print("Error percentage: ", LogRegressionRegularized.evaluate(thetas, data[0], data[1], 1)*100, "%")

def neuralNetwork(data, verbose, hideOutput, dontSave):
    print("\n*** Running neural network ***\n")
    thetas = NeuralNetwork.train(data[0], data[1], (IMAGE_WIDTH*IMAGE_HEIGHT*3, 50, 2), 2, 1)

    if (not hideOutput):
        print("Error percentage: ", NeuralNetwork.evaluate(thetas, data[0], data[1])*100, "%")

def runSVM(data, verbose, hideOutput, dontSave, c, sigma):
    print("\n*** Running SVM ***\n")
    trained_svm = SVM.train(data[0], data[1], c, sigma)

    if (not hideOutput):
        print("SVM Accuracy: ", SVM.test(trained_svm, data[0], data[1]) * 100)

# Run all algorithms in sequence
def runAllAlgorithms(data, verbose, hideOutput, dontSave, lamb, c, sigma):
    print("""
*** Running all algorithms ***
Depending on your machine specs, this may take some time\n""")

    logRegression(data, verbose, hideOutput, dontSave)
    logRegressionRegularized(data, verbose, hideOutput, dontSave, lambd)
    neuralNetwork(data, verbose, hideOutput, dontSave)
    runSVM(data, verbose, hideOutput, dontSave, svm_c, sigma)

# MAIN FUNCTION
def main():
    # Command line arguements
    parser = argparse.ArgumentParser(description='Detect infected malaria cells using different machine learning algorithms')

    # General
    parser.add_argument('--all', '-a', help='Run all algorithms. If no argument is passed, this is the default behaviour', action="store_true")
    parser.add_argument('--verbose', '-v', help='Output detailed information for each algorithm execution', action="store_true")
    parser.add_argument('--hideOutput', '-o', help='Do not show result graphs & statistics for the executed algorithms', action="store_true")
    parser.add_argument('--dontSave', '-d', help='Do not save to disk result graphs & statistics for the executed algorithms', action="store_true")
    parser.add_argument('--noCache', '-c', help='Do not load cached data and do not cache generated data', action="store_true")

    # Algorithms
    parser.add_argument('--logRegression', '-l', help='Execute the logistic regression algorithm', action="store_true")
    parser.add_argument('--logRegressionRegularized', '-r', help='Execute the regularized logistic regression algorithm', action="store_true")
    parser.add_argument('--neuralNetwork', '-n', help='Execute the neural network algorithm', action="store_true")
    parser.add_argument('--svm', '-s', help='Execute the support vector machine algorithm', action="store_true")

    # Extra parameters
    parser.add_argument('--lambd', help='Lambda parameter for the regularized logistic regression algorithm.', type=float, default=1)
    parser.add_argument('--svm_c', help='C parameter for the SVM algorithm.', type=float, default=1)
    parser.add_argument('--sigma', help='Sigma parameter for SVM algorithm.', type=float, default=1)


    args = parser.parse_args()
    runAll = True

    # Data must always be loaded
    loadResult = loadData(not args.noCache)
    if (not loadResult[0]):
        return
    data = loadResult[1]

    # Conditional execution depending on console arguments
    if (args.logRegression):
        logRegression(data, args.verbose, args.hideOutput, args.dontSave)
        runAll = False;

    if (args.logRegressionRegularized):
        logRegressionRegularized(data, args.verbose, args.hideOutput, args.dontSave, args.lambd)
        runAll = False;

    if (args.neuralNetwork):
        neuralNetwork(data, args.verbose, args.hideOutput, args.dontSave)
        runAll = False;

    if (args.svm):
        runSVM(data, args.verbose, args.hideOutput, args.dontSave, args.svm_c, args.sigma)
        runAll = False;

    if (runAll):
        runAllAlgorithms(data, args.verbose, args.hideOutput, args.dontSave, args.lambd, args.svm_c, args.sigma)

if __name__ == "__main__":
    main()
