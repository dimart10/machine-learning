import os
import argparse
import sys
import time
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
import neural_networks.neural_network as NeuralNetwork
import regression.regresion_logistic_regularized as LogRegressionRegularized
import svm.svm as SVM

# Paths
DATA_PATH = "../../data/heavy/malaria-cells/"
PARASITIZED_FOLDER = "Parasitized/"
UNINFECTED_FOLDER = "Uninfected/"
CACHE_FOLDER = ".cache/"
PARASITIZED_CACHE_FILE = "parasitized_cache"
UNINFECTED_CACHE_FILE = "uninfected_cache"

PARASITIZED_PATH = DATA_PATH + PARASITIZED_FOLDER
UNINFECTED_PATH = DATA_PATH + UNINFECTED_FOLDER
CACHE_PATH = DATA_PATH + CACHE_FOLDER

# Image configuration
FILE_EXTENSION = ".png"
CACHE_EXTENSION = ".npy"
IMAGE_WIDTH = 25
IMAGE_HEIGHT = 25

# Batch sizes
DEFAULT_TRAINING_BATCH = 2000
DEFAULT_VAL_BATCH = 500
DEFAULT_TEST_BATCH = 3000

# Load & preprocess data
def loadData(trainingBatch, testBatch, valBatch, cache = True):
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
            image = formatImage(cv2.imread(PARASITIZED_PATH + filename))
            parasitized.append(image.ravel())
        parasitized = np.array(parasitized)

        uninfected = []
        for filename in uninfected_filenames:
            image = formatImage(cv2.imread(UNINFECTED_PATH + filename))
            uninfected.append(image.ravel())
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
    trainingSet = formatData(parasitized[:trainingBatch,:],
                             uninfected[:trainingBatch,:])

    startIndex += trainingBatch
    validationSet = formatData(parasitized[startIndex:startIndex + valBatch,:],
                               uninfected[startIndex:startIndex + valBatch,:])

    startIndex += valBatch
    testSet = formatData(parasitized[startIndex:startIndex + testBatch,:],
                         uninfected[startIndex:startIndex + testBatch,:])

    return (True, trainingSet, validationSet, testSet)

def formatImage(image):
    formatted = cv2.resize(image, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    formatted = cv2.cvtColor(formatted, cv2.COLOR_BGR2GRAY) # Now use grayscale
    formatted[formatted < 50] = 0

    return formatted

# Returns the data loaded from disk in the standard format
def formatData(parasitized, uninfected):
    n = parasitized.shape[1]

    # Normalization
    parasitized = parasitized / 255
    uninfected = uninfected / 255

    formatted = np.vstack((np.insert(parasitized, n, 1, 1),
                            np.insert(uninfected, n, 0, 1)))
    formatted = Preprocessing.separate_data(formatted)

    return formatted

def logRegressionRegularized(trainData, testData, verbose, lamb):
    print("\n*** Running [regularized] logistic regression algorithm ***\n")

    elapsed = time.time()
    thetas = LogRegressionRegularized.train(trainData[0], trainData[1], 1, lamb, verbose)
    elapsed = time.time() - elapsed

    print("Accuracy [ lamb = ", lamb, "]: ", LogRegressionRegularized.evaluate(thetas, testData[0], testData[1], 1)*100, "%")
    print("Elapsed time: ", elapsed, "seconds")

def neuralNetwork(trainData, testData, verbose, hiddenLayers, lamb, maxIter):
    print("\n*** Running neural network ***\n")

    # Add the input and output layers to the hidden layers
    if (isinstance(hiddenLayers, list)):
        layers = hiddenLayers
    else:
        layers = list()
        layers.append(hiddenLayers)

    layers.insert(0, IMAGE_WIDTH*IMAGE_HEIGHT)
    layers.append(2) # Output with two possible tags
    layers = tuple(layers)

    elapsed = time.time()
    thetas = NeuralNetwork.train(trainData[0], trainData[1], layers, 2, lamb, maxIter, verbose)
    elapsed = time.time() - elapsed

    print("Accuracy [ layers = ", layers, "; lambda = ", lamb, "; max iterations = ", maxIter,"]: ",
            NeuralNetwork.evaluate(thetas, testData[0], testData[1])*100, "%")
    print("Elapsed time: ", elapsed, "seconds")

def runSVM(trainData, testData, verbose, c, sigma, degree, coef0, kernel):
    print("\n*** Running SVM ***\n")

    elapsed = time.time()
    trained_svm = SVM.train(trainData[0], trainData[1], c, sigma, degree, coef0, kernel, verbose)
    elapsed = time.time() - elapsed

    print("Accuracy [C = ", c, "; sigma = ", sigma, "; degree =", degree,"; coef0 = ", coef0, "]: ", SVM.test(trained_svm, testData[0], testData[1]) * 100, "%")
    print("Elapsed time: ", elapsed, "seconds")

# Run all algorithms in sequence
def runAllAlgorithms(trainData, testData, verbose, lamb, c, sigma, hiddenLayers, lambNeural, maxIter):
    print("""
*** Running all algorithms ***
Depending on your machine specs, this may take some time\n""")

    logRegressionRegularized(trainData, testData, verbose, hideOutput, dontSave, lambd)
    neuralNetwork(trainData, testData, verbose, hideOutput, dontSave)
    runSVM(trainData, testData, verbose, hideOutput, dontSave, svm_c, sigma)

# MAIN FUNCTION
def main():
    # Command line arguements
    parser = argparse.ArgumentParser(description='Detect infected malaria cells using different machine learning algorithms')

    # General
    parser.add_argument('--all', '-a', help='Run all algorithms. If no argument is passed, this is the default behaviour', action="store_true")
    parser.add_argument('--verbose', '-v', help='Output detailed information for each algorithm execution', action="store_true")
    parser.add_argument('--noCache', '-c', help='Do not load cached data and do not cache generated image data', action="store_true")

    # Algorithms
    parser.add_argument('--logRegression', '-l', help='Execute the [regularized] logistic regression algorithm', action="store_true")
    parser.add_argument('--neuralNetwork', '-n', help='Execute the neural network algorithm', action="store_true")
    parser.add_argument('--svm', '-s', help='Execute the support vector machine algorithm', action="store_true")

    # Specific algorithm parameters
    parser.add_argument('--lamb', help='Lambda parameter for the regularized logistic regression algorithm', type=float, default=1)

    parser.add_argument('--svm_degree', help='Degree parameter for SVM algorithm (poly kernel)', type=int, default=3)
    parser.add_argument('--svm_c', help='C parameter for the SVM algorithm', type=float, default=1)
    parser.add_argument('--sigma', help='Sigma parameter for SVM algorithm', type=float, default=0.1)
    parser.add_argument('--kernel', help='Kernel to use in the SVM algorithm', type=str, default='linear')
    parser.add_argument('--coef0', help='coef0 parameter for SVM algorithm', type=float, default=0)

    parser.add_argument('--maxIter', help='Maximum iterations while training the neural network', type=int, default=30)
    parser.add_argument('--hiddenLayers', help='Number of neurons of each hidden layer of the neural network', nargs="+", type=int, default=(50))
    parser.add_argument('--lambNeural', help='Lambda parameter for the neural network algorithm', type=float, default=1)

    # Extra parameters
    parser.add_argument('--trainingBatch', help='Total number of images of EACH GROUP used for training', type=int, default=DEFAULT_TRAINING_BATCH)
    parser.add_argument('--testBatch', help='Total number of images of EACH GROUP used for test', type=int, default=DEFAULT_TEST_BATCH)
    parser.add_argument('--valBatch', help='Total number of images of EACH GROUP used for validation', type=int, default=DEFAULT_VAL_BATCH)

    args = parser.parse_args()
    runAll = True

    # Data must always be loaded
    loadResult = loadData(args.trainingBatch, args.testBatch, args.valBatch, not args.noCache)
    if (not loadResult[0]):
        return
    trainData = loadResult[1]
    valData = loadResult[2] # Currently unused!
    testData = loadResult[3]

    # Conditional execution depending on console arguments
    if (args.logRegression):
        logRegressionRegularized(trainData, testData, args.verbose, args.lamb)
        runAll = False;

    if (args.neuralNetwork):
        neuralNetwork(trainData, testData, args.verbose, args.hiddenLayers, args.lambNeural, args.maxIter)
        runAll = False;

    if (args.svm):
        runSVM(trainData, testData, args.verbose, args.svm_c, args.sigma, args.svm_degree, args.coef0, args.kernel)
        runAll = False;

    if (runAll):
        runAllAlgorithms(trainData, testData, args.verbose, args.lamb, args.svm_c, args.sigma, args.hiddenLayers, args.lambNeural, args.maxIter)

if __name__ == "__main__":
    main()
