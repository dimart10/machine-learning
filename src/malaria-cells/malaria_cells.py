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
import regression.regresion_logistic as LogisticReg
import neural_networks.neural_network as NeuralNetwork


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
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80

TRAINING_BATCH = 15000
VAL_BATCH = 5000
TEST_BATCH = 7000

# Load & preprocess data
def loadData(cache = True):
    print("\nLoading data...")

    # First, check if the data has already been cached, if so load it and return
    if (cache and os.path.isfile(CACHE_PATH + PARASITIZED_CACHE_FILE + CACHE_EXTENSION)
            and os.path.isfile(CACHE_PATH + UNINFECTED_CACHE_FILE + CACHE_EXTENSION)):
        print("Image data is being loaded from cache")
        parasitized = np.load(CACHE_PATH + PARASITIZED_CACHE_FILE + CACHE_EXTENSION)
        uninfected = np.load(CACHE_PATH + UNINFECTED_CACHE_FILE + CACHE_EXTENSION)
        return(True, formatData(parasitized, uninfected))

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

    return (True, formatData(parasitized, uninfected))

# Returns the data loaded from disk in the standard format
def formatData(parasitized, uninfected):
    n = parasitized.shape[1]

    formatted = np.vstack((np.insert(parasitized, n, 1, 1),
                            np.insert(uninfected, n, 0, 1)))
    formatted = Preprocessing.separate_data(formatted)

    return formatted

def logisticRegression(data, verbose, hideOutput, dontSave):
    print("\n*** Running logistic regression algorithm ***\n")
    LogisticReg.train(Preprocessing.addInitialOnes(data[0]), data[1])

    if (not hideOutput):
        print("Error percentage: ", LogisticReg.evaluate(thetas, X, Y)*100, "%")

def neuralNetwork(data, verbose, hideOutput, dontSave):
    print("\n*** Running neural network ***\n")
    NeuralNetwork.train(data[0], data[1], (19200, 2000, 2), 2, 1)

    if (not hideOutput):
        print("Error percentage: ", NeuralNetwork.evaluate(thetas, X, Y)*100, "%")


# Run all algorithms in sequence
def runAllAlgorithms(data, verbose, hideOutput, dontSave):
    print("""
*** Running all algorithms ***
Depending on your machine specs, this may take some time\n""")

    logisticRegression(data, verbose, hideOutput, dontSave)
    neuralNetwork(data, verbose, hideOutput, dontSave)

# MAIN FUNCTION
def main():
    # Command line arguements
    parser = argparse.ArgumentParser(description='Detect infected malaria cells using different machine learning algorithms')

    parser.add_argument('--all', '-a', help='Run all algorithms. If no argument is passed, this is the default behaviour', action="store_true")
    parser.add_argument('--verbose', '-v', help='Output detailed information for each algorithm execution', action="store_true")
    parser.add_argument('--hideOutput', '-o', help='Do not show result graphs & statistics for the executed algorithms', action="store_true")
    parser.add_argument('--dontSave', '-d', help='Do not save to disk result graphs & statistics for the executed algorithms', action="store_true")
    parser.add_argument('--noCache', '-c', help='Do not load cached data and do not cache generated data', action="store_true")
    parser.add_argument('--logisticRegression', '-l', help='Execute the logistic regression algorithm', action="store_true")
    parser.add_argument('--neuralNetwork', '-n', help='Execute the neural network algorithm', action="store_true")

    args = parser.parse_args()
    runAll = True

    # Data must always be loaded
    loadResult = loadData(not args.noCache)
    if (not loadResult[0]):
        return
    data = loadResult[1]

    # Conditional execution depending on console arguments
    if (args.logisticRegression):
        logisticRegression(data, args.verbose, args.hideOutput, args.dontSave)
        runAll = False;

    if (args.neuralNetwork):
        neuralNetwork(data, args.verbose, args.hideOutput, args.dontSave)
        runAll = False;

    if (runAll):
        runAllAlgorithms(data, args.verbose, args.hideOutput, args.dontSave)

if __name__ == "__main__":
    main()
