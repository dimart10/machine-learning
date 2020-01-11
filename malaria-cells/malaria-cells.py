import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas

PARASITIZED_PATH = "../data/heavy/malaria-cells/Parasitized/"
UNINFECTED_PATH = "../data/heavy/malaria-cells/Uninfected/"

FILE_EXTENSION = "png"

TRAINING_BATCH = 15000
VAL_BATCH = 5000
TEST_BATCH = 7000

# DATA LOADING & PREPARATION
def loadData():
    print("\nLoading data...")

    if (not os.path.isdir(PARASITIZED_PATH) or not os.path.isdir(UNINFECTED_PATH)):
        print("\nData directories don't exist, please read the README.md of this folder"
                + "to learn how to download and setup the data\n")
        return False

    parasitized_filenames = [f for f in os.listdir(PARASITIZED_PATH) if f.endswith('.' + FILE_EXTENSION)]
    uninfected_filenames = [f for f in os.listdir(UNINFECTED_PATH) if f.endswith('.' + FILE_EXTENSION)]

    if (len(parasitized_filenames) <= 0 or len(uninfected_filenames) <= 0):
        print("\nData is missing or missplaced, please read the README.md of this folder"
                + "to learn how to download and setup the data")
        return False

    # Load every image as a 1D array
    parasitized = []
    for filename in parasitized_filenames:
        parasitized.append(imread(PARASITIZED_PATH + filename).ravel())
        parasitized[-1].ravel()
    parasitized = np.array(parasitized)

    uninfected = []
    for filename in uninfected_filenames:
        uninfected.append(imread(UNINFECTED_PATH + filename).ravel())
    uninfected = np.array(uninfected)

    return (True, parasitized, uninfected)

# LOGISTIC REGRESSION
def logisticRegression(verbose, hideOutput, dontSave):
    print("\n*** Running logistic regression ***\n")

def runAllAlgorithms(verbose, hideOutput, dontSave):
    print("""
*** Running all algorithms ***
Depending on your machine specs, this may take some time\n""")

# MAIN FUNCTION
def main():
    # Command line arguements
    parser = argparse.ArgumentParser(description='Detect infected malaria cells using different machine learning algorithms')

    parser.add_argument('--all', '-a', help='Run all algorithms. If no argument is passed, this is the default behaviour', action="store_true")
    parser.add_argument('--verbose', '-v', help='Output detailed information for each algorithm execution', action="store_true")
    parser.add_argument('--hideOutput', '-o', help='Do not show result graphs & statistics for the executed algorithms', action="store_true")
    parser.add_argument('--dontSave', '-d', help='Do not save to disk result graphs & statistics for the executed algorithms', action="store_true")
    parser.add_argument('--logisticRegression', '-l', help='Execute the logistic regression algorithm', action="store_true")

    args = parser.parse_args()
    runAll = True

    # Data must always be loaded
    if (not loadData()[0]):
        return

    # Conditional execution depending on console arguments
    if (args.logisticRegression):
        logisticRegression(args.verbose, args.hideOutput, args.dontSave)
        runAll = False;

    if (runAll):
        runAllAlgorithms(args.verbose, args.hideOutput, args.dontSave)

if __name__ == "__main__":
    main()
