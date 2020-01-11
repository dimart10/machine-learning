import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas

# DATA LOADING & PREPARATION
def loadData():
    print("\nLoading data...")
    return True

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
    if (not loadData()):
        return

    # Conditional execution depending on console arguments
    if (args.logisticRegression):
        logisticRegression(args.verbose, args.hideOutput, args.dontSave)
        runAll = False;

    if (runAll):
        runAllAlgorithms(args.verbose, args.hideOutput, args.dontSave)

if __name__ == "__main__":
    main()
