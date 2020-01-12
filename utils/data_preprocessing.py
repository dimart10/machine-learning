import numpy as np

# Receives data as one numpy array (one row for each example attributes, the last
# column representing its classification group/class). Then returns that data
# separeted in its componentes (theArrayOfXs, theColumnOfYs, numberOfExamples, numberOfAttributes)
def separate_data(data):
    X = data[:, :-1]
    Y = data[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    return (X, Y, m, n)

def addInitialOnes(X):
    m = np.shape(X)[0]
    return np.hstack([np.ones([m, 1]), X])
