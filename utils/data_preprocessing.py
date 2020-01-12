import numpy as np

# Receives data as one numpy array (one row for each example attributes, the last
# column representing its classification group/class). Then returns that data
# separeted in its componentes (theArrayOfXs, theColumnOfYs, numberOfExamples, numberOfAttributes)
def separate_data(data, addInitialOnes = True):
    X = data[:, :-1]
    Y = data[:, -1][np.newaxis].T

    m = np.shape(X)[0]
    if (addInitialOnes): X = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X)[1]

    return (X, Y, m, n)
