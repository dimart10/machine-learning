# This script uses the Montecarlo method to find the
# [approximated] area under a curve (definite integral)

import math
import random
import time

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def montecarlo_iterative_integration(fun, a, b, num_puntos=1000):
    # To keep track of time
    tic = time.time()

    #HALLAMOS M
    M = 0
    for i in np.arange(a, b, (b-a)/num_puntos):
        aux = fun(i)
        if aux > M :
            M = aux
    print("\n__Integral (iterative)__")
    print("Máximum: ", M)

    #QUEDA DEFINIDO EL CUADRADO ENTRE A-B Y 0-M
    #LANZAMOS PUNTOS ALEATORIOS DENTRO DEL CUADRADO

    nDebajo = 0

    for i in range(num_puntos):
        x = random.uniform(a, b)
        y = random.uniform(0, M)

        if fun(x)>y :
            nDebajo+=1

    print("Points under curve: ", nDebajo)
    print("Total points: ", num_puntos)

    I = (nDebajo/num_puntos)*(b-a)*M
    print ("Montecarlo integral: ", I)

    toc = time.time()
    print("time: ", 1000*(toc - tic))

    return 1000*(toc - tic)

def montecarlo_vectorized_integration(fun, a, b, num_puntos=1000):
    # To keep track of time
    tic = time.time()

    # Vectors with uniformly distributed points
    vectorX = np.random.uniform(low = a, high = b, size =(num_puntos))

    vectorizedFun = np.vectorize(fun)
    funResults = vectorizedFun(vectorX)
    M = funResults.max()
    vectorY = np.random.uniform(low = 0, high = M, size =(num_puntos))

    print("\n__Integral (vectorized)__")
    print("Máximum: ", M)

    # Points under the curve
    nDebajo = np.greater(funResults, vectorY).sum()

    print("Points under curve: ", nDebajo)
    print("Total points: ", num_puntos)

    # Value of the integral (Montecarlo approximation)
    I = (nDebajo/num_puntos)*(b-a)*M
    print ("Montecarlo integral: ", I)

    toc = time.time()
    print("Time: ", 1000*(toc - tic))

    return 1000*(toc - tic)


iterativeArray = []
vectorizedArray = []
iValues = []

# Compare both methods (iterative and vectorized) performance when using
# increasingly larger number of points
for i in range(1, 1000000, 100000):
    iterativeArray.append(montecarlo_iterative_integration(math.sin, 0, math.pi, i))
    vectorizedArray.append(montecarlo_vectorized_integration(math.sin, 0, math.pi, i))
    iValues.append(i)

# Show and save the results
plt.figure()
plt.scatter(iValues, iterativeArray, color='red', label="Iterative")
plt.scatter(iValues, vectorizedArray, color='blue', label="Vectorized")
plt.ylabel('Time comparison')
plt.legend()
plt.savefig("time_comparison.png")
plt.show()

print("\nReal integral value: ", integrate.quad(math.sin, 0, math.pi)[0], "\n")
