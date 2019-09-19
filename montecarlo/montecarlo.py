import math
import random
import scipy.integrate as integrate
import numpy as np
import time

def integra_mc_iterativa(fun, a, b, num_puntos=10000):

    #VARIABLE PARA CONTROLAR EL TIEMPO
    tic = time.time()

    #HALLAMOS M
    M = 0
    for i in np.arange(a, b, (b-a)/num_puntos):
        aux = fun(i)
        if aux > M :
            M = aux
    print("\n__Integral iterativa__")
    print("Máximo: ", M)

    #QUEDA DEFINIDO EL CUADRADO ENTRE A-B Y 0-M
    #LANZAMOS PUNTOS ALEATORIOS DENTRO DEL CUADRADO

    nDebajo = 0

    for i in range(num_puntos):
        x = random.uniform(a, b)
        y = random.uniform(0, M)

        if fun(x)>y :
            nDebajo+=1

    print("N Debajo: ", nDebajo)
    print("N Totales: ", num_puntos)

    I = (nDebajo/num_puntos)*(b-a)*M
    print ("I montecarlo: ", I)

    toc = time.time()
    print("Tiempo: ", 1000*(toc - tic))

def integra_mc_vectores(fun, a, b, num_puntos=10000):
    
    #VARIABLE PARA CONTROLAR EL TIEMPO
    tic = time.time()

   #HALLAMOS M
    vector = np.arange(a, b, (b-a)/num_puntos)

    vectorizedFun = np.vectorize(fun)
    M = vectorizedFun(vector).max()

    print("\n__Integral vectorial__")
    print("Máximo: ", M)
    
    #CREAMOS VECTORES CON PUNTOS
    vectorX = np.random.uniform(low = a, high = b, size =(num_puntos))
    vectorY = np.random.uniform(low = 0, high = M, size =(num_puntos))

    #CONTAMOS LOS PUNTOS DEBAJO
    nDebajo = np.greater(vectorizedFun(vectorX), vectorY).sum()

    print("N Debajo: ", nDebajo)
    print("N Totales: ", num_puntos)

    #HALLAMOS I CON LOS DATOS OBTENIDOS
    I = (nDebajo/num_puntos)*(b-a)*M
    print ("I montecarlo: ", I)

    toc = time.time()
    print("Tiempo: ", 1000*(toc - tic))

integra_mc_iterativa(math.sin, 0, math.pi)
integra_mc_vectores(math.sin, 0, math.pi)

print("\nI real: ", integrate.quad(math.sin, 0, math.pi)[0], "\n")