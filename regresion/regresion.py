import numpy as np
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    
 # suponemos que siempre trabajaremos con float
    return valores.astype(float)

print(carga_csv("ex1data1.csv"))