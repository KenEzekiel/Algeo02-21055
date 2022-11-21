import numpy
import math

def getKeigen(k, eigvalues, eigvectors, filecount):
    # eigvalues and eigvectors are numpy arrays
    eigvals = numpy.array([])
    eigvecs = numpy.empty((0, filecount), dtype=type(eigvectors))
    vt = eigvectors.transpose()
    for i in range(k):
        # print(i)
        a = eigvalues.max()
        # print("max:", a)
        eigvals = numpy.append(eigvals, [a])
        index = numpy.where(eigvalues == a)
        # print("idx:", index)
        eigvalues[index] = 0
        vec = vt[index]
        # print("vec:", vec)
        eigvecs = numpy.concatenate((eigvecs, vec), axis = 0)
    
    return eigvals, eigvecs

def getMagnitude(array):
    squaresum = 0
    for a in array:
        squaresum += a**2
    return math.sqrt(squaresum)
