import numpy

def getKeigen(vt, k, eigvalues, eigvectors):
    # eigvalues and eigvectors are numpy arrays
    vt = eigvectors.transpose()
    for i in range(k):
        print(i)
        a = eigvalues.max()
        print("max:", a)
        eigvals = numpy.append(eigvals, [a])
        index = numpy.where(eigvalues == a)
        print("idx:", index)
        eigvalues[index] = 0
        vec = vt[index]
        print("vec:", vec)
        eigvecs = numpy.concatenate((eigvecs, vec), axis = 0)
    
    return eigvals, eigvecs
