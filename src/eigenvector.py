import numpy as np


# M is a rectangular matrix, eigen are an array of its eigenvalues
def getVectors(A, eigen):
    I = np.identity(len(A))
    Zero = [0 for i in range(len(A))]
    allvector = []

    for val in eigen:
        MinusLambdaI = I * (-1) * val
        #print(MinusLambdaI)
        M = A + MinusLambdaI
        print("M:")
        print(M)
        
        #x = np.linalg.solve(M, Zero)
        #np.append(allvector, x)

    print(allvector)

    np.transpose(allvector)

    
    return


M = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
getVectors(M, [1, 5, 9])


