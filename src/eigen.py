import time
import numpy as np


'''
# Reading order of matrix
n = int(input('Enter order of matrix: '))

# Making numpy array of n x n size and initializing 
# to zero for storing matrix
a = np.zeros((n,n))

# Reading matrix
print('Enter Matrix Coefficients:')
for i in range(n):
    for j in range(n):
        a[i][j] = float(input( 'a['+str(i)+']['+ str(j)+']='))

# Making numpy array n x 1 size and initializing to zero
# for storing initial guess vector
x = np.zeros((n))

# Reading initial guess vector
print('Enter initial guess vector: ')
for i in range(n):
    x[i] = float(input( 'x['+str(i)+']='))

# Reading tolerable error
tolerable_error = float(input('Enter tolerable error: '))

# Reading maximum number of steps
max_iteration = int(input('Enter maximum number of steps: '))

# Power Method Implementation
lambda_old = 1.0
condition =  True
step = 1
while condition:
    # Multiplying a and x
    x = np.matmul(a,x)
    
    # Finding new Eigen value and Eigen vector
    lambda_new = max(abs(x))
    
    x = x/lambda_new
    
    # Displaying Eigen value and Eigen Vector
    print('\nSTEP %d' %(step))
    print('----------')
    print('Eigen Value = %0.4f' %(lambda_new))
    print('Eigen Vector: ')
    for i in range(n):
        print('%0.3f\t' % (x[i]))
    
    # Checking maximum iteration
    step = step + 1
    if step > max_iteration:
        print('Not convergent in given maximum iteration!')
        break
    
    # Calculating error
    error = abs(lambda_new - lambda_old)
    print('errror='+ str(error))
    lambda_old = lambda_new
    condition = error > tolerable_error
'''


def qr(A: np.matrix):
    """ QR decomposition dari matriks A menggunakan householder reflection """
    # https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections
    m, n = A.shape

    # Q0 = I ukuran mxm
    Q = np.eye(m)

    # R0 = A
    R = A.copy()

    # jumlah iterasi
    # setelah iterasi, R = Qt..Q2Q1A, Q = Q1..Qt
    t = min(m, n)

    for k in range(t - 1):
        # x adalah minor dari A pada iterasi ke k di-transpose
        x = R[k:, [k]]

        # e1 = [1 0 ... 0]T
        e1 = np.zeros_like(x)
        e1[0] = 1.0

        # alpha = kebalikan sign x[0] * ||x||
        alpha = -np.sign(x[0]) * np.linalg.norm(x)

        # u = x - alpha * e1
        u = (x - alpha * e1)

        # v = u / ||u||
        v = u / np.linalg.norm(u)

        # Q_k = householder matrix = I - 2vvT
        Qk = np.eye(m - k) - 2.0 * v @ v.T

        # karena x adalah matriks minor, berikan padding 1 pada diagonal dan 0 pada sisanya sehingga Q_k ukurannya m x m
        Qk = np.block([[np.eye(k), np.zeros((k, m - k))],
                       [np.zeros((m - k, k)), Qk]])

        # persiapan untuk iterasi selanjutnya
        Q = Q @ Qk.T
        R = Qk @ R

    return Q, R


def eigenvectors_qr(A: np.matrix, iters=50):
    """ Menghitung eigenvektor dari matriks A dengan metode QR eksplisit """

    Ak = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for _ in range(iters):
        Q, R = qr(Ak)
        Ak = R @ Q
        QQ = QQ @ Q
    return np.array(np.diag(Ak)), QQ
