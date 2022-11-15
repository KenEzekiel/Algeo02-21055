import numpy as np
import sys

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
a = np.array([[0.5, 0.5, 1.2], [0, 0.8, 1.5], [0, 0, 0.3]])
p = [1, 5, 10, 20]
for i in range(20):
    q, r = np.linalg.qr(a)
    a = np.dot(r, q)
    if i+1 in p:
        print(f'Iteration {i+1}:')
        print(a)