import numpy as np
import test
import math

def getMagnitude(array):
    squaresum = 0
    for a in array:
        squaresum += a**2
    return math.sqrt(squaresum)


# input image


inputimage = np.empty((256*256, 1))

# penyocokan image
# test.w : k x count (data set file)
wdata = np.transpose(test.w)
# wdata : count x k

# u local : k x N^2
ulocal = test.u

image = np.transpose(inputimage)
# image : 1 x N^2


# ui dimensinya 1 x N^2 karena ditranspose
# AT dimensinya M x N^2, diambil row nya jadi 1 x N^2
# Kaliin semua row (gambar) dengan semua row (ui) matriks u
# 1 gambar dikali k u, hasilnya ada k weight, vektor wi dimensinya 1 x k
# ada sebanyak count gambar, matriks w jadinya count x k, nanti ditranspose jadi k x count

wtest = np.empty(0, dtype = type(ulocal)) 
for j in range(test.k):
    uj = ulocal[j]
    temp = np.dot(image, uj)
    wtest = np.append(wtest, temp)
    # print(wi.size)
    # print(wi)

# wtest adalah weight dari test image jika dibandingkan dengan vektor u
# vektor u adalah eigen vector

# Cari euclidean distance terkecil dari w test dengan w data
distance = getMagnitude(wtest - wdata[0])
indexmin = 0
for i in range(1, test.count):
    temp = getMagnitude(wtest - wdata[i])
    if (temp < distance):
        distance = temp
        indexmin = i
# distance adalah jarak euclidean terkecil
# indexmin adalah index dengan w di wdata terdekat dengan wtest
print(distance)
print(indexmin)
