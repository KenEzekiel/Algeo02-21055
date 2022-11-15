import cv2
import numpy
import os

count = 0
data = []
sum = [0 for i in range(256 * 256)]
path = '..//test//contoh'
for foldername in os.listdir(path):
    print(f'Processing {foldername}')
    path2 = os.path.join(path, foldername)
    for imgname in os.listdir(path2):
        img = cv2.imread(os.path.join(path2, imgname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = numpy.array(img.T).flatten()
        data += [img]
        sum = numpy.add(sum, img)
        count += 1

print(f'File count: {count}')
mean = numpy.divide(sum, count)

for d in data:
    d = numpy.subtract(d, mean)

# ini kenapa jadi M x N^2? pas di add, di add ke row ya?
# data M x N^2
# A = [a1, a2, .., am], A = data
npdata = numpy.matrix(data)
print(npdata.shape)
# A : N^2 x M
npdata = npdata.transpose()
print('1')
# A : N^2 x M, At : M x N^2
# M x M : At x A
Cov = numpy.matmul(npdata.transpose(), npdata)
print('2')

w, v = numpy.linalg.eig(Cov)
print("w")
print(w)
print("v")
print(v.shape)
print(v)
print('3')

# Now we select k = 5 of w ( max eigenvalues )
k = 5
eigvals = numpy.array([])
eigvecs = numpy.empty((0, count), dtype=type(v))

vt = v.transpose()
for i in range(k):
    print(i)
    a = w.max()
    print("max:", a)
    eigvals = numpy.append(eigvals, [a])
    index = numpy.where(w == a)
    print("idx:", index)
    w[index] = 0
    vec = vt[index]
    print("vec:", vec)
    eigvecs = numpy.concatenate((eigvecs, vec), axis = 0)

# eigvals = lambda
print("eigvals:", eigvals)
# eigvecs = v
print("eigvecs (still transposed):", eigvecs)

# ui = A.vi
u = numpy.empty((0, count), dtype=type(v))
for i in range(k):
    # ambil row vi, terus transpose, terus dikali A, hasilnya langsung ui yang N^2 x 1 (A: N^2 x M, vi : M x 1)
    vi = eigvecs[i].transpose()
    print(vi)
    print(vi.size)
    print(npdata.size)
    ui = numpy.matmul(npdata, vi)
    u = numpy.concatenate((u, ui), axis = 1)

print("u:", u)





# contoh meanface
testmeanface = mean.reshape(-1, 256).T
# print(testmeanface)
cv2.imwrite(r"..\test\hasil\mean123.jpg", testmeanface)