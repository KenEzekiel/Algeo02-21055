import cv2
import numpy
import os

def getKeigen(k, eigvalues, eigvectors, filecount):
    # eigvalues and eigvectors are numpy arrays
    eigvals = numpy.array([])
    eigvecs = numpy.empty((0, filecount), dtype=type(eigvectors))
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

# A = N^2 x M
# A = [a1, a2, .., am], A = data
npdata = numpy.array(data)
# karena masih M x N^2
print(npdata.shape)
# A : N^2 x M
npdata = npdata.transpose()
# print('1')
# A : N^2 x M, At : M x N^2
# M x M : At x A
Cov = numpy.matmul(npdata.transpose(), npdata)
# print('2')

w, v = numpy.linalg.eig(Cov)
print("w")
print(w)
print("v")
print(v.shape)
print(v)
# print('3')

# Now we select k = 5 of w ( max eigenvalues )
k = 5
# eigvals = numpy.array([])
# eigvecs = numpy.empty((0, count), dtype=type(v))

eigvals, eigvecs = getKeigen(k, w, v, count)
# vt = v.transpose()
# for i in range(k):
#     print(i)
#     a = w.max()
#     print("max:", a)
#     eigvals = numpy.append(eigvals, [a])
#     index = numpy.where(w == a)
#     print("idx:", index)
#     w[index] = 0
#     vec = vt[index]
#     print("vec:", vec)
#     eigvecs = numpy.concatenate((eigvecs, vec), axis = 0)

# eigvals = lambda
print("eigvals:", eigvals)
# eigvecs = v
print("eigvecs (still transposed):", eigvecs)

# ui = A.vi
u = numpy.empty((0, 256*256), dtype=type(v))
for i in range(k):
    # ambil row vi, terus transpose, terus dikali A, hasilnya ui yang N^2 x 1 (A: N^2 x M, vi : M x 1)
    # ditranspose biar gampang masukinnya
    vi = eigvecs[i].transpose()
    print("vi: ")
    print(vi)
    print(vi.size)
    print(npdata.size)
    ui = numpy.matmul(npdata, vi)
    ui = ui.transpose()
    # print("u", u)
    print("ui", ui)
    print("size ui:", ui.size)
    u = numpy.vstack((u, ui))
    # u = numpy.concatenate((u, ui), axis = 0)

print("u:", u)
# u = u.transpose()
AT = npdata.transpose()


w = numpy.empty((0, k), dtype = type(u))
# ui dimensinya 1 x N^2 karena ditranspose
# AT dimensinya M x N^2, diambil row nya jadi 1 x N^2
# Kaliin semua row (gambar) dengan semua row (ui) matriks u
# 1 gambar dikali k u, hasilnya ada k weight, vektor wi dimensinya 1 x k
# ada sebanyak count gambar, matriks w jadinya count x k, nanti ditranspose jadi k x count
for i in range(count):
    wi = numpy.empty(0, dtype = type(u)) 
    Ai = AT[i]
    for j in range(k):
        uj = u[j]
        temp = numpy.dot(Ai, uj)
        wi = numpy.append(wi, temp)
        # print(wi.size)
        # print(wi)
    w = numpy.vstack((w, wi))

print("w:", w)
# count x k
print(w.size)
w = numpy.transpose(w)
# k x count
print(w.size)



# contoh meanface
testmeanface = mean.reshape(-1, 256).T
# print(testmeanface)
cv2.imwrite(r"..\test\hasil\mean123.jpg", testmeanface)