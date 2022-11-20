import numpy
import os
import cv2
import eigen
from image import resize_256


class Model:
    def __init__(self) -> None:
        self.path = None

        # select k = 5 ( max eigenvalues )
        self.k = 5

    def train(self, path: str) -> None:
        self.path = path
        sum = [0 for i in range(256 * 256)]
        self.count = 0
        self.data = []       # data yang diproses
        self.data2 = []      # data asli (untuk di display untuk hasil)
        self.filesname = []  # nama-nama dari file yang dibaca
        for imgname in os.listdir(path):
            print(f'Processing {imgname}')
            img = cv2.imread(os.path.join(path, imgname),
                             cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(
                path, imgname), cv2.COLOR_BGR2RGB)

            img = resize_256(img)
            img2 = resize_256(img2)

            img = numpy.array(img.T).flatten()

            self.data += [img]
            self.data2 += [img2]
            self.filesname += [imgname]

            sum = numpy.add(sum, img)
            self.count += 1

        # rawimg untuk yang ditampilin nanti
        self.rawimg = self.data2
        print(f'File count: {self.count}')
        mean = numpy.divide(sum, self.count)

        for d in self.data:
            d = numpy.subtract(d, mean)

        # A = N^2 x M
        # A = [a1, a2, .., am], A = data
        npdata = numpy.array(self.data)
        # karena masih M x N^2
        # A : N^2 x M
        npdata = npdata.transpose()
        # A : N^2 x M, At : M x N^2
        # M x M : At x A
        Cov = numpy.matmul(npdata.transpose(), npdata)

        # get eig vals and eig vectors
        w, v = eigen.eigenvectors_qr(Cov)
        # w, v = numpy.linalg.eig(Cov)

        eigvals, eigvecs = self.getKeigen(self.k, w, v, self.count)

        # ui = A.vi
        self.u = numpy.empty((0, 256*256), dtype=type(v))
        for i in range(self.k):
            # ambil row vi, terus transpose, terus dikali A, hasilnya ui yang N^2 x 1 (A: N^2 x M, vi : M x 1)
            # ditranspose biar gampang masukinnya
            vi = eigvecs[i].transpose()

            ui = numpy.matmul(npdata, vi)
            ui = ui.transpose()

            self.u = numpy.vstack((self.u, ui))

        AT = npdata.transpose()

        w = numpy.empty((0, self.k), dtype=type(self.u))
        # ui dimensinya 1 x N^2 karena ditranspose
        # AT dimensinya M x N^2, diambil row nya jadi 1 x N^2
        # Kaliin semua row (gambar) dengan semua row (ui) matriks u
        # 1 gambar dikali k u, hasilnya ada k weight, vektor wi dimensinya 1 x k
        # ada sebanyak count gambar, matriks w jadinya count x k, nanti ditranspose jadi k x count
        for i in range(self.count):
            wi = numpy.empty(0, dtype=type(self.u))
            Ai = AT[i]
            for j in range(self.k):
                uj = self.u[j]
                temp = numpy.dot(Ai, uj)
                wi = numpy.append(wi, temp)
            w = numpy.vstack((w, wi))

        w = numpy.transpose(w)
        # k x count

        # penyocokan image
        # test.w : k x count (data set file)
        self.wdata = numpy.transpose(w)

        # wdata : count x k

    @staticmethod
    def getKeigen(k, eigvalues, eigvectors, filecount):
        # eigvalues and eigvectors are numpy arrays
        eigvals = numpy.array([])
        eigvecs = numpy.empty((0, filecount), dtype=type(eigvectors))
        vt = eigvectors.transpose()
        for i in range(k):
            a = eigvalues.max()
            eigvals = numpy.append(eigvals, [a])
            index = numpy.where(eigvalues == a)
            eigvalues[index] = 0
            vec = vt[index]
            # print("vec:", vec)
            eigvecs = numpy.concatenate((eigvecs, vec), axis=0)

        return eigvals, eigvecs

    def save_cache(self, path: str) -> None:
        # save data
        numpy.savez(path, u=self.u, wdata=self.wdata, rawimg=self.rawimg,
                    filesname=self.filesname, count=self.count)

    def load_cache(self, path: str, encoded_path: str) -> None:
        # load data
        self.path = path
        data = numpy.load(encoded_path, allow_pickle=True)
        self.u = data['u']
        self.wdata = data['wdata']
        self.rawimg = data['rawimg']
        self.filesname = data['filesname']
        self.count = data['count']
