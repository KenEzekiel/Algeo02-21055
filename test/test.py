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

# data N^2 x M

# contoh meanface
testmeanface = mean.reshape(-1, 256).T
# print(testmeanface)
cv2.imwrite(r"..\test\hasil\mean123.jpg", testmeanface)