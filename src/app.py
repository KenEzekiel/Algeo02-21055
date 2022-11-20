from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import numpy
import os
import time
import eigen

window = Tk()

window.geometry("1080x600")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 600,
    width = 1080,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background/background.png")
background = canvas.create_image(
    540.0, 300.0,
    image=background_img)

img0 = PhotoImage(file = f"buttons/img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda: select_dataset(),
    relief = "flat")

b0.place(
    x = 75, y = 219,
    width = 160,
    height = 50)

img1 = PhotoImage(file = f"buttons/img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda: select_image(),
    relief = "flat")

b1.place(
    x = 75, y = 352,
    width = 160,
    height = 50)

img2 = PhotoImage(file = f"buttons/img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda: start(),
    relief = "flat")

b2.place(
    x = 75, y = 463,
    width = 160,
    height = 50)

entry0_img = PhotoImage(file = f"textbox/img_textBox0.png")
entry0_bg = canvas.create_image(
    584.0, 496.5,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#d2c6ff",
    highlightthickness = 0,
    font = ('Poppins', 10))

entry0.bind('<Button-1>', lambda e: 'break')

entry0.place(
    x = 545.5, y = 479,
    width = 77.0,
    height = 33)

entry1_img = PhotoImage(file = f"textbox/img_textBox1.png")
entry1_bg = canvas.create_image(
    155.0, 293.0,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#86a1ff",
    highlightthickness = 0,
    font = ('Poppins', 10))

entry1.bind('<Button-1>', lambda e: 'break')

entry1.place(
    x = 90.0, y = 278,
    width = 130.0,
    height = 24)

entry2_img = PhotoImage(file = f"textbox/img_textBox2.png")
entry2_bg = canvas.create_image(
    155.0, 427.0,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#86a1ff",
    highlightthickness = 0,
    font = ('Poppins', 10))

entry2.bind('<Button-1>', lambda e: 'break')

entry2.place(
    x = 90.0, y = 412,
    width = 130.0,
    height = 24)

entry3_img = PhotoImage(file = f"textbox/img_textBox3.png")
entry3_bg = canvas.create_image(
    873.5, 495.5,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#d2c6ff",
    highlightthickness = 0,
    font = ('Poppins', 10))

entry3.bind('<Button-1>', lambda e: 'break')

entry3.place(
    x = 799.5, y = 478,
    width = 148.0,
    height = 33)

canvas.pack()

def start():
    sum = [0 for i in range(256 * 256)]
    count = 0
    global imageRes
    data = []

    start_time = time.time()

    for imgname in os.listdir(pathdataset):
        print(f'Processing {imgname}')
        img = cv2.imread(os.path.join(pathdataset, imgname), cv2.IMREAD_GRAYSCALE)
        img = resize_256(img)
        img = numpy.array(img.T).flatten()
        data += [img]
        sum = numpy.add(sum, img)
        count += 1

    rawimg = data
    # print(f'shape rawimg {rawimg[0].shape}')
    print(f'File count: {count}')
    mean = numpy.divide(sum, count)

    for d in data:
        d = numpy.subtract(d, mean)

    print("done")

    # A = N^2 x M
    # A = [a1, a2, .., am], A = data
    npdata = numpy.array(data)
    # karena masih M x N^2
    # print(npdata.shape)
    # A : N^2 x M
    npdata = npdata.transpose()
    # print('1')
    # A : N^2 x M, At : M x N^2
    # M x M : At x A
    Cov = numpy.matmul(npdata.transpose(), npdata)
    # print('2')

    w, v = eigen.eigenvectors_qr(Cov)
    # print("w")
    # print(w)
    # print("v")
    # print(v.shape)
    # print(v)
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
    # print("eigvals:", eigvals)
    # eigvecs = v
    # print("eigvecs (still transposed):", eigvecs)

    # ui = A.vi
    u = numpy.empty((0, 256*256), dtype=type(v))
    for i in range(k):
        # ambil row vi, terus transpose, terus dikali A, hasilnya ui yang N^2 x 1 (A: N^2 x M, vi : M x 1)
        # ditranspose biar gampang masukinnya
        vi = eigvecs[i].transpose()
        # print("vi: ")
        # print(vi)
        # print(vi.size)
        # print(npdata.size)
        ui = numpy.matmul(npdata, vi)
        ui = ui.transpose()
        # print("u", u)
        # print("ui", ui)
        # print("size ui:", ui.size)
        u = numpy.vstack((u, ui))
        # u = numpy.concatenate((u, ui), axis = 0)

    # print("u:", u)
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

    # print("w:", w)
    # count x k
    print(w.size)
    w = numpy.transpose(w)
    # k x count
    print(w.size)

    inputimage = imageInput

    # penyocokan image
    # test.w : k x count (data set file)
    wdata = numpy.transpose(w)
    # wdata : count x k

    # u local : k x N^2
    ulocal = u

    image = numpy.transpose(inputimage)
    # image : 1 x N^2

    # ui dimensinya 1 x N^2 karena ditranspose
    # AT dimensinya M x N^2, diambil row nya jadi 1 x N^2
    # Kaliin semua row (gambar) dengan semua row (ui) matriks u
    # 1 gambar dikali k u, hasilnya ada k weight, vektor wi dimensinya 1 x k
    # ada sebanyak count gambar, matriks w jadinya count x k, nanti ditranspose jadi k x count

    wtest = numpy.empty(0, dtype = type(ulocal)) 
    for j in range(k):
        uj = ulocal[j]
        temp = numpy.dot(image, uj)
        wtest = numpy.append(wtest, temp)
        # print(wi.size)
        # print(wi)

    # wtest adalah weight dari test image jika dibandingkan dengan vektor u
    # vektor u adalah eigen vector

    # Cari euclidean distance terkecil dari w test dengan w data
    distance = getMagnitude(wtest - wdata[0])
    indexmin = 0
    for i in range(1, count):
        temp = getMagnitude(wtest - wdata[i])
        print(temp)
        if (temp < distance):
            distance = temp
            indexmin = i
    # distance adalah jarak euclidean terkecil
    # indexmin adalah index dengan w di wdata terdekat dengan wtest
    print(distance)
    print(indexmin)
    print(f'file count: {count}')

    # rawimg : N^2 x M, ditranspose jadi M x N^2
    # image closest : 1 x N^2
    rawimage = numpy.transpose(rawimg)
    imageclosest = rawimg[indexmin]
    # ditranspose jadi N^2 x 1
    imageclosest = imageclosest.T

    # contoh hasil
    testRes = imageclosest.reshape(-1, 256).T
    # print(testmeanface)
    cv2.imwrite(r"..\test\hasil\test1.jpg", testRes)

    print("donedone")

    current_time = time.time()
    execution_time = '{:.4f}'.format(current_time - start_time)

    current = entry0.get()
    entry0.delete(0, END)
    entry0.insert(0, execution_time + "s")

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

def getMagnitude(array):
    squaresum = 0
    for a in array:
        squaresum += a**2
    return math.sqrt(squaresum)

def select_dataset():
    global pathdataset
    pathdataset = filedialog.askdirectory()

    current = entry1.get()
    entry1.delete(0, END)
    entry1.insert(0, os.path.basename(pathdataset))

def resize_256(image):
    width, height = image.shape[1], image.shape[0]

    if (width > height):
        start_row = 0
        end_row = height

        start_col = (width - height) / 2
        end_col = width - start_col
    else:
        max = width
        start_row = (height - width) / 2
        end_row = height - start_row

        start_col = 0
        end_col = width

    crop_img = image[int(start_row):int(end_row), int(start_col):int(end_col)]

    dim = (256, 256)

    resized_image = cv2.resize(crop_img, dim, interpolation = cv2.INTER_LINEAR)

    # print(f'res {resized_image.shape}')

    return resized_image

def select_image():
    global image, imageInput

    path = filedialog.askopenfilename()

    current = entry2.get()
    entry2.delete(0, END)
    entry2.insert(0, os.path.basename(path))

    if len(path) > 0:
        image = cv2.imread(path)
        grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        resized_image = resize_256(image)
        resized_grayscale = resize_256(grayscale) 

        # print(f'resized {resized_grayscale.shape}')
        imageInput = numpy.array(resized_grayscale.T).flatten()

        # print(f'grayscale {imageInput.shape}')

        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        # grayscale = Image.fromarray(grayscale)

        image = ImageTk.PhotoImage(image)
        # grayscale = ImageTk.PhotoImage(grayscale)
        
        canvas.create_image(
            382, 188,
            anchor=NW,
            image=image)

        # canvas.create_image(
        #     707, 188,
        #     anchor=NW,
        #     image=grayscale)

window.resizable(False, False)
window.mainloop()