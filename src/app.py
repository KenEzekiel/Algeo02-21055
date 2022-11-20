# ---- Import Libraries ---- #
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import numpy
import os
import time
import eigen
from operator import itemgetter
import threading

# ---- Initialization Tkinter ---- #
window = Tk()
window.title('Face Recognition')
icon = PhotoImage(file='icon/logo.png')
window.tk.call('wm', 'iconphoto', window._w, icon)

# ---- Set Up Window ---- #
processing = False

window.geometry("1080x600")
window.configure(bg="#ffffff")
canvas = Canvas(
    window,
    bg="#ffffff",
    height=600,
    width=1080,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

background_img = PhotoImage(file=f"background/background.png")
background = canvas.create_image(
    540.0, 300.0,
    image=background_img)

img0 = PhotoImage(file=f"buttons/img0.png")
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: select_dataset(),
    relief="flat")

b0.place(
    x=75, y=219,
    width=160,
    height=50)

img1 = PhotoImage(file=f"buttons/img1.png")
b1 = Button(
    image=img1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: select_image(),
    relief="flat")

b1.place(
    x=75, y=352,
    width=160,
    height=50)

img2 = PhotoImage(file=f"buttons/img2.png")
b2 = Button(
    image=img2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: start_thread(),
    relief="flat")

b2.place(
    x=75, y=463,
    width=160,
    height=50)

entry0_img = PhotoImage(file=f"textbox/img_textBox0.png")
entry0_bg = canvas.create_image(
    584.0, 496.5,
    image=entry0_img)

entry0 = Label(
    bd=0,
    bg="#d2c6ff",
    highlightthickness=0,
    font=('Poppins 11 bold'),
    fg='#010030',
    justify='center')

entry0.bind('<Button-1>', lambda e: 'break')

entry0.place(
    x=545.5, y=479,
    width=77.0,
    height=33)

entry1_img = PhotoImage(file=f"textbox/img_textBox1.png")
entry1_bg = canvas.create_image(
    155.0, 293.0,
    image=entry1_img)

entry1 = Entry(
    bd=0,
    bg="#86a1ff",
    highlightthickness=0,
    font=('Poppins 11 bold'),
    fg='#ffffff',
    justify='center')

entry1.bind('<Button-1>', lambda e: 'break')

entry1.place(
    x=90.0, y=278,
    width=130.0,
    height=24)

entry2_img = PhotoImage(file=f"textbox/img_textBox2.png")
entry2_bg = canvas.create_image(
    155.0, 427.0,
    image=entry2_img)

entry2 = Entry(
    bd=0,
    bg="#86a1ff",
    highlightthickness=0,
    font=('Poppins 11 bold'),
    fg='#ffffff',
    justify='center')

entry2.bind('<Button-1>', lambda e: 'break')

entry2.place(
    x=90.0, y=412,
    width=130.0,
    height=24)

entry3_img = PhotoImage(file=f"textbox/img_textBox3.png")
entry3_bg = canvas.create_image(
    873.5, 495.5,
    image=entry3_img)

entry3 = Entry(
    bd=0,
    bg="#d2c6ff",
    highlightthickness=0,
    font=('Poppins 11 bold'),
    fg='#010030',
    justify='center')

entry3.bind('<Button-1>', lambda e: 'break')

entry3.place(
    x=799.5, y=478,
    width=148.0,
    height=33)

canvas.pack()


def counting():
    start = time.time()
    while processing:
        time.sleep(0.5)
        entry0.config(text=f"{(time.time() - start):.4f}s")


def start_thread():
    global processing
    if processing:
        return
    processing = True
    threading.Thread(target=start).start()
    threading.Thread(target=counting).start()


def check_entry():
    var1 = entry1.get()
    var2 = entry2.get()
    if var1 == '':
        entry1.insert(0, "No File Chosen")
    if var2 == '':
        entry2.insert(0, "No File Chosen")


check_entry()


def start():
    sum = [0 for i in range(256 * 256)]
    count = 0
    global imageRes
    data = []       # data yang diproses
    data2 = []      # data asli (untuk di display untuk hasil)
    filesname = []  # nama-nama dari file yang dibaca

    start_time = time.time()
    for imgname in os.listdir(pathdataset):
        print(f'Processing {imgname}')
        img = cv2.imread(os.path.join(pathdataset, imgname),
                         cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(
            pathdataset, imgname), cv2.COLOR_BGR2RGB)

        img = resize_256(img)
        img2 = resize_256(img2)

        img = numpy.array(img.T).flatten()

        data += [img]
        data2 += [img2]
        filesname += [imgname]

        sum = numpy.add(sum, img)
        count += 1

    # rawimg untuk yang ditampilin nanti
    rawimg = data2
    print(f'File count: {count}')
    mean = numpy.divide(sum, count)

    for d in data:
        d = numpy.subtract(d, mean)

    # A = N^2 x M
    # A = [a1, a2, .., am], A = data
    npdata = numpy.array(data)
    # karena masih M x N^2
    # A : N^2 x M
    npdata = npdata.transpose()
    # A : N^2 x M, At : M x N^2
    # M x M : At x A
    Cov = numpy.matmul(npdata.transpose(), npdata)

    # get eig vals and eig vectors
    w, v = eigen.eigenvectors_qr(Cov)
    # w, v = numpy.linalg.eig(Cov)

    # Now we select k = 5 of w ( max eigenvalues )
    k = 5

    eigvals, eigvecs = getKeigen(k, w, v, count)

    # ui = A.vi
    u = numpy.empty((0, 256*256), dtype=type(v))
    for i in range(k):
        # ambil row vi, terus transpose, terus dikali A, hasilnya ui yang N^2 x 1 (A: N^2 x M, vi : M x 1)
        # ditranspose biar gampang masukinnya
        vi = eigvecs[i].transpose()

        ui = numpy.matmul(npdata, vi)
        ui = ui.transpose()

        u = numpy.vstack((u, ui))

    AT = npdata.transpose()

    w = numpy.empty((0, k), dtype=type(u))
    # ui dimensinya 1 x N^2 karena ditranspose
    # AT dimensinya M x N^2, diambil row nya jadi 1 x N^2
    # Kaliin semua row (gambar) dengan semua row (ui) matriks u
    # 1 gambar dikali k u, hasilnya ada k weight, vektor wi dimensinya 1 x k
    # ada sebanyak count gambar, matriks w jadinya count x k, nanti ditranspose jadi k x count
    for i in range(count):
        wi = numpy.empty(0, dtype=type(u))
        Ai = AT[i]
        for j in range(k):
            uj = u[j]
            temp = numpy.dot(Ai, uj)
            wi = numpy.append(wi, temp)
        w = numpy.vstack((w, wi))

    w = numpy.transpose(w)
    # k x count

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

    wtest = numpy.empty(0, dtype=type(ulocal))
    for j in range(k):
        uj = ulocal[j]
        temp = numpy.dot(image, uj)
        wtest = numpy.append(wtest, temp)

    # wtest adalah weight dari test image jika dibandingkan dengan vektor u
    # vektor u adalah eigen vector

    # Cari euclidean distance terkecil dari w test dengan w data

    distance = numpy.linalg.norm(wtest - wdata[0])
    indexmin = 0

    # get distancelist
    distanceList = [(distance, indexmin)]

    for i in range(1, count):
        temp = numpy.linalg.norm(wtest - wdata[i])
        distanceList += [(temp, i)]
        if (temp < distance):
            distance = temp
            indexmin = i

    # distance adalah jarak euclidean terkecil
    # indexmin adalah index dengan w di wdata terdekat dengan wtest
    print(f'distance: {distance}')
    print(f'indexmin: {indexmin}')

    # rawimg : 256 x 256
    imageclosest = rawimg[indexmin]

    # Hasil
    testRes = cv2.cvtColor(imageclosest, cv2.COLOR_BGR2RGB)

    imageRes = Image.fromarray(testRes)

    imageRes = ImageTk.PhotoImage(imageRes)

    canvas.create_image(
        707, 188,
        anchor=NW,
        image=imageRes)

    print("3 most closest:")
    distanceList = sorted(distanceList, key=itemgetter(0))[
        :3]    # get 3 lowest
    for i in distanceList:
        print(f'- {filesname[i[1]]} - {i[0]}')

    print("done")

    entry3.delete(0, END)
    entry3.insert(0, filesname[indexmin])
    global processing
    processing = False

    current_time = time.time()
    execution_time = '{:.4f}'.format(current_time - start_time)

    entry0.config(text=execution_time + "s")


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


def select_dataset():
    global pathdataset
    pathdataset = filedialog.askdirectory()

    if (len(pathdataset) != 0):
        current = entry1.get()
        entry1.delete(0, END)
        entry1.insert(0, os.path.basename(pathdataset))
    else:
        check_entry()


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

    # dimensi yang diinginkan 256 x 256
    dim = (256, 256)

    resized_image = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LINEAR)

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

        # yang dibandingin
        imageInput = numpy.array(resized_grayscale.T).flatten()

        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        image = ImageTk.PhotoImage(image)

        canvas.create_image(
            382, 188,
            anchor=NW,
            image=image)
    else:
        check_entry()


window.resizable(False, False)
window.mainloop()
