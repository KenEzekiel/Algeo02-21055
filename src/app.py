# ---- Import Libraries ---- #
import sys
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import numpy
import os
import time
from image import resize_256
from operator import itemgetter
import threading
from model import Model
import base64
import math
from video import VideoCapture

# ---- Initialization Tkinter ---- #
window = Tk()
window.title('Face Recognition')
icon = PhotoImage(file='icon/logo.png')
window.tk.call('wm', 'iconphoto', window._w, icon)
video_capture = VideoCapture()

# ---- Set Up Window ---- #
processing = False

model = Model()

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

background_img = PhotoImage(file=f"background/background2.png")
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

img3 = PhotoImage(file = f"buttons/img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    # command = ,
    relief = "flat")

b3.place(
    x = 73, y = 522,
    width = 162,
    height = 52)

time_label_img = PhotoImage(file=f"textbox/img_textBox0.png")
time_label_bg = canvas.create_image(
    584.0, 496.5,
    image=time_label_img)

time_label = Label(
    bd=0,
    bg="#d2c6ff",
    highlightthickness=0,
    font=('Poppins 11 bold'),
    fg='#010030',
    justify='center')

time_label.bind('<Button-1>', lambda e: 'break')

time_label.place(
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

status_label_img = PhotoImage(file = f"textbox/img_textBox4.png")
status_label_bg = canvas.create_image(
    584.0, 547.5,
    image = status_label_img)

status_label = Label(
    bd = 0,
    bg = "#d2c6ff",
    highlightthickness = 0,
    font=('Poppins 11 bold'),
    fg='#010030',
    justify='center')

status_label.bind('<Button-1>', lambda e: 'break')

status_label.place(
    x = 545.5, y = 530,
    width = 77.0,
    height = 33)

result_label_img = PhotoImage(file=f"textbox/img_textBox3.png")
result_label_bg = canvas.create_image(
    873.5, 495.5,
    image=result_label_img)

result_label = Label(
    bd=0,
    bg="#d2c6ff",
    highlightthickness=0,
    font=('Poppins 11 bold'),
    fg='#010030',
    justify='center')

result_label.bind('<Button-1>', lambda e: 'break')

result_label.place(
    x=799.5, y=478,
    width=148.0,
    height=33)

left_img_label = Label()
left_img_label.place(
    x=382, y=188,
    anchor=NW, width=256, height=256
)

left_img_bg = PhotoImage(file="background/image_bg.png")
left_img_label.config(image=left_img_bg)
left_img_label.image = left_img_bg

canvas.pack()


def counting():
    start = time.time()
    while processing:
        time_label.config(text=f"{(time.time() - start):.4f}s")
        time.sleep(0.5)


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
    global imageRes
    start_time = time.time()

    encoded_path = base64.b64encode(
        pathdataset.encode('ascii')).decode('ascii')
    encoded_path = f"../test/_cache_/{encoded_path}"
    has_cache = os.path.exists(encoded_path + '.npz')

    if model.path != pathdataset:
        if has_cache:
            model.load_cache(pathdataset, encoded_path + '.npz')
        else:
            model.train(pathdataset)

    print(f"\n\nSIZE MODEL {sys.getsizeof(model)} \n\n")

    inputimage = imageInput

    image = numpy.transpose(inputimage)
    # image : 1 x N^2

    ulocal = model.u

    # ui dimensinya 1 x N^2 karena ditranspose
    # AT dimensinya M x N^2, diambil row nya jadi 1 x N^2
    # Kaliin semua row (gambar) dengan semua row (ui) matriks u
    # 1 gambar dikali k u, hasilnya ada k weight, vektor wi dimensinya 1 x k
    # ada sebanyak count gambar, matriks w jadinya count x k, nanti ditranspose jadi k x count

    wtest = numpy.empty(0, dtype=type(ulocal))
    for j in range(model.k):
        uj = ulocal[j]
        temp = numpy.dot(image, uj)
        wtest = numpy.append(wtest, temp)

    # wtest adalah weight dari test image jika dibandingkan dengan vektor u
    # vektor u adalah eigen vector

    # Cari euclidean distance terkecil dari w test dengan w data

    distance = getMagnitude(wtest - model.wdata[0])
    indexmin = 0

    # get distancelist
    distanceList = [(distance, indexmin)]

    for i in range(1, model.count):
        temp = getMagnitude(wtest - model.wdata[i])
        distanceList += [(temp, i)]
        if (temp < distance):
            distance = temp
            indexmin = i

    # distance adalah jarak euclidean terkecil
    # indexmin adalah index dengan w di wdata terdekat dengan wtest
    print(f'distance: {distance}')
    print(f'indexmin: {indexmin}')

    # rawimg : 256 x 256
    imageclosest = model.rawimg[indexmin]

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
        print(f'- {model.filesname[i[1]]} - {i[0]}')

    print("done")

    result_label.config(text=model.filesname[indexmin])
    global processing
    processing = False

    current_time = time.time()
    execution_time = '{:.4f}'.format(current_time - start_time)

    time_label.config(text=execution_time + "s")

    if not has_cache:
        model.save_cache(encoded_path)


def start_video():
    global video_capture, image, imageInput
    video_capture.start()
    last = time.time()
    while True:
        ret, frame = video_capture.get_image()
        if not ret:
            break
        resized_image = resize_256(frame)
        frame = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(frame)

        image = ImageTk.PhotoImage(frame)

        left_img_label.configure(image=image)
        left_img_label.image = image

    del video_capture


def getMagnitude(array):
    squaresum = 0
    for a in array:
        squaresum += a**2
    return math.sqrt(squaresum)


def select_dataset():
    if processing:
        return
    global pathdataset
    pathdataset = filedialog.askdirectory()

    if (len(pathdataset) != 0):
        current = entry1.get()
        entry1.delete(0, END)
        entry1.insert(0, os.path.basename(pathdataset))
    else:
        check_entry()


def select_image():
    if processing:
        return

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
