from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import numpy
import os

# def btn_clicked():
#     print("Button Clicked")

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

    for imgname in os.listdir(pathdataset):
        print(f'Processing {imgname}')
        img = cv2.imread(os.path.join(pathdataset, imgname), cv2.IMREAD_GRAYSCALE)
        img = resize_256(img)
        img = numpy.array(img.T).flatten()
        data += [img]
        sum = numpy.add(sum, img)
        count += 1

    print(f'File count: {count}')
    mean = numpy.divide(sum, count)

    for d in data:
        d = numpy.subtract(d, mean)

    print("done")

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

    return resized_image

def select_image():
    global image

    path = filedialog.askopenfilename()

    current = entry2.get()
    entry2.delete(0, END)
    entry2.insert(0, os.path.basename(path))

    if len(path) > 0:
        image = cv2.imread(path)

        resized_image = resize_256(image)

        # grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

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