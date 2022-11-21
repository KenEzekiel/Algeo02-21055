# Algeo02-21055

## Deskripsi Permasalahan

Pengenalan wajah (Face Recognition) adalah teknologi biometrik yang bisa dipakai untuk mengidentifikasi wajah seseorang untuk berbagai kepentingan khususnya keamanan. Program pengenalan wajah melibatkan kumpulan citra wajah yang sudah disimpan pada database lalu berdasarkan kumpulan citra wajah tersebut, program dapat mempelajari bentuk wajah lalu mencocokkan antara kumpulan citra wajah yang sudah dipelajari dengan citra yang akan diidentifikasi

Terdapat berbagai teknik untuk memeriksa citra wajah dari kumpulan citra yang sudah diketahui seperti jarak Euclidean dan cosine similarity, principal component analysis (PCA), serta Eigenface. Pada Projek ini, akan dibuat sebuah program pengenalan wajah menggunakan Eigenface.

Eigenface adalah teknik yang menggunakan matriks dari wajah-wajah, yang akan dihitung matriks eigenface nya, dan dalam pembentukannya, menggunakan eigenvector.


## Anggota Kelompok

| NIM      | Nama                         | Tanggung Jawab                                                       |
| -------- | ---------------------------- | -------------------------------------------------------------------- |
| 13521055 | Muhammad Bangkit Dwi Cahyono | GUI, Pemrosesan wajah menjadi matriks, resize                        |
| 13521089 | Kenneth Ezekiel Suprantoni   | Algoritma Pencocokan Utama dengan PCA dan Eigenface                  |
| 13521101 | Arsa Izdihar Islam           | Algoritma Eigenvalue & Eigenvector, implementasi pada video          |

## Struktur Program
```
.
│ 
├── doc
│   └── Algeo02-21055.pdf
│ 
├── src
│   ├── _pycache_
│   ├── background
│   │   ├─── background.png
│   │   ├─── background2.png
│   │   └─── image_bg.png
│   │
│   ├── buttons
│   │   ├─── img0.png		
│   │   ├─── img1.png		
│   │   ├─── img2.png	
│   │   ├─── img3.png		
│   │   └─── img4.png
│   │
│   ├── icon
│   │   └─── logo.png		
│   │
│   ├── textbox
│   │   ├─── img_textBox0.png
│   │   ├─── img_textBox1.png
│   │   ├─── img_textBox2.png
│   │   ├─── img_textBox3.png
│   │   └─── img_textBox4.png
│   │
│   ├── app.py
│   ├── eigen.py
│   ├── image.py
│   ├── model.py
│   ├── utilities.py
│   └── video.py
│  
├── test
│   ├── _pycache_
│   ├── dataset
│   └── pencocokan.py
│
└──README.md  
```

## Technologies Used
1. tkinter
2. numpy
3. cv2
4. PIL
5. threading

## How To Use
1. Please make sure you've installed tkinter and all the above technologies that we use
2. In `src/` folder type `python app.py`
3. Begin to use our app, select your own dataset and your own test-image

## Features
1. Image face recognition
2. Video face recognition

## Screenshots
![alt text](https://github.com/KenEzekiel/Algeo02-21055/blob/main/test/ss.jpg?raw=true)

## Project Status
Project is: complete
