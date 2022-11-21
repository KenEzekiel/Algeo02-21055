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
