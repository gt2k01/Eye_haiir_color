# Importing libraries

from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys
import pandas as pd
from ast import Lambda
from cProfile import label
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import font
from turtle import heading
from PIL import Image, ImageTk


# Importing trainged model facial landmarks detector
shape_predictor= "shape_predictor_68_face_landmarks.dat" 

# Making GUI window
top = tk.Tk()
top.geometry('800x600')
top.title('Hair and Eye Color Detector')
top.configure(background='#CDCDCD')

# Initializing the labels (1 for hair and 1 for eye)
label1 = Label(top, background = '#CDCDCD', font = ('arial', 15, "bold"))
label2 = Label(top, background = '#CDCDCD', font = ('arial', 15, 'bold'))
sign_image = Label(top)

# Reading from colors csv file that for which RGB value which color is predicted. We want its name
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


# This is our core function it predicts the color of eye and hair
def Detect(file_path):
    global label_packed
    img = cv2.imread(file_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
    (i, j)=(43, 45)
    eye_points= []
    for (x, y) in shape[i:j]:
        eye_points.append([x,y])

    hair_points= []
    for (x, y) in shape[i:j]:
        hair_points.append([x,y//4])

    rmin = min(hair_points[0][1], hair_points[1][1])
    rmax = max(hair_points[0][1], hair_points[1][1])
    cmin = min( hair_points[0][0], hair_points[1][0])
    cmax = max( hair_points[0][0], hair_points[1][0])
    arr = np.array( img[rmin : rmax, cmin : cmax] )
    b = int(arr[:,:, 0].mean())
    g = int(arr[:,:, 1].mean())
    r = int(arr[:, :, 2].mean())

    eye_row = max( eye_points[0][1], eye_points[1][1])
    eye_col = min( eye_points[0][0], eye_points[1][0])
    eye_arr = np.array( img[eye_row+5: eye_row+16, eye_col + 5])
    eb = int( eye_arr[:,0].mean() )
    eg = int( eye_arr[:,1].mean() )
    er = int( eye_arr[:,2].mean() )

    eye_color = get_color_name(er, eg, eb)
    hair_color = get_color_name(r, g, b)


    print( "Eye color is "+ eye_color)
    print( "Hair color is " + hair_color)
    label1.configure(foreground="#011638", text= 'Eye color is ' + eye_color)
    label2.configure(foreground='#011638', text= 'Hair color is ' + hair_color) 


# GUI botton to order the detection
def show_Detect_Button(file_path):
    Detect_b = Button(top, text="Detect Image", command= lambda: Detect(file_path), padx=10, pady=5 )
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# To select our image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail( ( ( top.winfo_width()/2.25 ), ( top.winfo_height()/2.25 ) ) )
        im = ImageTk.PhotoImage( uploaded )

        sign_image.configure( image = im )
        sign_image.image = im
        label1.configure( text='')
        label2.configure(text='')
        show_Detect_Button(file_path)
    except:
        pass


# GUI for our dialog
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font = ('arial', 10, 'bold'))
upload.pack(side='bottom', pady= 50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
heading = Label(top, text="Eye and Hair color detector", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()