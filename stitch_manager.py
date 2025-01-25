'''
MORE SOURCES:
https://www.geeksforgeeks.org/python-gui-tkinter/
https://docs.python.org/3/library/tk.html
https://www.geeksforgeeks.org/python-tkinter-mainloop/
https://pythonspot.com/tk-file-dialogs/#google_vignette
'''
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from image_stitcher import stitch
import os
from math import sqrt

#Base Root Window
root = tk.Tk()#Sets name of root/parent window
root.title('TJUAV Image Stitcher')#Window title
root.geometry("960x540")#Size

selectedFolder = None

def selectFolder():#Uses filedialog to select a director (folder)
    selectedFolder = filedialog.askdirectory()
    labelSelection.config(text = "Selected Folder: " + selectedFolder)

def runImageStitcher():
    folder = labelSelection.cget("text")
    listFolder = os.listdir(folder[17:])
    imgGridWidth = int(sqrt(len(listFolder)))
    if imgGridWidth % 2 != 0:
        imgGridWidth += 1
    loadImagesInGrid(folder[17:],imgGridWidth)

def loadImagesInGrid(folder, imgGridWidth):
    counter = 0
    for file in os.listdir(folder):
        img = Image.open(folder + "\\" + file)
        img = img.resize((20, 20))
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image = img)
        panel.image = img
        panel.grid(row = int(counter / imgGridWidth) + 3, column = int(counter % imgGridWidth) + 1, pady = 2)
        #print(str((counter % 10) + 3) + ", " + str(int(counter / 2)))
        counter += 1

#Menubar Configuration
menubar = tk.Menu(root)#Menubar is child of root
root.config(menu = menubar)#Sets root's menu field to menubar child

file = tk.Menu(menubar, tearoff = 0) #file is child of menu
#Adds stuff
menubar.add_cascade(label ='File', menu = file)#Opening style
#file.add_command(label ='New File', command = None) 
file.add_command(label ='Open Folder', command = selectFolder) 
file.add_command(label ='Export', command = None) 
file.add_separator() 
file.add_command(label ='Exit', command = root.quit)

#Main Stuff Configuration
labelIntro = tk.Label(root, text='Welcome to the TJUAV Image Stitcher. Select a folder then press run')
labelIntro.grid(row = 0, column = 0, pady = 2)

labelSelection = tk.Label(root, text='Selected Folder: None')
labelSelection.grid(row = 1, column = 0, pady = 2)

buttonRun = tk.Button(root, text = 'RUN IMAGE STITCHING', width = 25, command = runImageStitcher)
buttonRun.grid(row = 2, column = 0, pady = 2)

root.mainloop()



#stitch('C:\\Users\jaspe\Documents\Github_Local\mapping-tjuav\ImgSampleD', debug = 2, type = '.png', panorama_name = 'DimgsALL')