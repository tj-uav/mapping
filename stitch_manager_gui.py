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
import time

#Base Root Window
root = tk.Tk()#Sets name of root/parent window
root.title('TJUAV Image Stitcher')#Window title
root.geometry("960x540")#Size

selectedFolder = None

def selectFolder():#Uses filedialog to select a director (folder)
    selectedFolder = filedialog.askdirectory()
    labelSelection.config(text = "Selected Folder: " + selectedFolder)

def runImageStitcher():
    #Grab the folder
    folder = labelSelection.cget("text")
    listFolder = os.listdir(folder[17:])
    imgGridWidth = int(sqrt(len(listFolder)) * 0.75)#Make it a nice square-ish in the interface
    if imgGridWidth % 2 != 0:#But even bc it looks nicer
        imgGridWidth += 1
    load_imgs_into_grid(folder[17:],imgGridWidth)#Load into interface
    print('cool things happen now?')
    time.sleep(2)
    print('cool things happen now?')
    #stitching_manager(listFolder, len(listFolder))

def stitching_manager(folder, items):
    for item in folder:
        update_image(item, 'red')
        time.sleep(1)

def update_image(filename, new_color):
    for label in root.grid_slaves():
        if label.cget("text") == filename:
            label.configure(bg = new_color)
            return 'OK'
    return 'ERR_IMG_NOT_FOUND'

def load_imgs_into_grid(folder, imgGridWidth):
    counter = 0
    for file in os.listdir(folder):
        #Merci https://stackoverflow.com/questions/10133856/how-to-add-an-image-in-tkinter for getting this to work
        # img = Image.open(folder + "\\" + file)
        # imgSize = int(500/imgGridWidth)
        # img = img.resize((imgSize, imgSize))
        # img = ImageTk.PhotoImage(img)
        # panel = tk.Label(root, image = img)
        panel = tk.Label(root, text = file, background = 'gold')
        #panel.image = img # Yeah idk y u need to double declare, but it works and its just visual so idc
        #Put it into a grid
        panel.grid(row = int(counter / imgGridWidth) + 3, column = int(counter % imgGridWidth) + 1, padx = 2, pady = 2)
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

#root.mainloop()



#stitch('C:\\Users\jaspe\Documents\Github_Local\mapping-tjuav\ImgSampleD', debug = 2, type = '.png', panorama_name = 'DimgsALL')