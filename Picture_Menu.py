

## Interactive menu for changing picture

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 08:43:04 2022

@author: Bruger
"""

from asyncio.windows_events import NULL
from inputNumber import inputNumber
from displayMenu import displayMenu
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
# from autocofunc import *
# from autocolen import *
# from plot_acf import *
from inputFilename import *
from acf import *



# Create the options for the menus that are to be used later.
options = np.array(['Load picture', 'Set threshold, blur, margin, etc', 'Set threshold, sigma', 'Display modified picture', \
    'Get autocorrelation', 'Plot autocorrelation', 'Save', 'Quit'])
# optionsSettings = np.array(['Set blur', 'Set clipoutrange and pixel to mm conversionrate', 'Set threshold',\
#      'Set picture length', 'Display clipout', 'Done'])

fTypes = np.array(['null','Exponential','Gaussian','Exp Root'])
fit = np.array([1,2,3])


while True:
    choice = displayMenu(options)
    
    #----------------------------------------------------------

    #region Load Data
    if (choice == 'Load picture'):
        fileName = inputFilename()
        image = skimage.io.imread(fname=fileName, as_gray=True)
    #endregion Load Data
    
    #----------------------------------------------------------

    #region Dislay statistics
    elif (choice == 'Set threshold, blur, margin, etc'):
        
        # First we have to test wether or not the image is defined
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(x=image.flatten(), bins=256, range=(0,1))
        # ax1.title('frequency of grayscale pigments.')
        # ax1.xlabel('Brightness')
        ax2.imshow(image, cmap = 'gray')
        start, end = ax2.get_xlim()
        ax2.set_xticks(np.arange(start, end, 500))
        fig.set_figheight(5)
        fig.set_figwidth(11)
        plt.show(block=False)
        

        # If the data is defined we can display the statistics
        while True:
            try:
                blur = float(input('Please type the standard devation of the blur (0.25 or 2): '))
                break
            except ValueError:
                print('You did not input a real number')
            
        while True:
            try:
                top = int(input('Please type the top height you want to analyze (in pixels): '))
                break
            except ValueError:
                print('You did not input an integer and no value has been saved')
        
        while True:
            try:
                bottom = int(input('Please type the bottom height you want to analyze (in pixels): '))
                break
            except ValueError:
                print('You did not input an integer and no value has been saved')

        while True:
            try: 
                left = int(input('Please type the leftmost points height you want to analyze (in pixels): '))
                break
            except ValueError:
                print('You did not input an integer and no value has been saved')
        
        while True:
            try:
                right = int(input('Please type the rightmost points height you want to analyze (in pixels): '))
                break
            except ValueError:
                print('You did not input an integer and no value has been saved')
        
        while True:
            try:
                conversion = 90 / float(input('Please type the pixel length of the slab of ice in the horizontal direction: '))
                # The conversion is given as 90 mm / pixellength of the horizontal slab of ice
                break
            except ValueError:
                print('You did not input a number and no conversion has been saved')

        while True:
            try:
                threshold = float(input('Please type the threshold you would like: '))
                break
            except ValueError:
                print('You did not input a number')
        
        marginY = int((bottom - top) / 2)
        marginX = int((right - left) / 2)
        yMiddle = top + marginY
        xMiddle = left + marginX
        print('We have found you coordinates to be (x,y) = ({:d},{:d}) and your margins to be ({:d},{:d})'.format(xMiddle, \
            yMiddle, marginX, marginY))

        clip, blurredClip = clipBlur(fileName, xMiddle, yMiddle, marginX, marginY, sigma = blur)
        skimage.io.imshow(clip)
    #endregion Display Statistics        

    #----------------------------------------------------------

    elif(choice == 'Set threshold, sigma'):
        while True:
            try:
                threshold = float(input('Please type the threshold you would like (try 0.65): '))
                break
            except ValueError:
                print('You did not input a number')
        while True:
            try:
                blur = float(input('Please type the standard devation of the blur (0.25 or 2): '))
                break
            except ValueError:
                print('You did not input a real number')
    
    #----------------------------------------------------------

    #region Display modified picture
    elif(choice == 'Display modified picture'):
        try:
            clip, blurredClip = clipBlur(fileName, xMiddle, yMiddle, marginX, marginY, sigma = blur)
            clip[blurredClip > threshold] = 1
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(clip, cmap = 'gray')
            ax2.imshow(image, cmap = 'gray')
            fig.set_figheight(5)
            fig.set_figwidth(11)
            plt.show(block=False)
            clip[blurredClip > threshold] = 0
        except:
            print("You need to load picture first!")
    #endregion Display modified picture
    
    #----------------------------------------------------------

    #region Autocorrelation
    elif(choice == 'Get autocorrelation'):
        clip, blurredClip = clipBlur(fileName, xMiddle, yMiddle, marginX, marginY, sigma = blur)
        clip[blurredClip > threshold] = 0
        auflength  = np.empty(3)
        auflength, funcType, plotdata = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = fit)
        print("ACL is {:0.2f} mm, {:0.2f} mm and {:0.2f} mm and function type is {}".format(auflength[0], \
            auflength[1], auflength[2], funcType))
    #endregion Autocorrelation


    elif(choice == 'Plot autocorrelation'):
        xmax = inputNumber('Select length of plot in [mm]: ')
        plot_acf2(auflength, fTypes[fit], plotdata, xmax = xmax)

    #----------------------------------------------------------

    #region Save
    elif(choice == 'Save'):
        filePath = fileName.split("/")
        filename = filePath[-1].split(".")[0]
        Matrix = np.empty(11, dtype = 'object')
        auflength  = np.empty(3, dtype = 'U56')
        auflength,funcType = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = fit)
        txtName = 'variables/' + filename + '.txt'
        Matrix[0:11] = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
            auflength[1], auflength[2], funcType]
        np.savetxt(txtName, Matrix, delimiter=' ', newline = "\n", fmt = "%s")
    #endregion Save

    #----------------------------------------------------------

    #region Quit
    elif (choice == 'Quit'):
        print("Have a nice day :)")
        break
    #endregion Quit

    #----------------------------------------------------------

