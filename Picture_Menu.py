

## Interactive menu for changing picture

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 08:43:04 2022

@author: Bruger
"""

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
options = np.array(['Load picture', 'Set threshold, blur, margin, etc', 'Display modified picture', \
    'Get autocorrelation', 'Save', 'Quit'])
# optionsSettings = np.array(['Set blur', 'Set clipoutrange and pixel to mm conversionrate', 'Set threshold',\
#      'Set picture length', 'Display clipout', 'Done'])


while True:
    choice = displayMenu(options)
    
    #region Load Data
    if (choice == 'Load picture'):
        fileName = inputFilename()
        image = skimage.io.imread(fname=fileName, as_gray=True)
    #endregion Load Data
    
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
                blur = float(input('Please type the standard devation of the blur: '))
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
        print('We have found you coordinates to be (x,y) = ({:d},{:d}) and your margins \
            to be ({:d},{:d})'.format(xMiddle, yMiddle, marginX, marginY))

        clip, blurredClip = clipBlur(fileName, xMiddle, yMiddle, marginX, marginY, sigma = blur)
        skimage.io.imshow(clip)
    #endregion Display Statistics        
        
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
    
    #region Autocorrelation
    elif(choice == 'Get autocorrelation'):
        # try:
        clip, blurredClip = clipBlur(fileName, xMiddle, yMiddle, marginX, marginY, sigma = blur)
        clip[blurredClip > threshold] = 0
        auflength = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = 1)
        print(f"ACL er {auflength:0.2} mm")
        # except:
        #     print('You need to load some options before you can get the autocorrelation')
    

    #region Save
    elif(choice == 'Save'):
        filePath = fileName.split("/")
        filename = filePath[-1].split(".")[0]
        Matrix = np.empty((1,11))
        Matrix[:] = np.NaN
        txtName = filename + '.txt'
        Matrix[0,0:11] = np.array([blur, top, bottom, left, right, \
            conversion, threshold, marginY, marginX, yMiddle, xMiddle])
        np.savetxt(txtName, Matrix, delimiter=',')
    #endregion Save

    #region Quit
    elif (choice == 'Quit'):
        break
    #endregion Quit


