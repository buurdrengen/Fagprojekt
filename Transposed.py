
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
from acf import *

files = os.getcwd() + '\\variables'
images = os.getcwd() + '\\images'


fTypes = np.array(['null','Exponential','Gaussian','Exp Root', 'x Power', 'x Exponential'])
fit = np.array([1,2,4,5])


# The variables vector was defined as having the following positions
# variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
#     auflength[1], auflength[2], funcType]

variables = [1,1,1,1,1,1,1,[1,1,1],['','','']]


for filename in os.listdir(files):
    with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
    
        
        for i in np.arange(len(variables)):
            variables[i] = f.readline()
            variables[i] = variables[i][0:-1]
    

    # Now I find the fileplacement of the files 
    filenameM = images + '\\' + filename[0:-4] + '.jpg'
    filenameSave = 'variables_T\\' + filename[0:-4] + '_T.txt'
    imageSave = 'mod_image\\' + filename[0:-4] + '_mod.jpg'

    # print('')
    # print(filename)
    # print(imageSave)

    yMiddle = int(variables[0])
    xMiddle = int(variables[1])
    marginY = int(variables[2])
    marginX = int(variables[3])
    [conversion, blur, threshold] = np.float64(variables[4:7])

    ## Get the autocorrelation lengths.
    # image = skimage.io.imread(fname=filenameM, as_gray=True)

    clip, blurredClip = clipBlur(filenameM, xMiddle, yMiddle, marginX, marginY, sigma = blur)
    clip[blurredClip > threshold] = 0
    # Transpose the image so we get the autocorrelation on horizontally. image
    clip = clip.T
    blurredClip = blurredClip.T
    auflength  = np.empty(3)
    uncertainty = np.empty(3)
    confint = np.empty(3)
    RMSE = np.empty([3,4])
    kvalue = np.empty(1)
    xvalue = np.empty(1)

    # RMSE is the root mean square error of 
    auflength, confint, funcType, plotdata, RMSE, kvalue, xvalue = acf(clip, lags = marginY-1, conversion = conversion, plot = False, plotfunc = fit)
    print(funcType)
    
    variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
        auflength[1], auflength[2], funcType]
    


    np.savetxt(filenameSave, variables, delimiter=' ', newline = "\n", fmt = "%s")

    clip[blurredClip > threshold] = 1
    clip = np.uint8(clip*255)
    skimage.io.imsave(imageSave, clip)





    



        