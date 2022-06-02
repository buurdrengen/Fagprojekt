
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
from acf import *

files = os.getcwd() + '\\variables'
images = os.getcwd() + '\\images'

print('')
print(files)
print(images)
print(' ')

[yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength, funcType] = [1,1,1,1,1,1,1,[1,1,1],['','','']]

variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
    auflength[1], auflength[2], funcType]

print(len(variables))

filename = "20200213_170702.txt"

with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
# print(np.arange(len(variables)))

    for i in np.arange(len(variables)):
        variables[i] = f.readline()
        variables[i] = variables[i][0:-1]

    # Now I find the fileplacement of the files 
    filenameM = images + '\\' + filename[0:-4] + '.jpg'

    # Here i swap the ymiddle and xmiddle as well as the marginY and marginX because the picture will be transposed later on.
    xMiddle = int(variables[0])
    yMiddle = int(variables[1])
    marginX = int(variables[2])
    marginY = int(variables[3])
    [conversion, blur, threshold] = np.float64(variables[4:7])

    ## Get the autocorrelation lengths.
    image = skimage.io.imread(fname=filenameM, as_gray=True)
    # Transpose the image so we get the autocorrelation on horizontally.
    image = image.T
    print('')
    print(filename)

    clip, blurredClip = clipBlur(filenameM, xMiddle, yMiddle, marginX, marginY, sigma = blur)
    clip[blurredClip > threshold] = 0
    auflength  = np.empty(3)
    auflength, funcType, j = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = [1,2])



