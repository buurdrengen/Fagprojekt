


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


# The variables vector was defined as having the following positions
# variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
#     auflength[1], auflength[2], funcType]

variables = [1,1,1,1,1,1,1,[1,1,1],['','','']]


for filename in os.listdir(files):
    with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
        # print(np.arange(len(variables)))
        
        for i in np.arange(len(variables)):
            variables[i] = f.readline()
            variables[i] = variables[i][0:-1]
    

    # Now I find the fileplacement of the files 
    filenameM = images + '\\' + filename[0:-4] + '.jpg'
    filenameSave = 'variables_nonT\\' + filename


    # Here i swap the ymiddle and xmiddle as well as the marginY and marginX because the picture will be transposed later on.
    yMiddle = int(variables[0])
    xMiddle = int(variables[1])
    marginY = int(variables[2])
    marginX = int(variables[3])
    [conversion, blur, threshold] = np.float64(variables[4:7])

    ## Get the autocorrelation lengths.
    image = skimage.io.imread(fname=filenameM, as_gray=True)
    print('')
    print(filenameSave)

    clip, blurredClip = clipBlur(filenameM, xMiddle, yMiddle, marginX, marginY, sigma = blur)
    clip[blurredClip > threshold] = 0
    auflength  = np.empty(3)
    auflength, funcType, plotdata = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = [1,2])
    
    variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
        auflength[1], auflength[2], funcType]

    np.savetxt(filenameSave, variables, delimiter=' ', newline = "\n", fmt = "%s")



    


        