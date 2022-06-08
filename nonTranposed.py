


import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
from acf import acf, plot_acf2
from plot_threshold import plot_threshold
from plot_sigma import plot_sigma

files = os.getcwd() + '/variables'
images = os.getcwd() + '/images'

fit = np.array([1,2,4,5])


# The variables vector was defined as having the following positions
# variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
#     auflength[1], auflength[2], funcType]

variables = [1,1,1,1,1,1,1,[1,1,1],['','','']]
saveFile = [[1.0,1.0,1.0], 1.0, [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0, 1.0, 1.0], 1.0, 1.0]


for filename in os.listdir(files):
    with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
        # print(np.arange(len(variables)))
        
        for i in np.arange(len(variables)):
            variables[i] = f.readline()
            variables[i] = variables[i][0:-1]
    
    print('Processing file ' + filename[0:-4] + '...')
    # Now I find the fileplacement of the files 
    filenameM = images + '/' + filename[0:-4] + '.jpg'
    filenameSave = 'variables_nonT/' + filename
    imageSave = 'mod_image/' + filename[0:-4] + '_mod.jpg'


    yMiddle = int(variables[0])
    xMiddle = int(variables[1])
    marginY = int(variables[2])
    marginX = int(variables[3])
    [conversion, blur, threshold] = np.float64(variables[4:7])

    ## Get the autocorrelation lengths.
    image = skimage.io.imread(fname=filenameM, as_gray=True)

    clip, blurredClip = clipBlur(filenameM, xMiddle, yMiddle, marginX, marginY, sigma = blur)
    rawclip = np.copy(clip)


    clip[blurredClip > threshold] = 0
    auflength  = np.empty(3)
    uncertainty = np.empty(3)
    RMSE = np.empty([3,4])
    kvalue = np.empty(3)
    xvalue = np.empty(3)
    auflength, uncertainty, funcType, plotdata, RMSE, kvalue, xvalue = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = fit)

    #-----------------------------------------------------------------
    # Plot - Kan udkommenteres ->
    funcTypes = np.array(["Exponential","Gaussian", "x-Power", "x-Exponential"])
    # print(' -> Plotdata...')
    # plot_acf2(auflength, funcTypes, plotdata, xmax = 2, block = False, sectors = 3, saveas = filename[0:-4], plotshow=False)
    # print(' -> Threshold...')
    # plot_threshold(clip=np.copy(rawclip), blurredClip=np.copy(blurredClip), conversion=conversion, saveas = filename[0:-4], plotshow = False)
    # print(' -> Sigma...')
    # plot_sigma(clip=rawclip, threshold=threshold, conversion=conversion, saveas = filename[0:-4], plotshow = False)
    #-----------------------------------------------------------------
    
    variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
        auflength[1], auflength[2], funcType]
    
    saveFile = [np.round(auflength,3), np.round(uncertainty,3), np.round(RMSE[:,0]*10**4,3), np.round(RMSE[:,1]*10**2,3), np.round(RMSE[:,2]*10**3,3), np.round(RMSE[:,3]*10**5,3), np.round(kvalue,3), np.round(xvalue,3)]

    print(RMSE)
    #print('')

    np.savetxt(filenameSave, saveFile, delimiter=' ', newline = "\n", fmt = "%s")


    clip[blurredClip > threshold] = 1
    clip = np.uint8(clip*255)
    skimage.io.imsave(imageSave, clip)
    

    

    


        