
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
from acf import *

files = os.getcwd() + '\\variables'
images = os.getcwd() + '\\images'

fit = np.array([1,2,4,5])


# The variables vector was defined as having the following positions
# variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
#     auflength[1], auflength[2], funcType]

variables = [1,1,1,1,1,1,1,[1,1,1],['','','']]
saveFile = [[1.0,1.0,1.0], 1.0, [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0, 1.0, 1.0], 1.0, 1.0]



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

    num_pix = blurredClip < threshold
    n = np.shape(num_pix)[0]
    blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
    rho = np.empty(3)
    for i in range(3):
        section = num_pix[blox[i]:blox[i+1],:]
        rho[i] = np.sum(section) / np.size(section)
    # print(rho)

    # Transpose the image so we get the autocorrelation on horizontally. image
    clip = clip.T
    blurredClip = blurredClip.T
    auflength  = np.empty(3)
    uncertainty = np.empty(3)
    RMSE = np.empty([3,4])
    kvalue = np.empty(3)
    xvalue = np.empty(3)

    # RMSE is the root mean square error of Exponential, Gaussian, x-Power, X-exponential
    
    auflength, uncertainty, funcType, plotdata, RMSE, kvalue, xvalue = acf(clip, lags = marginY-1, conversion = conversion, plot = False, plotfunc = fit)
    print(funcType)
    
    variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
        auflength[1], auflength[2], funcType]
    
    saveFile = [np.round(auflength,3), np.round(uncertainty,3), np.round(RMSE[:,0]*10**3,3), np.round(RMSE[:,1]*10**2,3), np.round(RMSE[:,2]*10**4,3), np.round(RMSE[:,3]*10**5, 3), np.round(kvalue,3), np.round(xvalue,3), np.round(rho,3)]

    print(RMSE) 

    np.savetxt(filenameSave, saveFile, delimiter=' ', newline = "\n", fmt = "%s")

    clip[blurredClip > threshold] = 1
    clip = np.uint8(clip*255)
    skimage.io.imsave(imageSave, clip)


#-----------------------------------------------------------------
# L compared to depth - Kan udkommenteres ->
files_T = os.getcwd() + '/variables_T'
variables = [[1.0,1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0]]
N = len(os.listdir(files_T))
L_mean = [0,0,0]

plt.close()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True, sharex=True)
fig.suptitle('FYI error (PLEASE RETHINK THIS, DOESNT WORK RN CAUSE IT DOESNT GO DOWN PER SECTION)')
fig.supylabel('Autocorrelation [mm]')
fig.supxlabel('depth from surface [mm]')

RMSE_exp_all = np.array([])
RMSE_gauss_all = np.array([])
RMSE_xpow_all = np.array([])
RMSE_xexp_all = np.array([])
kval_all = np.array([])
xval_all = np.array([])

for filename in os.listdir(files_T):
    with open(os.path.join(files_T, filename), 'r') as f: # open in readonly mode
        # print(np.arange(len(variables)))
        
        for i in np.arange(len(variables)):
            variables[i] = f.readline().split(" ")
            # variables[i] = 
            variables[i] = np.float64(variables[i])
    
    L = variables[0]
    sigma = variables[1]
    RMSE_exp = variables[2] *10**(-4)
    RMSE_gauss = variables[3] *10**(-2)
    RMSE_xpow = variables[4] *10**(-3)
    RMSE_xexp = variables[5] *10**(-5)
    kval = variables[6]
    xval = variables[7]

    sections = np.empty(3)

    RMSE_exp_all = np.append(RMSE_exp_all, RMSE_exp)
    RMSE_gauss_all = np.append(RMSE_gauss_all, RMSE_gauss)
    RMSE_xpow_all = np.append(RMSE_xpow_all, RMSE_xpow)
    RMSE_xexp_all = np.append(RMSE_xexp_all, RMSE_xexp)
    xval_all = np.append(xval_all, xval)
    kval_all = np.append(kval_all, kval)

plt.close()
fig = plt.figure()
plt.title('RMSE Vertical')
logbins = np.logspace(np.log10(10**(-6)), np.log10(10), 13)
plt.hist([RMSE_exp_all, RMSE_gauss_all, RMSE_xpow_all, RMSE_xexp_all], stacked = True, bins=logbins)
plt.legend(['Exponential', 'Gaussian', 'k-power', 'k-exponential'])
plt.xscale('log')
fig.savefig('RMSE_T')
plt.close()

plt.close()
fig = plt.figure(1)
plt.title('k value for k-exponential vertically')
plt.hist(kval_all, bins=13, color = "orange")
fig.savefig('kval_vertically')
plt.close()

plt.close()
fig = plt.figure(1)
plt.title('k value for k-power vertically')
plt.hist(xval_all, bins=13, color = "skyblue")
fig.savefig('xval_vertically')
plt.close()

    



        