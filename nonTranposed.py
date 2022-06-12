


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


# for filename in os.listdir(files):
#     with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
#         # print(np.arange(len(variables)))
        
#         for i in np.arange(len(variables)):
#             variables[i] = f.readline()
#             variables[i] = variables[i][0:-1]
    
#     print('Processing file ' + filename[0:-4] + '...')
#     # Now I find the fileplacement of the files 
#     filenameM = images + '/' + filename[0:-4] + '.jpg'
#     filenameSave = 'variables_nonT/' + filename
#     imageSave = 'mod_image/' + filename[0:-4] + '_mod.jpg'


#     yMiddle = int(variables[0])
#     xMiddle = int(variables[1])
#     marginY = int(variables[2])
#     marginX = int(variables[3])
#     [conversion, blur, threshold] = np.float64(variables[4:7])

#     ## Get the autocorrelation lengths.
#     image = skimage.io.imread(fname=filenameM, as_gray=True)

#     clip, blurredClip = clipBlur(filenameM, xMiddle, yMiddle, marginX, marginY, sigma = blur)
#     rawclip = np.copy(clip)
#     clip[blurredClip > threshold] = 0

#     num_pix = blurredClip < threshold
#     n = np.shape(num_pix)[0]
#     blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
#     rho = np.empty(3)
#     for i in range(3):
#         section = num_pix[blox[i]:blox[i+1],:]
#         rho[i] = np.sum(section) / np.size(section)
#     # print(rho)
    
#     auflength  = np.empty(3)
#     uncertainty = np.empty(3)
#     RMSE = np.empty([3,4])
#     kvalue = np.empty(3)
#     xvalue = np.empty(3)
#     auflength, uncertainty, funcType, plotdata, RMSE, kvalue, xvalue = acf(clip, lags = marginX-1, conversion = conversion, plot = False, plotfunc = fit)

    #-----------------------------------------------------------------
    # Plot - Kan udkommenteres ->
    # funcTypes = np.array(["Exponential","Gaussian", "x-Power", "x-Exponential"])
    # print(' -> Plotdata...')
    # plot_acf2(auflength, funcTypes, plotdata, xmax = 2, block = False, sectors = 3, saveas = filename[0:-4], plotshow=False)
    # print(' -> Threshold...')
    # plot_threshold(clip=np.copy(rawclip), blurredClip=np.copy(blurredClip), conversion=conversion, saveas = filename[0:-4], plotshow = False)
    # print(' -> Sigma...')
    # plot_sigma(clip=rawclip, threshold=threshold, conversion=conversion, saveas = filename[0:-4], plotshow = False)
    #-----------------------------------------------------------------
    
    # variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
    #     auflength[1], auflength[2], funcType]
    
    # saveFile = [np.round(auflength,3), np.round(uncertainty,3), np.round(RMSE[:,0]*10**4,3), np.round(RMSE[:,1]*10**2,3), np.round(RMSE[:,2]*10**3,3), np.round(RMSE[:,3]*10**5,3), np.round(kvalue,3), np.round(xvalue,3), np.round(rho, 3)]

    # print(RMSE)
    # #print('')

    # np.savetxt(filenameSave, saveFile, delimiter=' ', newline = "\n", fmt = "%s")


    # clip[blurredClip > threshold] = 1
    # clip = np.uint8(clip*255)
    # skimage.io.imsave(imageSave, clip)
    

#-----------------------------------------------------------------
# L compared to depth - Kan udkommenteres ->
files_nonT = os.getcwd() + '/variables_nonT'
variables = [[1.0,1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0], [1.0, 1.0,1.0]]
N = len(os.listdir(files_nonT))
L_mean = [0,0,0]

plt.close()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True, sharex=True)
fig.suptitle('FYI error')
fig.supylabel('Autocorrelation [mm]')
fig.supxlabel('Depth from surface [mm]')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

RMSE_exp_all = np.array([])
RMSE_gauss_all = np.array([])
RMSE_xpow_all = np.array([])
RMSE_xexp_all = np.array([])

for filename in os.listdir(files_nonT):
    with open(os.path.join(files_nonT, filename), 'r') as f: # open in readonly mode
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


    if filename[0:-4] == "20200220_221521":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion
        ax1.scatter(sections, L, c = 'k')
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200220_223734":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 100
        ax1.scatter(sections, L, c = 'k')
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200220_225515":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 200
        ax1.scatter(sections, L, c = 'k')
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200220_231318":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 300
        ax1.scatter(sections, L, c = 'k')
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k')
        ax1.set_title('No. 1, 2, 3 and 4')

    if filename[0:-4] == "20200206_094159":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion
        ax2.scatter(sections, L, c = 'k')
        ax2.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200206_101354":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 100
        ax2.scatter(sections, L, c = 'k')
        ax2.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200206_103511":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 200
        ax2.scatter(sections, L, c = 'k')
        ax2.errorbar(sections, L, sigma, ls = 'none', c='k')
        ax2.set_title('No. 5,6 and 7')

    if filename[0:-4] == "20200122_215146":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion
        ax3.scatter(sections, L, c = 'k')
        ax3.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200122_221025":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 130
        ax3.scatter(sections, L, c = 'k')
        ax3.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200122_222713":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 225
        ax3.scatter(sections, L, c = 'k')
        ax3.errorbar(sections, L, sigma, ls = 'none', c='k')
        ax3.set_title('No. 8, 9 and 10')
        
    if filename[0:-4] == "20200219_135306":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion
        ax4.scatter(sections, L, c = 'k')
        ax4.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200219_141939":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 130
        ax4.scatter(sections, L, c = 'k')
        ax4.errorbar(sections, L, sigma, ls = 'none', c='k')
    if filename[0:-4] == "20200219_143636":
        imagePlacement = os.getcwd() + '\\variables\\' + filename[0:-4] + '.txt'
        with open(imagePlacement, 'r') as f:
            f.readline()
            f.readline()
            marginY = int(f.readline()[0:-1])
            f.readline()
            conversion = np.float64(f.readline()[0:-1])
        blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
        for i in range(3):
            sections[i] = (blox[i] + blox[i+1])/2 * conversion + 225
        ax4.scatter(sections, L, c = 'k')
        ax4.errorbar(sections, L, sigma, ls = 'none', c='k')
        ax4.set_title('No. 11, 12 and 13')


fig.savefig('errorDepth')


plt.close()
fig = plt.figure(1)
plt.title('RMSE horizontal')
plt.hist([RMSE_exp_all, RMSE_gauss_all, RMSE_xpow_all, RMSE_xexp_all], stacked = True, bins=20, range=(0,0.07), density=True)
plt.legend(['exp', 'gauss', 'xpow', 'xexp'])
fig.savefig('RMSE')
plt.close()



#-----------------------------------------------------------------


    


        