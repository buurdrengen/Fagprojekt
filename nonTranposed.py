


import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import time
from clipBlur import *
from acf import acf, plot_acf2
from plot_threshold import plot_threshold
from plot_sigma import plot_sigma

files = os.getcwd() + '/variables'
images = os.getcwd() + '/images'

fit = np.array([1,2,4,5])
mastertic = time.perf_counter()


# The variables vector was defined as having the following positions
# variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
#     auflength[1], auflength[2], funcType]

variables = [1,1,1,1,1,1,1,[1,1,1],['','','']]
saveFile = [[1.0,1.0,1.0], 1.0, [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0, 1.0, 1.0], 1.0, 1.0]
filesas = ["20200220_221521", #The files in numerical order...
    "20200220_223734",
    "20200220_225515",
    "20200220_231318",
    "20200206_094159",
    "20200206_101354",
    "20200206_103511",
    "20200122_215146",
    "20200122_221025",
    "20200122_222713",
    "20200219_135306",
    "20200219_141939",
    "20200219_143636",
    "20200201_145742",
    "20200121_231857",
    "20200121_231937",
    "20200129_205841_T",
    "20200212_202537",
    "20200212_202637",
    "20200212_204509",
    "hummock1_top_7cm_1",
    "hummock1_bot_7cm_1",
    "20200122_171409",
    "20200122_172102",
    "20200213_161609",
    "20200213_163440",
    "20200213_170702",
    "20200206_110750",
    "20200206_112710",
    "20200213_102626",
    "20200213_104615",
    "20200213_110428",
    "20200122_225922",
    "20200122_231339",
    "meltpond_ssmi_7cm_top",
    "meltpond_ssmi_7cm_bot_1"]
fileset = dict(zip(np.array(filesas),np.arange(np.size(filesas))))
filesize = np.size(filesas)
rhoset = np.zeros([filesize,3])
lset = np.zeros([filesize,3])
sigmaset = np.zeros([filesize,3])
rhoset_t = np.zeros([filesize,3])
lset_t = np.zeros([filesize,3])
sigmaset_t = np.zeros([filesize,3])

for nfo, filename in enumerate(os.listdir(files)):
    with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
        # print(np.arange(len(variables)))
        
        for i in np.arange(len(variables)):
            variables[i] = f.readline()
            variables[i] = variables[i][0:-1]
    
    print(f'Processing file {nfo+1}: {filename[0:-4]}...')
    tic = time.perf_counter()
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


#--------------
    num_pix = blurredClip < threshold
    n = np.shape(num_pix)[0]
    blox = np.int32(np.round(np.linspace(0,2*marginY,4)))
    rho = np.empty(3)

    for i in range(3):
        section = num_pix[blox[i]:blox[i+1],:]
        rho[i] = np.sum(section) / np.size(section)
    # print(rho)


    num_pix_T = blurredClip.T < threshold.T
    n_T = np.shape(num_pix_T)[0]
    blox_T = np.int32(np.round(np.linspace(0,2*marginX,4)))
    rho_T = np.empty(3)

    for i in range(3):
        section_T = num_pix_T[blox_T[i]:blox_T[i+1],:]
        rho_T[i] = np.sum(section_T) / np.size(section_T)
    # print(rho)
    
#------------------
    auflength  = np.empty(3)
    uncertainty = np.empty(3)
    RMSE = np.empty([3,4])
    kvalue = np.empty(3)
    xvalue = np.empty(3)
    auflength, uncertainty, funcType, plotdata, RMSE, kvalue, xvalue = acf(np.copy(clip), lags = marginX-1, conversion = conversion, plot = False, plotfunc = fit)
    auflengthT, uncertaintyT, funcTypeT, plotdataT, RMSET, kvalueT, xvalueT = acf(np.copy(clip.T), lags = marginX-1, conversion = conversion, plot = False, plotfunc = fit)
    #-----------------------------------------------------------------
    # Plot - Kan udkommenteres ->
    funcTypes = np.array(["Exponential","Gaussian", "x-Power", "x-Exponential"])
    print(' -> Plotdata H...')
    plot_acf2(auflength, funcTypes, plotdata, xmax = 2, block = True, sectors = 3, saveas = filename[0:-4], plotshow=False)
    print(' -> Plotdata V...')
    plot_acf2(auflength, funcTypes, plotdata, xmax = 2, block = True, sectors = 3, saveas = filename[0:-4] + "_T", plotshow=False)
    # print(' -> Threshold H...')
    # plot_threshold(clip=np.copy(rawclip), blurredClip=np.copy(blurredClip), conversion=conversion, saveas = filename[0:-4], plotshow = False)
    # print(' -> Threshold V...')
    # plot_threshold(clip=np.copy(rawclip.T), blurredClip=np.copy(blurredClip.T), conversion=conversion, saveas = filename[0:-4] + "_T", plotshow = False)
    # print(' -> Sigma H...')
    # plot_sigma(clip=np.copy(rawclip), threshold=threshold, conversion=conversion, saveas = filename[0:-4], plotshow = False)
    # print(' -> Sigma V...')
    # plot_sigma(clip=np.copy(rawclip.T), threshold=threshold, conversion=conversion, saveas = filename[0:-4] + "_T", plotshow = False)
    #-----------------------------------------------------------------
    
    variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
        auflength[1], auflength[2], funcType]
    
    saveFile = [np.round(auflength,3), np.round(uncertainty,3), np.round(RMSE[:,0]*10**4,3), np.round(RMSE[:,1]*10**2,3), np.round(RMSE[:,2]*10**3,3), np.round(RMSE[:,3]*10**5,3), np.round(kvalue,3), np.round(xvalue,3), np.round(rho, 3)]

    try:
        fileidx = fileset[filename[0:-4]]
        rhoset[fileidx] = rho
        lset[fileidx] = auflength
        sigmaset[fileidx] = uncertainty
        rhoset_t[fileidx] = rho_T
        lset_t[fileidx] = auflengthT
        sigmaset_t[fileidx] = uncertaintyT
    except KeyError:
        print(f"{filename[0:-4]} is not a member of test..")
        pass

    #print(RMSE)

    np.savetxt(filenameSave, saveFile, delimiter=' ', newline = "\n", fmt = "%s")


    clip[blurredClip > threshold] = 1
    clip = np.uint8(clip*255)
    skimage.io.imsave(imageSave, clip)
    toc = time.perf_counter()
    print(f"    Done in {time.strftime('%M:%S', time.gmtime(toc - tic))}")

    if nfo == 5:
        break
    

#-----------------------------------------------------------------
# L compared to depth - Kan udkommenteres ->
print('Postprocessing...')

compset = np.hstack([rhoset,lset,sigmaset,rhoset_t,lset_t,sigmaset_t])
print(f"Shape of compset = {np.shape(compset)}")
np.savetxt("rhoplotdata.txt",compset,delimiter=',',newline='\n')

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
kval_all = np.array([])
xval_all = np.array([])

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
    xval_all = np.append(xval_all, xval)
    kval_all = np.append(kval_all, kval)


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
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax1.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax2.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax2.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax2.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax3.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax3.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax3.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax4.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax4.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
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
        ax4.errorbar(sections, L, sigma, ls = 'none', c='k', capsize = 5)
        ax4.set_title('No. 11, 12 and 13')


fig.savefig('errorDepth')


print(np.min(RMSE_xexp_all))
print(np.max(RMSE_gauss_all))

plt.close()
fig = plt.figure(1)
plt.title('RMSE Horizontal')
logbins = np.logspace(np.log10(10**(-6)), np.log10(10), 13)
plt.hist([RMSE_exp_all, RMSE_gauss_all, RMSE_xpow_all, RMSE_xexp_all], stacked = True, bins=logbins)
plt.legend(['Exponential', 'Gaussian', 'k-power', 'k-exponential'])
plt.xscale('log')
fig.savefig('RMSE')
plt.close()

plt.close()
fig = plt.figure(1)
plt.title('k value for k-exponential horizontally')
plt.hist(kval_all, bins=13, color = "orange")
fig.savefig('kval_horizontally')
plt.close()

plt.close()
fig = plt.figure(1)
plt.title('k value for k-power horizontally')
plt.hist(xval_all, bins=13, color = "skyblue")
fig.savefig('xval_horizontally')
plt.close()
mastertoc = time.perf_counter()
print(f"Workload completed in {time.strftime( '%H:%M:%S', time.gmtime(mastertoc - mastertic))}!")

#-----------------------------------------------------------------


    


        