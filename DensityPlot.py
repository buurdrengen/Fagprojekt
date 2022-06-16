from distutils import filelist
import os
from matplotlib.markers import MarkerStyle
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import time
from clipBlur import *
from acf import acf, plot_acf2
from plot_threshold import plot_threshold
from plot_sigma import plot_sigma

# files = os.getcwd() + "/variables"
# images = os.getcwd() + "/images"

# fit = np.array([1,2,4,5])
# mastertic = time.perf_counter()
#print(np.shape(compdata))

def Density_plot(compdata,splitter,texts):
    splBack = 0
    for spl,tx in zip(splitter,texts):
        setH = compdata[splBack:spl]
        #print(f"Shape of section = {np.shape(setH)}")
        #print(f"Shape of rho = {np.shape(setH[:,0:3])}")
        sz = (spl-splBack)*3
        rhoH = np.reshape(setH[:,0:3],[sz])
        LH = np.reshape(setH[:,3:6],[sz])
        sigmaH = np.reshape(setH[:,6:9],[sz])
        rhoV = np.reshape(setH[:,9:12],[sz])
        LV = np.reshape(setH[:,12:15],[sz])
        sigmaV = np.reshape(setH[:,15:18],[sz])
        splBack = spl

        plt.errorbar(rhoH*100, LH, sigmaH, ls = 'none', c='m',ecolor='k', fmt='s', capsize=6, elinewidth=0.7, lw = 0.5)
        plt.xlabel('Inclusion Density [%]')
        plt.ylabel('Autocorrelation Length [mm]')
        plt.title(tx + ' - ' + 'Horizontal')
        plt.grid(True)
        plt.savefig('plotimg/' + tx.replace(' ','_') + '_H.png',dpi=300,format='png')
        #plt.show(block='True')
        plt.close()

        plt.errorbar(rhoV*100, LV, sigmaV, ls = 'none', c='m',  ecolor='k', fmt='o', capsize=6, elinewidth=0.7, lw = 0.5)
        plt.xlabel('Inclusion Density [%]')
        plt.ylabel('Autocorrelation Length [mm]')
        plt.title(tx + ' - ' + 'Vertical')
        plt.grid(True)
        plt.savefig('plotimg/' + tx.replace(' ','_') + '_V.png',dpi=300,format='png')
        #plt.show(block='True')
        plt.close()



    stemplot = compdata[:,18:22]
    colmap = np.array(['k','r','b','g','m'])
    idx = np.arange(np.size(stemplot,0)) + 1
    idxback = 0

    fig1, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex = True)
    ax1.grid()
    ax2.grid()


    for i in range(5):
        ax1.errorbar(idx[idxback:splitter[i]], stemplot[idxback:splitter[i],0], yerr=stemplot[idxback:splitter[i],2], ls = 'none', c=colmap[i],ecolor=colmap[i], fmt='o', capsize=6, elinewidth=0.7, lw = 0.5, label=funcnames[i])
        ax2.errorbar(idx[idxback:splitter[i]], stemplot[idxback:splitter[i],1], yerr=stemplot[idxback:splitter[i],3], ls = 'none', c=colmap[i],ecolor=colmap[i], fmt='o', capsize=6, elinewidth=0.7, lw = 0.5, label=funcnames[i])
        idxback = splitter[i]

    ax1.set_title('Horizontal')
    ax2.set_title('Vertical')
    fig1.supxlabel('Image No.')
    fig1.supylabel('Autocorrelation Length [mm]')
    fig1.savefig('Image_No_L')
    ax1.legend()
    fig1.savefig('StemplotHV.png',dpi=300,format='png')
    plt.close()



if __name__ == "__main__":

    compdata = np.loadtxt('rhoplotdata.txt',delimiter=',')
    print(f"Size of compdata = {np.shape(compdata)}")
    splitter = [14,20,27,29,36]
    funcnames = ["First-Year Ice","Second-Year Ice","Hummocks","Lead Ice","Melt Ponds"]

    Density_plot(compdata=compdata,splitter=splitter,texts=funcnames)