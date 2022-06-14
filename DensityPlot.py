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

        plt.errorbar(rhoH, LH, sigmaH, ls = 'none', c='k', fmt='.', capsize=6, elinewidth=0.7, lw = 0.5)
        plt.xlabel('Inclusion Density [%]')
        plt.ylabel('Autocorrelation Length [mm]')
        plt.title(tx + ' - ' + 'Horizontal')
        plt.grid(True)
        plt.savefig('plotimg/' + tx.replace(' ','_') + '_H.png',dpi=300,format='png')
        #plt.show(block='True')
        plt.close()

        plt.errorbar(rhoV, LV, sigmaV, ls = 'none', c='k', fmt='.', capsize=6, elinewidth=0.7, lw = 0.5)
        plt.xlabel('Inclusion Density [%]')
        plt.ylabel('Autocorrelation Length [mm]')
        plt.title(tx + ' - ' + 'Vertical')
        plt.grid(True)
        plt.savefig('plotimg/' + tx.replace(' ','_') + '_V.png',dpi=300,format='png')
        #plt.show(block='True')
        plt.close()

