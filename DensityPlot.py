from distutils import filelist
import os
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

compdata = np.loadtxt('rhoplotdata.txt',delimiter=',')
print(np.shape(compdata))

