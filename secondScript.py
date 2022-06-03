

from asyncio.windows_events import NULL
from inputNumber import inputNumber
from displayMenu import displayMenu
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
# from autocofunc import *
# from autocolen import *
# from plot_acf import *
from inputFilename import *
from acf import *


image = np.zeros([18,9])
image[8,4]=1
print(image)


blurredImage = skimage.filters.gaussian(image, sigma=2, truncate=3.5)
blurredImage = np.round(blurredImage*100, 2)
print(blurredImage[:,5])




