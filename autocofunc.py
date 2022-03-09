## Autocorrelation function ## 
import numpy as np
import skimage.io
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

x = 1300 
y = 3000 
margin = 1000
threshold = 0.5 

image = skimage.io.imread(fname = "20200220_221515.jpg")
clip = skimage.color.rgb2gray(image[y-margin:y+margin, x-margin:x+margin])
plt.imshow(clip,cmap = 'gray')
plt.show()

clip[clip > threshold] = 0
N = 2*margin

horizontal = np.sum(clip,axis = 0)/N
vertical = np.sum(clip,axis = 1)/N

linje = clip[1,:]
sm.tsa.acf(linje,nlags = 1999)
fig = tsaplots.plot_acf(linje,lags = 1999)
plt.show()
