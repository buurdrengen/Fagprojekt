# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:39:30 2022

@author: VIKTIS THE CHAMP
"""

import numpy as np
import skimage.io
import matplotlib.pyplot as plt

plt.close('all')

x = 1300
y = 3000
margin = 1000
threshhold = 0.5

image = skimage.io.imread(fname="data/isBillede.jpg")
clip = skimage.color.rgb2gray(image[y-margin:y+margin, x-margin:x+margin])

clip[clip > threshhold] = 1
N = 2*margin

plt.figure(1)
skimage.io.imshow(clip)   

clip[clip > threshhold] = 0


horizontal = np.sum(clip,axis=0) / N
vertical = np.sum(clip,axis=1) / N

X = np.arange(N) / N
plt.figure(2)
plt.plot(X,horizontal)


X = np.arange(N) / N
plt.figure(3)
plt.plot(X,vertical)


