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
<<<<<<< Updated upstream
# plt.imshow(clip,cmap = 'gray')
# plt.show()
=======
plt.imshow(clip,cmap = 'gray')
plt.show()
>>>>>>> Stashed changes

clip[clip > threshold] = 0
N = 2*margin

horizontal = np.sum(clip,axis = 0)/N
vertical = np.sum(clip,axis = 1)/N

<<<<<<< Updated upstream
# from clipBlur import clipBlur
# def autoCor()

## En linje 
linje = clip[1,:]
sm.tsa.acf(linje,nlags = 100)
fig = tsaplots.plot_acf(linje,lags = 100)
plt.show()

## Et helt billede
M = np.zeros(2000)
for i, clips in enumerate(clip):
    auto = sm.tsa.acf(clips,nlags = 1999)
    M = M + auto
    if i %10 == 0: 
        print(i)
M = 1/2000*M

plt.figure(1)
tsaplots.plot_acf(M,lags = 100)
plt.show()

# x1 = np.arange(0, 100)
# y1 = M[0,100]
# plt.plot(x1,y1)
# plt.show()
=======
## En linje 
linje = clip[1,:]
sm.tsa.acf(linje,nlags = 1999)
fig = tsaplots.plot_acf(linje,lags = 1999)
plt.show()

## Et helt billede
M = np.array([])
for i, clips in enumerate(clip):
    auto = sm.tsa.acf(clips,nlags = 1999)
    M = M + auto
print(np.size(M))
>>>>>>> Stashed changes
