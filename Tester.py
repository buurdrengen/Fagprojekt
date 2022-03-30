import numpy as np
from autocolen import autocolen
from clipBlur import clipBlur
from autocofunc import autoCor
from acf import acf
#from string import rfind
from plot_acf import func2
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

x = 1300
y = 3000
margin = 1000
threshold = 0.6
sigma = 2.0
filename = '20200220_221515.jpg'

clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)

clip[blurredClip > threshold] = 0
print('There are NaN: ', np.shape(np.isnan(clip)))
M = np.zeros(1999)
for i, clips in enumerate(clip):
    auto = sm.tsa.acf(clips,1999)
    M = M + auto
    # if i %10 == 0: 
        # print(i)
M = 1/(2000)*M
