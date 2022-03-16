import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.graphics import utils
from autocolen import autocolen
from clipBlur import clipBlur
from autocofunc import autoCor


x = 1300
y = 3000
margin = 1000
threshold = 0.6
sigma = 5.0
filename = "thin_slices/firstyearice/southerntransect/20200201_145725.jpg"

clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)
M = autoCor(blurredClip)

## Dette viser hvor smart github er 

print(np.arange(10))