import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.graphics import utils
from clipBlur import clipBlur
from autocofunc import autoCor


filename = "data/img01.jpg"

x = 1300
y = 3000
margin = 1000
threshold = 0.6
sigma = 5.0
filename = "data/img01.jpg"

clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)
M = autoCor(blurredClip)

