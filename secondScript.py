
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *
from autocofunc import *
from autocolen import *
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

# Define the variables used to analyze the picture
x = 1300
y = 3000
margin = 1000
threshold = 0.6
sigma = 5.0
filename = "data/img01.jpg"

# the variable that decides how many pixels we want to see the autocorrelation for. (length of displacement from 0 to end.) 
# Maximum is marginX
end = 500

# conversion variable is the distance (in mm) per pixel in the picture.
conversion = 90/2000

# Get a clipout of the picture (clip) and a clipout of the picture after it has been blurred (blurredClip)
clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)

# Define the modified clip that will be used in the autocorrelation function
clipMod = np.copy(clip)
clipMod[blurredClip > threshold] = 0

# Define a modified clipout of the picture that will be used to visualize the thresholding.
clipShow = np.copy(clip)
clipMod[blurredClip > threshold] = 1

# Make a greyscale histogram to show at which brightness there are peaks.
plt.figure(0)
plt.hist(x=blurredClip.flatten(), bins=256, range=(0,1))
plt.title('frequency of grayscale pigments.')
plt.xlabel('Brightness')
plt.show()

# show the clipout before it was modified
plt.figure(1)
skimage.io.imshow(clip)
plt.title('Clipout of image in grayscale, untreated')

# Show the clipout after it was modified.
plt.figure(2)
skimage.io.imshow(clipShow)
plt.title('modified picture clipout')

# Print all the figures created
plt.show()

# Make the autocorrelationfunction applying to out picture
autfct = autoCor(clipMod, plot = False)

# Define a displacement vector for the autocorrelation
Xbar = np.arange(end)
plt.figure(3)
plt.plot(Xbar, autfct[0:end])
plt.xlabel('displacement [pixels]')
plt.ylabel('autocorrelation')
plt.title('autocorrelation as a function of displacement')
plt.show()

autLength = autocolen(autfct) * conversion
print('the autocorrelation length is {:.2f} mm'.format(autLength))





