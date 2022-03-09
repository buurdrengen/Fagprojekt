
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from clipBlur import *


x = 1300
y = 3000
margin = 1000
threshold = 0.6
sigma = 5.0
filename = "data/img01.jpg"

clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)

clipMod = np.copy(clip)
clipMod[blurredClip > threshold] = 1

fig, ax = plt.subplots()
plt.figure(0)
plt.hist(x=blurredClip.flatten(), bins=256, range=(0,1))
plt.title('frequency of grayscale pigments.')
plt.xlabel('Brightness')
plt.show()

plt.figure(1)
skimage.io.imshow(clip)
plt.title('Clipout of image in grayscale, untreated')

plt.figure(2)
skimage.io.imshow(clipMod)
plt.title('modified picture clipout')


plt.show()






