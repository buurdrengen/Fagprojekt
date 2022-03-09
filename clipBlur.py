# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 08:38:25 2022

@author: Bruger
"""

def clipBlur(filename, x, y, marginX, marginY, sigma=1.0):
    # FUNC:
    # picture takes the variables and returns an image in grayscale that has been 
    # clipped and blurred
    #
    # VARIABLES:
    # filename is the position of the picture of a picture you want treated. 
    # 
    # (x,y) is the position of the middle of the outclip you want of the picture
    # Remember that x is the amount of pixels down from the top left corner
    # and y is the number of pixels to the right of the top left corner
    # 
    # marginX and marginY are the amount of pixels you want to include in the 
    # outclip from (x,y) 
    # 
    # sigma is the amount of blur you want.
    
    import numpy as np
    import skimage.io
    image = skimage.io.imread(fname=filename, as_gray=True)
    clip = image[y-marginY:y+marginY, x-marginX:x+marginX]

    blurredImage = skimage.filters.gaussian(image, sigma=sigma)
    blurredClip = blurredImage[y-marginY:y+marginY, x-marginX:x+marginX]


    
    return clip, blurredClip
    
    
    