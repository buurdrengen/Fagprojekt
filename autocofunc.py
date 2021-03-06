# ## Autocorrelation function ## 
# import numpy as np
# import skimage.io
# import matplotlib.pyplot as plt 
# import statsmodels.api as sm
# from statsmodels.graphics import tsaplots

# x = 1300 
# y = 3000 
# margin = 1000
# threshold = 0.5 

# image = skimage.io.imread(fname = "20200220_221515.jpg")
# clip = skimage.color.rgb2gray(image[y-margin:y+margin, x-margin:x+margin])
# # plt.imshow(clip,cmap = 'gray')
# # plt.show()

# clip[clip > threshold] = 0
# N = 2*margin

# horizontal = np.sum(clip,axis = 0)/N
# vertical = np.sum(clip,axis = 1)/N

#from clipBlur import clipBlur

from numpy import not_equal
from clipBlur import *

def autoCor(clipBlur, nlags = 1999):
    # Function 
    # Viser autocorrelation mellem alle pixels i det clipped/blurred billede. 

    # Variables
    # clipped/blurred billede loaded fra clipBlur funktionen.

    # Først importeres relevante pakker, især her statsmodels.
    import numpy as np
    import matplotlib.pyplot as plt 
    from statsmodels.tsa.stattools import acf as acff
    from statsmodels.graphics import tsaplots

    # Denne funktion tager det klippede og blurrede billede og bestemmer autocorrelation for alle pixels.
    # Nedenfor er testscript til hvordan autocorrelationsfunktionen fungerer for en enkelt linje. 

    # ## En linje 
    # linje = clip[1,:]
    # sm.tsa.acf(linje,nlags = 100)
    # fig = tsaplots.plot_acf(linje,lags = 100)
    # plt.show()

    ## Et helt billede
    # nlags bestemmer hvor mange pixels der medtages, 0 regnes ikke med og der er 1999 lig 2000. 
    M = np.zeros(nlags+1)
    for i, clips in enumerate(clipBlur):
        auto = acff(clips,nlags=nlags)
        if np.size(auto) != np.size(nlags)+1:
            print(f"Vector Mismatch, auto is {np.size(auto)} while clips is {np.size(clips)}")
        M = M + auto
        if i %10 == 0: 
            print(i)
    M = 1/i*M
    #C = 1/np.sqrt(2000)*C #Hacked konfidensinterval, check metoden 

    # lags bestemmer hvor mange punkter der plottes

    # # lags bestemmer hvor mange punkter der plottes 
    # plt.figure(1)   
    # tsaplots.plot_acf(M,lags = 100)
    # plt.show()
    return M
print(autoCor(clipBlur('data/img01.jpg',1300,3000,1000,1000)))


# x1 = np.arange(0, 100)
# y1 = M[0,100]
# plt.plot(x1,y1)
# plt.show()
