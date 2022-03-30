import numpy as np
from autocolen import autocolen
from clipBlur import clipBlur
from autocofunc import autoCor
from plot_acf import plot_acf


def acf(filename, xy=[1300,3000], margin=1000, threshold = 0.6, sigma = 5.0, plot=False, plotfunc=1):
    [x,y] = xy

    clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)
    M = autoCor(blurredClip)
    acl = autocolen(M,90/2000)


    if plot:
        x1 = filename.rfind('/')
        x2 = filename.rfind('.')
        fname = str(filename[x1+1:x2] + '-f' + str(plotfunc) + '.png')
        plot_acf(M, lags = 100, func = plotfunc, saveas = fname)

    print(f"Autokorrelationslængden er {acl:.2f}mm")


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
        # if i %10 == 0: 
            # print(i)
    M = 1/i*M
    #C = 1/np.sqrt(2000)*C #Hacked konfidensinterval, check metoden 

    # lags bestemmer hvor mange punkter der plottes

    # # lags bestemmer hvor mange punkter der plottes 
    # plt.figure(1)   
    # tsaplots.plot_acf(M,lags = 100)
    # plt.show()
    return M

def autocolen(acf,leng=1):
    """
    Beregner Autokorrelationslængden
    acf er en beregnet vektor af Auto korrelations funktionen
    leng er længden af billedet i enhed [længde per pixel], valgfri
    """
    t = np.exp(-1)
    n = np.arange(np.size(acf))[acf <= t]
   
    if np.size(n) == 0:
        return 0
    elif n >=3 and n <= (np.size(acf)-3):
        #dx = 1
        #dy = acf[n] - acf[n-1]
        print("s")

    return n[0]*leng