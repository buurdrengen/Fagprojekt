import numpy as np
from autocolen import autocolen
from clipBlur import clipBlur
from autocofunc import autoCor


def acf(filename, xy=[1300,3000], margin=1000, threshold = 0.6, sigma = 5.0, plot=False, plotfunc=1):
    [x,y] = xy

    clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)
    M = autoCor(blurredClip, plot=plot,  plotfunc=plotfunc)
    acl = autocolen(M,90/2000)
    print(f"Autokorrelationslængden er {acl:.2f}mm")