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
        fname = str(filename[x1+1:x2] + '.png')
        plot_acf(M, lags = 100, func = plotfunc, saveas = fname)

    print(f"Autokorrelationslængden er {acl:.2f}mm")