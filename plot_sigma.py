import numpy as np
import matplotlib.pyplot as plt

from acf import acf
from clipBlur import clipBlur


if __name__ == "__main__":
    fname = "thin_slices/meltponds/meltpond_20200211/20200213_110428.jpg"
    fTypes = np.array(['Empirical','Exponential','Gaussian','Exp Root'])
    fit = np.array([0])


    clip, blurredClip = clipBlur(fname,  x=1650, y=3100, marginX=1350, marginY=1200, sigma=0.25)

    plt.figure()
    plt.imshow(clip, cmap = 'gray')
    plt.show()
    plt.close()
    threshold = 0.5

    x = np.arange(0.1,4.05,0.05)
    s1 = np.zeros(np.size(x))
    s2 = np.zeros(np.size(x))
    s3 = np.zeros(np.size(x))
    for idx, i in enumerate(x):
        clip, blurredClip = clipBlur(fname,  x=1650, y=3100, marginX=1350, marginY=1200, sigma=i)
        clip[blurredClip > threshold] = 0

        auflength, functype, plotdata = acf(clip, lags = 200, conversion = 0.03214285714285714, plot = False, plotfunc = fit, ip=40, plotname="Testplot")
        s1[idx] = auflength[0]
        s2[idx] = auflength[1]
        s3[idx] = auflength[2]
        print(auflength)

    yscale = np.max([np.max(s1),np.max(s2),np.max(s3)])*1.5

    plt.figure()
    s1[s1 == 0] = 10
    s2[s2 == 0] = 10
    s3[s3 == 0] = 10
    plt.grid(True)
    plt.plot(x,s1,'r-',label='Section 1')
    plt.plot(x,s2,'b-',label='Section 2')
    plt.plot(x,s3,'k-',label='Section 3')
    plt.xlabel("Standard Deviation")
    plt.ylabel("ACL [mm]")
    plt.title("Autocorrelation Length")
    plt.legend()
    plt.xlim([-0.08,4.08])
    plt.ylim([-0.02*yscale,1.02*yscale])

    saveas = fname.split('/')[-1].split('.')[0]
    savename = str("sigma_images/" + saveas + ".png")
    #print(fname)
    plt.savefig(savename,dpi=300,format="png")
    plt.show(block = True)
    plt.close()
