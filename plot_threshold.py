import numpy as np
import matplotlib.pyplot as plt

from acf import acf
from clipBlur import clipBlur


if __name__ == "__main__":
    fname = "thin_slices/meltponds/meltpond_kuka/20200122_231339.jpg"
    fTypes = np.array(['Empirical','Exponential','Gaussian','Exp Root'])
    fit = np.array([0])


    clip, blurredClip = clipBlur(fname, x=1825, y=2150, marginX=1475, marginY=1150, sigma=0.25)

    plt.figure()
    plt.imshow(clip, cmap = 'gray')
    plt.show()
    plt.close()

    x = np.arange(1,0.005,-0.005)
    s1 = np.zeros(np.size(x))
    s2 = np.zeros(np.size(x))
    s3 = np.zeros(np.size(x))
    for idx, i in enumerate(x):
        clip[blurredClip > i] = 0

        auflength, functype, plotdata = acf(clip, lags = 200, conversion = 0.03214285714285714, plot = False, plotfunc = fit, ip=40, plotname="Testplot")
        s1[idx] = auflength[0]
        s2[idx] = auflength[1]
        s3[idx] = auflength[2]
        print(auflength)

    plt.figure()
    s1[s1 == 0] = 10
    s2[s2 == 0] = 10
    s3[s3 == 0] = 10
    x = x
    plt.grid(True)
    plt.plot(x,s1,'r-',label='Section 1')
    plt.plot(x,s2,'b-',label='Section 2')
    plt.plot(x,s3,'k-',label='Section 3')
    plt.xlabel("Brightness Threshold [Normalized 8-bit value]")
    plt.ylabel("ACL [mm]")
    plt.title("Autocorrelation Length")
    plt.legend()
    plt.xlim([-0.02,1.02])
    plt.ylim([-0.04,2.04])

    saveas = fname.split('/')[-1].split('.')[0]
    savename = str("threshold_images/" + saveas + ".png")
    #print(fname)
    plt.savefig(savename,dpi=300,format="png")
    plt.show(block = True)
    plt.close()
