import numpy as np
import matplotlib.pyplot as plt
import skimage

from acf import acf
from clipBlur import clipBlur


def plot_sigma(clip, threshold, conversion, saveas = None, plotshow = True):
    fit = np.array([0])
    x = np.arange(0.1,4.05,0.05)
    s1 = np.zeros(np.size(x))
    s2 = np.zeros(np.size(x))
    s3 = np.zeros(np.size(x))

    for idx, i in enumerate(x):
        print(f"Progress: {(idx/0.79):0.0f}%", end="\r")
        sclip = np.copy(clip)
        blurredClip = skimage.filters.gaussian(sclip, sigma=i, truncate=3.5)
        sclip[blurredClip > threshold] = 0

        auflength, _, _, _, _, _, _ = acf(sclip, lags = 200, conversion = conversion, plotfunc = fit, ip=np.exp(-2))
        s1[idx] = auflength[0]
        s2[idx] = auflength[1]
        s3[idx] = auflength[2]
        #print(auflength)

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
    plt.xlim([-0.08,4])
    plt.ylim([-0.02*yscale,1.02*yscale])

    #print(fname)
    if saveas != None:
        savename = str("sigma_images/" + saveas + ".png")
        plt.savefig(savename,dpi=300,format="png")
    if plotshow:
        plt.show(block = True)
    plt.close()


if __name__ == "__main__":
    fname = "thin_slices/meltponds/meltpond_kuka/meltpond_kuka_9cm_bottom.jpg"
    fTypes = np.array(['Empirical','Exponential','Gaussian','Exp Root'])


    clip, blurredClip = clipBlur(fname,  x=1800, y=2150, marginX=1500, marginY=1150, sigma=0.25)
    conversion=0.03

    plt.figure()
    plt.imshow(clip, cmap = 'gray')
    plt.show()
    plt.close()
    threshold = 0.52
    plot_sigma(clip, threshold, conversion, saveas = None, plotshow = True)
