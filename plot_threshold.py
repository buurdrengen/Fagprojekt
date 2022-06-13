import numpy as np
import matplotlib.pyplot as plt

from acf import acf
from clipBlur import clipBlur


def plot_threshold(clip, blurredClip, conversion, saveas = None, plotshow = True):
    fit = np.array([0])
    x = np.arange(1,0.005,-0.005)
    s1 = np.zeros(np.size(x))
    s2 = np.zeros(np.size(x))
    s3 = np.zeros(np.size(x))
    for idx, i in enumerate(x):
        print(f"Progress: {(idx/1.99):0.0f}%", end="\r")
        clip[blurredClip > i] = 0

        auflength, _, _, _, _, _, _  = acf(clip, lags = 200, conversion = conversion, plotfunc = fit, ip=np.exp(-2))
        s1[idx] = auflength[0]
        s2[idx] = auflength[1]
        s3[idx] = auflength[2]
        #print(auflength)

    plt.figure()
    s1[s1 == 0] = 0
    s2[s2 == 0] = 0
    s3[s3 == 0] = 0
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
    plt.ylim([-0.04,2])

    
    #print(fname)
    if saveas != None:
        savename = str("threshold_images/" + saveas + ".png")
        plt.savefig(savename,dpi=300,format="png")
    if plotshow:
        plt.show(block = True)
    plt.close()
    print("Done!")


if __name__ == "__main__":
    fname = "thin_slices/meltponds/meltpond_kuka/meltpond_kuka_9cm_bottom.jpg"
    fTypes = np.array(['Empirical','Exponential','Gaussian','Exp Root'])


    clip, blurredClip = clipBlur(fname, x=1800, y=2150, marginX=1500, marginY=1150, sigma=0.25)
    conversion = 0.03

    plot_threshold(clip, blurredClip, conversion, saveas = None, plotshow = True)
