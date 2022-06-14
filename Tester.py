from cProfile import label
import numpy as np
# from autocolen import autocolen
from clipBlur import clipBlur
# from autocofunc import autoCor
from acf import acf, plot_acf2
# #from string import rfind
# from plot_acf import func2
import matplotlib.pyplot as plt

if __name__ == "__main__":

#print(len(np.empty(12)))
# a = np.empty(12, dtype='object')
# a[0:3] = [1,2, "bananas"]
# print(a)

#---------------------------------------------------------

    fname = "images/hummock1_bot_7cm_1.jpg"
    fTypes = np.array(['Empirical','Exponential','Gaussian','Exp Root',"x-Power","x-Exponential"])
    fit = np.array([1,2,4,5])

    clip, blurredClip = clipBlur(fname, x=1800, y=2400, marginX=1400, marginY=1400, sigma=2)
    threshold = 0.65
    conversion = 0.03857
    clip[blurredClip > threshold] = 0

    plt.imshow(clip, cmap='gray')
    plt.show(block=True)

    #clip[blurredClip <= threshold] = 1

    auflength, confint, functype, plotdata, fitness, kvalue, xvalue = acf(clip, lags = 200, conversion = conversion, plot = False, plotfunc = fit, ip=np.exp(-2), plotname="Testplot",)
    print(auflength)
    print(confint)
    print(functype)
    print(np.round_(-np.log(fitness),decimals=2))
    print(kvalue)
    print(xvalue)
    plot_acf2(auflength, fTypes[fit], plotdata, block = True, xmax = 2)


#---------------------------------------------------------

#---------------------------------------------------------
# M = scanclip(clip)
   # l = input("Wait...")

#len,funct = acf(clip, plotname="Name", plot = True, plotfunc = [1, 2], lags=50, ip=40)
#print(f"Autocorrelation length is {len:.3}mm")
#print(f"The best fit is a {funct} function")
#acf("thin_slices/firstyearice/southerntransect/20200201_145739.jpg", plot=True, plotfunc=2)

# # plt.figure(10)
# # xx = np.arange(-3,3,0.05)
# # plt.plot(xx,func2([0,1,0,1],xx),'b-')
# # plt.show()
    print(int(6.9))


# fileName = "20200220_221515.jpg"
# filePath = fileName.split("/")
# print(filePath)
# filename = filePath[-1].split(".")[0]
# print(filename)
# # Matrix = np.empty((1,11))
# # Matrix[:] = np.NaN
# txtName = filename + '.txt'
# print(txtName)
# # Matrix[0,0:11] = np.array([blur, top, bottom, left, right, \
# #     conversion, threshold, marginY, marginX, yMiddle, xMiddle])
# # np.savetxt(txtName, Matrix, delimiter=',')

