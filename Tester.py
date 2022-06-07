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

    fname = "thin_slices/meltponds/meltpond_kuka/Meltpond_kuka_9cm_top.jpg"
    fTypes = np.array(['Empirical','Exponential','Gaussian','Exp Root'])
    fit = np.array([1,2,3])

    clip, blurredClip = clipBlur(fname, x=1800, y=2350, marginX=1350, marginY=1250, sigma=0.25)
    threshold = 0.55
    clip[blurredClip > threshold] = 0

    #plt.imshow(clip, cmap='gray')
    #plt.show()

    #clip[blurredClip <= threshold] = 1

    auflength, confint, functype, plotdata = acf(clip.T, lags = 200, conversion = 0.03214285714285714, plot = False, plotfunc = fit, ip=40, plotname="Testplot")
    print(auflength)
    print(confint)
    print(functype)
    plot_acf2(auflength, fTypes[fit], plotdata, block = True)


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

