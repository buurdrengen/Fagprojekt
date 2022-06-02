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

    # fname = "images/20200206_103511.jpg"
    # fTypes = np.array(['null','Exponential','Gaussian','New Function'])
    # fit = np.array([1,2,3])

    # clip, blurredClip = clipBlur(fname, y=2800, x=1825, marginY=1100, marginX=1175, sigma=0.25)
    # threshold = 0.5
    # clip[blurredClip > threshold] = 0
    # #clip[blurredClip <= threshold] = 1

    # auflength, functype, plotdata = acf(clip.T, lags = 200, conversion = 0.03214285714285714, plot = False, plotfunc = fit, ip=40, plotname="Testplot")
    # print(auflength)
    # plot_acf2(auflength, fTypes[fit], plotdata, block = True)


#---------------------------------------------------------

    fname = "images/20200206_103511.jpg"
    fTypes = np.array(['null','Exponential','Gaussian','New Function'])
    fit = np.array([1,2,3])

    s1 = np.array([0])
    s2 = np.array([0])
    s3 = np.array([0])
    clip, blurredClip = clipBlur(fname, y=2800, x=1825, marginY=1100, marginX=1175, sigma=0.25)

    x = np.arange(0.4,1,0.2)
    for blr in x:
        clip[blurredClip > blr] = 0

        auflength, functype, plotdata = acf(clip.T, lags = 200, conversion = 0.03214285714285714, plot = False, plotfunc = fit, ip=40, plotname="Testplot")
        s1 = np.hstack([s1,auflength[0]])
        s2 = np.hstack([s2,auflength[1]])
        s3 = np.hstack([s3,auflength[2]])

    plt.plot()
    #plt.grid(True)
    plt.plot(x,s1[1:],'r-.',label='Section 1')
    plt.plot(x,s2[1:],'b-.',label='Section 2')
    plt.plot(x,s3[1:],'k-.',label='Section 3')
    plt.xlabel("Threshold")
    plt.ylabel("ACL")
    plt.title(f"Autocorrelation Length")
    plt.legend()
    plt.show()

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
#print(int(6.9))


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

