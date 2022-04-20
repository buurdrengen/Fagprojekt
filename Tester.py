import numpy as np
# from autocolen import autocolen
from clipBlur import clipBlur
# from autocofunc import autoCor
from acf import acf, scanclip
# #from string import rfind
# from plot_acf import func2
# import matplotlib.pyplot as plt


print(np.zeros(12))
# a = np.empty(12, dtype='object')
# a[0:3] = [1,2, "bananas"]
# print(a)

# clip, blurredClip = clipBlur("20200220_221515.jpg", x=1700, y=3000, marginX=1100, marginY=1000, sigma=1)
# clip[blurredClip > 0.65] = 1


# auflength = acf(clip, lags = 1100-1, conversion = 90/2000, plot = False, plotfunc = [1])
# print(auflength)

# M = scanclip(clip)

#len,funct = acf(clip, plotname="Name", plot = True, plotfunc = [1, 2], lags=50, ip=40)
#print(f"Autocorrelation length is {len:.3}mm")
#print(f"The best fit is a {funct} function")
#acf("thin_slices/firstyearice/southerntransect/20200201_145739.jpg", plot=True, plotfunc=2)

# # plt.figure(10)
# # xx = np.arange(-3,3,0.05)
# # plt.plot(xx,func2([0,1,0,1],xx),'b-')
# # plt.show()
# print(int(6.9))


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

