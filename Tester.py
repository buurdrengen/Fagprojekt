import numpy as np
from autocolen import autocolen
from clipBlur import clipBlur
from autocofunc import autoCor
from acf import acf
#from string import rfind
from plot_acf import func2
import matplotlib.pyplot as plt


clip, blurredClip = clipBlur("20200220_221515.jpg", x=1700, y=3000, marginX=1100, marginY=1000, sigma=1)
clip[blurredClip > 0.65] = 0
len = acf(clip, plotname="Name", plot = True, plotfunc = [1])
print(f"ACL er {len:.3}mm")
#acf("thin_slices/firstyearice/southerntransect/20200201_145739.jpg", plot=True, plotfunc=2)

# plt.figure(10)
# xx = np.arange(-3,3,0.05)
# plt.plot(xx,func2([0,1,0,1],xx),'b-')
# plt.show()
# print(int(6.9))
