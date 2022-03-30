import numpy as np
from autocolen import autocolen
from clipBlur import clipBlur
from autocofunc import autoCor
from acf import acf
#from string import rfind
from plot_acf import func2
import matplotlib.pyplot as plt

acf("20200220_221515.jpg", plot=True, plotfunc=[1,2])
#acf("thin_slices/firstyearice/southerntransect/20200201_145739.jpg", plot=True, plotfunc=2)

# plt.figure(10)
# xx = np.arange(-3,3,0.05)
# plt.plot(xx,func2([0,1,0,1],xx),'b-')
# plt.show()
# print(int(6.9))
