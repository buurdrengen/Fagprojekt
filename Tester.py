import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.graphics import tsaplots


x = np.arange(20)

plt.figure(1)   
tsaplots.plot_acf(x,lags = 19)
plt.show()
