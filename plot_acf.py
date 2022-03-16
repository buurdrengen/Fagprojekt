from statsmodels.graphics import tsaplots
from statsmodels.graphics import utils
import numpy as np
import matplotlib.pyplot as plt

# This is a modified version of of statsmodels.graphics.plot_acf
#Original file: https://www.statsmodels.org/dev/_modules/statsmodels/graphics/tsaplots.html

def plot_acf(acf, lags, n=1):

    plt.figure(n)
    x = np.arange(lags)
    y = acf[x]
    plt.plot(x,y,'b-')
    plt.show()
