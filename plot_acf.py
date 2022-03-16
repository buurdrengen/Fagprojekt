from statsmodels.graphics import tsaplots
from statsmodels.graphics import utils
import numpy as np
import matplotlib.pyplot as plt

# This is a modified version of of statsmodels.graphics.plot_acf

def plot_acf(acf, lags, n=1):

    plt.figure
    x = np.arange(lags)
    y = acf[x]
    plt.plot(x,y,'b-')
    plt.show()



def lsm(x,y,m=[1, -1]):
    return 0

def dfda(m,x):
    return np.exp(-m[2]*x)
