from statsmodels.graphics import tsaplots
from statsmodels.graphics import utils
import numpy as np
import matplotlib.pyplot as plt

# This is a modified version of of statsmodels.graphics.plot_acf

def plot_acf(acf, lags, n=1):

    x = np.arange(lags)
    y = acf[x]
    x = x*90/2000
    m = lsm(x,y)
    fy = func(m,x)
    plt.figure
    plt.plot(x,y,'bo')
    plt.plot(x,fy,'k-')
    plt.show()


def lsm(x,y,m=[1, -1]):
    x = x
    y = y
    m = m
    for i in range(10):
        da = dfda(m,x)
        dk = dfdk(m,x)
        G = np.vstack((da,dk))
        GH = np.dot(G,np.transpose(G))
        yz = func(m,x)
        delta = np.dot(np.dot(G,y-yz),np.linalg.pinv(GH))
        res = np.dot(delta,np.transpose(delta))
        m = m + res
        print(f"Residuals: {res}")
        if res < 1e-5:
            break
        
    return m



    print(f"size of delta is {np.shape(delta)}")
    print(delta)
    return 0


def func(m,x):
    return m[0]*np.exp(m[1]*x)

def dfda(m,x):
    return np.exp(m[1]*x)

def dfdk(m,x):
    return m[0]*m[1]*np.exp(m[1]*x)