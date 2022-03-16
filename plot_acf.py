from statsmodels.graphics import tsaplots
from statsmodels.graphics import utils
import numpy as np
import matplotlib.pyplot as plt


def plot_acf(acf, lags, n=1):

    x = np.arange(lags)
    y = acf[x]
    x = x*90/2000
    m = lsm(x,y)
    fy = func(m,x)
    print(m)
    plt.figure
    plt.plot(x,y,'bo')
    plt.plot(x,fy,'k-')
    plt.grid(True)
    plt.show()


def lsm(x,y, m=[1, -1, 0.1], niter=20):
    """
    Least square method
    """
    y = y[:, np.newaxis]

    for i in range(niter):
        d1 = dfd1(m,x)
        d2 = dfd2(m,x)
        d3 = dfd3(m,x)
        G = np.transpose(np.vstack((d1,d2,d3)))
        yz = y - func(m,x)[:,np.newaxis]
        delta = np.linalg.pinv(G).dot(yz) #magic
        res = np.dot(np.transpose(delta)[0],delta)
        m = m + np.transpose(delta)[0]
        print(f"Residuals for {i}: {res}")
        if res < 1e-5:
            break
        
    return m



def func(m,x):
    return m[0]*np.exp(m[1]*x) + m[2]

def dfd1(m,x):
    return np.exp(m[1]*x)

def dfd2(m,x):
    return m[0]*m[1]*np.exp(m[1]*x)

def dfd3(m,x):
    return np.ones(np.size(x))