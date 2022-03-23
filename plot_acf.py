#from turtle import width
#from statsmodels.graphics import tsaplots
#from statsmodels.graphics import utils
import numpy as np
import matplotlib.pyplot as plt


def plot_acf(acf, lags, n=1, conversion=90/2000, niter=20, func=1):

    x = np.arange(lags)
    y = acf[x]
    x = x*conversion
    m = lsm(x,y, niter=niter, func=func)
    if func == 1:
        fy = func1(m,x)
    elif func == 2:
        fy = func2(m,x)
    elif func == 3:
        fy = func3(m,x)
    
    print(m)
    plt.figure
    plt.plot(x,y,'bo')
    plt.plot(x,fy,'k-')
    plt.grid(True)
    plt.xlabel("Længde [mm]")
    plt.ylabel("ACF")
    plt.title("Autokorrelation")
    plt.ylim([0, 1])
    plt.show()


def lsm(x,y, m=[0.1, 1, -1, 1, -1], niter=50, func=1):
    """
    Least square method
    """
    y = y[:, np.newaxis]
    if func in [1,3]: m = m[0:3]

    for i in range(niter):

        if func == 2:
            #dobbelt eksponentiel funktion exp(x^2) + exp(x) + a0
            d0 = dfd0(m,x)
            d1 = dfd1(m,x)
            d2 = dfd2(m,x)
            d3 = dfd3(m,x)
            d4 = dfd4(m,x)
            G = np.transpose(np.vstack((d0,d1,d2,d3,d4)))
            yz = y - func2(m,x)[:,np.newaxis]
        elif func == 1:
            # eksponentiel funktion exp(x) + a0
            d0 = dfd0(m,x)
            d1 = dfd1(m,x)
            d2 = dfd2(m,x)
            G = np.transpose(np.vstack((d0,d1,d2)))
            yz = y - func1(m,x)[:,np.newaxis]
        elif func == 3:
            #eksponentiel funktion exp(x^2) + a0
            d0 = dfd0(m,x)
            d1 = dfd3(np.array([m[0],0,0,m[1],m[2]]),x)
            d2 = dfd4(np.array([m[0],0,0,m[1],m[2]]),x)
            G = np.transpose(np.vstack((d0,d1,d2)))
            yz = y - func3(m,x)[:,np.newaxis]
        
        delta = np.linalg.lstsq(G,yz,rcond=None)[0]
        #print(np.shape(delta))
        m = m + np.transpose(delta)[0]
        res = np.transpose(delta).dot(delta)
        #print(f"Residuals for {i}: {res}")
        if res < 1e-8:
            break
        
    return m


def func1(m,x):
    return m[0] + m[1]*np.exp(m[2]*x)

def func2(m,x):
    return m[0] + m[1]*np.exp(m[2]*x) + m[3]*np.exp(m[4]*x**2)

def func3(m,x):
    return m[0] + m[1]*np.exp(m[2]*x**2)


def dfd0(m,x):
    return np.ones(np.size(x))

def dfd1(m,x):
    return np.exp(m[1]*x)

def dfd2(m,x):
    return m[1]*m[2]*np.exp(m[2]*x)

def dfd3(m,x):
    return np.exp(m[4]*x**2)

def dfd4(m,x):
    return m[3]*m[4]*np.exp(m[4]*x**2)






