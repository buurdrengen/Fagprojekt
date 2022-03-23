#from turtle import width
#from statsmodels.graphics import tsaplots
#from statsmodels.graphics import utils
import numpy as np
import matplotlib.pyplot as plt
import scipy


def plot_acf(acf, lags, n=1, conversion=90/2000, niter=20, func=1, saveas = None):

    x = np.arange(lags)
    y = acf[x]
    x = x*conversion
    m = lsm(x,y, niter=niter, func=func)

    if func == 1:
        fy = func1(m,x)
        plotlabel = r'$c_1 \exp(k_1 x) + c_0$'
    elif func == 2:
        fy = func2(m,x)
        plotlabel = r'$c_2 \exp(k_2 x^2) + c_1 \exp(k_1 x) + c_0$'
    
    #print(m)
    plt.figure()
    plt.plot(x,y,'bo',label="ACF")
    plt.plot(x,fy,'k-',label=plotlabel)
    plt.grid(True)
    plt.xlabel("Længde [mm]")
    plt.ylabel("ACF")
    plt.title("Autokorrelation")
    plt.ylim([0, 1])
    plt.legend()
    if saveas != None:
        fname = str("plotimg/" + saveas)
        #print(fname)
        plt.savefig(fname,dpi=300,format="png")
        plt.show(block=False)
    plt.close()

    rvs = np.cumsum(y)
    cdf = np.cumsum(fy)

    [stat,pval] = scipy.stats.kstest(rvs=rvs,cdf=cdf)
    alpha = 0.01
    n = len(rvs)
    m = len(cdf)
    test = np.sqrt(-np.log(alpha/2)*(1+m/n)/(2*m))

    print(f"Statistic is {stat:.04f} compared to {test:.04f}")
    print(f"p-value is {pval:.04f}")

def lsm(x,y, m=[0.1, 1, -1], niter=50, func=1):
    """
    Least square method
    """
    y = y[:, np.newaxis]

    for i in range(niter):

        if func == 1:
            # eksponentiel funktion m1*exp(m2*x) + m0
            d0 = dfd0(m,x)
            d1 = df1d1(m,x)
            d2 = df1d2(m,x)
            G = np.transpose(np.vstack((d0,d1,d2)))
            yz = y - func1(m,x)[:,np.newaxis]
        elif func == 2:
            #Gaussisk funktion exp(x^2) + a0
            d0 = dfd0(m,x)
            d1 = df2d1(m,x)
            d2 = df2d2(m,x)
            G = np.transpose(np.vstack((d0,d1,d2)))
            yz = y - func2(m,x)[:,np.newaxis]
        
        delta = np.linalg.lstsq(G,yz,rcond=None)[0]
        #print(np.shape(delta))
        m = m + np.transpose(delta)[0]
        res = np.transpose(delta).dot(delta)[0][0]
        #print(f"Residuals for {i}: {res}")
        if res < 1e-8:
            break
        
    print(m)
    return m


def func1(m,x):
    return m[0] + m[1]*np.exp(m[2]*x)

def func2(m,x):
    return m[0] + 1/(m[1]*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-m[2])/m[1])**2)



def dfd0(m,x):
    return np.ones(np.size(x))

def df1d1(m,x):
    return np.exp(m[2]*x)

def df1d2(m,x):
    return m[0]*m[1]*np.exp(m[2]*x)

def df2d1(m,x):
    return ((x - m[2])**2 - m[1]**2)/(m[1]**3) * func2([0,m[1],m[2]],x)

def df2d2(m,x):
    return (x-m[2])/(m[1]**2) * func2([0,m[1],m[2]],x)





