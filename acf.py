import numpy as np
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.stattools import acf as acff
from clipBlur import clipBlur


def acf(filename, xy=[1300,3000], margin=1000, threshold = 0.6, sigma = 5.0, lags=100, conversion = 90/2000, plot=False, plotfunc=1):
    [x,y] = xy

    clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)
    M = autoCor(blurredClip)
    acl = autocolen(M,conversion)


    if plot:
        for plt in plotfunc:
            x1 = filename.rfind('/')
            x2 = filename.rfind('.')
            fname = str(filename[x1+1:x2] + '-f' + str(plt) + '.png')
            plot_acf(M, lags = lags, func = plt, saveas = fname)

    print(f"Autokorrelationslængden (Lineær) er {acl:.4f}mm")


def autoCor(clipBlur, nlags = 1999):
    # Function 
    # Viser autocorrelation mellem alle pixels i det clipped/blurred billede. 

    # Variables
    # clipped/blurred billede loaded fra clipBlur funktionen.

    # Først importeres relevante pakker, især her statsmodels.

    # Denne funktion tager det klippede og blurrede billede og bestemmer autocorrelation for alle pixels.
    # Nedenfor er testscript til hvordan autocorrelationsfunktionen fungerer for en enkelt linje. 

    # ## En linje 
    # linje = clip[1,:]
    # sm.tsa.acf(linje,nlags = 100)
    # fig = tsaplots.plot_acf(linje,lags = 100)
    # plt.show()

    ## Et helt billede
    # nlags bestemmer hvor mange pixels der medtages, 0 regnes ikke med og der er 1999 lig 2000. 
    M = np.zeros(nlags+1)
    for i, clips in enumerate(clipBlur):
        auto = acff(clips,nlags=nlags)
        M = M + auto
        # if i %10 == 0: 
            # print(i)
    M = 1/i*M
    #C = 1/np.sqrt(2000)*C #Hacked konfidensinterval, check metoden 

    # lags bestemmer hvor mange punkter der plottes

    # # lags bestemmer hvor mange punkter der plottes 
    # plt.figure(1)   
    # tsaplots.plot_acf(M,lags = 100)
    # plt.show()
    return M

def autocolen(acf,scale=1):
    """
    Beregner Autokorrelationslængden
    acf er en beregnet vektor af Auto korrelations funktionen
    leng er længden af billedet i enhed [længde per pixel], valgfri
    """
    t = np.exp(-1)
    n = np.arange(np.size(acf))[acf <= t]
   
    if np.size(n) == 0:
        return 0
    elif n[0] >=1 and n[0] <= (np.size(acf)-1):
        dx = 1
        dy = acf[n[0]] - acf[n[0]-1]
        target = t - acf[n[0]-1]
        m = dx*dy/target
        n = n[0] - m
        #print(f"m er {m:.3f}, n er {n}")
    else:
        print("")
        #print(f"n er {n}")

    return n*scale


def plot_acf(acf, lags, n=1, conversion=90/2000, niter=20, func=1, saveas = None):

    x = np.arange(lags)
    y = acf[x]
    x = x*conversion
    if func == 1:
        m0 = [0.1, 1, -1]
    elif func == 2:
        m0 = [0.1,1,0.1,1]
    
    m = lsm(x, y, m=m0, niter=niter, func=func)

    if func == 1:
        fy = func1(m,x)
        plotlabel = r'$c_1 \exp(k_1 x) + c_0$'
    elif func == 2:
        fy = func2(m,x)
        plotlabel = r'$c_1 / \sqrt{2 \pi \sigma^2} \cdot \exp\left(-\frac{\left(x - \mu \right)^2}{2 \sigma}\right) + c_0$'

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

    #rvs = np.cumsum(y)
    #cdf = np.cumsum(fy)

    [stat,pval] = scipy.stats.kstest(y,fy)
    alpha = 0.01
    n = len(y)
    m1 = len(fy)
    test = np.sqrt(-np.log(alpha/2)*(1+m1/n)/(2*m1))

    print(f"Statistic is {stat:.04f} compared to {test:.04f}")
    print(f"p-value is {pval:.04f}")
    acl = np.sqrt(2)*m[1]-m[2]
    functype = {1: "eksponentiel", 2: "Gauss"}
    print(f"Autokorrelationslængde fra {functype[func]} lsm: {acl:.04f}mm")

#---------------------------------------------------------------


def lsm(x,y, m=[0.1, 1, -1], niter=50, func=1):
    """
    Least square method
    """
    # Omdanner til søjlevektorer
    y = np.hstack([np.flip(y),y])[:,np.newaxis]
    x = np.hstack([np.flip(-x),x])[:,np.newaxis]

    #Itererer
    for i in range(niter): 

        if func == 1:
            # eksponentiel funktion m1*exp(m2*x) + m0
            d0 = dfd0(m,x)
            d1 = df1d1(m,x)
            d2 = df1d2(m,x)
            G = np.hstack((d0,d1,d2))
            yz = y - func1(m,x)
        elif func == 2:
            #Gaussisk funktion exp(x^2) + a0
            d0 = dfd0(m,x)
            d1 = df2d1(m,x)
            d2 = df2d2(m,x)
            d3 = df2d3(m,x)
            G = np.hstack((d0,d1,d2,d3))
            yz = y - func2(m,x)
        
        delta = np.linalg.lstsq(G,yz,rcond=None)[0] #Magi
        m = m + np.transpose(delta)[0]
        res = np.transpose(delta).dot(delta)[0][0]
        #print(f"Residuals for {i}: {res}")
        if res < 1e-8:
            break
        
    #print(m)
    return m


#--------------------------------------------------------------

def func1(m,x):
    return m[0] + m[1]*np.exp(m[2]*x)

def func2(m,x):
    return m[0] + (m[3]/(m[1]*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-m[2])**2) / (m[1]**2))

#--------------------------------------------------------------

def dfd0(m,x): #c_0
    return np.ones(np.size(x))[:,np.newaxis]

def df1d1(m,x):
    return np.exp(m[2]*x)

def df1d2(m,x):
    return m[0]*m[1]*np.exp(m[2]*x)

def df2d1(m,x): #sigma
    return (((x - m[2])**2 - m[1]**2)/(m[1]**3)) * func2([0,m[1],m[2],m[3]],x)

def df2d2(m,x): #mu
    return ((x-m[2])/(m[1]**2)) * func2([0,m[1],m[2],m[3]],x)

def df2d3(m,x): #C_1
    return func2([0,m[1],m[2],1],x)