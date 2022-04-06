import numpy as np
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.stattools import acf as acff
from clipBlur import clipBlur
from numpy.linalg import solve


def acf(clip, lags=100, conversion = 90/2000, plot=False, plotfunc=1, plotname="Plot"):

    #clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)
    M = autoCor(clip)
    acl = autocolen(M,conversion)


    if plot:
        for plt in plotfunc:
            fname = f"{plotname}-f{plt}.png"
            plot_acf(M, lags = lags, func = plt, saveas = fname)

    #print(f"Autokorrelationslængden (Lineær) er {acl:.4f}mm")

    return acl


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
    n0 = n[0]

    if np.size(n) == 0:
        return 0
    elif n0 >=1 and n0 <= (np.size(acf)-1):
        dx = 1
        dy = acf[n0] - acf[n0-1]
        target = t - acf[n0-1]
        m = target/dy
        n = n0 + m - 1
        #print(f"m er {m:.3f}, n er {n}")
        #print(f"Test: p1: {acf[n0-1]}, p2: {acf[n0]}")
    else:
        print("")
        #print(f"n er {n}")

    return n*scale


def plot_acf(acf, lags, n=1, conversion=90/2000, niter=20, func=1, saveas = None, lsmpoints = 21):

    x = np.arange(lags)
    y = acf[x]
    x = x*conversion

    sx = x[0:lsmpoints+1]
    sy = y[0:lsmpoints+1]

    if func == 1:
        m0 = [0.1, 1.1, -1]
    elif func == 2:
        m0 = [0.1,1,0.1,1.1]
        sy = np.hstack([np.flip(y[1:]),y])
        sx = np.hstack([np.flip(-x[1:]),x])

    m = lsm(sx, sy, m=m0, niter=niter, func=func)

    if func == 1:
        fy = func1(m,x)
        plotlabel = r'$c_1 \exp(k_1 x) + c_0$'
    elif func == 2:
        fy = func2(m,x)
        plotlabel = r'$\frac{c_1}{\sqrt{2 \pi \sigma^2}} \cdot \exp\left(-\frac{\left(x - \mu \right)^2}{2 \sigma^2}\right) + c_0$'

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
    acl = [0,-1/m[2], np.sqrt(2)*np.abs(m[1])-m[2]]
    functype = ["null", "eksponentiel", "Gauss"]
    print(f"Autokorrelationslængde fra {functype[func]} lsm: {acl[func]:.04f}mm")

    return m

#---------------------------------------------------------------


def lsm(x,y, m=[0.1, 1.1, -1], niter=50, func=1):
    """
    Least square method
    """
    # Omdanner til søjlevektorer


    y = y[:,np.newaxis]
    x = x[:,np.newaxis]
    #print(f"size of x {np.shape(x)}")

    #Itererer
    for i in range(niter): 
        G = df(m=m, x=x, func=func)
        GT = G.T
        #print(f"size of G {np.shape(G)}")
        if func == 1:
            yz = y - func1(m,x)
        elif func == 2:
            yz = y - func2(m,x)
        
        sigma = 1 #Tilret sigma ved lejlighed
        #print(f"size of sigma {np.shape(sigma)}")
        Cobs = np.eye(np.size(x))*sigma
        #print(f"size of cobs {np.shape(Cobs)}")
        A = GT.dot(G)
        #print(A)
        #print(np.linalg.det(A))
        #print(A)
        b = GT.dot(yz)
        #print(b)

        delta = solve(A,b) #Magi
        m = m + np.transpose(delta)[0]
        res = np.transpose(delta).dot(delta)[0][0]
        #print(f"Residuals for {i}: {res}")
        if res < 1e-8:
            break
        
        
    if i == niter-1:
        print("Warning: Solution does not converge sufficiently fast!")

    #print(m)
    return m


#--------------------------------------------------------------

def func1(m,x):
    return m[0] + m[1]*np.exp(m[2]*x)

def func2(m,x):
    return m[0] + (m[3]/(m[1]*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-m[2])**2) / (m[1]**2))

#--------------------------------------------------------------

def df(m,x,func):

    if func == 1:
        G = np.hstack([np.ones(np.size(x))[:,np.newaxis], #dfd0
        np.exp(m[2]*x), #dfd1
        m[1]*x*np.exp(m[2]*x)]) #dfd2
    elif func == 2:
       G = np.hstack([np.ones(np.size(x))[:,np.newaxis], #dfd0
       (((x - m[2])**2 - m[1]**2)/(m[1]**3)) * func2([0,m[1],m[2],m[3]],x), #dfd1
       ((x-m[2])/(m[1]**2)) * func2([0,m[1],m[2],m[3]],x), #dfd2
       func2([0,m[1],m[2],1],x)]) #dfd3

    return G 


# def dfd0(m,x): #c_0
#     return np.ones(np.size(x))[:,np.newaxis]

# def df1d1(m,x):
#     return np.exp(m[2]*x)

# def df1d2(m,x):
#     return m[1]*m[2]*np.exp(m[2]*x)

# def df2d1(m,x): #sigma
#     return (((x - m[2])**2 - m[1]**2)/(m[1]**3)) * func2([0,m[1],m[2],m[3]],x)

# def df2d2(m,x): #mu
#     return ((x-m[2])/(m[1]**2)) * func2([0,m[1],m[2],m[3]],x)

# def df2d3(m,x): #C_1
#     return func2([0,m[1],m[2],1],x)