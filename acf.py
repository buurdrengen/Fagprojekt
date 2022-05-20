from distutils.log import error
import string
from matplotlib.colors import Normalize
import numpy as np
from numpy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.stattools import acf as acff
from scipy.linalg import pinv
from scipy.linalg import solve


def acf(clip, lags=50, conversion = 90/2000, plot=False, plotfunc=1, plotname="Plot", ip=0, sections=3):

    #Set plotfunc as iterable
    if type(plotfunc) == int:
        plotfunc = [plotfunc]

    #clip, blurredClip = clipBlur(filename, x, y, margin, margin, sigma)

    n = np.shape(clip)[0]
    M = np.zeros(n)
    bestfunc = np.empty(sections, dtype= 'U32')
    functype = ["null","Exponetial","Gaussian"]

    #print(f"n is {n}")

    if sections != 0:
        blocks = np.int32(np.round(np.linspace(0, n, sections + 1)))
        M2 = np.zeros(sections)

        for idx,_ in enumerate(M2):
            bestfitctrl = np.inf
            blck = clip[blocks[idx]:blocks[idx + 1]]
            auto = autoCor(blck, nlags=lags)
            acl = autocolen(auto, scale=conversion)
            bestfunc[idx] = functype[0]
            
            for plt in plotfunc:
                _, fitctrl = plot_acf(auto, lags = lags, init_acl=acl, func = plt, lsmpoints=ip, plot = plot)
                #print(f"Error for {plt} is {fitctrl:.4e}")
                if fitctrl < bestfitctrl:
                    #print(f"{fitctrl:.2e} is less than {bestfitctrl:.2e}")
                    bestfunc[idx] = functype[plt]
                    bestfitctrl = fitctrl
            # print(f"Summing block {blocks[idx]} to {blocks[idx + 1]}")
            
            M2[idx] = acl
            # print(f"idx is {idx} with sum {M2[idx]:.4f}")

        M = np.copy(M2)
    else:
        for idx, i in enumerate(clip):
            auto = autoCor([i], nlags=lags)
            acl = autocolen(auto,conversion)
            M[idx] = acl


    # if plot:
    #     for idx in range(sections):
    #         for plt in plotfunc:
    #             fname = f"{plotname}-f{plt}.png"
    #             m,fitctrl = plot_acf(M, lags = lags, func = plt, saveas = fname, lsmpoints=ip)
    #             #print(f"Error for {plt} is {fitctrl:.4e}")
    #             if fitctrl < bestfitctrl:
    #                 #print(f"{fitctrl:.2e} is less than {bestfitctrl:.2e}")
    #                 bestm = m
    #                 bestfunc = plt
    #                 bestfitctrl = fitctrl

    #print(f"Autokorrelationslængden (Lineær) er {acl:.4f}mm")

    return M, bestfunc

def autoCor(clipBlur, nlags = 1999):
    # Function 
    # Viser autocorrelation mellem alle pixels i det clipped/blurred billede. 

    # Variables
    # clipped/blurred billede loaded fra clipBlur funktionen.

    # Først importeres relevante pakker, især her statsmodels.

    # Denne funktion tager det klippede og blurrede billede og bestemmer autocorrelation for alle pixels.
    # Nedenfor er testscript til hvordan autocorrelationsfunktionen fungerer for en enkelt linje. 

    # En linje 
    # linje = clip[1,:]
    # sm.tsa.acf(linje,nlags = 100)
    # fig = tsaplots.plot_acf(linje,lags = 100)
    # plt.show()

    # sizeY = clipBlur.shape[1]


    ## Et helt billede
    # nlags bestemmer hvor mange pixels der medtages, 0 regnes ikke med og der er 1999 lig 2000. 
    M = np.zeros(nlags+1)
    for i, clips in enumerate(clipBlur):
        auto = acff(clips,nlags=nlags)
        M = M + auto
        
        # if i %10 == 0: 
            # print(i)
    M = 1/(i+1) * M
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
    else:
        n0 = n[0]
    
    if n0 >=1 and n0 <= (np.size(acf)-1):
        dx = 1
        dy = acf[n0] - acf[n0-1]
        target = t - acf[n0-1]
        m = target/dy
        n = n0-1 + m 
        #print(f"m er {m:.3f}, n er {n}")
        #print(f"Test: p1: {acf[n0-1]}, p2: {acf[n0]}")
    else:
        print("")
        #print(f"n er {n}")

    return n*scale


def plot_acf(acf, lags, init_acl = 0.5, n=1, conversion=90/2000, niter=20, func=1, plot=False, saveas = None, lsmpoints = 0):

    #set interp-points
    if lsmpoints == 0: lsmpoints = lags

    x = np.arange(lags)
    y = acf[x]
    x = x*conversion

    sx = x#[0:lsmpoints+1]
    sy = y#[0:lsmpoints+1]

    if func == 1:
        m0 = [0.1, 1.1, -1]
    elif func == 2:
        m0 = [0.1,0.5,0.1,0.75]
        sy = np.hstack([np.flip(sy[1:]),sy])
        sx = np.hstack([np.flip(-sx[1:]),sx])

    m, errorlevel = lsm(sx, sy, m=m0, niter=niter, func=func, guess = init_acl)

    if func == 1:
        fy = func1(m,x)
        plotlabel = r'$c_1 \exp(k_1 x) + c_0$'
    elif func == 2:
        fy = func2(m,x)
        plotlabel = r'$\frac{c_1}{\sqrt{2 \pi \sigma^2}} \cdot \exp\left(-\frac{\left(x - \mu \right)^2}{2 \sigma^2}\right) + c_0$'

    #print(m)
    if (saveas != None) or plot:
        plt.figure()
        plt.plot(x,y,'bo',label="ACF")
        plt.plot(x,fy,'k-',label=plotlabel)
        plt.grid(True)
        plt.xlabel("Længde [mm]")
        plt.ylabel("ACF")
        plt.title("Autokorrelation")
        plt.ylim([-0.1, 1])
        plt.legend()
        if saveas != None:
            fname = str("plotimg/" + saveas)
            #print(fname)
            plt.savefig(fname,dpi=300,format="png")
        if plot: 
            plt.show(block=False)

    rvs = np.cumsum(y)
    cdf = np.cumsum(fy)

    [stat,pval] = scipy.stats.kstest(rvs,cdf)
    alpha = 0.01
    n = len(y)
    m1 = len(fy)
    test = np.sqrt(-np.log(alpha/2)*(1+m1/n)/(2*m1))

    #print(f"Statistic is {stat:.04f} compared to {test:.04f}")
    #print(f"p-value is {pval:.04f}")
    #e = np.exp(1)
    acl = [0,
    -1/m[2], 
    np.sqrt(2)*np.abs(m[1])-m[2]]

    functype = ["null", "eksponentiel", "Gauss"]

    print(f"Autokorrelationslængde fra {functype[func]} lsm: {acl[func]:.04f}mm")

    if errorlevel == 0:
        fitctrl = 1/(lags-1) *np.sum((y-fy)**2)
    else:
        fitctrl = np.inf

    return m, fitctrl

#---------------------------------------------------------------


def lsm(x,y, m=[0.1, 1.1, -1], niter=50, func=1, guess = 0.5):
    """
    Least square method
    """

    y = y[:,np.newaxis]
    x = x[:,np.newaxis]
    s2 = (x - guess)**2
    s2[s2 < 1e-4] = 1e-4
    sigma = 1/s2
    Cobs = np.eye(np.size(x))*sigma
    #print(f"size of x {np.shape(x)}")

    # plt.figure(10+func)
    # plt.plot(x,y,'k-*')
    # plt.show()

    #Itererer
    for i in range(niter): 
        errorlevel = 0
        G = df(m=m, x=x, func=func)
        GT = np.dot(np.transpose(G),Cobs)
        #GT = G.T
        #print(f"size of G {np.shape(G)}")
        if func == 1:
            dy = y - func1(m,x)
        elif func == 2:
            dy = y - func2(m,x)
        
        #sigma = 1 #Tilret sigma ved lejlighed
        #print(f"size of sigma {np.shape(sigma)}")
        
        A = np.dot(GT,G)
        b = np.dot(GT,dy)

        #print(f"size of cobs {np.shape(Cobs)}")
        #print(f"Iteration {i}:")
        
        #Moore-Penrose pseudo-inverse
        #print(m)
        #print(A)q
        #print(b)
        try: 
            delta = np.linalg.inv(A).dot(b) #solve(A,b, assume_a='sym')
        except scipy.linalg.LinAlgError:
            errorlevel = 2
            break
        except ValueError:
            errorlevel = 3
            break
        m = m + np.transpose(delta)[0]
        res = np.transpose(delta).dot(delta)[0][0]
        #print(f"Residuals for {i}: {res}")
        if res < 1e-6:
            break
        errorlevel = 1
        
    errortype =  ["Solution converges too slowly","Coefficient matrix is singular","Solution is divergent"]

    if errorlevel:
        print(f"Warning: {errortype[errorlevel - 1]} - Solution discarded!!!")

    return m, errorlevel


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




def scanclip (clip, lags=100, conversion = 90/2000, sections = 3):   




    # plt.figure()
    # plt.plot(np.arange(n)*conversion,M,'b-.')
    # plt.xlabel('Depth [mm]')
    # plt.ylabel('ACL')
    # plt.title('Autocorrelation Length')
    # plt.show()

    # fclip = np.transpose(clip)

    # fn = np.shape(fclip)[0]
    # fM = np.zeros(fn)


    # plt.figure()
    # plt.imshow(fclip, cmap = 'gray')
    # plt.show(block = False)


    # print(f"fn is {fn}")
    # for idx, i in enumerate(fclip):
    #     auto = autoCor([i], nlags=lags)
    #     acl = autocolen(auto,conversion)
    #     fM[idx] = acl


    # cm1 = M[:,np.newaxis]
    # cm2 = fM[:,np.newaxis]

    # aci = np.sqrt(np.dot(cm1,cm2.T))

    # var1 = aci #fftshift(aci)
    # var2 = np.log(np.abs(np.real(fft2(var1))))

    #aci [aci > 0.35] = 1
    
    # if np.shape(aci) == np.shape(clip):
    #     plt.figure()
    #     plt.imshow(var2, cmap = 'gray')
    #     plt.show(block = True)
    # else:
    #     print("Image Mismatic!")
    #     print(f"Shape of clip is {np.shape(clip)} while shape of fclip is {np.shape(fclip)}..")

    return 0