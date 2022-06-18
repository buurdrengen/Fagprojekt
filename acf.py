import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf as acff


def acf(clip, lags=50, conversion = 90/2000, plot=False, plotfunc=[1,2], plotname="", ip=np.exp(-2), sections=3, errorlimit = np.inf, alpha =0.05):

    #Set plotfunc as iterable
    if type(plotfunc) == int:
        plotfunc = [plotfunc]

    n = np.shape(clip)[0]
    M = np.zeros(n)
    bestfunc = np.empty(sections, dtype= 'U32')
    functype = ["Empirical","Exponential","Gaussian","Exp Root", "x-Power", "x-Exponential"]

    plotdata = np.empty(lags)

    if sections != 0:
        blocks = np.int32(np.round(np.linspace(0, n, sections + 1)))
        M2 = np.zeros(sections)
        N2 = np.zeros(sections)
        kval = np.zeros(sections)
        xval = np.zeros(sections)
        fitness = np.zeros([sections,np.size(plotfunc)])

        for idx,_ in enumerate(M2):
            bestfitctrl = np.inf
            blck = clip[blocks[idx]:blocks[idx + 1]]
            auto, confint = autoCor(blck, nlags=lags, alpha = alpha)
            acl_est, std_est = autocolen(auto, confint=confint, scale=conversion)
            M2[idx] = acl_est
            N2[idx] = std_est

            if any(auto < -1) or any(auto > 1):
                print("Warning[acf]: Autocorrelation contains invalid values!")
            if any(np.isnan(auto)):
                print("Warning[acf]: Warning: Autocorrelation contains NaN entries!")

            #print(f"Estimated Autocorrelation Length: {acl_est:.04f}mm")

            bestfunc[idx] = functype[0]
            x = np.arange(lags)
            y = auto[x]
            x = x*conversion
            plotdata = np.vstack([plotdata,x,y])
            
            for jdx, pf in enumerate(plotfunc):
                coeffs, iidx = lsm3(x,y,pf, limit=ip,L=acl_est)
                c = np.hstack([0,coeffs])
                if pf == 1:
                    fy = func1(c,x)
                    #plotlabel = r'$c_1 \exp(k_1 x) + c_0$'
                    acl = 1/c[2]
                elif pf == 2:
                    fy = func2(c,x)
                    #plotlabel = r'$\frac{c_1}{\sqrt{2 \pi \sigma^2}} \cdot \exp\left(-\frac{\left(x - \mu \right)^2}{2 \sigma^2}\right) + c_0$'
                    acl = np.sqrt(2)*c[1]-c[2]
                elif pf == 3:
                    fy = func3(c,x)
                    acl = 1/(c[2]**2)
                elif pf == 4:
                    fy = func4(c,x)
                    acl = 1/(c[2])
                    xval[idx] = c[3]
                elif pf == 5:
                    fy = func5(c,x)
                    acl = 1/(c[2])
                    kval[idx] = c[3]
                elif pf == 0:
                    fy = y
                    acl = acl_est

                plotdata = np.vstack([plotdata,fy])

                # print(f"Autocorrelation legth from {functype[pf]} lsm: {acl:.04f}mm")

                fitctrl = 1/(iidx) *np.sum((y[:iidx]-fy[:iidx])**2)
                fitness[idx,jdx] = fitctrl

                if fitctrl < bestfitctrl:
                    #Error tolerance
                    if  np.not_equal(acl_est,0):
                        if abs(acl/acl_est - 1) > errorlimit:
                            acl = acl_est
                            #print('Linear approximation used due to too large deviation..')
                        else:
                            bestfunc[idx] = functype[pf]
                            bestfitctrl = fitctrl

        M = np.copy(M2)
    else:
        for idx, i in enumerate(clip):
            auto = autoCor([i], nlags=lags)
            acl = autocolen(auto,conversion)
            M[idx] = acl_est


    return M, N2, bestfunc, plotdata[1:], fitness, kval, xval

#-----------------------------------------------------------------------------

def autoCor(clipBlur, nlags = 1999, alpha = 0.05):
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
    N = np.zeros(nlags+1)
    err = 0
    for i, clips in enumerate(clipBlur):
        if all(clips == 0):
            auto = np.zeros(np.size(nlags))
            confint = np.zeros(np.size(nlags))

        else:
            auto, confint = acff(clips, nlags=nlags, missing='drop', alpha= alpha)
            cn0 = confint[:,0] - auto
            cn1 = confint[:,1] - auto
            confint = (cn1 - cn0)*0.5


        if any(np.isnan(auto)):
            print(f"Warning[autoCor]: NaN number in line {i}! - dropping line..")
            auto = np.zeros(np.size(nlags))
            err += 1
            if any(np.isnan(clips)):
                print(f"    This is due to erroneous input!!")

        M = M + auto
        N = N + confint**2

    M = 1/(i+1-err) * M
    N = 1/(i+1-err) * np.sqrt(N)

    return M, N

#-----------------------------------------------------------------------------

def autocolen(acf,confint,scale=1):
    """
    Beregner Autokorrelationslængden
    acf er en beregnet vektor af Auto korrelations funktionen
    leng er længden af billedet i enhed [længde per pixel], valgfri
    """
    t = np.exp(-1)
    n = np.arange(np.size(acf))[acf <= t]

    if np.size(n) < 2:
        n0 = np.size(acf) - 1
    else:
        n0 = n[0]

    if np.size(confint) < 2:
        confint = np.zeros(np.shape(acf))
    
    if n0 >= 1:
        y1 = acf[n0]; y2 = acf[n0-1]
        dy = y1 - y2
        target = t - y2
        m = target/dy
        n = n0-1 + m
        
        sy1 = confint[n0]
        sy2 = confint[n0-1]
        sa = np.sqrt(sy1**2 + sy2**2)

        sigma = np.sqrt( np.exp(-2) * sa**2 /dy**2  + (sy1**2 * y2**2 + sy2**2 * y1**2)/np.power(y1 - y2,4) )
    else:
        n = 0; sigma = 0
        #print(f"n er {n}")
    return n*scale, sigma*scale

#-----------------------------------------------------------------------------

def func1(m,x):
    return m[0] + m[1]*np.exp(-m[2]*x)

def func2(m,x):
    return m[0] + (m[3]/(m[1]*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-m[2])**2) / (m[1]**2))

def func3(m,x):
    return m[0] + m[1]*np.exp(-m[2]*np.sqrt(x))

def func4(m,x):
    return m[0] + m[1]*np.power(1 + (m[2]*x)**2,-m[3])

def func5(m,x):
    return m[0] + m[1]*np.exp(-np.power(m[2]*x,m[3]))

#--------------------------------------------------------------

def lsm3(x,y, func=1, limit = np.exp(-2), L=1):
    """
    Least square method
    """
    if func == 0:
        return [0,0,0],1
    idx = y>limit
    if all(idx):
        i = np.size(idx)
    else:
        i = np.where(idx == False)[0][0]
        

    Cobs = np.diag(1/((abs(y[:i] - np.exp(-1))**2)+1))

    y = y[:i,np.newaxis]
    x = x[:i,np.newaxis]

    ly = np.log(y)

    A = np.hstack([x])
    if func == 2:
        A = np.hstack([x**2])
        #print(np.shape(A))
    if func == 3:
        A = np.hstack([np.sqrt(abs(x))])

    if func == 4:
        ly = np.log(y[1:])
        A = np.hstack([np.log(1 + (x[1:]/L)**2)])

    if func == 5:
        ly = np.log(-ly[1:])
        A = np.hstack([np.log(x[1:]), np.ones(np.shape(x[1:]))])


    ATA = np.transpose(A) @ A
    b = np.transpose(A) @ ly

    try:
        m = np.linalg.inv(ATA) @ b # Computes 'inv(A^T A) A^T y' efficiently
    except np.linalg.LinAlgError:
        print(f"Unable to compute function {func}: Singular Matrix")
        m = [[0]]

    #print(m)

    if func == 1:
        a = m[0][0]

        k = -a
        return np.array([1,k,0]), i

    if func == 2:
        a = m[0][0]
        sigma = np.sqrt(np.divide(-1,2*a)) #sigma
        return np.array([sigma,0,sigma*np.sqrt(2*np.pi)]), i

    if func == 3:
        a = m[0][0]
        k = -a
        return np.array([1,k,0]), i

    if func == 4:
        a = m[0][0]
        k = -a
        A = np.hstack([x[1:]**2])
        ATA = np.transpose(A) @ A
        ly = np.power(y[1:],-1/k) - 1
        b = np.transpose(A) @ ly

        try:
            n = np.linalg.inv(ATA) @ b # Computes 'inv(A^T A) A^T y' efficiently
        except np.linalg.LinAlgError:
            print(f"Unable to compute function {func}: Singular Matrix")
            n = [[1]]
        
        if n[0][0] < 0:
            print('Warning: Fixed negative exponent!')
            n[0][0] = -n[0][0]

        L1 = np.sqrt(n[0][0])

        return np.array([1,L1,k]), i

    if func == 5:
        b = m[1][0]
        a = m[0][0]

        k = a#; print(f"k = {k}")
        L1 = np.exp(b/a)#; print(f"L = {1/L1}")

        return np.array([1,L1,k]), i


    return np.array([0,0,0]), i


def plot_acf2(auflength, funcTypes, plotdata, xmax = 5, block = True, sectors = 3, saveas = None, plotshow = True):

    #plt.figure()
    fig, axx = plt.subplots(1, sectors)

    kx = np.size(funcTypes) + 2

    colscheme = ['r-','b-','m-','c-','y-']
    for i in range(sectors):
        x = plotdata[kx*i]
        y = plotdata[kx*i + 1]
        ax = axx[i]
        ax.plot(x, y, 'k.', label="ACF")
        for j, fnc in enumerate(funcTypes):
            fy = plotdata[kx*i + j + 2]
            ax.plot(x, fy ,colscheme[j], label=fnc, lw = 0.75)
        
        ax.grid(True)
        ax.set_xlabel("Length [mm]")
        ax.set_ylabel("ACF")
        ax.set_title(f"Section {i+1}")
        ax.set_ylim([-0.02, 1])
        ax.set_xlim([-0.02*xmax, xmax])
        ax.legend()
    fig.set_figheight(4)
    fig.set_figwidth(4*sectors)
    if saveas != None:
        savename = str("plotimg/" + saveas + ".png")
        plt.savefig(savename,dpi=300,format="png")
    if plotshow:
        plt.show(block = block)
    else:
        plt.close()




