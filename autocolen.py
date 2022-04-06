import numpy as np

def autocolen(acf,leng=1):
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
        n = n[0] 
        print("m er {m:.3f}, n er {n}")

    return n*leng