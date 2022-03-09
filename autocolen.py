from re import T


import numpy as np

def autocolen(acf,leng=1):
    """
    Beregner Autokorrelationslængden
    acf er en beregnet vektor af ACF
    leng er længden af billedet i enhed [længde per pixel], valgfri
    """
    t = np.exp(-1)
    N = np.arange(np.size(acf))
    
    n = N[autocolen <= t][0]

    return n*leng