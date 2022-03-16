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

    return n[0]*leng