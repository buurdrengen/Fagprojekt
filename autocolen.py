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
    elif n >=3 and n <= (np.size(acf)-3):
        #dx = 1
        #dy = acf[n] - acf[n-1]
        print("s")

    return n[0]*leng