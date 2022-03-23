# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:50:02 2022

@author: Mikkel N. Schmidt + Viktor Isaksen
"""

import numpy as np
from inputNumber import inputNumber

def displayMenu(options):
    # displayMenu displays a menu of options, asks the user to choose an item and
    # returns the option
    # Display menu
    for i in range(len(options)):
        print("{:d}. {:s}".format(i + 1, options[i]))
    # Get a valid menu choice
    choice = 0
    while not(np.any(choice == np.arange(len(options))+1)):
        choice = int(inputNumber("Please choose a menu item: "))
    return options[choice - 1]



