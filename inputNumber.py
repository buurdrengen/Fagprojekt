# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:25:29 2022

@author: Mikkel N. Schmidt
"""

def inputNumber(prompt):
    # The function prompts the user to input a valid number
    #
    # The function will repeat until the user inputs a number
    while True:
        try:
            num = float(input(prompt))
            break
        except ValueError:
            pass
    return num


