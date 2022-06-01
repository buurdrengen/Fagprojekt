# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:38:29 2022

@author: Viktor Isaksen
"""

import numpy as np
import skimage.io


def inputFilename():
    # FUNCTION: This function aims to get a filepath from the user. The file 
    # can either be located inside the cwd or outside. If the file is in 
    # the cwd the user only has to input the filename and the function will 
    # check if the file is in the cwd. If the file is not in the cwd the 
    # function will require both a file directory as well as a filename, to 
    # see wether or not such a file exists within the specified directory.
    while True:
        # The file is not in the cwd
        filePath = str(input("Please enter the relative file path: (use / instead of \): "))
        try:
            skimage.io.imread(fname=filePath, as_gray=True)
            break
        except:
            print("The file was not found. Please check that the path and formatting is correct .")
            pass
            
             
    return filePath

# print(inputFilename())
