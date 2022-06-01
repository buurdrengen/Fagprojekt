
import os
import numpy as np

files = os.getcwd() + '\\variables'
images = os.getcwd() + '\\images'

print('')
print(files)
print(images)
print(' ')

[yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength, funcType] = [1,1,1,1,1,1,1,[1,1,1],['','','']]

variables = [yMiddle, xMiddle, marginY, marginX, conversion, blur, threshold, auflength[0], \
    auflength[1], auflength[2], funcType]


for filename in os.listdir(files):
    with open(os.path.join(files, filename), 'r') as f: # open in readonly mode
        # print(np.arange(len(variables)))
        
        for i in np.arange(len(variables)):
            variables[i] = f.readline()
            variables[i] = variables[i][0:-1]
    
    # with open(os.path.join(images, filename), 'r') as g:
    #     print(g)


print(os.listdir(files))
        