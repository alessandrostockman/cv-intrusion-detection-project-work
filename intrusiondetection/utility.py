import numpy as np

# Distance Functions

def distance_manhattan(img1, img2):
    ''' returns a matrix of floats64 that represents the manhattan distance between the two images
    '''
    return np.abs(img2.astype(float) - img1.astype(float))

def distance_euclidean(img1, img2):
    ''' returns a matrix of floats64 that represents the euclidean distance between the two images
    '''
    return np.sqrt((img2.astype(float) - img1.astype(float)) ** 2)
