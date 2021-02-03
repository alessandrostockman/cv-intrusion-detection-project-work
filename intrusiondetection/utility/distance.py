import numpy as np

# Distance Functions

def manhattan(img1, img2):
    ''' returns a matrix of floats64 that represents the manhattan distance between the two images
    '''
    return np.sum(np.abs(img2 - img1), axis=-1)

def euclidean(img1, img2):
    ''' returns a matrix of floats64 that represents the euclidean distance between the two images
    '''
    return np.sqrt(np.sum((img2 - img1) ** 2, axis=-1))

def maximum(img1, img2):
    '''returns a matrix of floats64 that represents the Chebyshev distance between the two images
    '''
    return np.max(img2 - img1, axis=-1)
