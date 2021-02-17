import numpy as np

# Distance Functions

def distance_manhattan(img1, img2):
    ''' returns a matrix of floats64 that represents the manhattan distance between the two images
    '''
    return np.abs(img2.astype(float) - img1.astype(float))

def distance_euclidean(img1, img2):
    ''' returns a matrix of floats64 that represents the euclidean distance between the two images
    '''
    #TODO Capire quale versione usare
    #return (np.sqrt((img2.astype(float) - img1.astype(float)) ** 2)).astype(np.uint8)
    
    img1 = np.tile(img1.astype(float)[:,:,np.newaxis], 3)
    img2 = np.tile(img2.astype(float)[:,:,np.newaxis], 3)
    return np.sqrt(np.sum((img2 - img1) ** 2, axis=-1))

