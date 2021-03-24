import numpy as np
from matplotlib import pyplot as plt

# Distance Functions

def distance_manhattan(img1, img2):
    ''' returns a matrix of floats64 that represents the manhattan distance between the two images
    '''
    return np.abs(img2.astype(float) - img1.astype(float))

def distance_euclidean(img1, img2):
    ''' returns a matrix of floats64 that represents the euclidean distance between the two images
    '''
    return (np.sqrt((img2.astype(float) - img1.astype(float)) ** 2)).astype(np.uint8)

# Plotting Functions

def subplot_images(image_dicts):
    plt.figure(figsize=(20, 10))
    for index, image_dict in enumerate(image_dicts):
        plt.subplot(1, len(image_dicts), index+1)
        plt.axis('off')
        if image_dict['title'] is None:
             image_dict['title'] = None
        image_dict['object'].display(image_dict['key'], title=image_dict['title'], show=False)
